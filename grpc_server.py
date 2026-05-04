# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
load_dotenv(".env")

import sys
import asyncio
import argparse
import json
import aiohttp

import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import AsyncIterator, Optional, Dict, Set, List, Any
import multiprocessing as mp

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from pathlib import Path

import grpc
from grpc_reflection.v1alpha import reflection

sys.path.append("./proto")
from proto import smart_notebook_pb2 as pb  # type: ignore
from proto import smart_notebook_pb2_grpc as pb_grpc  # type: ignore
from google.rpc import status_pb2, code_pb2

from langchain_openai import ChatOpenAI
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SmartNote")

from database.retriever import Retriever
from utils.browser import afetch_rendered_html, HTTPStatusError
from utils.emoji import generate_random_emoji
from agents.rag_agent import RagAgent
from agents.summary_agent import SummaryAgent

# ──────────────────────────────── 환경 설정

FILE_ROOT = str(os.getenv("FILE_UPLOAD_DIR", "/app/data"))
VECTORSTORE_ROOT = str(os.getenv("VECTORSTORE_DIR", "/vectorstore"))
OPENAI_MODEL_NAME = str(os.getenv("OPENAI_MODEL_NAME", "gpt-4.1"))
REMOVE_EMBEDDED_DOCUMENT = os.getenv("REMOVE_EMBEDDED_DOCUMENT").lower() in ('true', '1', 't')

# ──────────────────────────────── 전역 상태 (진행률/요청 인덱스/동시성)

# REQUEST_STATE: req_id -> { "docs": Set[str], "done": Set[str], "success": Set[str], "webhook": WebhookInfo }
REQUEST_STATE: Dict[int, Dict[str, Any]] = {}
# PROGRESS: req_id -> doc_id -> DocProgress
PROGRESS: Dict[int, Dict[str, pb.DocProgress]] = {}

# [수정] 동시에 실행될 수 있는 문서 처리 작업의 최대 수 (Semaphore로만 제어)
INFLIGHT_MAX = int(os.getenv("INFLIGHT_MAX", 8))
INFLIGHT_SEM = asyncio.Semaphore(INFLIGHT_MAX)

# [수정] I/O 작업과 파싱 작업을 위한 별도의 Executor
# I/O 작업은 스레드 수가 많아도 괜찮음
MAX_IO = int(os.getenv("MAX_IO", 8))
IO_SEM = asyncio.Semaphore(MAX_IO)

# 파서 동시 실행 상한 (CPU 코어 수에 맞추는 것이 좋음)
MAX_PARSERS = int(os.getenv("MAX_PARSERS", os.cpu_count() or 4))
# ProcessPoolExecutor를 전역으로 관리
CPU_EXECUTOR: Optional[ProcessPoolExecutor] = None

# 웹훅을 위한 aiohttp 세션
WEBHOOK_SESSION: Optional[aiohttp.ClientSession] = None
WEBHOOK_CONNECTOR_LIMIT = int(os.getenv("WEBHOOK_CONNECTOR_LIMIT", 50))
# [수정] 진행 중인 모든 백그라운드 작업을 추적하기 위한 Set
BACKGROUND_TASKS: Set[asyncio.Task] = set()

def _cleanup_req(req_id: int):
    """Clean up request status"""
    REQUEST_STATE.pop(req_id, None)
    PROGRESS.pop(req_id, None)
    logger.info(f"[{req_id}] Cleaned up request state.")

# ──────────────────────────────── gRPC Servicer 구현

def end_of_stream(req_id):
    return pb.RagResponse(
        req_id=req_id, 
        msg_role=pb.RagResponse.MessageRole.MSG_ROLE_ANSWER,
        status=status_pb2.Status(code=code_pb2.OK),
        result=pb.MessageContent(
            message_type=pb.MessageContent.MessageType.MSG_TYPE_UNKNOWN,
            text=pb.TextContent(
                text_segment="",
                sequence_index=0,
                end_of_stream=True
            )
        )
    )

def _set_progress(req_id: int, doc_id: str, stage: int, msg: str = ""):
    if req_id not in PROGRESS:
        PROGRESS[req_id] = {}
    dp = PROGRESS[req_id].get(doc_id) or pb.DocProgress(document_id=doc_id)
    dp.stage = stage
    if msg:
        dp.message = msg
    PROGRESS[req_id][doc_id] = dp
    

async def _send_webhook(session: aiohttp.ClientSession, url: str, token: str, payload: dict, timeout: int = 15):
    """웹훅 전송 로직을 통합하고 단순화합니다."""
    if not url:
        return
    headers = {
        "Content-Type": "application/json",
        "Authorization": token
    }
    try:
        async with session.post(url, data=json.dumps(payload), headers=headers, timeout=timeout) as resp:
            resp.raise_for_status()
            logger.info(f"Webhook to {url} sent successfully with status {resp.status}.")
            await resp.text()
    except aiohttp.ClientError as e:
        logger.error(f"[webhook] POST to {url} failed: {e}")
    except Exception as e:
        logger.error(f"[webhook] An unexpected error occurred during webhook POST to {url}: {e}")


def _parse_and_chunk_document(tmp_path_str: str, doc_id: str):
    """
    동기적으로 실행될 파싱 및 청킹 작업.
    자식 프로세스에서 발생하는 모든 예외를 처리하여 BrokenProcessPool을 방지합니다.
    Path 객체 대신 문자열을 인자로 받아 직렬화 문제를 방지합니다.
    """
    try:
        from database.document_parser import DocumentParser

        tmp_path = Path(tmp_path_str)
        parser = DocumentParser(tmp_path, document_id=doc_id)
        logger.info(f"Parsed document {tmp_path} ({doc_id})")
        markdown = parser.get_markdown()
        if not markdown:
            return None, None, "Document has no text"
        chunks = parser.get_chunk(markdown)
        return markdown, chunks, None  # 성공: (결과1, 결과2, 에러 없음)
    except AttributeError as ae:
        # docling의 OCR 관련 오류인지 확인
        if "'NoneType' object has no attribute" in str(ae):
            error_message = "OCR failed to detect any text in the document."
            print(f"INFO: Parser worker for doc '{doc_id}': {error_message}", file=sys.stderr, flush=True)
            return None, None, error_message
        else:
            # 그 외 다른 AttributeError
            import traceback
            print(f"CRITICAL: Parser worker failed for doc '{doc_id}': {ae}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            return None, None, str(ae)
    except Exception as e:
        import traceback
        # 자식 프로세스의 에러는 메인 로거에 기록되지 않을 수 있으므로 stderr에 직접 출력
        print(f"CRITICAL: Parser worker failed for doc '{doc_id}': {e}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        return None, None, str(e)


def _build_faiss_index(chunks: List[Any], index_path_str: str):
    """
    동기적으로 실행될 FAISS 인덱스 빌드 작업.
    자식 프로세스에서 발생하는 모든 예외를 처리하여 BrokenProcessPool을 방지합니다.
    """
    try:
        index_path = Path(index_path_str)
        Retriever.build_faiss(chunks, index_path)
        return None  # 성공: 에러 없음
    except Exception as e:
        import traceback
        print(f"CRITICAL: FAISS worker failed for index '{index_path_str}': {e}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        return str(e)


async def run_one_document(req_id: int, doc: pb.EmbedRequest.DocumentMeta, webhook: Optional[pb.WebhookInfo]):
    doc_id = doc.document_id
    tmp_path = None
    try:
        _set_progress(req_id, doc_id, pb.STAGE_PARSING, "parsing scheduled")
        logger.info(f"[{req_id}/{doc_id}] Scheduled for processing.")
        
        loop = asyncio.get_running_loop()

        # 1. 입력 파일 준비 (Async I/O)
        async with IO_SEM:
            if doc.HasField("file_url"):
                html = await afetch_rendered_html(doc.file_url)
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as fp:
                    fp.write(html)
                    tmp_path = Path(fp.name)
            elif doc.HasField("file_path"):
                tmp_path = Path(FILE_ROOT) / doc.file_path
            elif doc.HasField("file_data"):
                suffix = doc.suffix or ".bin"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fp:
                    fp.write(doc.file_data)
                    tmp_path = Path(fp.name)
            else:
                raise ValueError("No file source (file_url, file_path, file_data) provided")

        # 2. 파싱/청킹 (CPU-Bound -> ProcessPoolExecutor)
        _set_progress(req_id, doc_id, pb.STAGE_PARSING, "parsing")
        logger.info(f"[{req_id}/{doc_id}] Parsing and chunking in CPU executor...")
        # [수정] ProcessPoolExecutor에서 동기 함수 실행
        markdown, chunks, parse_error = await loop.run_in_executor(
            CPU_EXECUTOR, _parse_and_chunk_document, str(tmp_path), doc_id
        )
        if parse_error:
            raise RuntimeError(f"Parsing failed in worker: {parse_error}")

        # 3. FAISS 인덱싱 (CPU-Bound -> ProcessPoolExecutor)
        _set_progress(req_id, doc_id, pb.STAGE_INDEXING, "indexing")
        logger.info(f"[{req_id}/{doc_id}] Building FAISS index in CPU executor...")
        index_path = Path(VECTORSTORE_ROOT) / doc_id
        index_error = await loop.run_in_executor(
            CPU_EXECUTOR, _build_faiss_index, chunks, str(index_path)
        )
        if index_error:
            raise RuntimeError(f"Indexing failed in worker: {index_error}")

        # 4. 요약 (Network I/O)
        _set_progress(req_id, doc_id, pb.STAGE_SUMMARIZING, "summarizing")
        logger.info(f"[{req_id}/{doc_id}] Summarizing with LLM...")
        llm = ChatOpenAI(model=OPENAI_MODEL_NAME)
        summary = await SummaryAgent(llm).async_run([x.content for x in markdown])
        summary_text = summary.get("summary", "")
        
        _set_progress(req_id, doc_id, pb.STAGE_SUCCEEDED, "ok")
        
        # [수정] 성공한 문서 목록에 추가
        if req_id in REQUEST_STATE:
            REQUEST_STATE[req_id]["success"].add(doc_id)

        # 5. 개별 문서 완료 웹훅
        if webhook and webhook.endpoint and WEBHOOK_SESSION:
            markdown_sections_json = [
                {"sectionId": getattr(x, "id", str(i)), "header": getattr(x, "header", ""), "content": getattr(x, "content", "")}
                for i, x in enumerate(markdown)
            ]
            payload = {
                "status": "EMBEDDED",
                "markdowns": markdown_sections_json,
                "summary": summary_text
            }
            url = f"{webhook.endpoint.rstrip('/')}/notebooks/{webhook.notebook_id}/sources/{doc_id}/markdowns"
            await _send_webhook(WEBHOOK_SESSION, url, webhook.jwt_token, payload)

    except asyncio.CancelledError:
        _set_progress(req_id, doc_id, pb.STAGE_FAILED, "canceled")
        logger.warning(f"[{req_id}/{doc_id}] Task was cancelled.")
    except Exception as e:
        logger.exception(f"[{req_id}/{doc_id}] Failed to process document")
        _set_progress(req_id, doc_id, pb.STAGE_FAILED, f"error: {e}")
        # 실패 웹훅
        if webhook and webhook.endpoint and WEBHOOK_SESSION:
            payload = {"status": "EMBEDDED_FAILED"}
            url = f"{webhook.endpoint.rstrip('/')}/notebooks/{webhook.notebook_id}/sources/{doc_id}/markdowns"
            await _send_webhook(WEBHOOK_SESSION, url, webhook.jwt_token, payload)
    finally:
        # 6. 완료 처리 및 최종 웹훅 체크
        if req_id in REQUEST_STATE:
            state = REQUEST_STATE[req_id]
            state["done"].add(doc_id)
            # 모든 문서 작업이 완료되었는지 확인
            if state["docs"] == state["done"]:
                logger.info(f"[{req_id}] All documents processed. Sending final webhook.")
                final_webhook = state.get("webhook")
                if final_webhook and final_webhook.endpoint and WEBHOOK_SESSION:
                    payload = {"sourceIds": list(state["success"])}
                    url = f"{webhook.endpoint.rstrip('/')}/notebooks/{webhook.notebook_id}/initial-summary"
                    await _send_webhook(WEBHOOK_SESSION, url, webhook.jwt_token, payload)
                # 모든 작업 완료 후 상태 정리
                _cleanup_req(req_id)

        # 임시 파일 정리
        if REMOVE_EMBEDDED_DOCUMENT and tmp_path and tmp_path.exists():
            try:
                os.remove(tmp_path)
            except OSError as e:
                logger.error(f"Failed to remove temp file {tmp_path}: {e}")


async def _ingress_worker(req_id: int, d: pb.EmbedRequest.DocumentMeta, webhook: Optional[pb.WebhookInfo]):
    """하나의 문서를 처리하는 비동기 작업자"""
    try:
        async with INFLIGHT_SEM:
            await run_one_document(req_id, d, webhook)
    except Exception as e:
        logger.error(f"Unhandled exception in ingress worker for req_id={req_id}, doc_id={d.document_id}: {e}")


class SmartNoteService(pb_grpc.SmartNoteServiceServicer):
    """Single global GRAPH를 사용. 세션에는 cfg·interrupted 상태만 보관."""

    def __init__(self):
        self.sessions: dict[int, dict] = {}

    # util
    @staticmethod
    def _now_label(name: str):
        logging.info(f"[{datetime.now():%H:%M:%S.%f}] {name}")

    async def EmbedDocument(
        self,
        request: pb.EmbedRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.ProgressResponse:
        print(request)

        req_id = request.req_id
        docs = list(request.documents)
        
        if req_id in REQUEST_STATE:
            return pb.ProgressResponse(
                req_id=req_id,
                status=status_pb2.Status(code=code_pb2.ALREADY_EXISTS, message=f"Request ID {req_id} is already being processed."),
            )

        if not docs:
            return pb.ProgressResponse(
                req_id=req_id,
                status=status_pb2.Status(code=code_pb2.INVALID_ARGUMENT, message="no documents"),
            )

        REQUEST_STATE[req_id] = {
            "docs": {d.document_id for d in docs},
            "done": set(),
            "success": set(),
            "webhook": request.webhook_info
        }
        PROGRESS[req_id] = {}

        for d in docs:
            _set_progress(req_id, d.document_id, pb.STAGE_QUEUED, "queued")
            task = asyncio.create_task(_ingress_worker(req_id, d, request.webhook_info))
            BACKGROUND_TASKS.add(task)
            task.add_done_callback(BACKGROUND_TASKS.discard)

        logger.info(f"[{req_id}] Accepted {len(docs)} documents for processing.")
        return pb.ProgressResponse(
            req_id=req_id,
            status=status_pb2.Status(code=code_pb2.OK, message=f"accepted {len(docs)} docs"),
            progresses=list(PROGRESS.get(req_id, {}).values()),
        )
    
    
    async def EmbedProgress(
            self,
            request: pb.ProgressRequest,
            context: grpc.aio.ServicerContext,
    ) -> pb.ProgressResponse:
        req_id = request.req_id
        pmap = PROGRESS.get(req_id, {})
        if not pmap:
            return pb.ProgressResponse(
                req_id=req_id,
                status=status_pb2.Status(code=code_pb2.OK, message="no progress"),
                progresses=[]
            )
        items = [pmap[d] for d in request.document_ids if d in pmap] if request.document_ids else list(pmap.values())
        return pb.ProgressResponse(
            req_id=req_id,
            status=status_pb2.Status(code=code_pb2.OK),
            progresses=items
        )


    async def SummarizeDocument(
        self,
        request: pb.SummarizeRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.SummarizeResponse:

        try:
            llm = ChatOpenAI(model=OPENAI_MODEL_NAME)
            summary = await SummaryAgent(llm).async_run(
                list(request.summaries), single_document=False
            )

            return pb.SummarizeResponse(
                req_id=request.req_id,
                status=status_pb2.Status(code=code_pb2.OK),
                result=summary['summary'],
                title=summary['title'],
                emoji=generate_random_emoji()
            )

        except Exception as e:
            logging.exception("SummarizeDocument failed")
            _status = status_pb2.Status(code=code_pb2.ABORTED, message=str(e))
            return pb.SummarizeResponse(req_id=request.req_id, status=_status)

    
    async def RagChat(
        self,
        request: pb.RagRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb.RagResponse]:

        try:
            llm = ChatOpenAI(model=OPENAI_MODEL_NAME)
            index_paths = [os.path.join(VECTORSTORE_ROOT, x) for x in list(request.document_id)]
            answer = await RagAgent(llm, index_paths).async_run(request.msg)

            use_web_search = bool(request.use_web_search)
            answer = await RagAgent(llm, index_paths).async_run(request.msg, use_web_search)
            yield pb.RagResponse(
                req_id=request.req_id, 
                msg_role=pb.RagResponse.MessageRole.MSG_ROLE_ANSWER,
                status=status_pb2.Status(code=code_pb2.OK),
                result=pb.MessageContent(
                    message_type=pb.MessageContent.MessageType.MSG_TYPE_TEXT,
                    text=pb.TextContent(
                        full_text=answer['final_answer'], 
                        sequence_index=0,
                        end_of_stream=False
                    )
                ),
                title=answer['title']
            )

        except RuntimeError as e:
            import traceback
            _status = status_pb2.Status(
                code=code_pb2.ABORTED,
                message=f"Error during loading vectorstore (document_id : {list(request.document_id)})"
            )
            yield pb.RagResponse(
                req_id=request.req_id,
                status=_status
            )           

        except Exception as e:
            import traceback
            _status = status_pb2.Status(
                code=code_pb2.ABORTED,
                message=str(e)
            )
            yield pb.RagResponse(
                req_id=request.req_id,
                status=_status
            )

        finally:
            yield end_of_stream(request.req_id)


# ──────────────────────────────── gRPC 서버 부트스트랩

async def serve(port: int = 8085, max_workers: int = 32):
    global WEBHOOK_SESSION, CPU_EXECUTOR
    
    # [수정] CPU 집약적 작업을 위한 ProcessPoolExecutor 초기화
    ctx = mp.get_context("spawn")
    CPU_EXECUTOR = ProcessPoolExecutor(max_workers=MAX_PARSERS, mp_context=ctx)
    
    
    server = grpc.aio.server(
        ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 60_000),
            ("grpc.keepalive_timeout_ms", 20_000),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_ping_strikes", 0),
            ("grpc.http2.max_pings_without_data", 0)
        ],
    )
    pb_grpc.add_SmartNoteServiceServicer_to_server(SmartNoteService(), server)

    SERVICE_NAMES = (
        pb.DESCRIPTOR.services_by_name["SmartNoteService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    WEBHOOK_SESSION = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=WEBHOOK_CONNECTOR_LIMIT)
    )

    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    logger.info(f"Smart Notebook service started on PORT:{port} with {MAX_PARSERS} parser processes.")
    
    try:
        await server.wait_for_termination()
    except asyncio.CancelledError:
        logger.info("Server shutdown requested.")
    finally:
        logger.info("Starting graceful shutdown...")
        await server.stop(grace=10) # 10초의 유예 기간
        
        # 진행 중인 백그라운드 작업 취소
        if BACKGROUND_TASKS:
            logger.info(f"Cancelling {len(BACKGROUND_TASKS)} outstanding tasks.")
            for task in BACKGROUND_TASKS:
                task.cancel()
            await asyncio.gather(*BACKGROUND_TASKS, return_exceptions=True)

        if WEBHOOK_SESSION and not WEBHOOK_SESSION.closed:
            await WEBHOOK_SESSION.close()
            logger.info("Webhook session closed.")
        
        if CPU_EXECUTOR:
            CPU_EXECUTOR.shutdown(wait=True)
            logger.info("CPU executor shutdown.")
        logger.info("Shutdown complete.")


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8085)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--dev", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # .env 파일 결정
    if args.dev:
        os.environ["ENV"] = "dev"
    env_map = {
        "dev": ".env.dev",
        "qa": ".env.staging",
        "prod": ".env.production",
        "docker": ".env.docker",
    }

    asyncio.run(serve(port=args.port))
