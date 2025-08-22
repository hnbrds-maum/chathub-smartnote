# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
load_dotenv(".env")

import sys
import asyncio
import argparse
import json
import aiohttp
from urllib.parse import urljoin
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import AsyncIterator, Optional, Dict, Set, List
from pathlib import Path

import grpc
from grpc_reflection.v1alpha import reflection

sys.path.append("./proto")
from proto import smart_notebook_pb2 as pb  # type: ignore
from proto import smart_notebook_pb2_grpc as pb_grpc  # type: ignore
from google.rpc import status_pb2, code_pb2

from langchain_openai import ChatOpenAI

from database.document_parser import DocumentParser
from database.retriever import Retriever
from utils.browser import afetch_rendered_html, HTTPStatusError
from utils.emoji import generate_random_emoji
from agents.rag_agent import RagAgent
from agents.summary_agent import SummaryAgent

from scheduler_drr import DRRScheduler, DocTask

# ──────────────────────────────── 환경 설정

FILE_ROOT = str(os.getenv("FILE_UPLOAD_DIR"))
VECTORSTORE_ROOT = str(os.getenv("VECTORSTORE_DIR"))
OPENAI_MODEL_NAME = str(os.getenv("OPENAI_MODEL_NAME", "gpt-4.1"))
REMOVE_EMBEDDED_DOCUMENT = os.getenv("REMOVE_EMBEDDED_DOCUMENT").lower() in ('true', '1', 't')

# ──────────────────────────────── 전역 상태 (진행률/요청 인덱스/동시성)

# 진행률: req_id -> doc_id -> DocProgress
PROGRESS: Dict[int, Dict[str, pb.DocProgress]] = {}

# 요청 단위 인덱스/완료 추적/웹훅
REQ_INDEX: Dict[int, Set[str]] = {}         # 해당 req에 포함된 모든 doc_id
REQ_DONE: Dict[int, Set[str]] = {}          # 완료된 doc_id
REQ_WEBHOOK: Dict[int, pb.WebhookInfo] = {} # 웹훅 정보

# 파서 동시 실행 상한
MAX_PARSERS = int(os.getenv("MAX_PARSERS", 4))
parser_semaphore = asyncio.Semaphore(MAX_PARSERS)
# DRR 스케줄러
scheduler = DRRScheduler(quantum=1, aging_sec=8.0, aging_bonus=1)

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
    

async def _send_embed_webhook(webhook: Optional[pb.WebhookInfo], doc_id, payload: dict):
    """웹훅: Authorization Bearer + JSON. aiohttp가 있으면 사용, 없으면 표준라이브러리 fallback."""
    if not webhook or not webhook.endpoint:
        return
    headers = {
        "Content-Type": "application/json",
    }
    if webhook.jwt_token:
        headers["Authorization"] = f"Bearer {webhook.jwt_token}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(urljoin(base=webhook.endpoint,\
                                            url=f"/notebooks/{webhook.notebook_id}/sources/{doc_id}/markdowns"),
                                    data=json.dumps(payload),
                                    headers=headers,
                                    timeout=15) as resp:
                await resp.text()
    except Exception as e:
        # fallback (간단 로깅)
        print(f"[webhook] post failed: {e} -> will not retry here")


async def _send_done_webhook(webhook: Optional[pb.WebhookInfo], payload: dict):
    if not webhook or not webhook.endpoint:
        return
    headers = {
        "Content-Type": "application/json",
    }
    if webhook.jwt_token:
        headers["Authorization"] = f"Bearer {webhook.jwt_token}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(urljoin(base=webhook.endpoint, url=f"/notebooks/{webhook.notebook_id}/initial-summary"),
                                    data=json.dumps(payload),
                                    headers=headers,
                                    timeout=15) as resp:
                await resp.text()
    except Exception as e:
        # fallback (간단 로깅)
        print(f"[webhook] post failed: {e} -> will not retry here")


def _all_done(req_id: int) -> bool:
    return REQ_INDEX.get(req_id, set()) == REQ_DONE.get(req_id, set())


# ──────────────────────────────── 문서 하나 처리(워커)

async def run_one_document(job: DocTask):
    req_id = job.req_id
    doc = job.doc_meta
    webhook = job.webhook_info
    doc_id = doc.document_id

    markdown_sections_json = None
    summary_text = None

    try:
        _set_progress(req_id, doc_id, pb.STAGE_PARSING, "parsing scheduled")
        print(f"scheduled {req_id} / {doc_id}")

        # 입력 준비
        if doc.HasField("file_url"):
            html = await afetch_rendered_html(doc.file_url)
            tmp_path = Path(tempfile.NamedTemporaryFile(
                suffix=".html", delete=False, mode="w", encoding="utf-8").name)
            tmp_path.write_text(html, encoding="utf-8")
        elif doc.HasField("file_path"):
            tmp_path = Path(FILE_ROOT) / doc.file_path
        elif doc.HasField("file_data"):
            suffix = doc.suffix or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fp:
                fp.write(doc.file_data)
                tmp_path = Path(fp.name)
        else:
            raise ValueError("No file_* field is provided")

        # 파싱/인덱싱/요약: 동시성 5개 제한
        async with parser_semaphore:
            _set_progress(req_id, doc_id, pb.STAGE_PARSING, "parsing")
            parser = DocumentParser(tmp_path, document_id=doc_id)
            markdown = await asyncio.to_thread(parser.get_markdown)
            markdown_sections_json = [
                {
                    "sectionId": getattr(x, "id", str(i)),
                    "header": getattr(x, "header", ""),
                    "content": getattr(x, "content", ""),
                }
                for i, x in enumerate(markdown)
            ]

            if not markdown:
                raise RuntimeError("Document has no text")

            _set_progress(req_id, doc_id, pb.STAGE_CHUNKING, "chunking")
            chunks = await asyncio.to_thread(parser.get_chunk, markdown)

            _set_progress(req_id, doc_id, pb.STAGE_INDEXING, "indexing")
            index_path = Path(VECTORSTORE_ROOT) / doc_id
            await asyncio.to_thread(Retriever.build_faiss, chunks, index_path)

            _set_progress(req_id, doc_id, pb.STAGE_SUMMARIZING, "summarizing")
            llm = ChatOpenAI(model=OPENAI_MODEL_NAME)
            summary = await SummaryAgent(llm).async_run([x.content for x in markdown])
            summary_text = summary.get("summary", "")

        if REMOVE_EMBEDDED_DOCUMENT and tmp_path and tmp_path.exists():
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        _set_progress(req_id, doc_id, pb.STAGE_SUCCEEDED, "ok")
    except asyncio.CancelledError:
        _set_progress(req_id, doc_id, pb.STAGE_FAILED, "canceled")  # proto에 CANCELED 없음
    except Exception as e:
        _set_progress(req_id, doc_id, pb.STAGE_FAILED, f"error: {e}")
    finally:
        # per-document 완료 웹훅
        try:
            payload = {"status": "SOURCE_CREATED"}
            if markdown_sections_json is not None:
                payload["markdowns"] = markdown_sections_json
            if summary is not None:
                payload["summary"] = summary_text

            await _send_embed_webhook(webhook, doc_id, payload)
        except Exception as e:
            print(f"[webhook] per-doc failed: {e}")

        # 배치 완료 체크 & 최종 웹훅(여기는 기존처럼 메타만)
        REQ_DONE.setdefault(req_id, set()).add(doc_id)
        if _all_done(req_id):
            payload = {"sourceIds" : REQ_DONE.get(req_id, set())}
            try:
                await _send_done_webhook(webhook, payload)
            except Exception as e:
                print(f"[webhook] batch failed: {e}")


class SmartNoteService(pb_grpc.SmartNoteServiceServicer):
    """Single global GRAPH를 사용. 세션에는 cfg·interrupted 상태만 보관."""

    def __init__(self):
        self.sessions: dict[int, dict] = {}

    # util
    @staticmethod
    def _now_label(name: str):
        print(f"[{datetime.now():%H:%M:%S.%f}] {name}")

    async def EmbedDocument(
        self,
        request: pb.EmbedRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.ProgressResponse:

        req_id = request.req_id
        docs = list(request.documents)
        if not docs:
            return pb.ProgressResponse(
                req_id=req_id,
                status=status_pb2.Status(code=code_pb2.INVALID_ARGUMENT, message="no documents"),
                progresses=[]
            )

        # 요청 인덱스/웹훅 기록
        REQ_INDEX[req_id] = {d.document_id for d in docs}
        REQ_DONE[req_id] = set()
        REQ_WEBHOOK[req_id] = request.webhook_info

        # 초기 상태(QUEUED) 기록 & DRR enqueue
        for d in docs:
            _set_progress(req_id, d.document_id, pb.STAGE_QUEUED, "queued")
            scheduler.enqueue(req_id, DocTask(req_id=req_id, doc_meta=d, webhook_info=request.webhook_info))

        # 즉시 진행상태 반환(간소 LRO: 폴링은 EmbedProgress)
        return pb.ProgressResponse(
            req_id=req_id,
            status=status_pb2.Status(code=code_pb2.OK, message=f"accepted {len(docs)} docs"),
            progresses=list(PROGRESS[req_id].values()),
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
            import traceback
            traceback.print_exc()
            _status = status_pb2.Status(
                code=code_pb2.ABORTED,
                message=str(e)
            )
            return pb.SummarizeResponse(
                req_id=request.req_id,
                status=_status
            )

    
    async def RagChat(
        self,
        request: pb.RagRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb.RagResponse]:

        try:
            llm = ChatOpenAI(model=OPENAI_MODEL_NAME)
            index_paths = [os.path.join(VECTORSTORE_ROOT, x) for x in list(request.document_id)]
            answer = await RagAgent(llm, index_paths).async_run(request.msg)

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
                title=answer['title']#answer['search_queries'][0]
            )

        except RuntimeError as e:
            import traceback
            traceback.print_exc()
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
            traceback.print_exc()
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
    server = grpc.aio.server(
        ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            # 빠른 단절 감지를 위한 keep‑alive
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
    asyncio.create_task(scheduler.start(run_one_document))

    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    print(f"Smart Notebook service started on PORT:{port}")
    await server.wait_for_termination()


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
