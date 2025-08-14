# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
load_dotenv(".env")
import asyncio
import argparse
import tempfile
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import AsyncIterator, Optional
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

# ──────────────────────────────── 환경 설정

FILE_ROOT = str(os.getenv("FILE_UPLOAD_DIR"))
VECTORSTORE_ROOT = str(os.getenv("VECTORSTORE_DIR"))
OPENAI_MODEL_NAME = str(os.getenv("OPENAI_MODEL_NAME", "gpt-4.1"))
REMOVE_EMBEDDED_DOCUMENT = os.getenv("REMOVE_EMBEDDED_DOCUMENT").lower() in ('true', '1', 't')

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

class SmartNoteService(pb_grpc.SmartNoteServiceServicer):
    """Single global GRAPH를 사용. 세션에는 cfg·interrupted 상태만 보관."""

    def __init__(self):
        self.sessions: dict[int, dict] = {}

    # util
    @staticmethod
    def _now_label(name: str):
        print(f"[{datetime.now():%H:%M:%S.%f}] {name}")

    # DeepResearch bi‑directional streaming
    async def EmbedDocument(
        self,
        request: pb.EmbedRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.EmbedResponse:
        loop = asyncio.get_running_loop()
        doc = request.document

        try:
            llm = ChatOpenAI(model=OPENAI_MODEL_NAME)

            if doc.HasField("file_url"):
                # gRPC string -> read from url
                with tempfile.NamedTemporaryFile(
                    suffix=".html", delete=False, mode="w", encoding="utf-8"
                ) as fp:
                    html = await afetch_rendered_html(doc.file_url)
                    tmp_path = Path(tempfile.NamedTemporaryFile(
                            suffix=".html", delete=False, mode="w", encoding="utf-8").name)
                    tmp_path.write_text(html, encoding="utf-8") 

            elif doc.HasField("file_path"):
                # gRPC string -> read from docker volume
                tmp_path = Path(FILE_ROOT) / doc.file_path

            elif doc.HasField("file_data"):
                # gRPC bytes
                suffix = doc.suffix or ".bin"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fp:
                    fp.write(doc.file_data)
                    tmp_path = Path(fp.name)
            else:
                raise ValueError("No file_* field is provided")

            parser = DocumentParser(tmp_path)
            markdown = await asyncio.to_thread(parser.get_markdown)
            if not markdown:
                _status = status_pb2.Status(
                    code=code_pb2.ABORTED,
                    message="[ERROR] Document has not text"
                )
                return pb.EmbedResponse(
                    req_id=request.req_id,
                    status=_status
                )

            chunks = await asyncio.to_thread(parser.get_chunk, markdown)
            index_path = Path(VECTORSTORE_ROOT) / doc.document_id

            build_faiss_task = asyncio.to_thread(
                Retriever.build_faiss, chunks, index_path
            )

            summary_task = SummaryAgent(llm).async_run([x.content for x in markdown])
            summary_result, _ = await asyncio.gather(summary_task, build_faiss_task)

            if REMOVE_EMBEDDED_DOCUMENT and tmp_path and tmp_path.exists():
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            sections = [
                pb.MarkdownSection(id=x.id, header=x.header, content=x.content) \
                for x in markdown
            ]
            result = pb.EmbedResult(
                document_id=doc.document_id,
                sections=sections,
                summary=summary_result.get("summary", ""),
            )
            return pb.EmbedResponse(
                req_id=request.req_id,
                status=status_pb2.Status(code=code_pb2.OK),
                result=result
            )

        # ERROR Handling
        except HTTPStatusError as e:
            # browser url parse error
            _status = status_pb2.Status(
                code=code_pb2.ABORTED,
                message=str(f"URL 요청 실패 : {e.url} / {e.status}")
            )
            return pb.EmbedResponse(
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
            return pb.EmbedResponse(
                req_id=request.req_id,
                status=_status
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
