# -*- coding: utf-8 -*-
import asyncio
import argparse
import base64
import json
import tempfile
import os
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import AsyncIterator, Optional
import pathlib
from pathlib import Path

import grpc
from grpc_reflection.v1alpha import reflection
from dotenv import load_dotenv

sys.path.append("./proto")
from proto import smart_notebook_pb2 as pb  # type: ignore
from proto import smart_notebook_pb2_grpc as pb_grpc  # type: ignore
from google.rpc import status_pb2, code_pb2

from document_parser import DocumentParser
from browser import afetch_rendered_html

# ──────────────────────────────── 환경 설정
load_dotenv("/workspace/.env")
FILE_ROOT = os.getenv("FILE_UPLOAD_DIR")

# ──────────────────────────────── gRPC Servicer 구현

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
                ValueError("No file_* field is provided")

            parser = DocumentParser(tmp_path)
            markdown = await loop.run_in_executor(None, parser.get_markdown)
            os.remove(tmp_path)

            result = pb.EmbedResult(
                document_id=doc.document_id,
                markdown=markdown,
                summary="document summary",
            )
            return pb.EmbedResponse(req_id=request.req_id,
                                    status=status_pb2.Status(code=code_pb2.OK),
                                    result=result)
        
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

        return pb.SummarizeResponse(
            req_id=request.req_id,
            status=status_pb2.Status(code=code_pb2.OK),
            result="예시 Response",
        )
    
    async def RagChat(
        self,
        request: pb.RagRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb.RagResponse]:
        from sample import SAMPLE_TEXT

        last_idx = len(SAMPLE_TEXT) -1
        for idx, char in enumerate(SAMPLE_TEXT):
            result = pb.MessageContent(
                    message_type=pb.MessageContent.MessageType.MSG_TYPE_TEXT,
                    text=pb.TextContent(
                            text_segment=char,
                            sequence_index=idx,
                            end_of_stream=(idx == last_idx),
                        ),
                    )
            yield pb.RagResponse(req_id=request.req_id, 
                                 msg_role=pb.RagResponse.MessageRole.MSG_ROLE_ANSWER,
                                 status=status_pb2.Status(code=code_pb2.OK),
                                 result=result)
            await asyncio.sleep(0.01)


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
            ("grpc.http2.max_pings_without_data", 0),
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
    load_dotenv(env_map.get(os.getenv("ENV"), ".env"))

    asyncio.run(serve(port=args.port))
