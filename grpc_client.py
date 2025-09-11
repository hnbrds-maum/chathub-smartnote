# -*- coding: utf-8 -*-
"""
세 RPC(EmbedDocument, SummarizeDocument, RagChat) 테스트용 비동기 클라이언트
"""
import sys
import random
sys.path.append("./proto")
import asyncio, pathlib, grpc, os
from proto import smart_notebook_pb2 as pb
from proto import smart_notebook_pb2_grpc as pb_grpc

SERVER_ADDR = "localhost:8085"
TEST_URL = "https://docling-project.github.io/docling/examples/export_multimodal/"

async def embed_test(stub):
    req = pb.EmbedRequest(
        documents=[
            pb.EmbedRequest.DocumentMeta(
                document_id="doc-001",
                file_path="test2.pdf",
            ),
            pb.EmbedRequest.DocumentMeta(
                document_id="doc-002",
                file_path="test3.pdf"
            )
        ],
        req_id=random.randint(0, 100),
        webhook_info=pb.WebhookInfo(
            endpoint="http://10.50.3.3:2903",
            notebook_id="notebook_id_example",
            jwt_token="jwt_token_example"
        )
    )
    resp = await stub.EmbedDocument(req)

    print(resp)


async def embed_test_url(stub, url):
    req = pb.EmbedRequest(
        document=pb.EmbedRequest.DocumentMeta(
            document_id="doc-001",
            file_url=url
        ),
        req_id=1
    )
    resp = await stub.EmbedDocument(req)
    print(resp)

async def summarize_test(stub):
    req = pb.SummarizeRequest(
        summaries=["doc-001"],
        req_id=2
    )
    resp = await stub.SummarizeDocument(req)
    print("[Summarize]\n", resp.result)
    #print("[Summarize-ERR]", resp.status.message)

async def ragchat_test(stub):
    req = pb.RagRequest(
        document_id=["8c81af3df9244445b950b7bcde1a1f47"],
        msg="안전 규정에 대해 알려줘",
        req_id=3
    )
    print("[RagChat] stream\n")
    async for chunk in stub.RagChat(req):
        txt_content = chunk.result.text
        seg = txt_content.text_segment or txt_content.full_text
        print(seg, end="", flush=True)

async def main():
    async with grpc.aio.insecure_channel(SERVER_ADDR) as ch:
        stub = pb_grpc.SmartNoteServiceStub(ch)
        await embed_test(stub)
        await summarize_test(stub)
        await ragchat_test(stub)

if __name__ == "__main__":
    asyncio.run(main())