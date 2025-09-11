import os
from typing import Optional, List, Dict
from collections import defaultdict

from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

RRF_K = 60

class Retriever:
    """FAISS + InMemoryDocstore 기반 small-to-big 검색기."""

    def __init__(self, indices:List[str]):
        self.vector_db: FAISS = self._load_faiss(indices)
        self.docstore: InMemoryDocstore = self.vector_db.docstore
        self.bm25 = self._build_bm25()
        self.prev_to_curr: Dict[str, str] = self._build_prev_to_curr()

    @staticmethod
    def _load_faiss(indices: List[str]) -> FAISS:
        """디스크에 저장된 FAISS + Docstore 로드."""
        db = FAISS.load_local(
            indices[0],
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
        )
        for index in indices[1:]:
            db.merge_from(
                FAISS.load_local(
                    index,
                    OpenAIEmbeddings(),
                    allow_dangerous_deserialization=True
                )
            )
        return db
    

    def _build_bm25(self):
        texts = [doc.page_content for doc in self.docstore._dict.values()]
        metas = [doc.metadata for doc in self.docstore._dict.values()]
        return BM25Retriever.from_texts(texts=texts,
                                        metadatas=metas,
                                        ids=list(self.docstore._dict.keys()))
        

    def _build_prev_to_curr(self) -> Dict[str, str]:
        """
        역방향(prev_id → curr_id) 매핑 한 번만 만들어 둔다.
        previous_id 가 없는(=첫 청크) 경우는 건너뜀.
        """
        mapping: Dict[str, str] = {}
        for curr_id, doc in self.docstore._dict.items():
            prev_id: Optional[str] = doc.metadata.get("previous_id")
            if prev_id:                       # None 이면 문서 첫 페이지
                mapping[prev_id] = curr_id    # 하나의 prev_id 에 분기가 없다면 단일 매핑
        return mapping


    def _collect_neighbors(
        self,
        base_doc: Document,
        window: int,
    ) -> List[Document]:
        """
        기준 청크(base_doc)를 중심으로 앞/뒤 window 만큼 인접 청크 수집.
        * 뒤(next) : self.prev_to_curr 사전 활용
        * 앞(prev) : previous_id 체인 활용
        """
        neighbors: List[Document] = []

        # 뒤쪽(next) 확장
        cur_id, steps = base_doc.metadata["chunk_id"], 0
        while cur_id in self.prev_to_curr and steps < window:
            next_id = self.prev_to_curr[cur_id]
            next_doc = self.docstore.search(next_id)
            if next_doc is None:
                break
            neighbors.append(next_doc)
            cur_id = next_id
            steps += 1

        # 앞쪽(previous) 확장
        prev_id, steps = base_doc.metadata.get("previous_id"), 0
        while prev_id and steps < window:
            prev_doc = self.docstore.search(prev_id)
            if prev_doc is None:
                break
            neighbors.append(prev_doc)
            prev_id = prev_doc.metadata.get("previous_id")
            steps += 1

        return neighbors

    
    def _merge_documents(self, docs: List[Document]) -> Dict:
        """
        여러 Document를 하나의 dict(content, metadata)로 병합.
        • content  : '\n\n' 으로 이어붙임
        • metadata : 문서 이름, 헤더, 페이지 범위 등 집계
        """
        if not docs:
            return {"content": "", "metadata": {}}

        # 정렬 (page_start 기준)
        docs_sorted = sorted(docs, key=lambda d: d.metadata.get("start_page", 0))

        content = "\n\n".join(d.page_content for d in docs_sorted)

        meta_first = docs_sorted[0].metadata
        meta_last = docs_sorted[-1].metadata

        merged_meta = {
            "document_id": meta_first.get("document_id"),
            "heading_id": meta_first.get("heading_id"),
            "chunk_type": meta_first.get("chunk_type"),
            "file_path": meta_last.get("file_path"),
        }
        return {"content": content, "metadata": merged_meta}

    @staticmethod
    def _rrf_fuse(rank_lists: List[List[str]], k: int = RRF_K) -> List[str]:
        """Reciprocal Rank Fusion → ID 순서 리스트 반환"""
        score = defaultdict(float)
        for lst in rank_lists:
            for rank, did in enumerate(lst, 1):
                score[did] += 1.0 / (k + rank)
        return [did for did, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)]

    def retrieve_small_to_big(
        self,
        query: str,
        top_k: int = 3,
        window: int = 2,
    ) -> List[Document]:
        """
        Hybrid RAG:
        1. Dense(FAISS)·Sparse(BM25) 각 top_k 검색
        2. RRF 로 랭크 융합
        3. 각 선정 청크 → window 이웃 확장 → 병합 결과 반환
        """
        dense_docs = self.vector_db.similarity_search(query, k=top_k)
        sparse_docs = self.bm25.invoke(query, k=top_k)

        dense_ids = [d.metadata["chunk_id"] for d in dense_docs]
        sparse_ids = [d.metadata["chunk_id"] for d in sparse_docs]

        fused_ids = self._rrf_fuse([dense_ids, sparse_ids])
        fused_ids = fused_ids[:top_k]

        results = []
        seen = set()
        for cid in fused_ids:
            base_doc = self.docstore.search(cid)
            if not base_doc or cid in seen:
                continue
            neighbors = self._collect_neighbors(base_doc, window)
            merged = self._merge_documents([base_doc, *neighbors])
            results.append(merged)
            seen.update(d.metadata["chunk_id"] for d in [base_doc, *neighbors])

        return results
    
    @staticmethod
    def build_faiss(docs, index_path: str = "faiss_index"):
        ids = [d.metadata["chunk_id"] for d in docs]
        vs = FAISS.from_documents(
            documents=docs,
            embedding=OpenAIEmbeddings(),
            ids=ids 
        )
        vs.save_local(index_path)
        return vs