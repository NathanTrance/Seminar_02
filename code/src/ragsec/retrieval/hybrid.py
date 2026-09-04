import logging

from .base import RetrievedDoc
from .bm25 import BM25Retriever
from .dense import DenseRetriever

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedDoc]],
    k: int = 60,
    top_n: int = 5,
) -> list[RetrievedDoc]:
    scores: dict[str, tuple[float, RetrievedDoc]] = {}
    for rank_list in rankings:
        for rank, doc in enumerate(rank_list):
            if doc.doc_id not in scores:
                scores[doc.doc_id] = [0.0, doc]
            scores[doc.doc_id][0] += 1.0 / (k + rank + 1)

    sorted_docs = sorted(
        scores.values(), key=lambda x: x[0], reverse=True
    )[:top_n]
    results = []
    for rank, (score, doc) in enumerate(sorted_docs):
        results.append(
            RetrievedDoc(
                doc_id=doc.doc_id,
                content=doc.content,
                score=score,
                rank=rank + 1,
                metadata=doc.metadata,
                pool=doc.pool,
            )
        )
    return results


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        rrf_k: int = 60,
    ):
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        dense_results = self.dense.retrieve(query, k=k)
        bm25_results = self.bm25.retrieve(query, k=k)
        return reciprocal_rank_fusion(
            [dense_results, bm25_results], k=self.rrf_k, top_n=k
        )
