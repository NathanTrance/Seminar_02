import logging
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from .base import RetrievedDoc

logger = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self, index_path: str | Path | None = None):
        self.index_path = Path(index_path) if index_path else None
        self.corpus: list[str] = []
        self.doc_ids: list[str] = []
        self.doc_metadata: list[dict] = []
        self.bm25: BM25Okapi | None = None

    def build_index(self, documents: list[dict]) -> None:
        self.corpus = []
        self.doc_ids = []
        self.doc_metadata = []
        for doc in documents:
            self.corpus.append(doc.get("content", ""))
            self.doc_ids.append(doc.get("doc_id", ""))
            self.doc_metadata.append(doc.get("metadata", {}))
        tokenized = [text.split() for text in self.corpus]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built with %d documents", len(self.corpus))

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built")
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        results = []
        for rank, idx in enumerate(top_indices):
            results.append(
                RetrievedDoc(
                    doc_id=self.doc_ids[idx],
                    content=self.corpus[idx],
                    score=float(scores[idx]),
                    rank=rank + 1,
                    metadata=self.doc_metadata[idx],
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "corpus": self.corpus,
                    "doc_ids": self.doc_ids,
                    "doc_metadata": self.doc_metadata,
                    "bm25": self.bm25,
                },
                f,
            )
        logger.info("BM25 index saved to %s", path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.corpus = data["corpus"]
        self.doc_ids = data["doc_ids"]
        self.doc_metadata = data["doc_metadata"]
        self.bm25 = data["bm25"]
        logger.info("BM25 index loaded from %s with %d docs", path, len(self.corpus))
