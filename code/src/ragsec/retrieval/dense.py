import logging
import pickle
from pathlib import Path

import numpy as np
import faiss

from ..clients.embeddings import EmbeddingClient
from .base import RetrievedDoc

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        index_path: str | Path | None = None,
    ):
        self.embedding_client = embedding_client or EmbeddingClient()
        self.index_path = Path(index_path) if index_path else None
        self.doc_ids: list[str] = []
        self.doc_metadata: list[dict] = []
        self.corpus: list[str] = []
        self.index: faiss.Index | None = None
        self.dimension: int = 0

    def build_index(self, documents: list[dict]) -> None:
        texts = [doc.get("content", "") for doc in documents]
        self.doc_ids = [doc.get("doc_id", "") for doc in documents]
        self.doc_metadata = [doc.get("metadata", {}) for doc in documents]
        self.corpus = texts

        logger.info("Embedding %d documents...", len(texts))
        embeddings = self.embedding_client.embed(texts)
        emb_array = np.array(embeddings, dtype=np.float32)
        self.dimension = emb_array.shape[1]

        self.index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(emb_array)
        self.index.add(emb_array)
        logger.info(
            "Dense index built with %d documents (dim=%d)",
            len(self.doc_ids),
            self.dimension,
        )

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        if self.index is None:
            raise RuntimeError("Dense index not built")
        query_emb = self.embedding_client.embed_one(query)
        query_array = np.array([query_emb], dtype=np.float32)
        faiss.normalize_L2(query_array)
        scores, indices = self.index.search(query_array, k)
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                break
            results.append(
                RetrievedDoc(
                    doc_id=self.doc_ids[idx],
                    content=self.corpus[idx],
                    score=float(scores[0][rank]),
                    rank=rank + 1,
                    metadata=self.doc_metadata[idx],
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "doc_ids": self.doc_ids,
            "doc_metadata": self.doc_metadata,
            "corpus": self.corpus,
            "dimension": self.dimension,
        }
        with open(path, "wb") as f:
            pickle.dump(meta, f)
        faiss.write_index(self.index, str(path) + ".faiss")
        logger.info("Dense index saved to %s", path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "rb") as f:
            meta = pickle.load(f)
        self.doc_ids = meta["doc_ids"]
        self.doc_metadata = meta["doc_metadata"]
        self.corpus = meta["corpus"]
        self.dimension = meta["dimension"]
        self.index = faiss.read_index(str(path) + ".faiss")
        logger.info(
            "Dense index loaded from %s with %d docs", path, len(self.doc_ids)
        )
