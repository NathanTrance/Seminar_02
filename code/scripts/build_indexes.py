#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import dotenv

dotenv.load_dotenv()

from src.ragsec.datasets.ease_rag import EaseRagDataset
from src.ragsec.retrieval.bm25 import BM25Retriever
from src.ragsec.retrieval.dense import DenseRetriever
from src.ragsec.static.ast_features import ASTFeatureExtractor
from src.ragsec.utils.logging import setup_logging

logger = logging.getLogger(__name__)

DATASET_REGISTRY = {
    "ease_rag": EaseRagDataset,
}


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indexes")
    parser.add_argument(
        "--dataset", "-d", type=str, default="ease_rag",
        choices=list(DATASET_REGISTRY.keys()),
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/raw",
    )
    parser.add_argument(
        "--index-dir", type=str, default="data/indexes",
    )
    parser.add_argument(
        "--no-dense", action="store_true",
        help="Skip dense indexes (no embedding endpoint needed)",
    )
    args = parser.parse_args()

    setup_logging("INFO")
    data_root = Path(args.data_dir)
    index_root = Path(args.index_dir)
    index_root.mkdir(parents=True, exist_ok=True)

    dataset_cls = DATASET_REGISTRY[args.dataset]
    dataset = dataset_cls(path=data_root / args.dataset)
    samples = dataset.load(split="train")

    mal_samples = [s for s in samples if s.label == 1]
    ben_samples = [s for s in samples if s.label == 0]
    all_samples = mal_samples + ben_samples

    logger.info(
        "Building indexes for %s: %d mal + %d ben = %d total",
        args.dataset, len(mal_samples), len(ben_samples), len(all_samples),
    )

    def to_doc(s, pool):
        content = s.raw_code or s.behavior_text or ""
        return {
            "doc_id": s.sample_id,
            "content": content,
            "metadata": {
                "package_name": s.package_name,
                "label": s.label,
                "pool": pool,
                "dataset": s.dataset,
            },
        }

    # BM25 indexes
    logger.info("Building BM25 indexes...")
    bm25_all = BM25Retriever()
    bm25_all.build_index([to_doc(s, "all") for s in all_samples])
    bm25_all.save(index_root / "bm25_all.pkl")

    bm25_mal = BM25Retriever()
    bm25_mal.build_index([to_doc(s, "malicious") for s in mal_samples])
    bm25_mal.save(index_root / "bm25_mal.pkl")

    if not args.no_dense:
        logger.info("Building dense indexes...")
        dense_all = DenseRetriever()
        dense_all.build_index([to_doc(s, "all") for s in all_samples])
        dense_all.save(index_root / "dense_all.pkl")

        dense_mal = DenseRetriever()
        dense_mal.build_index([to_doc(s, "malicious") for s in mal_samples])
        dense_mal.save(index_root / "dense_mal.pkl")

        dense_ben = DenseRetriever()
        dense_ben.build_index([to_doc(s, "benign") for s in ben_samples])
        dense_ben.save(index_root / "dense_ben.pkl")

        # Behavior indexes
        logger.info("Building behavior dense indexes...")
        beh_mal_docs = []
        beh_ben_docs = []
        for s in mal_samples:
            ext = ASTFeatureExtractor(s.raw_code or "")
            beh_mal_docs.append({
                "doc_id": s.sample_id,
                "content": ext.to_behavior_text(),
                "metadata": {"label": 1, "pool": "malicious"},
            })
        for s in ben_samples:
            ext = ASTFeatureExtractor(s.raw_code or "")
            beh_ben_docs.append({
                "doc_id": s.sample_id,
                "content": ext.to_behavior_text(),
                "metadata": {"label": 0, "pool": "benign"},
            })

        dense_beh_mal = DenseRetriever()
        dense_beh_mal.build_index(beh_mal_docs)
        dense_beh_mal.save(index_root / "dense_beh_mal.pkl")

        dense_beh_ben = DenseRetriever()
        dense_beh_ben.build_index(beh_ben_docs)
        dense_beh_ben.save(index_root / "dense_beh_ben.pkl")

    logger.info("All indexes built in %s", index_root)


if __name__ == "__main__":
    main()
