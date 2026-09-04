#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path

import dotenv

dotenv.load_dotenv()

from src.ragsec.clients.llm import LLMClient
from src.ragsec.datasets.ease_rag import EaseRagDataset
from src.ragsec.experiments.runner import ExperimentRunner
from src.ragsec.retrieval.bm25 import BM25Retriever
from src.ragsec.retrieval.dense import DenseRetriever
from src.ragsec.retrieval.hybrid import HybridRetriever
from src.ragsec.retrieval.contrastive import ContrastiveRetriever
from src.ragsec.static.ast_features import ASTFeatureExtractor
from src.ragsec.utils.cache import Cache
from src.ragsec.utils.logging import setup_logging

logger = logging.getLogger(__name__)

DATASET_REGISTRY = {
    "ease_rag": EaseRagDataset,
}

KNOWN_METHODS = {
    "no_rag",
    "dense",
    "bm25",
    "hybrid",
    "behavior",
    "contrastive",
    "behavior_contrastive",
}

RETRIEVAL_METHODS = KNOWN_METHODS - {"no_rag"}


def build_indexes(samples, method, cache, data_root="data"):
    data_root = Path(data_root)
    indexes_root = data_root / "indexes"
    indexes_root.mkdir(parents=True, exist_ok=True)

    mal_samples = [s for s in samples if s.label == 1]
    ben_samples = [s for s in samples if s.label == 0]

    logger.info(
        "Building indexes: %d malicious, %d benign",
        len(mal_samples),
        len(ben_samples),
    )

    def sample_to_doc(s, pool="malicious"):
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

    mal_docs_bm25 = [sample_to_doc(s, "malicious") for s in mal_samples]
    ben_docs_bm25 = [sample_to_doc(s, "benign") for s in ben_samples]
    all_docs_bm25 = mal_docs_bm25 + ben_docs_bm25

    if method in ("bm25", "hybrid", "contrastive", "behavior_contrastive", "behavior"):
        bm25_path = indexes_root / "bm25_mal.pkl"
        bm25_all_path = indexes_root / "bm25_all.pkl"

        if bm25_path.exists():
            bm25_mal = BM25Retriever()
            bm25_mal.load(bm25_path)
        else:
            bm25_mal = BM25Retriever()
            bm25_mal.build_index(mal_docs_bm25)
            bm25_mal.save(bm25_path)

        if bm25_all_path.exists():
            bm25_all = BM25Retriever()
            bm25_all.load(bm25_all_path)
        else:
            bm25_all = BM25Retriever()
            bm25_all.build_index(all_docs_bm25)
            bm25_all.save(bm25_all_path)

    if method in ("dense", "hybrid", "contrastive", "behavior_contrastive", "behavior"):
        dense_path = indexes_root / "dense_mal.pkl"
        dense_all_path = indexes_root / "dense_all.pkl"

        if dense_path.exists():
            dense_mal = DenseRetriever()
            dense_mal.load(dense_path)
        else:
            dense_mal = DenseRetriever()
            dense_mal.build_index(mal_docs_bm25)
            dense_mal.save(dense_path)

        dense_all = DenseRetriever()
        if dense_all_path.exists():
            dense_all.load(dense_all_path)
        else:
            dense_all.build_index(all_docs_bm25)
            dense_all.save(dense_all_path)


def get_retrieval_func(method, samples, cache, data_root="data"):
    data_root = Path(data_root)
    indexes_root = data_root.parent / "indexes"

    mal_samples = [s for s in samples if s.label == 1]
    ben_samples = [s for s in samples if s.label == 0]

    def sample_to_doc(s, pool="malicious"):
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

    mal_docs = [sample_to_doc(s, "malicious") for s in mal_samples]
    ben_docs = [sample_to_doc(s, "benign") for s in ben_samples]

    if method == "dense":
        ret = DenseRetriever()
        p = indexes_root / "dense_all.pkl"
        if p.exists():
            ret.load(p)
        else:
            ret.build_index(mal_docs + ben_docs)
            ret.save(p)
        return ret.retrieve

    elif method == "bm25":
        ret = BM25Retriever()
        p = indexes_root / "bm25_all.pkl"
        if p.exists():
            ret.load(p)
        else:
            ret.build_index(mal_docs + ben_docs)
            ret.save(p)
        return ret.retrieve

    elif method == "hybrid":
        dense = DenseRetriever()
        dp = indexes_root / "dense_all.pkl"
        if dp.exists():
            dense.load(dp)
        else:
            dense.build_index(mal_docs + ben_docs)
            dense.save(dp)
        bm25 = BM25Retriever()
        bp = indexes_root / "bm25_all.pkl"
        if bp.exists():
            bm25.load(bp)
        else:
            bm25.build_index(mal_docs + ben_docs)
            bm25.save(bp)
        hybrid = HybridRetriever(dense, bm25)
        return hybrid.retrieve

    elif method == "behavior":
        beh_docs_mal = []
        for s in mal_samples:
            extractor = ASTFeatureExtractor(s.raw_code or "")
            content = extractor.to_behavior_text()
            beh_docs_mal.append({
                "doc_id": s.sample_id,
                "content": content,
                "metadata": {"package_name": s.package_name, "label": s.label, "pool": "malicious", "dataset": s.dataset},
            })
        ret = DenseRetriever()
        p = indexes_root / "dense_behavior.pkl"
        if p.exists():
            ret.load(p)
        else:
            ret.build_index(beh_docs_mal)
            ret.save(p)
        return ret.retrieve

    elif method == "contrastive":
        mal_docs_all = [sample_to_doc(s, "malicious") for s in mal_samples]
        ben_docs_all = [sample_to_doc(s, "benign") for s in ben_samples]

        dense_mal = DenseRetriever()
        mp = indexes_root / "dense_mal.pkl"
        if mp.exists():
            dense_mal.load(mp)
        else:
            dense_mal.build_index(mal_docs_all)
            dense_mal.save(mp)

        dense_ben = DenseRetriever()
        bp = indexes_root / "dense_ben.pkl"
        if bp.exists():
            dense_ben.load(bp)
        else:
            dense_ben.build_index(ben_docs_all)
            dense_ben.save(bp)

        cont = ContrastiveRetriever(dense_mal, dense_ben)
        return cont.retrieve

    elif method == "behavior_contrastive":
        def make_beh_docs(samples_list, pool):
            docs = []
            for s in samples_list:
                extractor = ASTFeatureExtractor(s.raw_code or "")
                content = extractor.to_behavior_text()
                docs.append({
                    "doc_id": s.sample_id,
                    "content": content,
                    "metadata": {"package_name": s.package_name, "label": s.label, "pool": pool, "dataset": s.dataset},
                })
            return docs

        beh_mal = make_beh_docs(mal_samples, "malicious")
        beh_ben = make_beh_docs(ben_samples, "benign")

        dense_mal = DenseRetriever()
        mp = indexes_root / "dense_beh_mal.pkl"
        if mp.exists():
            dense_mal.load(mp)
        else:
            dense_mal.build_index(beh_mal)
            dense_mal.save(mp)

        dense_ben = DenseRetriever()
        bp = indexes_root / "dense_beh_ben.pkl"
        if bp.exists():
            dense_ben.load(bp)
        else:
            dense_ben.build_index(beh_ben)
            dense_ben.save(bp)

        cont = ContrastiveRetriever(dense_mal, dense_ben)
        return cont.retrieve

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run a single RAGsec experiment"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default="ease_rag",
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset name",
    )
    parser.add_argument(
        "--model", "-m", type=str, default="qwen3_coder_30b",
        help="Model ID from configs/models.yaml",
    )
    parser.add_argument(
        "--method", "-M", type=str, default="no_rag",
        choices=sorted(KNOWN_METHODS),
        help="Experiment method",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["all", "train", "dev", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of samples (for testing)",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Top-k retrieved documents",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results/raw",
        help="Results output directory",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="data/cache",
        help="Cache directory",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/raw",
        help="Data root directory",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable LLM response caching",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Generation temperature",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Maximum generation tokens",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    setup_logging("INFO")

    data_root = Path(args.data_dir)
    dataset_root = data_root / args.dataset

    dataset_cls = DATASET_REGISTRY[args.dataset]
    dataset = dataset_cls(path=dataset_root)
    samples = dataset.load(split=args.split)

    logger.info(
        "Loaded %d samples from %s (%s split)",
        len(samples), args.dataset, args.split,
    )

    cache = Cache(path=Path(args.cache_dir) / "cache.db")
    llm_client = LLMClient()
    runner = ExperimentRunner(
        llm_client=llm_client,
        cache=cache,
        results_dir=args.results_dir,
        seed=args.seed,
    )

    retrieval_func = None
    if args.method in RETRIEVAL_METHODS:
        logger.info("Setting up retrieval for method: %s", args.method)
        kb_samples = dataset.load(split="train")
        retrieval_func = get_retrieval_func(
            method=args.method,
            samples=kb_samples,
            cache=cache,
            data_root=str(data_root),
        )

    experiment_id = f"{args.method}_{args.model}_{args.dataset}"

    results = runner.run_experiment(
        method=args.method,
        model=args.model,
        dataset_name=args.dataset,
        samples=samples,
        experiment_id=experiment_id,
        retrieval_func=retrieval_func,
        top_k_malicious=args.top_k,
        top_k_benign=args.top_k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        limit=args.limit,
    )

    runner.save_results(results, experiment_id)

    metrics = runner.evaluate_results(results)
    logger.info("Metrics for %s:", experiment_id)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
        else:
            logger.info("  %s: %s", k, v)

    metrics_path = Path("results/metrics") / f"{experiment_id}_metrics.json"
    from src.ragsec.evaluation.classification import save_metrics
    save_metrics(metrics, metrics_path)

    print(f"\nDone. Results: results/raw/{experiment_id}.jsonl")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
