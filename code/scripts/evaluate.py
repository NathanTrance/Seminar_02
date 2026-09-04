#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

from src.ragsec.evaluation.classification import (
    compute_classification_metrics,
    save_metrics,
)
from src.ragsec.evaluation.faithfulness import compute_faithfulness_metrics
from src.ragsec.static.behaviors import extract_behavior_flags
from src.ragsec.datasets.ease_rag import EaseRagDataset
from src.ragsec.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate experiment results")
    parser.add_argument(
        "results_file", type=str,
        help="Path to results JSONL file",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output path for metrics JSON",
    )
    args = parser.parse_args()

    setup_logging("INFO")

    results_path = Path(args.results_file)
    results = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    logger.info("Loaded %d results", len(results))

    y_true = [r["gold_label"] for r in results]
    y_pred = [r["predicted_label"] for r in results]
    failure_mask = [not r["parse_ok"] for r in results]

    cls_metrics = compute_classification_metrics(y_true, y_pred, failure_mask)

    faithfulness_scores = []
    for r in results:
        code = r.get("raw_response", "")
        response = {"behaviors": r.get("behaviors", [])}
        fm = compute_faithfulness_metrics(code, response)
        faithfulness_scores.append(fm)

    avg_faithfulness = {}
    if faithfulness_scores:
        avg_faithfulness = {
            "avg_unsupported_claim_rate": sum(
                f["unsupported_claim_rate"] for f in faithfulness_scores
            ) / len(faithfulness_scores),
            "avg_behavior_precision": sum(
                f["behavior_precision"] for f in faithfulness_scores
            ) / len(faithfulness_scores),
            "avg_behavior_recall": sum(
                f["behavior_recall"] for f in faithfulness_scores
            ) / len(faithfulness_scores),
        }

    metrics = {**cls_metrics, **avg_faithfulness}

    print("\n=== Classification Metrics ===")
    for k, v in cls_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n=== Faithfulness Metrics ===")
    for k, v in avg_faithfulness.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if args.output:
        save_metrics(metrics, args.output)
        print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
