#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "results" / "raw" / "jev_direct_typesafe_jev_1_13_ease_rag.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "metrics" / "jev_direct_typesafe_jev_1_13.json"


def compute_metrics(rows: list[dict]) -> dict:
    tp = fp = tn = fn = failures = 0
    for row in rows:
        if not row.get("parse_ok"):
            failures += 1
            continue
        gold = row["gold_label"]
        predicted = row["predicted_label"]
        if gold == 1 and predicted == 1:
            tp += 1
        elif gold == 1 and predicted == 0:
            fn += 1
        elif gold == 0 and predicted == 1:
            fp += 1
        elif gold == 0 and predicted == 0:
            tn += 1
        else:
            raise ValueError(f"Unexpected labels: gold={gold!r}, predicted={predicted!r}")

    mal_precision = tp / (tp + fp) if tp + fp else 0.0
    mal_recall = tp / (tp + fn) if tp + fn else 0.0
    benign_precision = tn / (tn + fn) if tn + fn else 0.0
    benign_recall = tn / (tn + fp) if tn + fp else 0.0
    mal_f1 = 2 * mal_precision * mal_recall / (mal_precision + mal_recall) if mal_precision + mal_recall else 0.0
    benign_f1 = 2 * benign_precision * benign_recall / (benign_precision + benign_recall) if benign_precision + benign_recall else 0.0
    return {
        "n": len(rows),
        "n_fail": failures,
        "coverage": (len(rows) - failures) / len(rows) if rows else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "mal_precision": mal_precision,
        "mal_recall": mal_recall,
        "mal_f1": mal_f1,
        "ben_precision": benign_precision,
        "ben_recall": benign_recall,
        "ben_f1": benign_f1,
        "macro_f1": (mal_f1 + benign_f1) / 2,
        "balanced_accuracy": (mal_recall + benign_recall) / 2,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "cost_usd": sum(row.get("cost_usd") or 0.0 for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate direct Jev full-test results")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = []
    sample_ids = set()
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample_id at line {line_number}: {sample_id}")
            sample_ids.add(sample_id)
            rows.append(row)

    metrics = compute_metrics(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
