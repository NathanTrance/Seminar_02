#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ragsec.static.ast_features import ASTFeatureExtractor

from evaluate_jev_full_test import compute_metrics


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "jev_behavior_guided_typesafe_jev_1_13_ease_rag.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "jev_behavior_guided_typesafe_jev_1_13.json"
)
DEFAULT_DATA_DIR = PROJECT_ROOT / "mal-LLM" / "RQ_experiments" / "data"


def load_source_by_id(data_dir: Path) -> dict[str, str]:
    sources = {}
    for source_name, filename in (
        ("malicious", "test_malicious_packages_final.json"),
        ("benign", "test_benign_packages_final.json"),
    ):
        entries = json.loads((data_dir / filename).read_text(encoding="utf-8"))
        for index, entry in enumerate(entries):
            sources[f"ease_test_{source_name}_{index:05d}"] = entry.get("setup.py") or ""
    return sources


def compute_behavior_metrics(rows: list[dict], sources: dict[str, str]) -> dict:
    claims_total = supported_total = 0
    sample_precisions = []
    sample_recalls = []
    evaluated = 0
    for row in rows:
        if not row.get("behavior_parse_ok"):
            continue
        evaluated += 1
        claims = {behavior["type"] for behavior in row.get("behaviors", [])}
        detected = ASTFeatureExtractor(sources[row["sample_id"]]).detect_behaviors()
        observed = {behavior for behavior, lines in detected.items() if lines}
        supported = claims & observed
        claims_total += len(claims)
        supported_total += len(supported)
        if claims:
            sample_precisions.append(len(supported) / len(claims))
        if observed:
            sample_recalls.append(len(supported) / len(observed))

    return {
        "n_behavior_evaluated": evaluated,
        "n_relevant_claims": claims_total,
        "n_supported": supported_total,
        "unsupported_claim_rate": (
            1 - supported_total / claims_total if claims_total else 0.0
        ),
        "behavior_precision": (
            sum(sample_precisions) / len(sample_precisions) if sample_precisions else 0.0
        ),
        "behavior_recall": (
            sum(sample_recalls) / len(sample_recalls) if sample_recalls else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate behavior-guided Jev classification and typed behaviors"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.open(encoding="utf-8")]
    result = {
        "classification": compute_metrics(rows),
        "typed_behavior_alignment": compute_behavior_metrics(
            rows, load_source_by_id(args.data_dir)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
