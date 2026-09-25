#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import dotenv
except ModuleNotFoundError:
    dotenv = None

if dotenv is not None:
    dotenv.load_dotenv()

from ragsec.clients.jev import OPENROUTER_JEV_MODEL, OpenRouterJevClient
from ragsec.experiments.jev_full_test import load_ease_test_samples, run_full_test


DEFAULT_DATA_DIR = PROJECT_ROOT / "mal-LLM" / "RQ_experiments" / "data"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "raw" / "jev_direct_typesafe_jev_1_13_ease_rag.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct OpenRouter Jev decisions on the EASE RAG test split"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=OPENROUTER_JEV_MODEL)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples, excluded = load_ease_test_samples(args.data_dir)
    print(f"eligible={len(samples)} excluded_missing_source={len(excluded)}")

    if args.dry_run:
        print(f"would_write={args.output}")
        return 0

    summary = run_full_test(
        samples=samples,
        client=OpenRouterJevClient(timeout=args.timeout),
        model=args.model,
        output_path=args.output,
        limit=args.limit,
        delay_seconds=args.delay,
    )
    print(" ".join(f"{key}={value}" for key, value in summary.items()))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
