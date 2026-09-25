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

from ragsec.clients.jev import (
    JEV_BASE_URL,
    JEV_DECISIONS_PATH,
    OPENROUTER_DECISIONS_URL,
    OPENROUTER_JEV_MODEL,
    JevClient,
    OpenRouterJevClient,
)
from ragsec.experiments.jev_demo import (
    DEFAULT_JEV_MODEL,
    format_results,
    load_run,
    load_samples,
    run_samples,
    save_run,
)


DEFAULT_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "jev_demo.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send source text directly to the JEV decision API"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate without network calls")
    mode.add_argument("--live", action="store_true", help="Make bounded live JEV calls")
    mode.add_argument("--replay", type=Path, help="Display a saved run without network calls")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--source-file", type=Path, help="Use one source text file instead")
    parser.add_argument("--sample-id", default="user_source")
    parser.add_argument("--provider", choices=("openrouter", "jev"), default="openrouter")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, choices=(1, 2), default=None)
    parser.add_argument("--max-requests", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "jev_demo",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.replay:
        print(format_results(load_run(args.replay)))
        return 0

    if args.source_file:
        samples = [
            {
                "sample_id": args.sample_id,
                "source_code": args.source_file.read_text(encoding="utf-8"),
                "provenance": str(args.source_file),
            }
        ]
    else:
        samples = load_samples(args.fixture)
    if args.limit is not None:
        samples = samples[: args.limit]

    if args.provider == "openrouter":
        model = args.model or OPENROUTER_JEV_MODEL
        endpoint = OPENROUTER_DECISIONS_URL
        client_class = OpenRouterJevClient
    else:
        model = args.model or DEFAULT_JEV_MODEL
        endpoint = f"{JEV_BASE_URL}{JEV_DECISIONS_PATH}"
        client_class = JevClient

    client = None if args.dry_run else client_class(timeout=args.timeout)
    run = run_samples(
        samples=samples,
        model=model,
        client=client,
        dry_run=args.dry_run,
        max_requests=args.max_requests,
        endpoint=endpoint,
    )
    output_path = save_run(run, args.results_dir)
    print(format_results(run))
    print(f"saved={output_path}")

    if args.live and any(
        item["response"]["status"] != "ok" for item in run["results"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
