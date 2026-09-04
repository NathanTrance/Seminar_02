#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import yaml
import dotenv

dotenv.load_dotenv()

from src.ragsec.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment matrix from config"
    )
    parser.add_argument(
        "--config", "-c", type=str, default="configs/experiments.yaml",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print experiments without running",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit samples per experiment",
    )
    args = parser.parse_args()

    setup_logging("INFO")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    experiments = config.get("experiments", [])

    for exp in experiments:
        stage = exp.get("stage", "B")
        methods = exp.get("methods", [])
        models = exp.get("models", [exp.get("model", "qwen3_coder_30b")])
        datasets = exp.get("datasets", [exp.get("dataset", "ease_rag")])
        split = exp.get("split", "test")
        top_k = exp.get("retrieval", {}).get("top_k_malicious", 3)

        for dataset in datasets:
            for model in models:
                for method in methods:
                    cmd = (
                        f"python scripts/run_experiment.py "
                        f"--dataset {dataset} "
                        f"--model {model} "
                        f"--method {method} "
                        f"--split {split} "
                        f"--top-k {top_k} "
                    )
                    if args.limit:
                        cmd += f"--limit {args.limit} "

                    if args.dry_run:
                        print(f"[DRY RUN] {cmd}")
                    else:
                        logger.info("Running: %s", cmd)
                        import subprocess
                        subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
