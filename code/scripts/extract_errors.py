#!/usr/bin/env python3
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str)
    args = parser.parse_args()

    with open(args.results_file) as f:
        results = [json.loads(line) for line in f]

    fps = [r for r in results if r["parse_ok"] and r["gold_label"] == 0 and r["predicted_label"] == 1]
    fns = [r for r in results if r["parse_ok"] and r["gold_label"] == 1 and r["predicted_label"] == 0]

    print(json.dumps({
        "experiment": Path(args.results_file).stem,
        "total": len(results),
        "false_positives": {
            "count": len(fps),
            "samples": fps
        },
        "false_negatives": {
            "count": len(fns),
            "samples": fns
        }
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
