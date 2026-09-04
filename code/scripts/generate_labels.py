#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd
import dotenv

dotenv.load_dotenv()


def find_xlsx() -> Path:
    for c in [
        Path("data/raw/ease_rag/MalBenignDataset.xlsx"),
        Path("mal-LLM/RQ_experiments/data/MalBenignDataset.xlsx"),
    ]:
        if c.exists():
            return c
    raise FileNotFoundError("MalBenignDataset.xlsx not found")


def main():
    parser = argparse.ArgumentParser(description="Generate label files for datasets")
    parser.add_argument(
        "--method", type=str, default="has_susp_url",
        choices=["has_susp_url", "all_zero", "all_one"],
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
    )
    args = parser.parse_args()

    xlsx_path = find_xlsx()
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    labels = {}
    for _, row in df.iterrows():
        pkg = str(row["package_name"])
        if args.method == "all_zero":
            labels[pkg] = 0
        elif args.method == "all_one":
            labels[pkg] = 1
        else:
            raw = row.get("has_susp_url", 0)
            labels[pkg] = int(raw) if pd.notna(raw) else 0

    mal = sum(1 for v in labels.values() if v == 1)
    print(f"Labels: {mal} malicious, {len(labels) - mal} benign")

    out = Path(args.output or "data/raw/ease_rag/labels.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
