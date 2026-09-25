import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from ..clients.jev import OpenRouterJevClient
from .jev_demo import QUESTION_ID, build_binary_decision_request


MISSING_SOURCE = {None, "", "Not Available"}


def load_ease_test_samples(data_dir: str | Path) -> tuple[list[dict], list[dict]]:
    data_dir = Path(data_dir)
    samples = []
    excluded = []
    for source_name, label, filename in (
        ("malicious", 1, "test_malicious_packages_final.json"),
        ("benign", 0, "test_benign_packages_final.json"),
    ):
        with (data_dir / filename).open(encoding="utf-8") as handle:
            entries = json.load(handle)
        if not isinstance(entries, list):
            raise ValueError(f"{filename} must contain a JSON list")
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"{filename} entry {index} is not an object")
            sample_id = f"ease_test_{source_name}_{index:05d}"
            source_code = entry.get("setup.py")
            sample = {
                "sample_id": sample_id,
                "gold_label": label,
                "package_name": entry.get("package_name"),
                "source_code": source_code,
            }
            if source_code in MISSING_SOURCE:
                excluded.append(sample)
            else:
                samples.append(sample)
    return samples, excluded


def load_completed_sample_ids(output_path: str | Path) -> set[str]:
    path = Path(output_path)
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                result = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            sample_id = result.get("sample_id")
            if isinstance(sample_id, str):
                completed.add(sample_id)
    return completed


def remove_failed_rows(output_path: str | Path) -> int:
    path = Path(output_path)
    if not path.exists():
        return 0
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    retained = [row for row in rows if row.get("parse_ok")]
    removed = len(rows) - len(retained)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return removed


def build_result(sample: dict, model: str, response: dict) -> dict:
    success = response["status"] == "ok"
    decision = response["decision"] if success else None
    predicted_label = 1 if decision == "malicious" else 0
    usage = response.get("usage") or {}
    return {
        "experiment_id": "jev_direct_typesafe_jev_1_13_ease_rag",
        "method": "jev_direct",
        "dataset": "ease_rag",
        "model": model,
        "resolved_model": response.get("model"),
        "sample_id": sample["sample_id"],
        "gold_label": sample["gold_label"],
        "predicted_label": predicted_label if success else None,
        "confidence": response.get("confidence"),
        "probabilities": response.get("probabilities"),
        "behaviors": [],
        "retrieved_ids": [],
        "retrieval_scores": [],
        "raw_response": response.get("raw_response"),
        "latency_ms": response.get("latency_ms", 0),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cost_usd": usage.get("cost"),
        "parse_ok": success,
        "error": response.get("error"),
        "error_type": response.get("error_type"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_full_test(
    samples: list[dict],
    client: OpenRouterJevClient,
    model: str,
    output_path: str | Path,
    limit: int | None = None,
    delay_seconds: float = 0.0,
) -> dict:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if delay_seconds < 0:
        raise ValueError("delay_seconds cannot be negative")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_sample_ids(output_path)
    pending = [sample for sample in samples if sample["sample_id"] not in completed]
    if limit is not None:
        pending = pending[:limit]

    successful = 0
    failed = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for position, sample in enumerate(pending, start=1):
            request = build_binary_decision_request(sample["source_code"], model=model)
            response = client.decide(request, question_id=QUESTION_ID)
            result = build_result(sample, model, response)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if result["parse_ok"]:
                successful += 1
            else:
                failed += 1
            if delay_seconds and position < len(pending):
                time.sleep(delay_seconds)

    return {
        "total_eligible": len(samples),
        "already_completed": len(completed),
        "attempted": len(pending),
        "successful": successful,
        "failed": failed,
        "output_path": str(output_path),
    }
