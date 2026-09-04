import json
import logging
from pathlib import Path

from ..static.behaviors import BEHAVIOR_LABELS, extract_behavior_flags

logger = logging.getLogger(__name__)

STATIC_BEHAVIORS = [
    "shell_execution",
    "process_creation",
    "network_access",
    "file_write",
    "file_delete",
    "dynamic_execution",
    "environment_access",
    "base64_decode",
    "remote_download",
    "persistence",
    "obfuscation",
    "credential_access",
]


def parse_model_behavior_claims(response: dict) -> dict[str, bool]:
    claims = {}
    for behavior in response.get("behaviors", []):
        btype = behavior.get("type", "").lower().replace(" ", "_")
        evidence = behavior.get("evidence", [])
        if btype in STATIC_BEHAVIORS:
            claims[btype] = len(evidence) > 0
    return claims


def compute_faithfulness_metrics(
    code: str | None,
    response: dict,
) -> dict:
    if not code:
        return {
            "unsupported_claim_rate": 0.0,
            "behavior_precision": 0.0,
            "behavior_recall": 0.0,
            "total_model_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "static_behaviors": {},
            "model_claims": {},
        }

    static_flags = extract_behavior_flags(code)
    model_claims = parse_model_behavior_claims(response)

    total_claims = sum(1 for v in model_claims.values() if v)
    supported = sum(
        1
        for b, claimed in model_claims.items()
        if claimed and static_flags.get(b, False)
    )
    unsupported = total_claims - supported

    static_positive = sum(1 for v in static_flags.values() if v)
    recall_denom = static_positive if static_positive > 0 else 1
    recall = (
        sum(
            1
            for b in STATIC_BEHAVIORS
            if model_claims.get(b, False) and static_flags.get(b, False)
        )
        / recall_denom
    )

    return {
        "unsupported_claim_rate": (
            unsupported / total_claims if total_claims > 0 else 0.0
        ),
        "behavior_precision": (
            supported / total_claims if total_claims > 0 else 0.0
        ),
        "behavior_recall": float(recall),
        "total_model_claims": total_claims,
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "static_behaviors": static_flags,
        "model_claims": model_claims,
    }
