import logging

logger = logging.getLogger(__name__)


def compute_label_precision_at_k(
    retrieved_ids: list[str],
    gold_labels: dict[str, int],
    k: int | None = None,
) -> float:
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    if not retrieved_ids:
        return 0.0
    relevant = sum(
        1 for rid in retrieved_ids if gold_labels.get(rid, 0) == 1
    )
    return relevant / len(retrieved_ids)


def compute_same_behavior_hit_at_k(
    retrieved_ids: list[str],
    query_behaviors: set[str],
    doc_behaviors: dict[str, set[str]],
    k: int | None = None,
) -> int:
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    if not query_behaviors:
        return 0
    for rid in retrieved_ids:
        if query_behaviors & doc_behaviors.get(rid, set()):
            return 1
    return 0
