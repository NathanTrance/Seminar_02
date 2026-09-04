import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: list[int], y_pred: list[int], failure_mask: list[bool] | None = None
) -> dict:
    coverage = 1.0
    if failure_mask is not None and sum(failure_mask) > 0:
        n_fail = sum(failure_mask)
        n_total = len(failure_mask)
        coverage = (n_total - n_fail) / n_total

    valid_mask = (
        [not f for f in failure_mask] if failure_mask else [True] * len(y_true)
    )
    y_true_v = [y for y, m in zip(y_true, valid_mask) if m]
    y_pred_v = [y for y, m in zip(y_pred, valid_mask) if m]

    if len(set(y_true_v)) < 2 or len(set(y_pred_v)) < 2:
        return {
            "malicious_precision": 0.0,
            "malicious_recall": 0.0,
            "malicious_f1": 0.0,
            "benign_precision": 0.0,
            "benign_recall": 0.0,
            "benign_f1": 0.0,
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "coverage": coverage,
            "n_samples": len(y_true_v),
            "n_failures": sum(failure_mask) if failure_mask else 0,
            "confusion_matrix": [],
        }

    cm = confusion_matrix(y_true_v, y_pred_v, labels=[0, 1]).tolist()

    metrics = {
        "malicious_precision": float(precision_score(y_true_v, y_pred_v, pos_label=1, zero_division=0)),
        "malicious_recall": float(recall_score(y_true_v, y_pred_v, pos_label=1, zero_division=0)),
        "malicious_f1": float(f1_score(y_true_v, y_pred_v, pos_label=1, zero_division=0)),
        "benign_precision": float(precision_score(y_true_v, y_pred_v, pos_label=0, zero_division=0)),
        "benign_recall": float(recall_score(y_true_v, y_pred_v, pos_label=0, zero_division=0)),
        "benign_f1": float(f1_score(y_true_v, y_pred_v, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(y_true_v, y_pred_v, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_v, y_pred_v)),
        "coverage": coverage,
        "n_samples": len(y_true_v),
        "n_failures": sum(failure_mask) if failure_mask else 0,
        "confusion_matrix": cm,
    }

    if len(cm) == 2:
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return metrics


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved to %s", path)


def load_metrics(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
