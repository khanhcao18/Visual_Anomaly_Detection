from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve


def percentile_threshold(normal_scores: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(normal_scores, percentile))


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return percentile_threshold(scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def threshold_sweep(y_true: np.ndarray, scores: np.ndarray, thresholds: list[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(np.sum((y_true == 1) & (y_pred == 1)) / max(np.sum(y_pred == 1), 1)),
                "recall": float(np.sum((y_true == 1) & (y_pred == 1)) / max(np.sum(y_true == 1), 1)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return rows
