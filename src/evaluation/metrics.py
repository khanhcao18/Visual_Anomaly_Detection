from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    result = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    result["roc_auc"] = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")
    result["pr_auc"] = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")
    return result
