import numpy as np

from src.evaluation.threshold import best_f1_threshold, percentile_threshold, threshold_sweep


def test_percentile_threshold() -> None:
    assert percentile_threshold(np.array([0.1, 0.2, 0.3]), 50) == 0.2


def test_threshold_sweep_returns_metrics() -> None:
    rows = threshold_sweep(np.array([0, 1, 1]), np.array([0.1, 0.4, 0.9]), [0.3])
    assert rows[0]["precision"] == 1.0
    assert rows[0]["recall"] == 1.0
    assert rows[0]["f1"] == 1.0


def test_best_f1_threshold_is_float() -> None:
    threshold = best_f1_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert isinstance(threshold, float)
