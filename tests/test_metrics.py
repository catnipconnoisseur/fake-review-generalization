"""
tests/test_metrics.py — Unit tests for evaluation metrics and delta calculations.
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_deltas,
    interpret_delta_f1,
)


def test_compute_classification_metrics():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])
    y_score = np.array([0.9, 0.4, 0.1, 0.2, 0.8, 0.7])

    metrics = compute_classification_metrics(y_true, y_pred, y_score)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 2
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_interpret_delta_f1():
    assert interpret_delta_f1(0.02) == "excellent_generalization"
    assert interpret_delta_f1(0.10) == "moderate_degradation"
    assert interpret_delta_f1(0.22) == "severe_degradation"
    assert interpret_delta_f1(0.45) == "catastrophic_failure"


def test_compute_deltas():
    within = {"f1": 0.90, "roc_auc": 0.95, "precision": 0.88, "recall": 0.92, "accuracy": 0.89}
    cross = {"f1": 0.60, "roc_auc": 0.65, "precision": 0.62, "recall": 0.58, "accuracy": 0.61}

    deltas = compute_deltas(within, cross)
    assert pytest.approx(deltas["delta_f1"], 1e-4) == 0.30
    assert pytest.approx(deltas["delta_roc_auc"], 1e-4) == 0.30
    assert pytest.approx(deltas["delta_precision"], 1e-4) == 0.26
    assert pytest.approx(deltas["delta_recall"], 1e-4) == 0.34
    assert deltas["interpretation"] == "severe_degradation"
