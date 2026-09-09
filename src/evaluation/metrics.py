"""
src/evaluation/metrics.py — Evaluation Metrics & Delta Engine.

Provides standard binary classification performance metrics and computes
cross-dataset transfer degradation deltas:
- F1, ROC-AUC, Precision, Recall, Accuracy, Macro F1
- Delta metrics: ΔF1, ΔROC-AUC, ΔPrecision, ΔRecall
- Generalization interpretation categories
"""

from typing import Any, Dict, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_score: Optional[Union[np.ndarray, list]] = None,
) -> Dict[str, Any]:
    """
    Compute full suite of classification metrics.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (0=genuine, 1=fake)
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels
    y_score : Optional array-like of shape (n_samples,)
        Continuous decision scores or predicted probabilities for class 1.
        Used to compute ROC-AUC.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    metrics: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "f1_macro": round(f1_macro, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }

    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        # Handle cases where score array might be constant or NaN
        try:
            auc = float(roc_auc_score(y_true, y_score))
            metrics["roc_auc"] = round(auc, 4)
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def interpret_delta_f1(delta_f1: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    """
    Categorize transfer degradation based on ΔF1:
    - < 0.05: excellent_generalization
    - 0.05 - 0.15: moderate_degradation
    - 0.15 - 0.30: severe_degradation
    - > 0.30: catastrophic_failure
    """
    t = thresholds or {
        "excellent": 0.05,
        "moderate": 0.15,
        "severe": 0.30,
    }

    if delta_f1 < t["excellent"]:
        return "excellent_generalization"
    elif delta_f1 <= t["moderate"]:
        return "moderate_degradation"
    elif delta_f1 <= t["severe"]:
        return "severe_degradation"
    else:
        return "catastrophic_failure"


def compute_deltas(
    within_metrics: Dict[str, Any],
    cross_metrics: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute degradation metrics: Δ = Within - Cross.
    Positive values indicate drop in performance when transferred.
    """
    delta_f1 = round(float(within_metrics["f1"] - cross_metrics["f1"]), 4)
    delta_prec = round(float(within_metrics["precision"] - cross_metrics["precision"]), 4)
    delta_rec = round(float(within_metrics["recall"] - cross_metrics["recall"]), 4)
    delta_acc = round(float(within_metrics["accuracy"] - cross_metrics["accuracy"]), 4)

    delta_auc = None
    if within_metrics.get("roc_auc") is not None and cross_metrics.get("roc_auc") is not None:
        delta_auc = round(float(within_metrics["roc_auc"] - cross_metrics["roc_auc"]), 4)

    interpretation = interpret_delta_f1(delta_f1, thresholds)

    return {
        "delta_f1": delta_f1,
        "delta_roc_auc": delta_auc,
        "delta_precision": delta_prec,
        "delta_recall": delta_rec,
        "delta_accuracy": delta_acc,
        "interpretation": interpretation,
    }
