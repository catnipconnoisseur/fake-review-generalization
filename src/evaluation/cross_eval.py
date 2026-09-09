"""
src/evaluation/cross_eval.py — Cross-Dataset Transfer Evaluation Engine.

Evaluates models across domains (Amazon GPT-2 ↔ DOSC MTurk):
1. Evaluates model on its own held-out test set (within-domain baseline).
2. Evaluates model on the opposing dataset's held-out test set (out-of-domain transfer).
3. Computes vocabulary coverage and out-of-vocabulary rate (for TF-IDF baselines).
4. Computes performance degradation deltas (ΔF1, ΔROC-AUC, ΔPrecision, ΔRecall).
5. Classifies generalization severity according to predefined research thresholds.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_classification_metrics, compute_deltas
from src.models.registry import get_model_info, load_model


def evaluate_model_on_dataset(
    model: Any,
    texts: Iterable[str],
    y_true: Iterable[int],
    dataset_name: str,
) -> Dict[str, Any]:
    """
    Evaluate a model instance on a given text dataset.
    Automatically computes predictions, decision scores, metrics,
    and lexical vocabulary coverage if the model uses TF-IDF.
    """
    texts_list = list(texts)
    y_arr = np.array(list(y_true), dtype=int)

    if hasattr(model, "predict_and_scores"):
        preds, scores = model.predict_and_scores(texts_list)
    else:
        preds = model.predict(texts_list)
        scores = model.predict_scores(texts_list)

    metrics = compute_classification_metrics(y_arr, preds, scores)
    metrics["dataset_name"] = dataset_name
    metrics["n_samples"] = len(texts_list)

    # Compute vocabulary coverage for TF-IDF based models
    if hasattr(model, "vectorizer") and hasattr(model.vectorizer, "compute_vocabulary_coverage"):
        cov_info = model.vectorizer.compute_vocabulary_coverage(texts_list)
        metrics["vocab_coverage"] = cov_info

    return metrics


def run_single_cross_eval(
    model_id: str,
    models_dir: Union[str, Path],
    amazon_df: pd.DataFrame,
    amazon_splits: Dict[str, Any],
    dosc_df: pd.DataFrame,
    dosc_splits: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform full within-dataset and cross-dataset evaluation for a single model ID.
    """
    cfg = cfg or {}
    model_info = get_model_info(model_id)
    source_dataset = model_info["dataset"]
    target_dataset = "dosc" if source_dataset == "amazon" else "amazon"

    # Prepare datasets and test splits
    if source_dataset == "amazon":
        source_df, source_test_idx = amazon_df, amazon_splits["test_idx"]
        target_df, target_test_idx = dosc_df, dosc_splits["test_idx"]
    else:
        source_df, source_test_idx = dosc_df, dosc_splits["test_idx"]
        target_df, target_test_idx = amazon_df, amazon_splits["test_idx"]

    source_test_texts = source_df.iloc[source_test_idx]["text"].tolist()
    source_test_y = source_df.iloc[source_test_idx]["target"].values

    target_test_texts = target_df.iloc[target_test_idx]["text"].tolist()
    target_test_y = target_df.iloc[target_test_idx]["target"].values

    # Load trained model
    model = load_model(model_id, models_dir)

    # 1. Within-dataset evaluation
    within_metrics = evaluate_model_on_dataset(
        model=model,
        texts=source_test_texts,
        y_true=source_test_y,
        dataset_name=source_dataset,
    )

    # 2. Cross-dataset evaluation
    cross_metrics = evaluate_model_on_dataset(
        model=model,
        texts=target_test_texts,
        y_true=target_test_y,
        dataset_name=target_dataset,
    )

    # 3. Delta degradation calculation
    thresholds = cfg.get("cross_eval", {}).get("interpretation_thresholds")
    deltas = compute_deltas(within_metrics, cross_metrics, thresholds=thresholds)

    return {
        "model_id": model_id,
        "family": model_info["family"],
        "algorithm": model_info["algorithm"],
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "within_eval": within_metrics,
        "cross_eval": cross_metrics,
        "deltas": deltas,
    }


def build_transfer_dataframe(eval_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten list of cross-evaluation results into a publication-ready tabular DataFrame.
    """
    rows = []
    for res in eval_results:
        m_id = res["model_id"]
        src = res["source_dataset"]
        tgt = res["target_dataset"]
        w = res["within_eval"]
        c = res["cross_eval"]
        d = res["deltas"]

        # Vocabulary coverage on cross dataset
        cov = c.get("vocab_coverage", {}).get("coverage", None)

        rows.append({
            "Model ID": m_id,
            "Family": res["family"],
            "Algorithm": res["algorithm"],
            "Source (Train)": src.upper(),
            "Target (Eval)": tgt.upper(),
            "Within F1": w["f1"],
            "Cross F1": c["f1"],
            "Delta F1 (Drop)": d["delta_f1"],
            "Within ROC-AUC": w.get("roc_auc"),
            "Cross ROC-AUC": c.get("roc_auc"),
            "Delta ROC-AUC": d.get("delta_roc_auc"),
            "Within Acc": w["accuracy"],
            "Cross Acc": c["accuracy"],
            "Cross Vocab Coverage": round(cov, 4) if cov is not None else "N/A",
            "Degradation Category": d["interpretation"],
        })

    return pd.DataFrame(rows)
