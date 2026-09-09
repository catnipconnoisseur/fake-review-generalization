"""
scripts/run_training.py — Train Classical Baselines with Reproducible Cross-Validation.

Trains 4 classical baseline models:
- A_logreg: Logistic Regression on Amazon GPT-2 synthetic reviews (GroupKFold CV)
- A_svm:    Linear SVC on Amazon GPT-2 synthetic reviews (GroupKFold CV)
- B_logreg: Logistic Regression on DOSC MTurk deceptive reviews (StratifiedKFold CV)
- B_svm:    Linear SVC on DOSC MTurk deceptive reviews (StratifiedKFold CV)

Saves models to results/models/ and outputs metrics to results/metrics/baseline_within_metrics.json.
"""

import argparse
import json
import time
from pathlib import Path
import pandas as pd

from src.data.ingest import load_config
from src.data.splitter import load_splits
from src.evaluation.metrics import compute_classification_metrics
from src.models.baseline import BaselineModel, train_baseline_model
from src.models.registry import load_model, save_model


def run_baseline_training(project_root: Path, target_model: str = "all", n_jobs: int = -1):
    print("=" * 80)
    print("  PHASE 2: CLASSICAL BASELINE TRAINING PIPELINE")
    print("=" * 80)

    cfg = load_config(project_root)
    models_dir = project_root / cfg.get("results", {}).get("models_dir", "results/models")
    metrics_dir = project_root / cfg.get("results", {}).get("metrics_dir", "results/metrics")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Processed Parquets
    amazon_parquet = project_root / cfg["data"]["amazon_clean"]
    dosc_parquet = project_root / cfg["data"]["dosc_clean"]

    print(f"Loading datasets:\n  Amazon: {amazon_parquet}\n  DOSC:   {dosc_parquet}")
    amazon_df = pd.read_parquet(amazon_parquet)
    dosc_df = pd.read_parquet(dosc_parquet)

    # 2. Load Splits
    amazon_splits_path = project_root / cfg["data"]["amazon_splits"]
    dosc_splits_path = project_root / cfg["data"]["dosc_splits"]

    print(f"Loading splits:\n  Amazon: {amazon_splits_path}\n  DOSC:   {dosc_splits_path}")
    amazon_splits = load_splits(amazon_splits_path)
    dosc_splits = load_splits(dosc_splits_path)

    # Prepare datasets dictionary
    datasets = {
        "amazon": {
            "df": amazon_df,
            "train_idx": amazon_splits["train_idx"],
            "val_idx": amazon_splits["val_idx"],
            "test_idx": amazon_splits["test_idx"],
            "has_groups": True,
        },
        "dosc": {
            "df": dosc_df,
            "train_idx": dosc_splits["train_idx"],
            "val_idx": dosc_splits["val_idx"],
            "test_idx": dosc_splits["test_idx"],
            "has_groups": False,
        },
    }

    baseline_model_ids = ["A_logreg", "A_svm", "B_logreg", "B_svm"]
    if target_model != "all":
        if target_model not in baseline_model_ids:
            raise ValueError(f"Unknown baseline target model: {target_model}. Choose from {baseline_model_ids} or 'all'")
        models_to_train = [target_model]
    else:
        models_to_train = baseline_model_ids

    all_results = {}

    for model_id in models_to_train:
        ds_name = "amazon" if model_id.startswith("A_") else "dosc"
        ds_info = datasets[ds_name]
        df = ds_info["df"]

        train_sub = df.iloc[ds_info["train_idx"]]
        val_sub = df.iloc[ds_info["val_idx"]]
        test_sub = df.iloc[ds_info["test_idx"]]

        train_texts = train_sub["text"].tolist()
        train_y = train_sub["target"].values
        train_groups = train_sub["group_id"].values if ds_info["has_groups"] else None

        val_texts = val_sub["text"].tolist()
        val_y = val_sub["target"].values

        test_texts = test_sub["text"].tolist()
        test_y = test_sub["target"].values

        print("\n" + "-" * 70)
        print(f"  Training Model: {model_id} (Dataset: {ds_name.upper()})")
        print(f"  Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")
        if train_groups is not None:
            print(f"  GroupKFold Active: {len(set(train_groups))} distinct prefix groups")
        else:
            print("  StratifiedKFold Active")
        print("-" * 70)

        start_time = time.time()
        model: BaselineModel = train_baseline_model(
            model_id=model_id,
            train_texts=train_texts,
            train_labels=train_y,
            groups=train_groups,
            cfg=cfg,
            n_jobs=n_jobs,
        )
        elapsed = time.time() - start_time
        print(f"  Training + Tuning completed in {elapsed:.2f}s")
        print(f"  Best Params: {model.best_params}")
        print(f"  CV F1 Score: {model.cv_score:.4f}")

        # Save model
        saved_path = save_model(model, model_id, models_dir)
        print(f"  Model saved to: {saved_path}")

        # Evaluate on Validation Set
        val_preds = model.predict(val_texts)
        val_scores = model.predict_scores(val_texts)
        val_metrics = compute_classification_metrics(val_y, val_preds, val_scores)

        # Evaluate on Test Set
        test_preds = model.predict(test_texts)
        test_scores = model.predict_scores(test_texts)
        test_metrics = compute_classification_metrics(test_y, test_preds, test_scores)

        print(f"  Validation Metrics -> F1: {val_metrics['f1']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Test Metrics       -> F1: {test_metrics['f1']:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f} | Acc: {test_metrics['accuracy']:.4f}")

        # Top diagnostic features
        top_features = model.get_top_features(n_top=5)
        top_fake_str = ", ".join([f"{name} ({weight:+.3f})" for name, weight in top_features["top_fake"][:5]])
        top_gen_str = ", ".join([f"{name} ({weight:+.3f})" for name, weight in top_features["top_genuine"][:5]])
        print(f"  Top 5 Fake Cues:    {top_fake_str}")
        print(f"  Top 5 Genuine Cues: {top_gen_str}")

        all_results[model_id] = {
            "model_id": model_id,
            "dataset": ds_name,
            "training_time_sec": round(elapsed, 2),
            "best_params": model.best_params,
            "cv_f1": round(model.cv_score, 4),
            "validation": val_metrics,
            "test": test_metrics,
            "top_features": top_features,
        }

    # Save summary metrics to JSON
    metrics_file = metrics_dir / "baseline_within_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved within-dataset baseline metrics to: {metrics_file}")

    # Print Comparison Table
    print("\n" + "=" * 80)
    print("  WITHIN-DATASET BASELINE SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Model ID':<10} | {'Dataset':<8} | {'Best C':<8} | {'CV F1':<8} | {'Val F1':<8} | {'Test F1':<8} | {'Test AUC':<8} | {'Test Acc':<8}"
    print(header)
    print("-" * len(header))
    for m_id, res in all_results.items():
        best_c = res["best_params"].get("C", "N/A")
        print(
            f"{m_id:<10} | {res['dataset']:<8} | {str(best_c):<8} | "
            f"{res['cv_f1']:<8.4f} | {res['validation']['f1']:<8.4f} | "
            f"{res['test']['f1']:<8.4f} | {res['test']['roc_auc']:<8.4f} | "
            f"{res['test']['accuracy']:<8.4f}"
        )
    print("=" * 80)

    # Verification Gates
    print("\n[VERIFICATION GATES]")
    for m_id in models_to_train:
        # Check artifact existence and loadability
        loaded = load_model(m_id, models_dir)
        assert isinstance(loaded, BaselineModel), f"Failed loading {m_id}"
        print(f"  [PASS] {m_id} loaded successfully from disk")

        test_f1 = all_results[m_id]["test"]["f1"]
        if m_id.startswith("A_"):
            assert test_f1 >= 0.85, f"{m_id} Test F1 {test_f1} is below sanity threshold 0.85"
            print(f"  [PASS] {m_id} within-dataset Test F1 ({test_f1:.4f}) >= 0.85")
        elif m_id.startswith("B_"):
            assert test_f1 >= 0.75, f"{m_id} Test F1 {test_f1} is below sanity threshold 0.75"
            print(f"  [PASS] {m_id} within-dataset Test F1 ({test_f1:.4f}) >= 0.75")

    print("\nAll baseline training and verification gates PASSED successfully!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classical baseline models for fake review detection.")
    parser.add_argument("--model", type=str, default="all", help="Model ID to train (A_logreg, A_svm, B_logreg, B_svm, or all)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of CPU cores for GridSearchCV")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    run_baseline_training(project_root, target_model=args.model, n_jobs=args.n_jobs)
