"""
scripts/run_transformers.py — Train & Fine-Tune Transformer Models (BERT & RoBERTa).

Fine-tunes BERT-base and RoBERTa-base on Amazon GPT-2 or DOSC MTurk datasets
using Apple Silicon MPS acceleration, with early stopping on validation F1.
Saves checkpoints to results/models/{model_id}/.
"""

import argparse
import json
import time
from pathlib import Path
import pandas as pd
import torch

from src.data.ingest import load_config
from src.data.splitter import load_splits
from src.evaluation.metrics import compute_classification_metrics
from src.models.registry import save_model
from src.models.transformer_ft import get_default_device, train_transformer


def run_transformer_training(
    project_root: Path,
    target_model: str = "B_bert",
    epochs_override: int = None,
):
    print("=" * 80)
    print("  PHASE 3: TRANSFORMER FINE-TUNING PIPELINE")
    print("=" * 80)

    cfg = load_config(project_root)
    models_dir = project_root / cfg.get("results", {}).get("models_dir", "results/models")
    metrics_dir = project_root / cfg.get("results", {}).get("metrics_dir", "results/metrics")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = get_default_device()
    print(f"Hardware Acceleration Device: {device.type.upper()}")

    # 1. Load Data
    amazon_df = pd.read_parquet(project_root / cfg["data"]["amazon_clean"])
    dosc_df = pd.read_parquet(project_root / cfg["data"]["dosc_clean"])
    amazon_splits = load_splits(project_root / cfg["data"]["amazon_splits"])
    dosc_splits = load_splits(project_root / cfg["data"]["dosc_splits"])

    datasets = {
        "amazon": {
            "df": amazon_df,
            "train_idx": amazon_splits["train_idx"],
            "val_idx": amazon_splits["val_idx"],
            "test_idx": amazon_splits["test_idx"],
        },
        "dosc": {
            "df": dosc_df,
            "train_idx": dosc_splits["train_idx"],
            "val_idx": dosc_splits["val_idx"],
            "test_idx": dosc_splits["test_idx"],
        },
    }

    transformer_models = ["B_bert", "B_roberta", "A_bert", "A_roberta"]
    if target_model != "all":
        if target_model not in transformer_models:
            raise ValueError(f"Unknown transformer model: {target_model}. Choose from {transformer_models} or 'all'")
        models_to_train = [target_model]
    else:
        models_to_train = transformer_models

    if epochs_override is not None:
        cfg["transformers"]["bert"]["epochs"] = epochs_override
        cfg["transformers"]["roberta"]["epochs"] = epochs_override

    results = {}

    for model_id in models_to_train:
        ds_name = "amazon" if model_id.startswith("A_") else "dosc"
        ds_info = datasets[ds_name]
        df = ds_info["df"]

        train_sub = df.iloc[ds_info["train_idx"]]
        val_sub = df.iloc[ds_info["val_idx"]]
        test_sub = df.iloc[ds_info["test_idx"]]

        train_texts = train_sub["text"].tolist()
        train_y = train_sub["target"].values.tolist()
        val_texts = val_sub["text"].tolist()
        val_y = val_sub["target"].values.tolist()
        test_texts = test_sub["text"].tolist()
        test_y = test_sub["target"].values.tolist()

        print("\n" + "-" * 70)
        print(f"  Fine-Tuning Transformer: {model_id} (Dataset: {ds_name.upper()})")
        print(f"  Train samples: {len(train_texts)} | Val samples: {len(val_texts)} | Test: {len(test_texts)}")
        print("-" * 70)

        start_time = time.time()
        model = train_transformer(
            model_id=model_id,
            train_texts=train_texts,
            train_labels=train_y,
            val_texts=val_texts,
            val_labels=val_y,
            cfg=cfg,
            device=device,
        )
        elapsed = time.time() - start_time
        print(f"  Training finished in {elapsed:.2f}s ({elapsed/60:.2f} min)")

        # Save model
        saved_path = save_model(model, model_id, models_dir)
        print(f"  Saved transformer checkpoint to: {saved_path}")

        # Evaluate on Test
        print("  Evaluating on within-dataset test split ...")
        test_preds = model.predict(test_texts, batch_size=32)
        test_scores = model.predict_scores(test_texts, batch_size=32)
        test_metrics = compute_classification_metrics(test_y, test_preds, test_scores)

        print(f"  Within Test Metrics -> F1: {test_metrics['f1']:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f} | Acc: {test_metrics['accuracy']:.4f}")

        results[model_id] = {
            "model_id": model_id,
            "dataset": ds_name,
            "training_time_sec": round(elapsed, 2),
            "test_metrics": test_metrics,
        }

    # Save summary
    out_json = metrics_dir / "transformer_within_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved transformer within-dataset metrics to: {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune transformer models.")
    parser.add_argument("--model", type=str, default="B_bert", help="Transformer model ID (B_bert, B_roberta, A_bert, A_roberta, or all)")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    run_transformer_training(project_root, target_model=args.model, epochs_override=args.epochs)
