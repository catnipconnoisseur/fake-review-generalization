"""
scripts/run_cross_eval.py — Execute Cross-Dataset Evaluation & Generate Transfer Matrix.

Evaluates all trained models (A_* models evaluated on Amazon test + DOSC test;
B_* models evaluated on DOSC test + Amazon test).

Computes performance degradation deltas (ΔF1, ΔROC-AUC) and vocabulary coverage.
Outputs:
- results/metrics/transfer_matrix.csv
- results/metrics/cross_eval_results.json
"""

import argparse
import json
from pathlib import Path
import pandas as pd

from src.data.ingest import load_config
from src.data.splitter import load_splits
from src.evaluation.cross_eval import build_transfer_dataframe, run_single_cross_eval
from src.models.registry import list_models


def execute_cross_evaluation(project_root: Path, target_models: str = "all"):
    print("=" * 85)
    print("  PHASE 4: CROSS-DATASET TRANSFER EVALUATION & DELTA METRIC PIPELINE")
    print("=" * 85)

    cfg = load_config(project_root)
    models_dir = project_root / cfg.get("results", {}).get("models_dir", "results/models")
    metrics_dir = project_root / cfg.get("results", {}).get("metrics_dir", "results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load DataFrames
    amazon_df = pd.read_parquet(project_root / cfg["data"]["amazon_clean"])
    dosc_df = pd.read_parquet(project_root / cfg["data"]["dosc_clean"])

    # 2. Load Split Indices
    amazon_splits = load_splits(project_root / cfg["data"]["amazon_splits"])
    dosc_splits = load_splits(project_root / cfg["data"]["dosc_splits"])

    # 3. Identify models available to evaluate
    all_known_models = list_models()
    if target_models == "all":
        # Check which models exist on disk
        models_to_eval = []
        for m_id in all_known_models:
            classical_file = models_dir / f"{m_id}.joblib"
            tf_dir = models_dir / m_id
            if classical_file.exists() or tf_dir.exists():
                models_to_eval.append(m_id)
    else:
        models_to_eval = [m.strip() for m in target_models.split(",")]

    print(f"Models to evaluate ({len(models_to_eval)}): {', '.join(models_to_eval)}\n")

    eval_results = []

    for model_id in models_to_eval:
        print(f"Evaluating {model_id} ...")
        res = run_single_cross_eval(
            model_id=model_id,
            models_dir=models_dir,
            amazon_df=amazon_df,
            amazon_splits=amazon_splits,
            dosc_df=dosc_df,
            dosc_splits=dosc_splits,
            cfg=cfg,
        )
        eval_results.append(res)

        src = res["source_dataset"].upper()
        tgt = res["target_dataset"].upper()
        w_f1 = res["within_eval"]["f1"]
        c_f1 = res["cross_eval"]["f1"]
        d_f1 = res["deltas"]["delta_f1"]
        cat = res["deltas"]["interpretation"]

        print(f"  Within ({src} → {src}) F1: {w_f1:.4f}")
        print(f"  Cross  ({src} → {tgt}) F1: {c_f1:.4f}")
        print(f"  Degradation ΔF1: {d_f1:+.4f} [{cat.upper()}]")
        if "vocab_coverage" in res["cross_eval"]:
            cov = res["cross_eval"]["vocab_coverage"]["coverage"]
            oov = res["cross_eval"]["vocab_coverage"]["oov_rate"]
            print(f"  Target Vocabulary Coverage: {cov*100:.1f}% (OOV rate: {oov*100:.1f}%)")
        print()

    # 4. Save JSON Results
    json_path = metrics_dir / "cross_eval_results.json"
    with open(json_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved detailed JSON metrics to: {json_path}")

    # 5. Build and Save CSV Transfer Matrix
    df_transfer = build_transfer_dataframe(eval_results)
    csv_path = metrics_dir / "transfer_matrix.csv"
    df_transfer.to_csv(csv_path, index=False)
    print(f"Saved transfer matrix table to: {csv_path}\n")

    # 6. Display Transfer Matrix Table
    print("=" * 110)
    print("  CROSS-DATASET GENERALIZATION TRANSFER MATRIX")
    print("=" * 110)
    cols_to_print = [
        "Model ID", "Source (Train)", "Target (Eval)",
        "Within F1", "Cross F1", "Delta F1 (Drop)",
        "Within ROC-AUC", "Cross ROC-AUC", "Delta ROC-AUC",
        "Degradation Category"
    ]
    print(df_transfer[cols_to_print].to_string(index=False))
    print("=" * 110)

    return df_transfer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-dataset transfer evaluation.")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated model IDs or 'all'")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    execute_cross_evaluation(project_root, target_models=args.models)
