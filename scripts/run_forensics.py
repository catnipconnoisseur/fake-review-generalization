"""
scripts/run_forensics.py — Forensic Failure Mode & Visualization Suite Execution.

Executes complete Phase 5 forensic pipeline:
1. Feature Weight Divergence & Overlap Analysis (A_logreg vs B_logreg).
2. Lexical & Psycholinguistic Feature Diagnosis.
3. 6-Category Error Taxonomy Classification on Cross-Dataset Failures.
4. Generation of 8 publication-ready figures (300 DPI) in results/figures/.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from src.data.ingest import load_config
from src.data.splitter import load_splits
from src.evaluation.error_analysis import analyze_errors, compute_training_baselines
from src.features.psycholinguistic import compute_psycholinguistic_features
from src.features.text_stats import extract_tokens
from src.models.registry import load_model
from src.visualization.forensics import (
    plot_confusion_matrices,
    plot_delta_f1_bars,
    plot_error_taxonomy,
    plot_feature_divergence,
    plot_length_distributions,
    plot_psycholinguistic_scatter,
    plot_roc_curves,
    plot_vocab_overlap,
)


def run_forensics_pipeline(project_root: Path):
    print("=" * 85)
    print("  PHASE 5: FORENSIC FAILURE MODE & VISUALIZATION SUITE")
    print("=" * 85)

    cfg = load_config(project_root)
    models_dir = project_root / cfg.get("results", {}).get("models_dir", "results/models")
    metrics_dir = project_root / cfg.get("results", {}).get("metrics_dir", "results/metrics")
    figures_dir = project_root / cfg.get("results", {}).get("figures_dir", "results/figures")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data and Splits
    amazon_df = pd.read_parquet(project_root / cfg["data"]["amazon_clean"])
    dosc_df = pd.read_parquet(project_root / cfg["data"]["dosc_clean"])
    amazon_splits = load_splits(project_root / cfg["data"]["amazon_splits"])
    dosc_splits = load_splits(project_root / cfg["data"]["dosc_splits"])

    amazon_train_texts = amazon_df.iloc[amazon_splits["train_idx"]]["text"].tolist()
    amazon_test_texts = amazon_df.iloc[amazon_splits["test_idx"]]["text"].tolist()
    amazon_test_y = amazon_df.iloc[amazon_splits["test_idx"]]["target"].values

    dosc_train_texts = dosc_df.iloc[dosc_splits["train_idx"]]["text"].tolist()
    dosc_test_texts = dosc_df.iloc[dosc_splits["test_idx"]]["text"].tolist()
    dosc_test_y = dosc_df.iloc[dosc_splits["test_idx"]]["target"].values

    # 2. Load Trained LogReg Models
    print("Loading A_logreg and B_logreg models ...")
    a_logreg = load_model("A_logreg", models_dir)
    b_logreg = load_model("B_logreg", models_dir)

    # ------------------------------------------------------------------------
    # Task 5.3: Feature Weight Divergence & Overlap Analysis
    # ------------------------------------------------------------------------
    print("\n--- [Task 5.3] Feature Weight Extraction & Overlap Ratio ---")
    a_top = a_logreg.get_top_features(n_top=50)
    b_top = b_logreg.get_top_features(n_top=50)

    # Strip prefixes to compare core vocabulary/n-gram overlap
    a_fake_set = {name.split("__", 1)[-1] for name, _ in a_top["top_fake"]}
    b_fake_set = {name.split("__", 1)[-1] for name, _ in b_top["top_fake"]}
    shared_fake = a_fake_set.intersection(b_fake_set)
    fake_overlap_ratio = len(shared_fake) / 50.0

    a_gen_set = {name.split("__", 1)[-1] for name, _ in a_top["top_genuine"]}
    b_gen_set = {name.split("__", 1)[-1] for name, _ in b_top["top_genuine"]}
    shared_gen = a_gen_set.intersection(b_gen_set)
    gen_overlap_ratio = len(shared_gen) / 50.0

    print(f"Top 50 Fake Cues Overlap:    {len(shared_fake)}/50 ({fake_overlap_ratio*100:.1f}%) -> Shared: {shared_fake}")
    print(f"Top 50 Genuine Cues Overlap: {len(shared_gen)}/50 ({gen_overlap_ratio*100:.1f}%) -> Shared: {shared_gen}")

    # Build Comparative Table
    comp_rows = []
    for rank in range(50):
        comp_rows.append({
            "Rank": rank + 1,
            "Amazon_Fake_Signal": a_top["top_fake"][rank][0],
            "Amazon_Weight": round(a_top["top_fake"][rank][1], 4),
            "DOSC_Fake_Signal": b_top["top_fake"][rank][0],
            "DOSC_Weight": round(b_top["top_fake"][rank][1], 4),
        })
    df_top_comp = pd.DataFrame(comp_rows)
    comp_csv = metrics_dir / "top_features_comparison.csv"
    df_top_comp.to_csv(comp_csv, index=False)
    print(f"Saved feature comparison table to: {comp_csv}")

    # ------------------------------------------------------------------------
    # Task 5.2: Error Analysis & 6-Category Taxonomy
    # ------------------------------------------------------------------------
    print("\n--- [Task 5.2] Misclassification Taxonomy Analysis ---")
    amazon_train_baselines = compute_training_baselines(amazon_train_texts)
    dosc_train_baselines = compute_training_baselines(dosc_train_texts)

    # Errors when A_logreg is evaluated on DOSC test
    errors_a_to_b, summary_a_to_b = analyze_errors(
        model=a_logreg,
        test_texts=dosc_test_texts,
        test_labels=dosc_test_y,
        train_baselines=amazon_train_baselines,
        source_dataset="amazon",
        target_dataset="dosc",
    )

    # Errors when B_logreg is evaluated on Amazon test
    errors_b_to_a, summary_b_to_a = analyze_errors(
        model=b_logreg,
        test_texts=amazon_test_texts,
        test_labels=amazon_test_y,
        train_baselines=dosc_train_baselines,
        source_dataset="dosc",
        target_dataset="amazon",
    )

    taxonomy_summary = {
        "A_to_B (Amazon -> DOSC)": summary_a_to_b,
        "B_to_A (DOSC -> Amazon)": summary_b_to_a,
    }
    tax_json = metrics_dir / "error_taxonomy_summary.json"
    with open(tax_json, "w") as f:
        json.dump(taxonomy_summary, f, indent=2)
    print(f"Saved error taxonomy summary to: {tax_json}")

    print("\nError Prevalence Breakdown:")
    for key, summ in taxonomy_summary.items():
        print(f"  {key} (Total Errors: {summ['total_errors']}/{summ['total_samples']}):")
        for code, prev in summ["prevalence"].items():
            print(f"    - {code}: {prev*100:.1f}%")

    # ------------------------------------------------------------------------
    # Task 5.4: Generate 8 Publication-Ready Figures
    # ------------------------------------------------------------------------
    print("\n--- [Task 5.4] Generating 8 Publication-Ready Figures (300 DPI) ---")

    # Figure 1: Feature Divergence
    p1 = figures_dir / "feature_divergence.png"
    plot_feature_divergence(a_top, b_top, p1, n_top=15)
    print(f"  [1/8] Generated: {p1}")

    # Figure 2: Vocabulary Overlap Venn
    vocab_a = set(extract_tokens(" ".join(amazon_df["text"])))
    vocab_b = set(extract_tokens(" ".join(dosc_df["text"])))
    p2 = figures_dir / "vocab_overlap_venn.png"
    plot_vocab_overlap(vocab_a, vocab_b, p2)
    print(f"  [2/8] Generated: {p2}")

    # Figure 3: Length Distributions
    p3 = figures_dir / "length_distributions.png"
    plot_length_distributions(amazon_df, dosc_df, p3)
    print(f"  [3/8] Generated: {p3}")

    # Figure 4: Confusion Matrices (Classical + Transformers)
    a_within_pred = a_logreg.predict(amazon_test_texts)
    a_cross_pred = a_logreg.predict(dosc_test_texts)
    b_within_pred = b_logreg.predict(dosc_test_texts)
    b_cross_pred = b_logreg.predict(amazon_test_texts)

    cms = {
        "A_logreg Within (Amazon→Amazon)": confusion_matrix(amazon_test_y, a_within_pred, labels=[0, 1]),
        "A_logreg Cross (Amazon→DOSC)": confusion_matrix(dosc_test_y, a_cross_pred, labels=[0, 1]),
        "B_logreg Within (DOSC→DOSC)": confusion_matrix(dosc_test_y, b_within_pred, labels=[0, 1]),
        "B_logreg Cross (DOSC→Amazon)": confusion_matrix(amazon_test_y, b_cross_pred, labels=[0, 1]),
    }

    # If BERT models exist, include them in Figure 4
    if (models_dir / "A_bert").exists():
        try:
            a_bert = load_model("A_bert", models_dir)
            a_bert_within_pred = a_bert.predict(amazon_test_texts, batch_size=64)
            a_bert_cross_pred = a_bert.predict(dosc_test_texts, batch_size=64)
            cms["A_bert Within (Amazon→Amazon)"] = confusion_matrix(amazon_test_y, a_bert_within_pred, labels=[0, 1])
            cms["A_bert Cross (Amazon→DOSC)"] = confusion_matrix(dosc_test_y, a_bert_cross_pred, labels=[0, 1])
        except Exception as e:
            print(f"  Note: Could not evaluate A_bert for Figure 4: {e}")

    if (models_dir / "B_bert").exists():
        try:
            b_bert = load_model("B_bert", models_dir)
            b_bert_within_pred = b_bert.predict(dosc_test_texts, batch_size=64)
            b_bert_cross_pred = b_bert.predict(amazon_test_texts, batch_size=64)
            cms["B_bert Within (DOSC→DOSC)"] = confusion_matrix(dosc_test_y, b_bert_within_pred, labels=[0, 1])
            cms["B_bert Cross (DOSC→Amazon)"] = confusion_matrix(amazon_test_y, b_bert_cross_pred, labels=[0, 1])
        except Exception as e:
            print(f"  Note: Could not evaluate B_bert for Figure 4: {e}")

    p4 = figures_dir / "confusion_matrices.png"
    plot_confusion_matrices(cms, p4)
    print(f"  [4/8] Generated: {p4}")

    # Figure 5: ROC Curves
    a_within_scores = a_logreg.predict_scores(amazon_test_texts)
    a_cross_scores = a_logreg.predict_scores(dosc_test_texts)
    b_within_scores = b_logreg.predict_scores(dosc_test_texts)
    b_cross_scores = b_logreg.predict_scores(amazon_test_texts)

    fpr_aw, tpr_aw, _ = roc_curve(amazon_test_y, a_within_scores)
    fpr_ac, tpr_ac, _ = roc_curve(dosc_test_y, a_cross_scores)
    fpr_bw, tpr_bw, _ = roc_curve(dosc_test_y, b_within_scores)
    fpr_bc, tpr_bc, _ = roc_curve(amazon_test_y, b_cross_scores)

    roc_data = {
        "A_logreg Within": {"fpr": fpr_aw, "tpr": tpr_aw, "auc": roc_auc_score(amazon_test_y, a_within_scores)},
        "A_logreg Cross":  {"fpr": fpr_ac, "tpr": tpr_ac, "auc": roc_auc_score(dosc_test_y, a_cross_scores)},
        "B_logreg Within": {"fpr": fpr_bw, "tpr": tpr_bw, "auc": roc_auc_score(dosc_test_y, b_within_scores)},
        "B_logreg Cross":  {"fpr": fpr_bc, "tpr": tpr_bc, "auc": roc_auc_score(amazon_test_y, b_cross_scores)},
    }
    p5 = figures_dir / "roc_curves.png"
    plot_roc_curves(roc_data, p5)
    print(f"  [5/8] Generated: {p5}")

    # Figure 6: Delta F1 Degradation Bar Chart
    transfer_csv = metrics_dir / "transfer_matrix.csv"
    if transfer_csv.exists():
        transfer_df = pd.read_csv(transfer_csv)
        p6 = figures_dir / "delta_f1_comparison.png"
        plot_delta_f1_bars(transfer_df, p6)
        print(f"  [6/8] Generated: {p6}")
    else:
        print("  [6/8] Skipped (transfer_matrix.csv not found)")

    # Figure 7: Error Taxonomy Stacked Bar Chart
    tax_prevalence = {
        "Amazon → DOSC (A_logreg)": summary_a_to_b["prevalence"],
        "DOSC → Amazon (B_logreg)": summary_b_to_a["prevalence"],
    }
    p7 = figures_dir / "error_taxonomy.png"
    plot_error_taxonomy(tax_prevalence, p7)
    print(f"  [7/8] Generated: {p7}")

    # Figure 8: Psycholinguistic Scatter Plot
    # Combine test samples from DOSC evaluated by A_logreg
    scatter_records = []
    for text, y_true, y_pred in zip(dosc_test_texts, dosc_test_y, a_cross_pred):
        psych = compute_psycholinguistic_features(text)
        scatter_records.append({
            "pronoun_1st_ratio": psych["pronoun_1st_total_ratio"],
            "spatial_density": psych["spatial_detail_density"],
            "Classification": "Correct" if y_true == y_pred else "Misclassified",
        })
    df_scatter = pd.DataFrame(scatter_records)
    p8 = figures_dir / "psycholinguistic_scatter.png"
    plot_psycholinguistic_scatter(df_scatter, p8)
    print(f"  [8/8] Generated: {p8}")

    print("\n" + "=" * 85)
    print("  ALL 8 PUBLICATION-READY FIGURES SUCCESSFULLY GENERATED!")
    print("=" * 85)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    run_forensics_pipeline(project_root)
