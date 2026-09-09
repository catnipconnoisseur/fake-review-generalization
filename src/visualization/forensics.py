"""
src/visualization/forensics.py — Publication-Ready Forensic Visualizations.

Generates 8 publication-quality figures (300 DPI) for thesis and paper presentation:
1. feature_divergence.png: Side-by-side top-20 positive & negative feature weights.
2. vocab_overlap_venn.png: Lexical overlap and vocabulary isolation visualization.
3. length_distributions.png: Review length KDE overlays (genuine vs. fake by dataset).
4. confusion_matrices.png: Within vs. cross transfer confusion matrix heatmaps.
5. roc_curves.png: ROC curves comparing within-domain vs. cross-domain discrimination.
6. delta_f1_comparison.png: Bar chart of performance degradation (ΔF1) across models.
7. error_taxonomy.png: Failure mode taxonomy prevalence distribution.
8. psycholinguistic_scatter.png: 1st-person pronoun vs. spatial density scatter.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

# Global publication aesthetics
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def plot_feature_divergence(
    a_top: Dict[str, List[Any]],
    b_top: Dict[str, List[Any]],
    output_path: Path,
    n_top: int = 15,
) -> None:
    """Plot side-by-side comparative horizontal bar charts for A_logreg and B_logreg."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=False)

    # Amazon A_logreg top fake cues
    a_fake_names = [f[0].replace("word__", "w:").replace("char__", "c:") for f in a_top["top_fake"][:n_top]][::-1]
    a_fake_vals = [f[1] for f in a_top["top_fake"][:n_top]][::-1]

    axes[0].barh(a_fake_names, a_fake_vals, color="#3498db", alpha=0.85, edgecolor="#2980b9")
    axes[0].set_title(f"A_logreg (Amazon GPT-2)\nTop {n_top} Synthetic Deception Cues", fontweight="bold")
    axes[0].set_xlabel("Regression Weight (Positive = Fake)")
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    # DOSC B_logreg top fake cues
    b_fake_names = [f[0].replace("word__", "w:").replace("char__", "c:") for f in b_top["top_fake"][:n_top]][::-1]
    b_fake_vals = [f[1] for f in b_top["top_fake"][:n_top]][::-1]

    axes[1].barh(b_fake_names, b_fake_vals, color="#e74c3c", alpha=0.85, edgecolor="#c0392b")
    axes[1].set_title(f"B_logreg (DOSC MTurk)\nTop {n_top} Human Deception Cues", fontweight="bold")
    axes[1].set_xlabel("Regression Weight (Positive = Fake)")
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_vocab_overlap(
    vocab_a: set,
    vocab_b: set,
    output_path: Path,
) -> None:
    """Plot proportional circular Euler/Venn diagram of vocabulary overlap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    overlap = len(vocab_a.intersection(vocab_b))
    only_a = len(vocab_a) - overlap
    only_b = len(vocab_b) - overlap

    # Circles
    circle_a = Circle((0.38, 0.5), 0.32, color="#3498db", alpha=0.5, ec="#2980b9", lw=2)
    circle_b = Circle((0.62, 0.5), 0.22, color="#e74c3c", alpha=0.5, ec="#c0392b", lw=2)

    ax.add_patch(circle_a)
    ax.add_patch(circle_b)

    ax.text(0.24, 0.5, f"Amazon Only\n{only_a:,}\n({only_a/(only_a+overlap)*100:.1f}%)", ha="center", va="center", fontweight="bold")
    ax.text(0.50, 0.5, f"Shared\n{overlap:,}", ha="center", va="center", fontweight="bold", color="#2c3e50")
    ax.text(0.72, 0.5, f"DOSC Only\n{only_b:,}\n({only_b/(only_b+overlap)*100:.1f}%)", ha="center", va="center", fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Lexical Vocabulary Overlap: Amazon (GPT-2) vs. DOSC (MTurk)", fontsize=13, fontweight="bold", pad=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_length_distributions(
    amazon_df: pd.DataFrame,
    dosc_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot KDE review length distributions (genuine vs. fake) across datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Compute word length
    amazon_lens = amazon_df["text"].apply(lambda t: len(t.split()))
    dosc_lens = dosc_df["text"].apply(lambda t: len(t.split()))

    # Amazon
    sns.kdeplot(amazon_lens[amazon_df["target"] == 0], ax=axes[0], label="Genuine (OR)", color="#2ecc71", fill=True, alpha=0.3)
    sns.kdeplot(amazon_lens[amazon_df["target"] == 1], ax=axes[0], label="Synthetic Fake (CG)", color="#e74c3c", fill=True, alpha=0.3)
    axes[0].set_title("Amazon Reviews Length (Words)", fontweight="bold")
    axes[0].set_xlabel("Word Count")
    axes[0].set_xlim(0, 300)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # DOSC
    sns.kdeplot(dosc_lens[dosc_df["target"] == 0], ax=axes[1], label="Genuine (Truthful)", color="#2ecc71", fill=True, alpha=0.3)
    sns.kdeplot(dosc_lens[dosc_df["target"] == 1], ax=axes[1], label="Deceptive (MTurk)", color="#e74c3c", fill=True, alpha=0.3)
    axes[1].set_title("DOSC Hotel Reviews Length (Words)", fontweight="bold")
    axes[1].set_xlabel("Word Count")
    axes[1].set_xlim(0, 300)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrices(
    cms: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Plot multi-panel grid of confusion matrices comparing within vs. cross transfer."""
    valid_items = [(k, v) for k, v in cms.items() if v is not None]
    n_items = len(valid_items)
    if n_items <= 4:
        nrows, ncols = 2, 2
        figsize = (10, 8)
    elif n_items <= 6:
        nrows, ncols = 2, 3
        figsize = (14, 8)
    else:
        nrows, ncols = 2, 4
        figsize = (18, 8)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    labels = ["Genuine (0)", "Fake (1)"]

    for ax, (title, cm) in zip(axes_flat, valid_items):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)

    # Hide any unused axes
    for ax in axes_flat[len(valid_items):]:
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_curves(
    roc_data: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot within-domain vs. cross-domain ROC curve overlay."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        "A_logreg Within": "#2980b9",
        "A_logreg Cross":  "#e67e22",
        "B_logreg Within": "#27ae60",
        "B_logreg Cross":  "#c0392b",
    }

    for name, data in roc_data.items():
        fpr, tpr = data["fpr"], data["tpr"]
        auc_val = data["auc"]
        color = colors.get(name, "#7f8c8d")
        linestyle = "-" if "Within" in name else "--"
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=color, lw=2, linestyle=linestyle)

    ax.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("ROC Curves: Within-Dataset vs. Cross-Dataset Generalization", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_delta_f1_bars(
    transfer_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot bar chart comparing Within F1 vs. Cross F1 and Delta drop."""
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(transfer_df))
    width = 0.35

    ax.bar(x - width/2, transfer_df["Within F1"], width, label="Within-Domain F1", color="#2ecc71", edgecolor="#27ae60")
    ax.bar(x + width/2, transfer_df["Cross F1"], width, label="Cross-Domain F1", color="#e74c3c", edgecolor="#c0392b")

    ax.set_ylabel("F1 Score")
    ax.set_title("Cross-Dataset Performance Degradation (F1 Score)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['Model ID']}\n({row['Source (Train)']}→{row['Target (Eval)']})" for _, row in transfer_df.iterrows()])
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate drop
    for i, row in transfer_df.iterrows():
        drop = row["Delta F1 (Drop)"]
        ax.annotate(f"Δ = {drop:+.3f}",
                    (i, max(row["Within F1"], row["Cross F1"]) + 0.04),
                    ha="center", fontweight="bold", color="#c0392b")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_error_taxonomy(
    error_prevalence: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Plot stacked or grouped bar chart of failure modes prevalence."""
    df_tax = pd.DataFrame(error_prevalence).T
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#e74c3c", "#3498db", "#9b59b6", "#f1c40f", "#e67e22", "#95a5a6"]
    df_tax.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="#333333", alpha=0.85)

    ax.set_ylabel("Proportion of Misclassified Samples")
    ax.set_title("Error Taxonomy Distribution across Cross-Dataset Failures", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Failure Mode", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_psycholinguistic_scatter(
    scatter_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot 1st-person pronoun ratio vs spatial detail density."""
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.scatterplot(
        data=scatter_df,
        x="pronoun_1st_ratio",
        y="spatial_density",
        hue="Classification",
        palette={"Correct": "#2ecc71", "Misclassified": "#e74c3c"},
        alpha=0.6,
        s=40,
        ax=ax,
    )

    ax.set_title("Psycholinguistic Divergence: Pronoun Usage vs. Spatial Detail", fontweight="bold")
    ax.set_xlabel("1st-Person Pronoun Ratio (Self-Reference Density)")
    ax.set_ylabel("Spatial & Sensory Detail Density")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
