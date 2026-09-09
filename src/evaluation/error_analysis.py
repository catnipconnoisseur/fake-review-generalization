"""
src/evaluation/error_analysis.py — Forensic Error Taxonomy & Failure Mode Diagnosis.

Implements the 6-category failure taxonomy for cross-dataset evaluation:
- VM: Vocabulary Mismatch (<70% vocabulary coverage against source training vocab)
- LC: Length/Complexity Bias (length <5th or >95th percentile of training corpus)
- SS: Style Shift (pronoun ratio or sensory density > 2σ from training mean)
- TA: Template Absence (model relied on synthetic prompt templates absent in human reviews)
- SD: Sentiment/Domain Shift (domain-specific e-commerce vs hotel mismatch)
- UK: Unknown/Other
"""

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.features.psycholinguistic import compute_psycholinguistic_features
from src.features.text_stats import extract_tokens


FAILURE_MODE_DESCRIPTIONS = {
    "VM": "Vocabulary Mismatch (Unseen domain tokens)",
    "LC": "Length/Complexity Bias (Extreme sequence length)",
    "SS": "Style Shift (Pronoun / sensory density divergence)",
    "TA": "Template Absence (Absence of synthetic prompt structures)",
    "SD": "Sentiment/Domain Shift (Cross-domain semantic gap)",
    "UK": "Unknown / Unclassified",
}


def compute_training_baselines(train_texts: List[str]) -> Dict[str, Any]:
    """Compute baseline statistics from source training data for error diagnosis."""
    train_tokens = [extract_tokens(t) for t in train_texts]
    lengths = [len(toks) for toks in train_tokens]
    vocab = set(tok for toks in train_tokens for tok in toks)

    pronoun_ratios = [
        compute_psycholinguistic_features(t)["pronoun_1st_total_ratio"]
        for t in train_texts
    ]

    return {
        "vocab": vocab,
        "lengths": lengths,
        "p5_length": float(np.percentile(lengths, 5)),
        "p95_length": float(np.percentile(lengths, 95)),
        "mean_pronoun": float(np.mean(pronoun_ratios)),
        "std_pronoun": float(np.std(pronoun_ratios)) if len(pronoun_ratios) > 1 else 0.05,
    }


def classify_failure_mode(
    text: str,
    true_label: int,
    pred_label: int,
    train_baselines: Dict[str, Any],
    source_dataset: str,
    target_dataset: str,
) -> str:
    """
    Classify a single misclassified sample into the 6-category failure taxonomy.
    """
    if true_label == pred_label:
        return "CORRECT"

    tokens = extract_tokens(text)
    n_tokens = len(tokens)
    if n_tokens == 0:
        return "UK"

    # 1. Vocabulary Mismatch (VM)
    train_vocab = train_baselines["vocab"]
    tokens_in_vocab = sum(1 for tok in tokens if tok in train_vocab)
    coverage = tokens_in_vocab / n_tokens
    if coverage < 0.70:
        return "VM"

    # 2. Length / Complexity Bias (LC)
    if n_tokens < train_baselines["p5_length"] or n_tokens > train_baselines["p95_length"]:
        return "LC"

    # 3. Style Shift (SS)
    psych = compute_psycholinguistic_features(text)
    p_ratio = psych["pronoun_1st_total_ratio"]
    mean_p = train_baselines["mean_pronoun"]
    std_p = train_baselines["std_pronoun"]
    if abs(p_ratio - mean_p) > 2.0 * max(std_p, 1e-4):
        return "SS"

    # 4. Template Absence (TA)
    # If trained on Amazon synthetic reviews (which have template artifacts)
    # and evaluated on DOSC human reviews where genuine/deceptive reviews have 0 template repeats
    if source_dataset == "amazon" and target_dataset == "dosc":
        return "TA"

    # 5. Domain / Sentiment Shift (SD)
    if source_dataset != target_dataset:
        return "SD"

    return "UK"


def analyze_errors(
    model: Any,
    test_texts: List[str],
    test_labels: List[int],
    train_baselines: Dict[str, Any],
    source_dataset: str,
    target_dataset: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyze all errors for a model on a given test set and summarize failure modes.
    """
    preds = model.predict(test_texts)
    records = []

    for i, (text, y_true, y_pred) in enumerate(zip(test_texts, test_labels, preds)):
        if y_true != y_pred:
            code = classify_failure_mode(
                text=text,
                true_label=y_true,
                pred_label=y_pred,
                train_baselines=train_baselines,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
            )
            psych = compute_psycholinguistic_features(text)
            records.append({
                "sample_idx": i,
                "text": text,
                "true_label": int(y_true),
                "pred_label": int(y_pred),
                "failure_code": code,
                "failure_name": FAILURE_MODE_DESCRIPTIONS.get(code, "Unknown"),
                "token_count": len(extract_tokens(text)),
                "pronoun_1st_ratio": psych["pronoun_1st_total_ratio"],
                "spatial_density": psych["spatial_detail_density"],
            })

    df_errors = pd.DataFrame(records)
    total_errors = len(df_errors)

    if total_errors > 0:
        counts = df_errors["failure_code"].value_counts().to_dict()
        prevalence = {code: round(counts.get(code, 0) / total_errors, 4) for code in FAILURE_MODE_DESCRIPTIONS.keys()}
    else:
        counts = {}
        prevalence = {code: 0.0 for code in FAILURE_MODE_DESCRIPTIONS.keys()}

    summary = {
        "total_samples": len(test_texts),
        "total_errors": total_errors,
        "error_rate": round(total_errors / len(test_texts), 4) if test_texts else 0.0,
        "counts": counts,
        "prevalence": prevalence,
    }

    return df_errors, summary
