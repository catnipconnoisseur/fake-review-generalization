"""
tests/test_forensics.py — Unit tests for forensic features, text stats, and error taxonomy.
"""

import pandas as pd
import pytest

from src.evaluation.error_analysis import (
    analyze_errors,
    classify_failure_mode,
    compute_training_baselines,
)
from src.features.psycholinguistic import compute_psycholinguistic_features
from src.features.text_stats import compute_corpus_statistics, compute_text_statistics


def test_text_statistics():
    text = "The room was very clean. The bed was comfortable and soft."
    stats = compute_text_statistics(text)

    assert stats["token_count"] > 0
    assert stats["sentence_count"] == 2
    assert 0.0 < stats["ttr"] <= 1.0
    assert 0.0 <= stats["hapax_ratio"] <= 1.0


def test_corpus_statistics():
    corpus = [
        "A simple review with nice words.",
        "Another review talking about clean bathrooms and good service.",
    ]
    c_stats = compute_corpus_statistics(corpus)
    assert c_stats["n_documents"] == 2
    assert c_stats["total_tokens"] > 0
    assert c_stats["vocab_size"] > 0


def test_psycholinguistic_features():
    text = "I loved my stay at this hotel. We had a great time near the downtown elevator room!"
    feats = compute_psycholinguistic_features(text)

    assert feats["pronoun_1st_singular_ratio"] > 0.0  # "i", "my"
    assert feats["pronoun_1st_plural_ratio"] > 0.0    # "we"
    assert feats["spatial_detail_density"] > 0.0      # "hotel", "room", "elevator", "downtown"


def test_error_taxonomy_categorization():
    train_texts = [
        "Product works great and arrived quickly in the mail.",
        "Affordable price, durable quality, highly satisfied.",
        "Nice item for daily use at home.",
    ]
    baselines = compute_training_baselines(train_texts)

    # Review with unfamiliar hotel vocabulary
    text_unseen = "Concierge at luxury Chicago Hyatt suite was unresponsive."
    failure = classify_failure_mode(
        text=text_unseen,
        true_label=1,
        pred_label=0,
        train_baselines=baselines,
        source_dataset="amazon",
        target_dataset="dosc",
    )
    assert failure in ["VM", "TA", "LC", "SS", "SD"]
