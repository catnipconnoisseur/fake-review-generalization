"""
tests/test_cross_eval.py — Unit tests for cross-dataset evaluation engine.
"""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.evaluation.cross_eval import (
    build_transfer_dataframe,
    evaluate_model_on_dataset,
    run_single_cross_eval,
)
from src.models.baseline import train_baseline_model
from src.models.registry import save_model


@pytest.fixture
def mock_cross_env():
    amazon_df = pd.DataFrame({
        "text": [
            "Great electronic product, fast shipping and works well.",
            "Fake generated bot review with weird repetitive phrases.",
            "Authentic genuine product arrived safely in good condition.",
            "Artificial generated deceptive text with spam words.",
            "Good quality headphones, decent bass and clean sound.",
            "Synthetic computer generated fake praise text.",
            "Nice compact design and reliable performance daily.",
            "Deceptive fake bot advertisement text.",
        ],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
        "group_id": ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"],
    })
    amazon_splits = {
        "train_idx": [0, 1, 2, 3],
        "val_idx": [4, 5],
        "test_idx": [6, 7],
    }

    dosc_df = pd.DataFrame({
        "text": [
            "Terrible hotel experience, dirty bathroom and rude staff.",
            "I stayed at this luxury Chicago hotel with my husband.",
            "Comfortable bed and clean room, good stay overall.",
            "My vacation in Chicago was ruined by the fake service.",
            "The room rate was affordable and close to downtown.",
            "When I visited Chicago for my vacation it was awful.",
            "Great location and friendly front desk team.",
            "Fake deceptive review written for money.",
        ],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    dosc_splits = {
        "train_idx": [0, 1, 2, 3],
        "val_idx": [4, 5],
        "test_idx": [6, 7],
    }

    cfg = {
        "seed": 42,
        "cross_eval": {
            "cv_folds": 2,
            "interpretation_thresholds": {"excellent": 0.05, "moderate": 0.15, "severe": 0.30},
        },
        "tfidf": {
            "word_ngram_range": [1, 2],
            "word_max_features": 30,
            "char_ngram_range": [3, 4],
            "char_max_features": 30,
            "sublinear_tf": True,
        },
        "models": {
            "logreg": {"C": [0.1, 1.0], "solver": "liblinear", "class_weight": "balanced", "max_iter": 50},
            "svm": {"C": [0.1, 1.0], "class_weight": "balanced", "max_iter": 50},
        },
    }

    return amazon_df, amazon_splits, dosc_df, dosc_splits, cfg


def test_cross_eval_pipeline(mock_cross_env):
    amazon_df, amazon_splits, dosc_df, dosc_splits, cfg = mock_cross_env

    # Train a mock A_logreg model
    model = train_baseline_model(
        model_id="A_logreg",
        train_texts=amazon_df.iloc[amazon_splits["train_idx"]]["text"],
        train_labels=amazon_df.iloc[amazon_splits["train_idx"]]["target"],
        groups=amazon_df.iloc[amazon_splits["train_idx"]]["group_id"],
        cfg=cfg,
        n_jobs=1,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir)
        save_model(model, "A_logreg", models_dir)

        result = run_single_cross_eval(
            model_id="A_logreg",
            models_dir=models_dir,
            amazon_df=amazon_df,
            amazon_splits=amazon_splits,
            dosc_df=dosc_df,
            dosc_splits=dosc_splits,
            cfg=cfg,
        )

        assert result["model_id"] == "A_logreg"
        assert result["source_dataset"] == "amazon"
        assert result["target_dataset"] == "dosc"
        assert "within_eval" in result
        assert "cross_eval" in result
        assert "deltas" in result
        assert "vocab_coverage" in result["cross_eval"]

        df_table = build_transfer_dataframe([result])
        assert len(df_table) == 1
        assert "Delta F1 (Drop)" in df_table.columns
        assert "Cross Vocab Coverage" in df_table.columns
