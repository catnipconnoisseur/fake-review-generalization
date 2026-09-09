"""
tests/test_baseline.py — Unit tests for BaselineModel, train_baseline_model, and registry.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.models.baseline import BaselineModel, train_baseline_model
from src.models.registry import (
    EXPERIMENT_MODELS,
    get_model_info,
    is_classical,
    is_transformer,
    list_models,
    load_model,
    save_model,
)


@pytest.fixture
def mock_dataset():
    texts = [
        "Great quality product, completely authentic and works well.",
        "Fake generated repetitive sentence with fake quality.",
        "Loved the stay, friendly hotel staff and comfortable room.",
        "Scam advertisement, deceptive text written by bot.",
        "Standard genuine experience, arrived promptly as described.",
        "Totally artificial review praising nonexistent hotel amenities.",
        "Room was clean and spacious with a nice city view.",
        "Another deceptive bot post repeating artificial words.",
        "Great value for money, highly recommend this item.",
        "Fake synthetic text generated to mislead customers.",
    ]
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    groups = ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4", "g5", "g5"]
    cfg = {
        "seed": 42,
        "cross_eval": {"cv_folds": 2},
        "tfidf": {
            "word_ngram_range": [1, 2],
            "word_max_features": 50,
            "char_ngram_range": [3, 4],
            "char_max_features": 50,
            "sublinear_tf": True,
        },
        "models": {
            "logreg": {"C": [0.1, 1.0], "solver": "liblinear", "class_weight": "balanced", "max_iter": 100},
            "svm": {"C": [0.1, 1.0], "class_weight": "balanced", "max_iter": 500},
        },
    }
    return texts, labels, groups, cfg


def test_registry_metadata():
    assert len(EXPERIMENT_MODELS) == 8
    assert is_classical("A_logreg")
    assert is_classical("B_svm")
    assert is_transformer("A_bert")
    assert is_transformer("B_roberta")

    amazon_models = list_models(dataset="amazon")
    assert len(amazon_models) == 4
    assert "A_logreg" in amazon_models

    dosc_classical = list_models(dataset="dosc", family="classical")
    assert sorted(dosc_classical) == ["B_logreg", "B_svm"]


def test_train_logreg_with_groups(mock_dataset):
    texts, labels, groups, cfg = mock_dataset
    model = train_baseline_model(
        model_id="A_logreg",
        train_texts=texts,
        train_labels=labels,
        groups=groups,
        cfg=cfg,
        n_jobs=1,
    )

    assert isinstance(model, BaselineModel)
    assert model.model_id == "A_logreg"
    assert model.metadata["group_cv_used"] is True
    assert "C" in model.best_params

    # Check predictions
    preds = model.predict(texts[:4])
    assert len(preds) == 4
    assert set(preds).issubset({0, 1})

    # Check scores
    scores = model.predict_scores(texts[:4])
    assert len(scores) == 4
    assert all(0.0 <= s <= 1.0 for s in scores)  # predict_proba probabilities


def test_train_svm_stratified(mock_dataset):
    texts, labels, _, cfg = mock_dataset
    model = train_baseline_model(
        model_id="B_svm",
        train_texts=texts,
        train_labels=labels,
        groups=None,  # Stratified CV
        cfg=cfg,
        n_jobs=1,
    )

    assert isinstance(model, BaselineModel)
    assert model.model_id == "B_svm"
    assert model.metadata["group_cv_used"] is False

    scores = model.predict_scores(texts[:4])
    assert len(scores) == 4
    # LinearSVC decision_function returns signed floats
    assert all(isinstance(float(s), float) for s in scores)


def test_top_features_extraction(mock_dataset):
    texts, labels, _, cfg = mock_dataset
    model = train_baseline_model(
        model_id="B_logreg",
        train_texts=texts,
        train_labels=labels,
        cfg=cfg,
        n_jobs=1,
    )

    top_feats = model.get_top_features(n_top=5)
    assert "top_fake" in top_feats
    assert "top_genuine" in top_feats
    assert len(top_feats["top_fake"]) <= 5
    assert len(top_feats["top_genuine"]) <= 5
    assert all(isinstance(name, str) for name, _ in top_feats["top_fake"])
    assert all(isinstance(val, float) for _, val in top_feats["top_fake"])


def test_save_and_load_registry(mock_dataset):
    texts, labels, _, cfg = mock_dataset
    model = train_baseline_model(
        model_id="A_svm",
        train_texts=texts,
        train_labels=labels,
        cfg=cfg,
        n_jobs=1,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir)
        saved_path = save_model(model, "A_svm", models_dir)
        assert saved_path.exists()

        loaded_model = load_model("A_svm", models_dir)
        assert isinstance(loaded_model, BaselineModel)
        assert loaded_model.model_id == "A_svm"

        orig_preds = model.predict(texts)
        loaded_preds = loaded_model.predict(texts)
        np.testing.assert_array_equal(orig_preds, loaded_preds)
