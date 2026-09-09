"""
tests/test_tfidf.py — Unit tests for DualTfidfVectorizer.
"""

import tempfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import pytest

from src.features.tfidf_features import DualTfidfVectorizer


@pytest.fixture
def sample_corpus():
    return [
        "This is an amazing genuine hotel review with great service.",
        "Terrible experience, the room was dirty and noisy all night.",
        "Affordable price and very convenient location near the downtown market.",
        "Computer generated fake text with unusual repetitive phrasing.",
    ]


@pytest.fixture
def sample_cfg():
    return {
        "tfidf": {
            "word_ngram_range": [1, 2],
            "word_max_features": 50,
            "char_ngram_range": [3, 4],
            "char_max_features": 50,
            "sublinear_tf": True,
        }
    }


def test_init_and_from_config(sample_cfg):
    vec = DualTfidfVectorizer.from_config(sample_cfg)
    assert vec.word_ngram_range == (1, 2)
    assert vec.word_max_features == 50
    assert vec.char_ngram_range == (3, 4)
    assert vec.char_max_features == 50
    assert vec.sublinear_tf is True
    assert not vec._is_fitted


def test_fit_transform_shape_and_format(sample_corpus, sample_cfg):
    vec = DualTfidfVectorizer.from_config(sample_cfg)
    X = vec.fit_transform(sample_corpus)

    assert sp.isspmatrix_csr(X)
    assert X.shape[0] == len(sample_corpus)
    assert X.shape[1] > 0
    assert vec._is_fitted
    assert vec.n_features == X.shape[1]


def test_transform_unseen(sample_corpus, sample_cfg):
    vec = DualTfidfVectorizer.from_config(sample_cfg)
    vec.fit(sample_corpus)

    unseen = ["A brand new review with some overlapping words like hotel and room."]
    X_unseen = vec.transform(unseen)
    assert sp.isspmatrix_csr(X_unseen)
    assert X_unseen.shape[0] == 1
    assert X_unseen.shape[1] == vec.n_features


def test_not_fitted_error(sample_corpus):
    vec = DualTfidfVectorizer()
    with pytest.raises(RuntimeError):
        vec.transform(sample_corpus)
    with pytest.raises(RuntimeError):
        vec.get_feature_names_out()


def test_feature_names_prefixes(sample_corpus, sample_cfg):
    vec = DualTfidfVectorizer.from_config(sample_cfg)
    vec.fit(sample_corpus)

    feature_names = vec.get_feature_names_out()
    assert len(feature_names) == vec.n_features
    assert any(f.startswith("word__") for f in feature_names)
    assert any(f.startswith("char__") for f in feature_names)


def test_save_and_load_roundtrip(sample_corpus, sample_cfg):
    vec = DualTfidfVectorizer.from_config(sample_cfg)
    X_orig = vec.fit_transform(sample_corpus)

    with tempfile.TemporaryDirectory() as tmpdir:
        dump_path = Path(tmpdir) / "tfidf_vec.joblib"
        vec.save(dump_path)

        loaded_vec = DualTfidfVectorizer.load(dump_path)
        X_loaded = loaded_vec.transform(sample_corpus)

        np.testing.assert_allclose(X_orig.toarray(), X_loaded.toarray())
        assert loaded_vec.n_features == vec.n_features


def test_vocabulary_coverage(sample_corpus, sample_cfg):
    vec = DualTfidfVectorizer.from_config(sample_cfg)
    vec.fit(sample_corpus)

    target_texts = ["amazing hotel service", "completely unknown alien vocabulary"]
    cov = vec.compute_vocabulary_coverage(target_texts)

    assert "coverage" in cov
    assert "oov_rate" in cov
    assert 0.0 <= cov["coverage"] <= 1.0
    assert pytest.approx(cov["coverage"] + cov["oov_rate"], 1e-5) == 1.0
    assert cov["overlap_count"] >= 2  # 'amazing', 'hotel', 'service'
