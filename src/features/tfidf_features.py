"""
src/features/tfidf_features.py — Dual TF-IDF Feature Extraction.

Extracts combined word-level and character-level TF-IDF representations:
1. Word n-grams (1, 2) up to 30,000 features with sublinear TF scaling.
2. Char n-grams (3, 5) up to 20,000 features with sublinear TF scaling.
Stacked into a single sparse CSR matrix (up to 50,000 features).
"""

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


class DualTfidfVectorizer:
    """
    Combined Word-level and Character-level TF-IDF Vectorizer.

    Produces a concatenated sparse CSR representation of text documents,
    capturing both lexical semantic n-grams (words) and stylistic / morphological
    patterns (character n-grams).
    """

    def __init__(
        self,
        word_ngram_range: Tuple[int, int] = (1, 2),
        word_max_features: int = 30000,
        char_ngram_range: Tuple[int, int] = (3, 5),
        char_max_features: int = 20000,
        sublinear_tf: bool = True,
    ):
        self.word_ngram_range = tuple(word_ngram_range)
        self.word_max_features = word_max_features
        self.char_ngram_range = tuple(char_ngram_range)
        self.char_max_features = char_max_features
        self.sublinear_tf = sublinear_tf

        self.word_vectorizer = TfidfVectorizer(
            ngram_range=self.word_ngram_range,
            analyzer="word",
            max_features=self.word_max_features,
            sublinear_tf=self.sublinear_tf,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.char_vectorizer = TfidfVectorizer(
            ngram_range=self.char_ngram_range,
            analyzer="char",
            max_features=self.char_max_features,
            sublinear_tf=self.sublinear_tf,
        )
        self._is_fitted = False

    @classmethod
    def from_config(cls, cfg: dict) -> "DualTfidfVectorizer":
        """Factory method to instantiate from the project config dictionary."""
        tfidf_cfg = cfg.get("tfidf", {})
        return cls(
            word_ngram_range=tuple(tfidf_cfg.get("word_ngram_range", [1, 2])),
            word_max_features=tfidf_cfg.get("word_max_features", 30000),
            char_ngram_range=tuple(tfidf_cfg.get("char_ngram_range", [3, 5])),
            char_max_features=tfidf_cfg.get("char_max_features", 20000),
            sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
        )

    def fit(self, raw_documents: Iterable[str]) -> "DualTfidfVectorizer":
        """Fit both word and char TF-IDF vectorizers on training documents."""
        docs = list(raw_documents)
        self.word_vectorizer.fit(docs)
        self.char_vectorizer.fit(docs)
        self._is_fitted = True
        return self

    def transform(self, raw_documents: Iterable[str]) -> sp.csr_matrix:
        """Transform documents to dual TF-IDF sparse matrix."""
        if not self._is_fitted:
            raise RuntimeError("DualTfidfVectorizer is not fitted yet. Call fit() first.")
        docs = list(raw_documents)
        X_word = self.word_vectorizer.transform(docs)
        X_char = self.char_vectorizer.transform(docs)
        return sp.hstack([X_word, X_char], format="csr")

    def fit_transform(self, raw_documents: Iterable[str]) -> sp.csr_matrix:
        """Fit to documents, then transform them to sparse CSR matrix."""
        docs = list(raw_documents)
        X_word = self.word_vectorizer.fit_transform(docs)
        X_char = self.char_vectorizer.fit_transform(docs)
        self._is_fitted = True
        return sp.hstack([X_word, X_char], format="csr")

    def get_feature_names_out(self) -> np.ndarray:
        """
        Return concatenated feature names with 'word__' and 'char__' prefixes.
        """
        if not self._is_fitted:
            raise RuntimeError("DualTfidfVectorizer is not fitted yet.")
        word_names = np.array([f"word__{f}" for f in self.word_vectorizer.get_feature_names_out()])
        char_names = np.array([f"char__{f}" for f in self.char_vectorizer.get_feature_names_out()])
        return np.concatenate([word_names, char_names])

    @property
    def n_features(self) -> int:
        """Total number of extracted features across word and char extractors."""
        if not self._is_fitted:
            return 0
        return len(self.word_vectorizer.vocabulary_) + len(self.char_vectorizer.vocabulary_)

    def compute_vocabulary_coverage(self, target_texts: Iterable[str]) -> Dict[str, Union[int, float]]:
        """
        Compute vocabulary overlap and out-of-vocabulary rate against target texts.
        Uses word-level tokens from target texts to evaluate lexical coverage.
        """
        if not self._is_fitted:
            raise RuntimeError("Vectorizer must be fitted before computing vocabulary coverage.")

        source_vocab = set(self.word_vectorizer.vocabulary_.keys())
        # Fit a temporary unconstrained word vectorizer on target texts to get target word vocabulary
        target_vec = TfidfVectorizer(
            ngram_range=(1, 1),
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )
        target_vec.fit(list(target_texts))
        target_vocab = set(target_vec.vocabulary_.keys())

        overlap = source_vocab.intersection(target_vocab)
        coverage = len(overlap) / len(target_vocab) if len(target_vocab) > 0 else 0.0
        oov_rate = 1.0 - coverage

        return {
            "source_word_vocab_size": len(source_vocab),
            "target_word_vocab_size": len(target_vocab),
            "overlap_count": len(overlap),
            "coverage": float(coverage),
            "oov_rate": float(oov_rate),
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize fitted vectorizer using joblib."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DualTfidfVectorizer":
        """Deserialize a fitted vectorizer from disk."""
        obj = joblib.load(filepath)
        if not isinstance(obj, DualTfidfVectorizer):
            raise TypeError(f"Expected DualTfidfVectorizer, got {type(obj)}")
        return obj
