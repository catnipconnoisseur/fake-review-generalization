"""
src/features/text_stats.py — Quantitative Text & Vocabulary Statistics.

Computes forensic lexical metrics across review corpora:
- Token counts, character counts, average word length
- Type-Token Ratio (TTR) — lexical diversity
- Hapax Legomena Ratio — fraction of words occurring exactly once
- Sentence count and mean sentence length
"""

import re
from collections import Counter
from typing import Any, Dict, List, Union


_TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")
_SENTENCE_PATTERN = re.compile(r"[.!?]+(?:\s+|$)")


def extract_tokens(text: str) -> List[str]:
    """Extract lowercase alphanumeric word tokens."""
    return _TOKEN_PATTERN.findall(text.lower())


def extract_sentences(text: str) -> List[str]:
    """Split text into sentence strings."""
    sentences = [s.strip() for s in _SENTENCE_PATTERN.split(text) if s.strip()]
    return sentences if sentences else [text.strip()]


def compute_text_statistics(text: str) -> Dict[str, float]:
    """
    Compute lexical and structural statistics for an individual document.
    """
    tokens = extract_tokens(text)
    sentences = extract_sentences(text)

    n_chars = len(text)
    n_tokens = len(tokens)
    n_sentences = len(sentences)

    if n_tokens == 0:
        return {
            "char_count": float(n_chars),
            "token_count": 0.0,
            "sentence_count": float(n_sentences),
            "avg_word_length": 0.0,
            "avg_sentence_length": 0.0,
            "ttr": 0.0,
            "hapax_ratio": 0.0,
        }

    token_counts = Counter(tokens)
    n_unique_tokens = len(token_counts)
    ttr = n_unique_tokens / n_tokens

    n_hapax = sum(1 for _, count in token_counts.items() if count == 1)
    hapax_ratio = n_hapax / n_unique_tokens if n_unique_tokens > 0 else 0.0

    avg_word_len = sum(len(t) for t in tokens) / n_tokens
    avg_sentence_len = n_tokens / n_sentences if n_sentences > 0 else float(n_tokens)

    return {
        "char_count": float(n_chars),
        "token_count": float(n_tokens),
        "sentence_count": float(n_sentences),
        "avg_word_length": round(avg_word_len, 2),
        "avg_sentence_length": round(avg_sentence_len, 2),
        "ttr": round(ttr, 4),
        "hapax_ratio": round(hapax_ratio, 4),
    }


def compute_corpus_statistics(texts: List[str]) -> Dict[str, Any]:
    """
    Aggregate lexical statistics across an entire corpus or subset.
    """
    all_tokens = []
    total_chars = 0
    total_sentences = 0

    doc_ttrs = []
    doc_token_counts = []

    for text in texts:
        stats = compute_text_statistics(text)
        doc_ttrs.append(stats["ttr"])
        doc_token_counts.append(stats["token_count"])
        all_tokens.extend(extract_tokens(text))
        total_chars += int(stats["char_count"])
        total_sentences += int(stats["sentence_count"])

    total_tokens = len(all_tokens)
    vocab = Counter(all_tokens)
    vocab_size = len(vocab)
    corpus_ttr = vocab_size / total_tokens if total_tokens > 0 else 0.0

    n_hapax = sum(1 for _, count in vocab.items() if count == 1)
    corpus_hapax_ratio = n_hapax / vocab_size if vocab_size > 0 else 0.0

    return {
        "n_documents": len(texts),
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "corpus_ttr": round(corpus_ttr, 4),
        "corpus_hapax_ratio": round(corpus_hapax_ratio, 4),
        "mean_doc_tokens": round(float(sum(doc_token_counts) / len(texts)), 2) if texts else 0.0,
        "mean_doc_ttr": round(float(sum(doc_ttrs) / len(texts)), 4) if texts else 0.0,
    }
