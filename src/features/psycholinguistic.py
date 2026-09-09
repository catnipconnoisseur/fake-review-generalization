"""
src/features/psycholinguistic.py — Psycholinguistic & Forensic Feature Extraction.

Extracts psychological and deception-related markers from review texts:
- 1st-person pronouns (singular vs. plural)
- 2nd-person and 3rd-person pronouns
- Spatial & sensory detail density (concrete perceptual terms)
- Numerical & temporal detail density
- Punctuation and emotional intensity cues (exclamations, uppercase ratio)
"""

import re
from typing import Any, Dict, List, Set

from src.features.text_stats import extract_tokens


# Canonical word categories for psycholinguistic deception analysis
PRONOUNS_1ST_SINGULAR = {"i", "me", "my", "mine", "myself"}
PRONOUNS_1ST_PLURAL = {"we", "us", "our", "ours", "ourselves"}
PRONOUNS_2ND = {"you", "your", "yours", "yourself", "yourselves"}
PRONOUNS_3RD = {
    "he", "she", "it", "they", "him", "her", "them",
    "his", "hers", "their", "theirs", "himself", "herself", "itself", "themselves"
}

SPATIAL_CONCRETE_WORDS = {
    "room", "bed", "bathroom", "shower", "floor", "door", "window",
    "desk", "lobby", "elevator", "hallway", "street", "building", "corner",
    "downtown", "city", "location", "near", "view", "balcony", "pool",
    "restaurant", "bar", "wall", "carpet", "tv", "tub", "sink"
}

_NUMERICAL_REGEX = re.compile(r"\b\d+(?:[\.,]\d+)?\b|\$|\%|\bdollar(?:s)?\b|\bcent(?:s)?\b")


def compute_psycholinguistic_features(text: str) -> Dict[str, float]:
    """
    Compute psycholinguistic ratios and density metrics for a single document.
    """
    tokens = extract_tokens(text)
    n_tokens = len(tokens)
    n_chars = len(text)

    if n_tokens == 0:
        return {
            "pronoun_1st_singular_ratio": 0.0,
            "pronoun_1st_plural_ratio": 0.0,
            "pronoun_1st_total_ratio": 0.0,
            "pronoun_2nd_ratio": 0.0,
            "pronoun_3rd_ratio": 0.0,
            "spatial_detail_density": 0.0,
            "numerical_density": 0.0,
            "exclamation_rate": 0.0,
            "question_rate": 0.0,
            "uppercase_ratio": 0.0,
        }

    c1s = sum(1 for t in tokens if t in PRONOUNS_1ST_SINGULAR)
    c1p = sum(1 for t in tokens if t in PRONOUNS_1ST_PLURAL)
    c2 = sum(1 for t in tokens if t in PRONOUNS_2ND)
    c3 = sum(1 for t in tokens if t in PRONOUNS_3RD)
    c_spatial = sum(1 for t in tokens if t in SPATIAL_CONCRETE_WORDS)

    n_numbers = len(_NUMERICAL_REGEX.findall(text.lower()))
    n_exclamations = text.count("!")
    n_questions = text.count("?")
    n_upper = sum(1 for c in text if c.isupper())

    return {
        "pronoun_1st_singular_ratio": round(c1s / n_tokens, 4),
        "pronoun_1st_plural_ratio": round(c1p / n_tokens, 4),
        "pronoun_1st_total_ratio": round((c1s + c1p) / n_tokens, 4),
        "pronoun_2nd_ratio": round(c2 / n_tokens, 4),
        "pronoun_3rd_ratio": round(c3 / n_tokens, 4),
        "spatial_detail_density": round(c_spatial / n_tokens, 4),
        "numerical_density": round(n_numbers / n_tokens, 4),
        "exclamation_rate": round(n_exclamations / n_tokens, 4),
        "question_rate": round(n_questions / n_tokens, 4),
        "uppercase_ratio": round(n_upper / n_chars, 4) if n_chars > 0 else 0.0,
    }
