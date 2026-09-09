"""
src/data/splitter.py — Reproducible, leakage-free data splitting.

Amazon: GroupShuffleSplit on prefix-hashed group_id to prevent
        train/test contamination from shared GPT-2 prompt templates.
DOSC:   StratifiedShuffleSplit with multi-label stratification on
        (target, polarity) to preserve class and sentiment balance.

All split indices are serialized to JSON with SHA-256 checksums
for full reproducibility.
"""

import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from src.data.ingest import load_config


def _compute_checksum(data: dict) -> str:
    """SHA-256 checksum of the JSON-serialized split data."""
    raw = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def split_amazon(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = 42,
) -> dict:
    """
    Split Amazon dataset using GroupShuffleSplit on `group_id`.

    Two-stage split:
      1. train+val / test  (GroupShuffleSplit, test_size)
      2. train / val       (GroupShuffleSplit, val_size relative to train+val)

    Returns:
        dict with keys: train_idx, val_idx, test_idx, metadata, checksum
    """
    assert "group_id" in df.columns, "Run assign_prefix_groups() first"

    groups = df["group_id"].values
    y = df["target"].values

    # Stage 1: Split off test set (grouped)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(df, y, groups=groups))

    # Stage 2: Split train from val within trainval (grouped)
    trainval_df = df.iloc[trainval_idx]
    trainval_groups = trainval_df["group_id"].values
    trainval_y = trainval_df["target"].values
    relative_val_size = val_size / (1.0 - test_size)

    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=relative_val_size, random_state=seed
    )
    train_local_idx, val_local_idx = next(
        gss_val.split(trainval_df, trainval_y, groups=trainval_groups)
    )

    # Map local indices back to global DataFrame indices
    train_idx = trainval_idx[train_local_idx]
    val_idx = trainval_idx[val_local_idx]

    # === CRITICAL VALIDATION: Zero prefix overlap ===
    train_prefixes = set(df.iloc[train_idx]["group_id"].unique())
    val_prefixes = set(df.iloc[val_idx]["group_id"].unique())
    test_prefixes = set(df.iloc[test_idx]["group_id"].unique())

    train_test_overlap = train_prefixes & test_prefixes
    train_val_overlap = train_prefixes & val_prefixes
    val_test_overlap = val_prefixes & test_prefixes

    assert len(train_test_overlap) == 0, (
        f"LEAKAGE: {len(train_test_overlap)} prefix groups overlap between train and test!"
    )
    assert len(train_val_overlap) == 0, (
        f"LEAKAGE: {len(train_val_overlap)} prefix groups overlap between train and val!"
    )
    assert len(val_test_overlap) == 0, (
        f"LEAKAGE: {len(val_test_overlap)} prefix groups overlap between val and test!"
    )

    # Build output
    split_data = {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "metadata": {
            "dataset": "amazon",
            "total_rows": len(df),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "seed": seed,
            "test_ratio": test_size,
            "val_ratio": val_size,
            "prefix_min_len": 80,
            "train_label_dist": pd.Series(y[train_idx]).value_counts().to_dict(),
            "val_label_dist": pd.Series(y[val_idx]).value_counts().to_dict(),
            "test_label_dist": pd.Series(y[test_idx]).value_counts().to_dict(),
            "train_test_prefix_overlap": len(train_test_overlap),
            "train_val_prefix_overlap": len(train_val_overlap),
            "val_test_prefix_overlap": len(val_test_overlap),
        },
    }
    split_data["checksum"] = _compute_checksum(split_data)

    return split_data


def split_dosc(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = 42,
) -> dict:
    """
    Split DOSC dataset using StratifiedShuffleSplit.

    Stratification key: compound of (target, polarity) to ensure
    balanced representation of both class and sentiment in each split.

    Returns:
        dict with keys: train_idx, val_idx, test_idx, metadata, checksum
    """
    y = df["target"].values

    # Create compound stratification key: target × polarity
    if "polarity" in df.columns:
        strat_key = df["target"].astype(str) + "_" + df["polarity"].astype(str)
    else:
        strat_key = df["target"].astype(str)

    strat_values = strat_key.values

    # Stage 1: Split off test set
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    trainval_idx, test_idx = next(sss_test.split(df, strat_values))

    # Stage 2: Split train from val within trainval
    trainval_strat = strat_values[trainval_idx]
    relative_val_size = val_size / (1.0 - test_size)

    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_val_size, random_state=seed
    )
    train_local_idx, val_local_idx = next(
        sss_val.split(np.arange(len(trainval_idx)), trainval_strat)
    )

    # Map back to global indices
    train_idx = trainval_idx[train_local_idx]
    val_idx = trainval_idx[val_local_idx]

    # Build output
    split_data = {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "metadata": {
            "dataset": "dosc",
            "total_rows": len(df),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "seed": seed,
            "test_ratio": test_size,
            "val_ratio": val_size,
            "train_label_dist": pd.Series(y[train_idx]).value_counts().to_dict(),
            "val_label_dist": pd.Series(y[val_idx]).value_counts().to_dict(),
            "test_label_dist": pd.Series(y[test_idx]).value_counts().to_dict(),
        },
    }
    split_data["checksum"] = _compute_checksum(split_data)

    return split_data


def save_splits(split_data: dict, output_path: Path) -> None:
    """Save split indices + metadata to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(split_data, f, indent=2)
    print(f"  Splits saved to {output_path}")
    print(f"  Checksum: {split_data['checksum'][:16]}...")


def load_splits(split_path: Path) -> dict:
    """Load and verify split indices from JSON."""
    with open(split_path, "r") as f:
        split_data = json.load(f)

    # Verify checksum
    stored_checksum = split_data.pop("checksum")
    computed_checksum = _compute_checksum(split_data)
    split_data["checksum"] = stored_checksum

    assert stored_checksum == computed_checksum, (
        f"Checksum mismatch! Stored: {stored_checksum[:16]}..., "
        f"Computed: {computed_checksum[:16]}..."
    )
    return split_data
