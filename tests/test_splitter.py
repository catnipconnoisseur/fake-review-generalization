"""
tests/test_splitter.py — Automated verification for data splitting.

Critical gate: zero prefix overlap between train/val/test splits for Amazon.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from src.data.ingest import ingest_amazon, ingest_dosc, load_config
from src.data.dedup import deduplicate_dosc, assign_prefix_groups
from src.data.splitter import split_amazon, split_dosc


@pytest.fixture(scope="module")
def cfg():
    return load_config(PROJECT_ROOT)


@pytest.fixture(scope="module")
def amazon_df(cfg):
    df = ingest_amazon(PROJECT_ROOT)
    return assign_prefix_groups(df, prefix_min_len=cfg["preprocessing"]["prefix_min_len"])


@pytest.fixture(scope="module")
def dosc_df():
    df = ingest_dosc(PROJECT_ROOT)
    return deduplicate_dosc(df)


@pytest.fixture(scope="module")
def amazon_splits(amazon_df, cfg):
    return split_amazon(
        amazon_df,
        test_size=cfg["preprocessing"]["test_size"],
        val_size=cfg["preprocessing"]["val_size"],
        seed=cfg["seed"],
    )


@pytest.fixture(scope="module")
def dosc_splits(dosc_df, cfg):
    return split_dosc(
        dosc_df,
        test_size=cfg["preprocessing"]["test_size"],
        val_size=cfg["preprocessing"]["val_size"],
        seed=cfg["seed"],
    )


class TestAmazonSplits:
    """Verify Amazon GroupShuffleSplit with prefix decontamination."""

    def test_no_index_overlap_train_test(self, amazon_splits):
        """Train and test must share zero row indices."""
        train = set(amazon_splits["train_idx"])
        test = set(amazon_splits["test_idx"])
        assert len(train & test) == 0, "Train/test index overlap detected!"

    def test_no_index_overlap_train_val(self, amazon_splits):
        """Train and val must share zero row indices."""
        train = set(amazon_splits["train_idx"])
        val = set(amazon_splits["val_idx"])
        assert len(train & val) == 0, "Train/val index overlap detected!"

    def test_no_index_overlap_val_test(self, amazon_splits):
        """Val and test must share zero row indices."""
        val = set(amazon_splits["val_idx"])
        test = set(amazon_splits["test_idx"])
        assert len(val & test) == 0, "Val/test index overlap detected!"

    def test_all_indices_covered(self, amazon_splits, amazon_df):
        """Union of train+val+test must equal all row indices."""
        all_idx = set(amazon_splits["train_idx"]) | set(amazon_splits["val_idx"]) | set(amazon_splits["test_idx"])
        assert all_idx == set(range(len(amazon_df))), "Some indices are missing from splits!"

    def test_zero_prefix_overlap_train_test(self, amazon_splits, amazon_df):
        """CRITICAL GATE: No shared 80-char prefix groups between train and test."""
        train_groups = set(amazon_df.iloc[amazon_splits["train_idx"]]["group_id"].unique())
        test_groups = set(amazon_df.iloc[amazon_splits["test_idx"]]["group_id"].unique())
        overlap = train_groups & test_groups
        assert len(overlap) == 0, (
            f"LEAKAGE: {len(overlap)} prefix groups overlap between train and test!"
        )

    def test_zero_prefix_overlap_train_val(self, amazon_splits, amazon_df):
        """No shared prefix groups between train and val."""
        train_groups = set(amazon_df.iloc[amazon_splits["train_idx"]]["group_id"].unique())
        val_groups = set(amazon_df.iloc[amazon_splits["val_idx"]]["group_id"].unique())
        overlap = train_groups & val_groups
        assert len(overlap) == 0, (
            f"LEAKAGE: {len(overlap)} prefix groups overlap between train and val!"
        )

    def test_approximate_split_ratios(self, amazon_splits, amazon_df):
        """Split sizes should be approximately 70/10/20 of total."""
        n = len(amazon_df)
        assert abs(len(amazon_splits["test_idx"]) / n - 0.20) < 0.05, "Test ratio way off"
        assert abs(len(amazon_splits["val_idx"]) / n - 0.10) < 0.05, "Val ratio way off"

    def test_label_presence_in_all_splits(self, amazon_splits, amazon_df):
        """Both labels must be present in every split."""
        for split_name in ("train_idx", "val_idx", "test_idx"):
            labels = set(amazon_df.iloc[amazon_splits[split_name]]["target"].unique())
            assert labels == {0, 1}, f"{split_name} missing a label class: {labels}"

    def test_metadata_prefix_overlap_zero(self, amazon_splits):
        """Metadata should record zero overlap."""
        meta = amazon_splits["metadata"]
        assert meta["train_test_prefix_overlap"] == 0
        assert meta["train_val_prefix_overlap"] == 0
        assert meta["val_test_prefix_overlap"] == 0


class TestDOSCSplits:
    """Verify DOSC StratifiedShuffleSplit."""

    def test_no_index_overlap(self, dosc_splits):
        """All three splits must be disjoint."""
        train = set(dosc_splits["train_idx"])
        val = set(dosc_splits["val_idx"])
        test = set(dosc_splits["test_idx"])
        assert len(train & test) == 0
        assert len(train & val) == 0
        assert len(val & test) == 0

    def test_all_indices_covered(self, dosc_splits, dosc_df):
        """Union must cover all rows."""
        all_idx = set(dosc_splits["train_idx"]) | set(dosc_splits["val_idx"]) | set(dosc_splits["test_idx"])
        assert all_idx == set(range(len(dosc_df)))

    def test_approximate_split_ratios(self, dosc_splits, dosc_df):
        """Split sizes should be approximately 70/10/20."""
        n = len(dosc_df)
        assert abs(len(dosc_splits["test_idx"]) / n - 0.20) < 0.05
        assert abs(len(dosc_splits["val_idx"]) / n - 0.10) < 0.05

    def test_label_balance_in_splits(self, dosc_splits, dosc_df):
        """Each split should have roughly balanced labels (within 10%)."""
        for split_name in ("train_idx", "val_idx", "test_idx"):
            labels = dosc_df.iloc[dosc_splits[split_name]]["target"]
            ratio = labels.mean()  # proportion of class 1
            assert 0.35 < ratio < 0.65, (
                f"{split_name} label imbalance: {ratio:.2f} (expected ~0.50)"
            )

    def test_checksum_present(self, dosc_splits):
        """Split data must include a SHA-256 checksum."""
        assert "checksum" in dosc_splits
        assert len(dosc_splits["checksum"]) == 64  # SHA-256 hex length
