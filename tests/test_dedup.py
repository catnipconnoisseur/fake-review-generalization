"""
tests/test_dedup.py — Automated verification for deduplication and leakage control.
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.data.ingest import ingest_amazon, ingest_dosc
from src.data.dedup import deduplicate_dosc, assign_prefix_groups


class TestDOSCDeduplication:
    """Verify DOSC deduplication and leakage controls."""

    @pytest.fixture(scope="class")
    def dosc_raw(self):
        return ingest_dosc(PROJECT_ROOT)

    @pytest.fixture(scope="class")
    def dosc_deduped(self, dosc_raw):
        return deduplicate_dosc(dosc_raw)

    def test_raw_dosc_has_1600_rows(self, dosc_raw):
        """DOSC raw ingestion should produce exactly 1600 rows."""
        assert len(dosc_raw) == 1600, f"Expected 1600, got {len(dosc_raw)}"

    def test_deduped_dosc_has_1596_rows(self, dosc_deduped):
        """After removing 4 duplicates, DOSC should have 1596 rows."""
        assert len(dosc_deduped) == 1596, f"Expected 1596, got {len(dosc_deduped)}"

    def test_source_column_absent(self, dosc_raw):
        """The `source` column MUST be absent to prevent target leakage."""
        assert "source" not in dosc_raw.columns, (
            "CRITICAL LEAKAGE: `source` column is still present!"
        )

    def test_labels_are_binary_integers(self, dosc_deduped):
        """Labels must be {0, 1} integers only."""
        assert set(dosc_deduped["target"].unique()) == {0, 1}
        assert dosc_deduped["target"].dtype in ("int64", "int32")

    def test_no_null_text(self, dosc_deduped):
        """No null or empty text values allowed."""
        assert dosc_deduped["text"].notna().all()
        assert (dosc_deduped["text"].str.len() > 0).all()

    def test_no_remaining_duplicates(self, dosc_deduped):
        """After dedup, no exact text duplicates should remain."""
        assert dosc_deduped["text"].duplicated().sum() == 0


class TestAmazonPrefixGroups:
    """Verify Amazon prefix-group assignment."""

    @pytest.fixture(scope="class")
    def amazon_grouped(self):
        df = ingest_amazon(PROJECT_ROOT)
        return assign_prefix_groups(df, prefix_min_len=80)

    def test_amazon_has_40432_rows(self, amazon_grouped):
        """Amazon dataset should have exactly 40432 rows."""
        assert len(amazon_grouped) == 40432, (
            f"Expected 40432, got {len(amazon_grouped)}"
        )

    def test_group_id_column_exists(self, amazon_grouped):
        """group_id column must be present after prefix assignment."""
        assert "group_id" in amazon_grouped.columns

    def test_group_ids_are_nonempty_strings(self, amazon_grouped):
        """Every row must have a non-empty group_id."""
        assert amazon_grouped["group_id"].notna().all()
        assert (amazon_grouped["group_id"].str.len() > 0).all()

    def test_labels_are_binary_integers(self, amazon_grouped):
        """Labels must be {0, 1} integers only."""
        assert set(amazon_grouped["target"].unique()) == {0, 1}

    def test_prefix_groups_are_label_pure(self, amazon_grouped):
        """No prefix group should contain both CG and OR labels."""
        cross = (
            amazon_grouped.groupby("group_id")["target"]
            .nunique()
            .loc[lambda x: x > 1]
        )
        assert len(cross) == 0, (
            f"{len(cross)} groups span both labels — possible prefix collision"
        )

    def test_shared_prefix_groups_exist(self, amazon_grouped):
        """Forensic finding: ~550 rows share prefixes, so >1-member groups must exist."""
        group_sizes = amazon_grouped["group_id"].value_counts()
        n_shared = (group_sizes > 1).sum()
        assert n_shared > 0, "Expected shared prefix groups but found none"
