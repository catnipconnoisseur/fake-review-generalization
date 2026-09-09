"""
src/data/dedup.py — Deduplication and prefix-group assignment.

1. DOSC: Remove exact-duplicate text rows (keep first occurrence).
2. Amazon: Assign sha256-based group IDs to rows sharing ≥80-char
   text prefixes, ensuring grouped rows stay in the same fold.
"""

import hashlib
import pandas as pd
from pathlib import Path

from src.data.ingest import load_config


def deduplicate_dosc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact-duplicate rows from DOSC based on the `text` column.

    Forensic audit found 4 exact duplicates.  We keep the first
    occurrence and drop subsequent copies.

    Returns:
        Deduplicated DataFrame (expected: 1596 rows from 1600).
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    n_removed = n_before - len(df)
    print(f"  DOSC dedup: removed {n_removed} exact duplicate(s) "
          f"({n_before} → {len(df)})")
    return df


def assign_prefix_groups(
    df: pd.DataFrame,
    prefix_min_len: int = 80,
) -> pd.DataFrame:
    """
    Assign sha256-hashed group IDs to Amazon reviews based on the first
    `prefix_min_len` characters of each review's text.

    Reviews sharing the same prefix (≥80 chars) get the same group_id,
    ensuring they are never split across train/test folds.

    Args:
        df: Amazon DataFrame with a `text` column.
        prefix_min_len: Number of leading characters to hash.

    Returns:
        DataFrame with an added `group_id` column.
    """
    def _hash_prefix(text: str) -> str:
        prefix = text[:prefix_min_len] if len(text) >= prefix_min_len else text
        return hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:16]

    df = df.copy()
    df["group_id"] = df["text"].apply(_hash_prefix)

    # Diagnostic: how many groups, how many rows share a prefix
    n_groups = df["group_id"].nunique()
    group_sizes = df["group_id"].value_counts()
    n_shared = (group_sizes > 1).sum()
    max_group = group_sizes.max()

    print(f"  Amazon prefix groups: {n_groups} unique groups from {len(df)} rows")
    print(f"  Groups with >1 member (shared prefix): {n_shared}")
    print(f"  Largest group size: {max_group}")

    # Safety check: verify no group_id spans both labels
    cross_label_groups = (
        df.groupby("group_id")["target"]
        .nunique()
        .loc[lambda x: x > 1]
    )
    if len(cross_label_groups) > 0:
        print(f"  ⚠ WARNING: {len(cross_label_groups)} groups span both labels!")
        print(f"    This may indicate prefix collisions, not true shared prefixes.")
    else:
        print(f"  ✓ All prefix groups are label-pure (no cross-label contamination)")

    return df


if __name__ == "__main__":
    from src.data.ingest import ingest_amazon, ingest_dosc

    project_root = Path(__file__).resolve().parent.parent.parent
    cfg = load_config(project_root)

    print("=== DOSC Deduplication ===")
    dosc = ingest_dosc(project_root)
    dosc = deduplicate_dosc(dosc)
    assert len(dosc) == 1596, f"Expected 1596 rows, got {len(dosc)}"
    print(f"  ✓ DOSC: {len(dosc)} rows after dedup")

    print("\n=== Amazon Prefix Group Assignment ===")
    amz = ingest_amazon(project_root)
    prefix_len = cfg["preprocessing"]["prefix_min_len"]
    amz = assign_prefix_groups(amz, prefix_min_len=prefix_len)
    assert "group_id" in amz.columns, "group_id column missing!"
    print(f"  ✓ Amazon: {len(amz)} rows with group_id assigned")
