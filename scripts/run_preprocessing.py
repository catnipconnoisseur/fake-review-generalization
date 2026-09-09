#!/usr/bin/env python3
"""
scripts/run_preprocessing.py — End-to-end data preprocessing pipeline.

Executes the full Phase 1 pipeline:
  1. Ingest raw CSVs (unify labels, drop source column)
  2. Deduplicate DOSC, assign Amazon prefix groups
  3. Split both datasets (GroupShuffleSplit / StratifiedShuffleSplit)
  4. Save clean parquet files and split JSONs
  5. Print verification summary

Usage:
    python scripts/run_preprocessing.py
"""

import hashlib
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.ingest import ingest_amazon, ingest_dosc, load_config
from src.data.dedup import deduplicate_dosc, assign_prefix_groups
from src.data.splitter import split_amazon, split_dosc, save_splits


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    cfg = load_config(PROJECT_ROOT)
    seed = cfg["seed"]
    prefix_len = cfg["preprocessing"]["prefix_min_len"]
    test_size = cfg["preprocessing"]["test_size"]
    val_size = cfg["preprocessing"]["val_size"]

    # Set global seeds
    np.random.seed(seed)

    print("=" * 70)
    print("  PHASE 1: Data Preprocessing & Splitting Pipeline")
    print("=" * 70)
    print(f"  Seed: {seed}")
    print(f"  Prefix min length: {prefix_len}")
    print(f"  Test size: {test_size}, Val size: {val_size}")
    print()

    # ─── Step 1: Ingest ────────────────────────────────────────────────
    print("─── Step 1: Ingestion ───")
    amz = ingest_amazon(PROJECT_ROOT)
    dosc = ingest_dosc(PROJECT_ROOT)
    print(f"  Amazon: {len(amz)} rows, labels: {amz['target'].value_counts().to_dict()}")
    print(f"  DOSC:   {len(dosc)} rows, labels: {dosc['target'].value_counts().to_dict()}")
    assert "source" not in dosc.columns, "LEAKAGE: `source` column present in DOSC!"
    print("  ✓ DOSC `source` column correctly excluded")
    print()

    # ─── Step 2: Deduplication & Prefix Groups ─────────────────────────
    print("─── Step 2: Deduplication & Prefix Groups ───")
    dosc = deduplicate_dosc(dosc)
    amz = assign_prefix_groups(amz, prefix_min_len=prefix_len)
    print()

    # ─── Step 3: Split ─────────────────────────────────────────────────
    print("─── Step 3: Splitting ───")
    print("  Splitting Amazon (GroupShuffleSplit on prefix groups)...")
    amz_splits = split_amazon(amz, test_size=test_size, val_size=val_size, seed=seed)
    print(f"    Train: {amz_splits['metadata']['train_size']}")
    print(f"    Val:   {amz_splits['metadata']['val_size']}")
    print(f"    Test:  {amz_splits['metadata']['test_size']}")
    print(f"    ✓ Prefix overlap (train↔test): {amz_splits['metadata']['train_test_prefix_overlap']}")
    print()

    print("  Splitting DOSC (StratifiedShuffleSplit on target×polarity)...")
    dosc_splits = split_dosc(dosc, test_size=test_size, val_size=val_size, seed=seed)
    print(f"    Train: {dosc_splits['metadata']['train_size']}")
    print(f"    Val:   {dosc_splits['metadata']['val_size']}")
    print(f"    Test:  {dosc_splits['metadata']['test_size']}")
    print()

    # ─── Step 4: Save ──────────────────────────────────────────────────
    print("─── Step 4: Saving Outputs ───")

    # Save clean parquet files
    amz_parquet = PROJECT_ROOT / cfg["data"]["amazon_clean"]
    dosc_parquet = PROJECT_ROOT / cfg["data"]["dosc_clean"]
    amz_parquet.parent.mkdir(parents=True, exist_ok=True)
    dosc_parquet.parent.mkdir(parents=True, exist_ok=True)

    amz.to_parquet(amz_parquet, index=False)
    dosc.to_parquet(dosc_parquet, index=False)
    print(f"  ✓ Amazon clean: {amz_parquet} ({amz_parquet.stat().st_size:,} bytes)")
    print(f"  ✓ DOSC clean:   {dosc_parquet} ({dosc_parquet.stat().st_size:,} bytes)")

    # Save split JSONs
    amz_split_path = PROJECT_ROOT / cfg["data"]["amazon_splits"]
    dosc_split_path = PROJECT_ROOT / cfg["data"]["dosc_splits"]
    save_splits(amz_splits, amz_split_path)
    save_splits(dosc_splits, dosc_split_path)
    print()

    # ─── Step 5: Verification Summary ──────────────────────────────────
    print("=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)

    checks = []

    # F1.1: Amazon row count
    c = len(amz) == 40432
    checks.append(c)
    print(f"  [{'✓' if c else '✗'}] F1.1: Amazon has {len(amz)} rows (expected 40432)")

    # F1.2: DOSC row count
    c = len(dosc) == 1596
    checks.append(c)
    print(f"  [{'✓' if c else '✗'}] F1.2: DOSC has {len(dosc)} rows (expected 1596)")

    # F1.3: DOSC source column absent
    c = "source" not in dosc.columns
    checks.append(c)
    print(f"  [{'✓' if c else '✗'}] F1.3: DOSC `source` column excluded")

    # F1.4: Zero prefix overlap
    c = amz_splits["metadata"]["train_test_prefix_overlap"] == 0
    checks.append(c)
    print(f"  [{'✓' if c else '✗'}] F1.4: Zero prefix overlap (train↔test)")

    c = amz_splits["metadata"]["train_val_prefix_overlap"] == 0
    checks.append(c)
    print(f"  [{'✓' if c else '✗'}] F1.4b: Zero prefix overlap (train↔val)")

    # F1.5: Binary labels
    c = set(amz["target"].unique()) == {0, 1} and set(dosc["target"].unique()) == {0, 1}
    checks.append(c)
    print(f"  [{'✓' if c else '✗'}] F1.5: All labels are binary {{0, 1}}")

    # F1.6: No null text
    c = amz["text"].notna().all() and dosc["text"].notna().all()
    checks.append(c)
    print(f"  [{'✓' if c else '✗'}] F1.6: No null/empty text values")

    # F1.7: Output files exist
    c = amz_parquet.exists() and dosc_parquet.exists()
    checks.append(c)
    print(f"  [{'✓' if c else '✗'}] F1.7a: Parquet files exist")

    c = amz_split_path.exists() and dosc_split_path.exists()
    checks.append(c)
    print(f"  [{'✓' if c else '✗'}] F1.7b: Split JSON files exist")

    # File checksums
    print()
    print("  File checksums:")
    for p in [amz_parquet, dosc_parquet, amz_split_path, dosc_split_path]:
        print(f"    {p.name}: {sha256_file(p)[:32]}...")

    print()
    n_pass = sum(checks)
    n_total = len(checks)
    status = "ALL PASSED ✓" if n_pass == n_total else f"FAILED ({n_total - n_pass} failures)"
    print(f"  Result: {n_pass}/{n_total} checks — {status}")
    print("=" * 70)

    if n_pass < n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
