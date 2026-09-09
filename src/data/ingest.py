"""
src/data/ingest.py — Raw CSV ingestion and label unification.

Loads the two raw datasets, renames columns to a canonical schema,
unifies labels to binary integers {0=genuine, 1=fake}, and drops
null/empty text rows.  NEVER modifies the raw files on disk.
"""

import pandas as pd
import yaml
from pathlib import Path


def load_config(project_root: Path) -> dict:
    """Load the central YAML configuration."""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ingest_amazon(project_root: Path) -> pd.DataFrame:
    """
    Ingest Amazon synthetic reviews dataset.

    Raw columns: category, rating, label, text_
    Target mapping: CG (Computer Generated) → 1, OR (Original) → 0
    """
    cfg = load_config(project_root)
    raw_path = project_root / cfg["data"]["amazon_raw"]

    df = pd.read_csv(raw_path)

    # Rename columns to canonical schema
    df = df.rename(columns={"text_": "text", "label": "target"})

    # Unify labels to binary integers
    label_map = {"CG": 1, "OR": 0}
    df["target"] = df["target"].map(label_map)

    # Validate no unmapped labels
    assert df["target"].notna().all(), (
        f"Unmapped labels found: {df.loc[df['target'].isna(), 'target'].unique()}"
    )
    df["target"] = df["target"].astype(int)

    # Drop rows with null or empty text
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().str.len() > 0].copy()
    df["text"] = df["text"].str.strip()

    # Keep only relevant columns: text + target + metadata
    df = df[["text", "target", "category", "rating"]].reset_index(drop=True)

    return df


def ingest_dosc(project_root: Path) -> pd.DataFrame:
    """
    Ingest Deceptive Opinion Spam Corpus (DOSC).

    Raw columns: deceptive, hotel, polarity, source, text
    Target mapping: deceptive → 1, truthful → 0

    CRITICAL: The `source` column is DROPPED immediately because it
    perfectly separates classes (MTurk = all deceptive, TripAdvisor/Web
    = all truthful), causing 100% target leakage if retained.
    """
    cfg = load_config(project_root)
    raw_path = project_root / cfg["data"]["dosc_raw"]

    df = pd.read_csv(raw_path)

    # === LEAKAGE CONTROL: Drop `source` column FIRST ===
    if "source" in df.columns:
        df = df.drop(columns=["source"])

    # Unify labels to binary integers
    label_map = {"deceptive": 1, "truthful": 0}
    df["target"] = df["deceptive"].map(label_map)

    # Validate no unmapped labels
    assert df["target"].notna().all(), (
        f"Unmapped labels found: {df.loc[df['target'].isna(), 'deceptive'].unique()}"
    )
    df["target"] = df["target"].astype(int)

    # Drop rows with null or empty text
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().str.len() > 0].copy()
    df["text"] = df["text"].str.strip()

    # Keep only relevant columns: text + target + metadata (hotel, polarity)
    # NOTE: `source` is already dropped above
    df = df[["text", "target", "hotel", "polarity"]].reset_index(drop=True)

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    print(f"Project root: {project_root}")

    amz = ingest_amazon(project_root)
    print(f"\nAmazon: {len(amz)} rows")
    print(f"  Columns: {list(amz.columns)}")
    print(f"  Labels:  {amz['target'].value_counts().to_dict()}")
    print(f"  Null text: {amz['text'].isna().sum()}")

    dosc = ingest_dosc(project_root)
    print(f"\nDOSC: {len(dosc)} rows")
    print(f"  Columns: {list(dosc.columns)}")
    print(f"  Labels:  {dosc['target'].value_counts().to_dict()}")
    print(f"  Null text: {dosc['text'].isna().sum()}")
    assert "source" not in dosc.columns, "LEAKAGE: `source` column still present!"
    print("  ✓ `source` column correctly excluded")
