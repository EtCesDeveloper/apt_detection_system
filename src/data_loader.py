"""
Data loading and cleaning for CICIDS2017.

CICIDS2017 is delivered as multiple CSV files (one per day/attack type).
This module:
  - Loads and concatenates them
  - Cleans column names (they have leading whitespace in the original files)
  - Drops infinite/NaN values
  - Builds a binary label: 0 = BENIGN, 1 = ATTACK (any non-benign label)
  - Drops non-numeric / leakage columns (Flow ID, IPs, timestamp, etc.)
"""

from __future__ import annotations
import glob
import numpy as np
import pandas as pd
from pathlib import Path

from . import config


# Columns that must be removed before feeding the model
# (identifiers and metadata that would cause data leakage)
LEAKAGE_COLUMNS = [
    "Flow ID", "Source IP", "Src IP",
    "Destination IP", "Dst IP",
    "Source Port", "Src Port",
    "Destination Port", "Dst Port",
    "Protocol", "Timestamp",
]


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """CICIDS2017 column names contain leading/trailing spaces — normalize them."""
    df.columns = [c.strip() for c in df.columns]
    return df


def _drop_leakage_and_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifier-like columns that should not be used as features."""
    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop, errors="ignore")


def _binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the multi-class 'Label' column into a binary target.
    BENIGN -> 0, anything else (DoS, PortScan, Bot, Infiltration, etc.) -> 1
    """
    if "Label" not in df.columns:
        raise KeyError("Expected a 'Label' column in CICIDS2017 CSV files.")
    df["Label"] = (df["Label"].astype(str).str.upper() != "BENIGN").astype(np.int8)
    return df


def load_cicids2017(raw_dir: Path = config.RAW_DATA_DIR) -> pd.DataFrame:
    """
    Load every CSV under data/raw/ and return a single cleaned DataFrame.
    Numeric features only + binary 'Label'.
    """
    csv_paths = sorted(glob.glob(str(raw_dir / "**" / "*.csv"), recursive=True))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")

    print(f"[data_loader] Found {len(csv_paths)} CSV files.")
    frames = []
    for p in csv_paths:
        # low_memory=False prevents dtype guessing chunk-by-chunk
        df = pd.read_csv(p, low_memory=False, encoding="latin-1")
        df = _clean_column_names(df)
        frames.append(df)
        print(f"  - {Path(p).name}: {len(df):,} rows")

    df = pd.concat(frames, ignore_index=True)
    print(f"[data_loader] Concatenated rows: {len(df):,}")

    # Cleaning pipeline
    df = _drop_leakage_and_meta(df)
    df = _binary_label(df)

    # Replace inf/-inf with NaN, then drop NaN rows.
    # CICIDS2017 has many inf values in rate-based columns (e.g. Flow Bytes/s).
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Keep numeric columns only (some versions still leave object columns around)
    label = df["Label"]
    features = df.drop(columns=["Label"]).select_dtypes(include=[np.number])
    df = pd.concat([features, label], axis=1)

    print(f"[data_loader] Final rows: {len(df):,} | features: {features.shape[1]}")
    print(f"[data_loader] Class balance:\n{df['Label'].value_counts(normalize=True)}")
    return df
