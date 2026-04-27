"""
Diagnostic script — run BEFORE training to validate the dataset.

Checks:
  1. Which CICIDS2017 variant is loaded (MachineLearningCSV vs GeneratedLabelledFlows)
  2. Column inventory (count, names, dtypes)
  3. Class balance (BENIGN vs each attack type)
  4. Inf/NaN counts per column
  5. Timestamp ordering (relevant for sequence construction)
  6. Memory footprint estimate

Run with:
    python -m src.diagnose
"""

from __future__ import annotations
import glob
import numpy as np
import pandas as pd
from pathlib import Path

from . import config


# Columns specific to each variant — we use these to identify which one we have
GENERATED_FLOWS_MARKERS = {"Flow ID", "Source IP", "Destination IP"}
ML_CSV_MARKERS_ONLY    = {"Bwd Avg Bytes/Bulk", "Idle Mean"}  # present in both, kept for ref


def detect_variant(columns: set[str]) -> str:
    """Identify which CICIDS2017 variant we have based on column names."""
    cols_clean = {c.strip() for c in columns}
    has_ids = bool(GENERATED_FLOWS_MARKERS & cols_clean)
    if has_ids:
        return "GeneratedLabelledFlows  (forensics — has IPs/ports/timestamps)"
    return "MachineLearningCSV  (the right one for ML — features only)"


def main() -> None:
    csv_paths = sorted(glob.glob(str(config.RAW_DATA_DIR / "**" / "*.csv"),
                                  recursive=True))

    print(f"\n{'='*70}")
    print("CICIDS2017 DATASET DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"data/raw/ contains {len(csv_paths)} CSV files:")
    for p in csv_paths:
        size_mb = Path(p).stat().st_size / (1024 * 1024)
        print(f"  - {Path(p).name}  ({size_mb:.1f} MB)")

    if not csv_paths:
        print("\n[ERROR] No CSVs found in data/raw/")
        return

    # Inspect just the first file to detect variant
    print(f"\n--- Inspecting {Path(csv_paths[0]).name} ---")
    df_sample = pd.read_csv(csv_paths[0], low_memory=False, encoding="latin-1",
                             nrows=1000)
    df_sample.columns = [c.strip() for c in df_sample.columns]
    cols = set(df_sample.columns)

    print(f"\n[1] Variant detected: {detect_variant(cols)}")
    print(f"[2] Total columns: {len(cols)}")
    print(f"    First 10: {list(df_sample.columns[:10])}")

    # Now load all files in full to check class balance
    print(f"\n[3] Loading all {len(csv_paths)} files (this may take a minute)...")
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p, low_memory=False, encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print(f"    Total rows: {len(df):,}")

    # Class balance with attack-type breakdown
    if "Label" in df.columns:
        print(f"\n[4] Label distribution (top 15):")
        vc = df["Label"].value_counts().head(15)
        total = len(df)
        for label, count in vc.items():
            pct = 100 * count / total
            print(f"    {label:<30s} {count:>10,d}  ({pct:5.2f}%)")

        binary = (df["Label"].astype(str).str.upper() != "BENIGN").astype(int)
        print(f"\n    Binary  BENIGN: {(binary == 0).sum():,}  "
              f"({(binary == 0).mean() * 100:.2f}%)")
        print(f"    Binary  ATTACK: {(binary == 1).sum():,}  "
              f"({(binary == 1).mean() * 100:.2f}%)")

    # Inf / NaN inventory on numeric columns
    numeric = df.select_dtypes(include=[np.number])
    inf_total = np.isinf(numeric.values).sum()
    nan_total = numeric.isna().sum().sum()
    print(f"\n[5] Numeric data quality:")
    print(f"    Inf values:  {inf_total:,}")
    print(f"    NaN values:  {nan_total:,}")
    if inf_total or nan_total:
        bad_rows = numeric.replace([np.inf, -np.inf], np.nan).isna().any(axis=1).sum()
        print(f"    Rows with at least one inf/NaN: {bad_rows:,} "
              f"({100 * bad_rows / len(df):.2f}% of total)")
        print(f"    [These will be dropped by data_loader.]")

    # Timestamp ordering
    if "Timestamp" in df.columns:
        print(f"\n[6] Timestamp column present — first 3 values:")
        print(f"    {df['Timestamp'].head(3).tolist()}")
        print(f"    [data_loader currently does NOT sort by timestamp.]")
    else:
        print(f"\n[6] No 'Timestamp' column — natural file order will be used.")

    # Memory footprint after cleaning estimate
    rows_after_clean = len(df) - (numeric.replace([np.inf, -np.inf], np.nan)
                                  .isna().any(axis=1).sum() if (inf_total or nan_total) else 0)
    n_features = numeric.shape[1] - (1 if "Label" in numeric.columns else 0)
    seq_count = rows_after_clean // config.TIMESTEPS
    bytes_per_seq = config.TIMESTEPS * n_features * 4   # float32
    total_mb = (seq_count * bytes_per_seq) / (1024 * 1024)
    print(f"\n[7] Memory estimate after preprocessing:")
    print(f"    Clean rows:   ~{rows_after_clean:,}")
    print(f"    Sequences:    ~{seq_count:,}  (timesteps={config.TIMESTEPS})")
    print(f"    Features:     ~{n_features}")
    print(f"    Total memory: ~{total_mb:.1f} MB (X only, train+val+test combined)")

    print(f"\n{'='*70}")
    print("Diagnostics done. If everything above looks good, run:")
    print("    python main.py train")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
