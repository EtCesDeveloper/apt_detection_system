"""
Preprocessing: turn the flat CICIDS2017 DataFrame into 3D sequences
of shape (n_samples, timesteps, n_features) suitable for a CNN+LSTM.

Why this version differs from a naive sliding-window approach
-------------------------------------------------------------
Building sliding windows AFTER shuffling rows inflates positive rates
(probability that a random window contains at least one attack flow
approaches 1 - (1-p)^T, which is ~89% for p=0.20 and T=10). Worse, it
destroys the temporal structure the LSTM is supposed to learn.

The correct pipeline:
  1. Build NON-overlapping windows of length T from the flow-ordered
     DataFrame. Each window represents an actual burst of consecutive
     traffic, so the temporal dimension carries real signal.
  2. Stratified train/val/test split AT THE SEQUENCE LEVEL (after the
     windows are built, never before).
  3. Fit StandardScaler on the train fold ONLY (no leakage).
  4. Persist the scaler for inference.

A window is labelled 1 (ATTACK) iff at least one of its T flows is an
attack — standard sequence-level intrusion-detection formulation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from . import config


def _make_non_overlapping_windows(
    X: np.ndarray, y: np.ndarray, timesteps: int
):
    """
    Reshape (N, F) -> (N // timesteps, timesteps, F) by truncating the
    tail. Non-overlapping windows preserve temporal structure and avoid
    the data-leakage / label-inflation that sliding windows introduce.

    Window label = 1 if ANY flow in the window is an attack.
    """
    n_full = (len(X) // timesteps) * timesteps   # drop the tail
    X = X[:n_full]
    y = y[:n_full]

    n_windows = n_full // timesteps
    X_seq = X.reshape(n_windows, timesteps, X.shape[1])
    y_per_step = y.reshape(n_windows, timesteps)
    y_seq = (y_per_step.sum(axis=1) > 0).astype(np.int8)

    return X_seq, y_seq


def build_dataset(df: pd.DataFrame, timesteps: int = config.TIMESTEPS):
    """
    Full preprocessing pipeline:
      1. Build non-overlapping sequences from the (already file-ordered) df.
      2. Stratified train/val/test split AT THE SEQUENCE LEVEL.
      3. Fit StandardScaler on the train fold only, transform all folds
         (we flatten across timesteps, scale, and reshape back).
      4. Persist the scaler for inference.
    """
    y = df["Label"].values.astype(np.int8)
    X = df.drop(columns=["Label"]).values.astype(np.float32)

    # 1. Build sequences first (preserves natural ordering of flows)
    X_seq, y_seq = _make_non_overlapping_windows(X, y, timesteps)
    print(f"[preprocessing] Sequences built: {X_seq.shape}")
    print(f"[preprocessing] Positive rate at sequence level: {y_seq.mean():.4f}")

    # 2. Stratified split at the SEQUENCE level
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_seq, y_seq,
        test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
        random_state=config.SEED,
        stratify=y_seq,
        shuffle=True,
    )
    rel_test = config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=rel_test,
        random_state=config.SEED,
        stratify=y_temp,
        shuffle=True,
    )

    # 3. Scale: fit on train, apply to all. We have to flatten the time
    #    axis because StandardScaler expects 2D input.
    n_features = X_train.shape[2]

    def _flatten(a):
        return a.reshape(-1, n_features)

    def _unflatten(a, n_seq):
        return a.reshape(n_seq, timesteps, n_features)

    scaler = StandardScaler()
    X_train_s = _unflatten(scaler.fit_transform(_flatten(X_train)), len(X_train))
    X_val_s   = _unflatten(scaler.transform(_flatten(X_val)),       len(X_val))
    X_test_s  = _unflatten(scaler.transform(_flatten(X_test)),      len(X_test))

    # Persist scaler for later inference / evaluation
    joblib.dump(scaler, config.PROCESSED_DATA_DIR / "scaler.joblib")

    print(f"[preprocessing] Train: {X_train_s.shape}, "
          f"Val: {X_val_s.shape}, Test: {X_test_s.shape}")
    print(f"[preprocessing] Positive rate (train): {y_train.mean():.4f}")
    print(f"[preprocessing] Positive rate (val):   {y_val.mean():.4f}")
    print(f"[preprocessing] Positive rate (test):  {y_test.mean():.4f}")

    return {
        "X_train": X_train_s, "y_train": y_train,
        "X_val":   X_val_s,   "y_val":   y_val,
        "X_test":  X_test_s,  "y_test":  y_test,
        "n_features": n_features,
        "scaler": scaler,
    }