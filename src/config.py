"""
Global configuration for the APT detection project.
Centralizes paths, hyperparameters, and reproducibility settings.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in (PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sequence shape
# ---------------------------------------------------------------------------
# CICIDS2017 flows are independent rows. To exploit temporal dependencies
# we group consecutive flows (sorted by timestamp) into sequences.
TIMESTEPS = 10           # Number of consecutive flows per sequence
N_FEATURES = 70          # Final feature count after cleaning (set dynamically)

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Multitask loss weights — tune these based on validation performance
LOSS_WEIGHT_CLASSIFIER = 1.0
LOSS_WEIGHT_AUTOENCODER = 0.3

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
