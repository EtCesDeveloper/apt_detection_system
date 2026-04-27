"""
Visualization module — produces plots to inspect model performance.

Generates four figures saved to ./logs/plots/ :
  1. training_history.png   — loss + accuracy curves over epochs
  2. confusion_matrix.png   — TP / FP / FN / TN heatmap on test set
  3. roc_pr_curves.png      — ROC and Precision-Recall curves
  4. reconstruction.png     — reconstruction error distribution per class
                              (benign vs attack — useful as anomaly score)

It also prints a summary table of the key metrics.
"""

from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report,
)

from . import config
from .data_loader import load_cicids2017
from .preprocessing import build_dataset


# Output directory for plots
PLOTS_DIR = config.LOGS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------
def plot_training_history(history_path: Path = config.MODELS_DIR / "history.json"):
    """Plot loss and classifier accuracy over epochs from saved history.json."""
    if not history_path.exists():
        print(f"[viz] No history file at {history_path}. Skipping.")
        return

    with open(history_path) as f:
        h = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total loss
    axes[0].plot(h["loss"], label="train")
    if "val_loss" in h:
        axes[0].plot(h["val_loss"], label="val")
    axes[0].set_title("Total multitask loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Classifier accuracy
    acc_key = "classifier_output_accuracy"
    val_acc_key = "val_classifier_output_accuracy"
    if acc_key in h:
        axes[1].plot(h[acc_key], label="train")
    if val_acc_key in h:
        axes[1].plot(h[val_acc_key], label="val")
    axes[1].set_title("Classifier accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.tight_layout()
    out = PLOTS_DIR / "training_history.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved {out}")


# ---------------------------------------------------------------------------
# 2. Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["BENIGN", "ATTACK"],
        yticklabels=["BENIGN", "ATTACK"],
        cbar=False, ax=ax,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")

    # Annotate percentages below counts
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.75, f"({pct:.2f}%)",
                    ha="center", va="center", fontsize=10, color="gray")

    out = PLOTS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved {out}")


# ---------------------------------------------------------------------------
# 3. ROC + Precision-Recall curves
# ---------------------------------------------------------------------------
def plot_roc_pr(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend(loc="lower right")

    axes[1].plot(rec, prec, label=f"AP = {ap:.4f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    out = PLOTS_DIR / "roc_pr_curves.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved {out}")


# ---------------------------------------------------------------------------
# 4. Reconstruction error per class (anomaly-style view)
# ---------------------------------------------------------------------------
def plot_reconstruction_error(X_test, ae_pred, y_test):
    err = np.mean((ae_pred - X_test) ** 2, axis=(1, 2))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(err[y_test == 0], bins=60, label="BENIGN",
                 color="#2E86C1", stat="density", alpha=0.6, ax=ax)
    sns.histplot(err[y_test == 1], bins=60, label="ATTACK",
                 color="#C0392B", stat="density", alpha=0.6, ax=ax)
    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Density")
    ax.set_title("Autoencoder reconstruction error by class")
    ax.legend()
    # log scale on x helps when distributions are skewed
    ax.set_xscale("log")

    out = PLOTS_DIR / "reconstruction.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved {out}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def main():
    print("[viz] Loading data...")
    df = load_cicids2017()
    data = build_dataset(df, timesteps=config.TIMESTEPS)

    model_path = config.MODELS_DIR / "best_model.keras"
    if not model_path.exists():
        model_path = config.MODELS_DIR / "final_model.keras"
    print(f"[viz] Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    X_test, y_test = data["X_test"], data["y_test"]
    cls_pred, ae_pred = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=1)
    cls_score = cls_pred.ravel()
    y_pred = (cls_score >= 0.5).astype(np.int8)

    # Print metrics
    print("\n=== Test-set metrics ===")
    print(classification_report(
        y_test, y_pred, target_names=["BENIGN", "ATTACK"], digits=4
    ))

    # Generate all plots
    plot_training_history()
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_pr(y_test, cls_score)
    plot_reconstruction_error(X_test, ae_pred, y_test)

    print(f"\n[viz] All plots saved under {PLOTS_DIR}")


if __name__ == "__main__":
    main()
