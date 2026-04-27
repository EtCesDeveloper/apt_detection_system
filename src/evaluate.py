"""
Evaluation script.

Loads the best model saved during training and reports:
  - Classification metrics on the test set (precision, recall, F1, AUC)
  - Autoencoder reconstruction error per class (useful for anomaly-style
    detection: attacks should reconstruct worse than benign traffic)

Note on model loading
---------------------
The training compiles the model with a custom weighted-BCE loss (defined
in model.py for class-imbalance handling). To load the model for
INFERENCE, we set compile=False — we only need predictions, not gradients,
so we can skip recreating the loss function at load time.
"""

from __future__ import annotations
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)

from . import config
from .data_loader import load_cicids2017
from .preprocessing import build_dataset


def main() -> None:
    df = load_cicids2017()
    data = build_dataset(df, timesteps=config.TIMESTEPS)

    # compile=False to skip rebuilding the custom loss at load time
    model = tf.keras.models.load_model(
        config.MODELS_DIR / "best_model.keras",
        compile=False,
    )
    print("[evaluate] Loaded best model.")

    X_test, y_test = data["X_test"], data["y_test"]

    # Predict both heads
    cls_pred, ae_pred = model.predict(X_test, batch_size=config.BATCH_SIZE)
    cls_pred = cls_pred.ravel()
    y_pred = (cls_pred >= 0.5).astype(np.int8)

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred,
                                target_names=["BENIGN", "ATTACK"], digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, cls_pred):.4f}")

    # Reconstruction error per class — informative diagnostic
    recon_err = np.mean((ae_pred - X_test) ** 2, axis=(1, 2))
    print("\n=== Reconstruction MSE by class ===")
    print(f"  BENIGN: mean={recon_err[y_test == 0].mean():.6f} "
          f"std={recon_err[y_test == 0].std():.6f}")
    print(f"  ATTACK: mean={recon_err[y_test == 1].mean():.6f} "
          f"std={recon_err[y_test == 1].std():.6f}")


if __name__ == "__main__":
    main()
