"""
Training script for the multitask CNN+LSTM model.

Conventions
-----------
- Class imbalance is handled INSIDE the model (weighted BCE, see model.py).
  We do NOT pass `class_weight` or `sample_weight` to model.fit() because
  these are unsupported / buggy with multi-output models in Keras 3.

- Targets are passed as a POSITIONAL LIST in the same order as
  model.outputs:
      y[0] = y_cls  (B, 1)  -> classifier_output
      y[1] = y_ae   (B, T, F) -> autoencoder_output

- y_cls is reshaped to (N, 1) explicitly to match the model output shape
  (B, 1) and avoid silent broadcasting.
"""

from __future__ import annotations
import json
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from . import config
from .data_loader import load_cicids2017
from .preprocessing import build_dataset
from .model import build_multitask_model


def set_seeds(seed: int = config.SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main() -> None:
    set_seeds()

    # 1. Load + clean
    df = load_cicids2017()

    # 2. Build sequences
    data = build_dataset(df, timesteps=config.TIMESTEPS)
    n_features = data["n_features"]
    print(f"[train] n_features detected: {n_features}")

    # 3. Compute class weights from training labels (BEFORE building the
    #    model — they get baked into the model's classifier loss).
    classes = np.array([0, 1])
    cw = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=data["y_train"],
    )
    class_weight_dict = {0: float(cw[0]), 1: float(cw[1])}
    print(f"[train] class_weight (baked into the model loss): "
          f"{class_weight_dict}")

    # 4. Build model with the class weights embedded in the classifier loss
    model = build_multitask_model(
        timesteps=config.TIMESTEPS,
        n_features=n_features,
        class_weight=class_weight_dict,
    )
    model.summary()

    # 5. Prepare targets — POSITIONAL LIST matching model.outputs order:
    #    [classifier_output (B, 1), autoencoder_output (B, T, F)]
    X_train = data["X_train"].astype(np.float32)
    X_val   = data["X_val"].astype(np.float32)

    y_cls_train = data["y_train"].astype(np.float32).reshape(-1, 1)
    y_cls_val   = data["y_val"].astype(np.float32).reshape(-1, 1)

    y_train_targets = [y_cls_train, X_train]   # AE target = X (reconstruction)
    y_val_targets   = [y_cls_val,   X_val]

    # 6. Detect the ACTUAL Keras 3 metric names by running a tiny probe.
    #    Keras 3 sometimes names multi-output metrics as
    #    "<layer_name>_<metric>" and sometimes as "output_<i>_<metric>".
    #    We use whichever key exists in the live history.
    print("[train] Probing metric names with 1-step evaluation...")
    probe_metrics = model.evaluate(
        x=X_train[:config.BATCH_SIZE],
        y=[y_cls_train[:config.BATCH_SIZE], X_train[:config.BATCH_SIZE]],
        batch_size=config.BATCH_SIZE,
        verbose=0,
        return_dict=True,
    )
    print(f"[train] Available metric keys: {list(probe_metrics.keys())}")

    # Pick the AUC metric of the classifier head, whichever name Keras chose.
    # Candidate patterns we accept, in order of preference:
    auc_candidates = [
        "classifier_output_auc",
        "auc",
        "compile_metrics",
    ]
    # Find the first auc-like key
    auc_key = None
    for candidate in probe_metrics.keys():
        if "auc" in candidate.lower():
            auc_key = candidate
            break
    if auc_key is None:
        # Fallback: use total loss for early stopping
        monitor_metric = "val_loss"
        monitor_mode = "min"
        print(f"[train] No AUC metric found, monitoring {monitor_metric} instead.")
    else:
        monitor_metric = f"val_{auc_key}"
        monitor_mode = "max"
        print(f"[train] Monitoring: {monitor_metric} (mode={monitor_mode})")

    # 7. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode=monitor_mode,
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODELS_DIR / "best_model.keras"),
            monitor=monitor_metric,
            mode=monitor_mode,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(config.LOGS_DIR),
            histogram_freq=1,
        ),
    ]

    # 8. Fit — clean call, no class_weight, no sample_weight
    history = model.fit(
        x=X_train,
        y=y_train_targets,
        validation_data=(X_val, y_val_targets),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    # 9. Save final model + history
    model.save(config.MODELS_DIR / "final_model.keras")
    with open(config.MODELS_DIR / "history.json", "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f, indent=2,
        )

    print("[train] Training finished. Models saved under ./models/")


if __name__ == "__main__":
    main()
