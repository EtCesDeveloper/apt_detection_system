"""
=============================================================================
 Multitask CNN-LSTM Architecture for Malicious Traffic Detection
 Dataset: CICIDS2017 | Framework: TensorFlow / Keras Functional API
=============================================================================
 Architecture overview:
   INPUT → PREPROCESSING → CNN BLOCK → LSTM BLOCK → ┬─ CLASSIFIER (sigmoid)
                                                     └─ AUTOENCODER (reconstruction)
 Two task heads share the same feature extractor backbone (CNN + LSTM).
 The model is trained jointly with a weighted sum of:
   - Binary Cross-Entropy  (malicious vs. benign)
   - Mean Squared Error    (sequence reconstruction)
=============================================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
import os

# ─────────────────────────────────────────────────────────────────────────────
# 0. REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ═════════════════════════════════════════════════════════════════════════════
# 1. HYPERPARAMETERS  (edit these to match your actual dataset)
# ═════════════════════════════════════════════════════════════════════════════

# --- Input shape ─────────────────────────────────────────────────────────────
TIMESTEPS = 20       # Number of consecutive packets / windows per sample
N_FEATURES = 78      # CICIDS2017 has 78 flow-level features after cleaning

# --- CNN block ───────────────────────────────────────────────────────────────
CNN_FILTERS_1   = 64    # Filters in first Conv1D
CNN_FILTERS_2   = 128   # Filters in second Conv1D
CNN_KERNEL_SIZE = 3     # Local receptive field (3 timesteps)
CNN_POOL_SIZE   = 2     # MaxPooling stride

# --- LSTM block ──────────────────────────────────────────────────────────────
LSTM_UNITS_1 = 128   # First LSTM layer (return sequences → feeds second)
LSTM_UNITS_2 = 64    # Second LSTM layer (output: final hidden state)

# --- Decoder / Autoencoder head ──────────────────────────────────────────────
DECODER_UNITS = 128  # Dense units in the reconstruction decoder

# --- Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 64
EPOCHS          = 50
LEARNING_RATE   = 1e-3
DROPOUT_RATE    = 0.3

# --- Loss weights (sum does not need to equal 1) ─────────────────────────────
# Increase ALPHA to prioritise detection; increase BETA to prioritise
# reconstruction (anomaly-score quality).
ALPHA = 1.0   # weight for classification loss (Binary Cross-Entropy)
BETA  = 0.5   # weight for reconstruction loss (MSE)


# ═════════════════════════════════════════════════════════════════════════════
# 2. MODEL BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_model(
    timesteps:    int   = TIMESTEPS,
    n_features:   int   = N_FEATURES,
    cnn_filters1: int   = CNN_FILTERS_1,
    cnn_filters2: int   = CNN_FILTERS_2,
    kernel_size:  int   = CNN_KERNEL_SIZE,
    pool_size:    int   = CNN_POOL_SIZE,
    lstm_units1:  int   = LSTM_UNITS_1,
    lstm_units2:  int   = LSTM_UNITS_2,
    decoder_units:int   = DECODER_UNITS,
    dropout_rate: float = DROPOUT_RATE,
) -> Model:
    """
    Build and return the multitask CNN-LSTM model using Keras Functional API.

    Tensor flow (shapes shown for default hyperparameters):
    ─────────────────────────────────────────────────────────────────────────
      INPUT          (None, 20, 78)
        ↓  LayerNorm
      PREPROC        (None, 20, 78)
        ↓  TimeDistributed(Conv1D×2 + MaxPool)
      CNN OUT        (None, 20, 128)   ← local patterns per timestep window
        ↓  Reshape (flatten spatial dim into features)
      RESHAPE        (None, 20, 128)   ← ready for LSTM sequential scan
        ↓  LSTM(128, return_sequences=True)
      LSTM1 OUT      (None, 20, 128)   ← hidden state at every timestep
        ↓  LSTM(64, return_sequences=False)
      LSTM2 OUT      (None, 64)        ← single context vector for classifier
                                         AND seed for decoder
        ├──────────────────── HEAD A: CLASSIFIER ─────────────────────────┐
        │  Dense(32) → Dropout → Dense(1, sigmoid)                        │
        │  OUTPUT: (None, 1)   ← P(malicious)                             │
        └──────────────────────────────────────────────────────────────────┘
        ├──────────────────── HEAD B: AUTOENCODER ────────────────────────┐
        │  RepeatVector(timesteps) → LSTM(128, ret_seq=True)              │
        │  → TimeDistributed(Dense(n_features))                           │
        │  OUTPUT: (None, 20, 78)  ← reconstructed input sequence         │
        └──────────────────────────────────────────────────────────────────┘
    """

    # ── 2.1  INPUT ──────────────────────────────────────────────────────────
    # Each sample is a sequence of `timesteps` flow records, each with
    # `n_features` numerical features extracted from the CICIDS2017 CSV.
    inputs = keras.Input(
        shape=(timesteps, n_features),
        name="input_traffic_sequence"
    )

    # ── 2.2  PREPROCESSING LAYER  ────────────────────────────────────────────
    # LayerNormalization normalises EACH sample independently (unlike
    # BatchNorm it does not depend on batch statistics → safer at inference).
    # axis=-1 normalises across the feature dimension at every timestep.
    x = layers.LayerNormalization(axis=-1, name="layer_norm")(inputs)

    # ── 2.3  CNN BLOCK  ──────────────────────────────────────────────────────
    # Goal: extract LOCAL patterns within short windows of timesteps.
    # We use TimeDistributed so that IDENTICAL Conv1D weights are applied
    # to every "local view" — this is equivalent to a 2-D convolution where
    # one dimension is the feature axis and the other is the time axis.
    #
    # Why TimeDistributed?
    #   Conv1D inside TimeDistributed operates on the FEATURE dimension of
    #   each individual timestep, treating the sequence of features as a
    #   1-D signal. This extracts intra-timestep correlations (e.g., the
    #   relationship between byte count and flag fields at the same instant).
    #   The LSTM will then capture INTER-timestep (temporal) dependencies.

    # First convolutional layer — low-level feature detectors
    x = layers.TimeDistributed(
        layers.Conv1D(
            filters=cnn_filters1,
            kernel_size=kernel_size,
            padding="same",          # keep length identical to input
            activation="relu",
        ),
        name="td_conv1d_1"
    )(x)
    # Shape: (None, timesteps, n_features, cnn_filters1)
    # After TimeDistributed: (None, timesteps, n_features, 64)

    x = layers.TimeDistributed(
        layers.BatchNormalization(),
        name="td_bn_1"
    )(x)

    # Second convolutional layer — higher-level pattern detectors
    x = layers.TimeDistributed(
        layers.Conv1D(
            filters=cnn_filters2,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
        ),
        name="td_conv1d_2"
    )(x)
    # Shape: (None, timesteps, n_features, 128)

    x = layers.TimeDistributed(
        layers.BatchNormalization(),
        name="td_bn_2"
    )(x)

    # Max-pooling compresses the feature dimension, keeping the most
    # activated local features and reducing the subsequent LSTM input size.
    x = layers.TimeDistributed(
        layers.MaxPooling1D(pool_size=pool_size, padding="same"),
        name="td_maxpool"
    )(x)
    # Shape: (None, timesteps, n_features // pool_size, 128)
    #      = (None, 20, 39, 128)

    # Flatten the inner (feature × filter) dimensions so each timestep is
    # represented as a single flat vector that the LSTM can process.
    x = layers.TimeDistributed(
        layers.Flatten(),
        name="td_flatten"
    )(x)
    # Shape: (None, 20, 39*128) = (None, 20, 4992)
    # Each of the 20 timesteps is now a 4992-dim CNN embedding.

    # Optional projection to keep LSTM input size manageable
    x = layers.TimeDistributed(
        layers.Dense(lstm_units1, activation="relu"),
        name="td_projection"
    )(x)
    # Shape: (None, 20, 128)  ← projected down to lstm_units1

    x = layers.Dropout(dropout_rate, name="dropout_cnn")(x)

    # ── 2.4  LSTM BLOCK  ─────────────────────────────────────────────────────
    # Goal: capture TEMPORAL DEPENDENCIES across the sequence of CNN embeddings.
    # The CNN already knows WHAT is happening at each timestep;
    # the LSTM now learns WHEN and HOW these patterns evolve over time.

    # First LSTM: return_sequences=True → outputs a hidden state at every
    # timestep. This is required because the second LSTM needs the full
    # sequence to refine temporal context before producing a summary.
    x = layers.LSTM(
        lstm_units1,
        return_sequences=True,   # (None, 20, 128)
        dropout=dropout_rate,
        recurrent_dropout=0.1,
        name="lstm_1"
    )(x)

    # Second LSTM: return_sequences=False → outputs ONE context vector that
    # summarises the entire sequence. Used as input to the classifier head.
    # We save this as `context` because the decoder will also use it.
    context = layers.LSTM(
        lstm_units2,
        return_sequences=False,  # (None, 64)
        dropout=dropout_rate,
        recurrent_dropout=0.1,
        name="lstm_2"
    )(x)
    # `context` shape: (None, 64)
    # This single vector encodes the full temporal + local-pattern information
    # of the input sequence.

    # ── 2.5a  HEAD A — BINARY CLASSIFIER  ───────────────────────────────────
    # Classifies each traffic sequence as malicious (APT/attack) or benign.

    clf = layers.Dense(32, activation="relu", name="clf_dense")(context)
    clf = layers.Dropout(dropout_rate, name="clf_dropout")(clf)
    clf_output = layers.Dense(
        1,
        activation="sigmoid",
        name="classifier_output"     # ← loss: binary_crossentropy
    )(clf)
    # Output shape: (None, 1)  — probability of being malicious

    # ── 2.5b  HEAD B — SEQUENCE AUTOENCODER  ────────────────────────────────
    # Reconstructs the (normalised) input sequence.
    # A reconstruction error that is HIGH at inference time is a strong
    # indicator of anomaly / zero-day traffic, complementing the classifier.
    #
    # Decoder design:
    #   RepeatVector expands the context vector back to `timesteps` steps,
    #   then an LSTM decoder refines each step, and a Dense projects back
    #   to the original feature space.

    # Expand: (None, 64) → (None, 20, 64)
    dec = layers.RepeatVector(timesteps, name="dec_repeat")(context)

    # Decode temporal structure
    dec = layers.LSTM(
        decoder_units,
        return_sequences=True,       # (None, 20, 128)
        dropout=dropout_rate,
        name="dec_lstm"
    )(dec)

    # Project each decoded timestep back to original feature space
    ae_output = layers.TimeDistributed(
        layers.Dense(n_features, activation="linear"),  # linear = no clipping
        name="autoencoder_output"    # ← loss: mse
    )(dec)
    # Output shape: (None, 20, 78)  ← matches input shape exactly

    # ── 2.6  ASSEMBLE MODEL  ─────────────────────────────────────────────────
    model = Model(
        inputs=inputs,
        outputs=[clf_output, ae_output],
        name="MultiTask_CNN_LSTM_IDS"
    )

    return model


# ═════════════════════════════════════════════════════════════════════════════
# 3. COMPILE
# ═════════════════════════════════════════════════════════════════════════════

def compile_model(model: Model, alpha: float = ALPHA, beta: float = BETA) -> Model:
    """
    Compile the multitask model.

    Loss:
        total_loss = alpha * BCE(y_true, y_clf) + beta * MSE(x, x_hat)

    Metrics per output:
        classifier_output → accuracy, precision, recall, AUC
        autoencoder_output → mse (tracks reconstruction quality)
    """
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss={
            "classifier_output":  "binary_crossentropy",
            "autoencoder_output": "mse",
        },
        loss_weights={
            "classifier_output":  alpha,   # ALPHA
            "autoencoder_output": beta,    # BETA
        },
        metrics={
            "classifier_output": [
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
            "autoencoder_output": ["mse"],
        },
    )
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 4. CALLBACKS
# ═════════════════════════════════════════════════════════════════════════════

def get_callbacks(checkpoint_dir: str = "./checkpoints") -> list:
    """
    Return a standard set of training callbacks.

    - EarlyStopping:    stops when val_loss stops improving (patience=7)
    - ReduceLROnPlateau: halves LR when val_loss plateaus (patience=3)
    - ModelCheckpoint:  saves best weights to disk
    - TensorBoard:      logs for visualisation (run: tensorboard --logdir ./logs)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(log_dir="./logs", histogram_freq=1),
    ]


# ═════════════════════════════════════════════════════════════════════════════
# 5. DATA UTILS  (replace with your real CICIDS2017 loader)
# ═════════════════════════════════════════════════════════════════════════════

def load_cicids2017(csv_path: str, timesteps: int = TIMESTEPS) -> tuple:
    """
    Placeholder loader — replace the body with your actual CICIDS2017 pipeline.

    Expected CSV format (after standard CIC preprocessing):
        Columns 0..77 → 78 flow features
        Column 78     → label ("BENIGN" or attack name)

    Returns
    -------
    X : np.ndarray, shape (n_samples, timesteps, n_features)
    y : np.ndarray, shape (n_samples,)  — 0=benign, 1=malicious
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Drop non-numeric / identifier columns if present
    drop_cols = [c for c in ["Flow ID", "Source IP", "Destination IP",
                              "Timestamp", "Source Port", "Destination Port"]
                 if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Binary label: 0 = BENIGN, 1 = attack
    label_col = "Label"
    df[label_col] = (df[label_col].str.upper() != "BENIGN").astype(int)
    labels = df[label_col].values
    features = df.drop(columns=[label_col]).select_dtypes(include=[np.number])

    # Replace inf / nan
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(features.median(), inplace=True)

    # Scale — note: in the model we add LayerNorm, but a global StandardScaler
    # beforehand is still a good practice to remove gross magnitude differences.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)

    # Reshape into (n_windows, timesteps, n_features)
    n_full = (len(X_scaled) // timesteps) * timesteps
    X_seq = X_scaled[:n_full].reshape(-1, timesteps, X_scaled.shape[1])
    # For labels, take the majority label within each window
    y_seq = labels[:n_full].reshape(-1, timesteps).max(axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=SEED, stratify=y_seq
    )
    return X_train, X_val, y_train, y_val


def make_dummy_data(n_samples: int = 2000) -> tuple:
    """
    Generate synthetic data for architecture testing (shape validation only).
    Delete / ignore this when using real CICIDS2017 data.
    """
    print("[INFO] Using DUMMY data for shape/smoke test. "
          "Replace with load_cicids2017() for real training.")
    X = np.random.randn(n_samples, TIMESTEPS, N_FEATURES).astype(np.float32)
    y = np.random.randint(0, 2, size=(n_samples,)).astype(np.float32)
    split = int(n_samples * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


# ═════════════════════════════════════════════════════════════════════════════
# 6. TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def train(
    csv_path:    str | None = None,   # set to your CSV path for real data
    use_dummy:   bool = True,          # flip to False when CSV is ready
    epochs:      int  = EPOCHS,
    batch_size:  int  = BATCH_SIZE,
) -> tuple[Model, dict]:
    """
    End-to-end training entry point.

    Parameters
    ----------
    csv_path  : path to the CICIDS2017 CSV (or folder of CSVs).
    use_dummy : if True, generates random data for a smoke test.
    epochs    : max training epochs (EarlyStopping may stop sooner).
    batch_size: mini-batch size.

    Returns
    -------
    model   : trained Keras Model
    history : dict with loss/metric curves
    """

    # --- Load data -----------------------------------------------------------
    if use_dummy or csv_path is None:
        X_train, X_val, y_train, y_val = make_dummy_data()
    else:
        X_train, X_val, y_train, y_val = load_cicids2017(csv_path)

    n_features_actual = X_train.shape[2]
    timesteps_actual  = X_train.shape[1]
    print(f"[INFO] Train shape: {X_train.shape}  |  Val shape: {X_val.shape}")
    print(f"[INFO] Class balance (train): "
          f"{int(y_train.sum())} malicious / {int((1-y_train).sum())} benign")

    # --- Build & compile -----------------------------------------------------
    model = build_model(
        timesteps=timesteps_actual,
        n_features=n_features_actual,
    )
    model = compile_model(model)

    # --- Summary -------------------------------------------------------------
    model.summary(line_length=100)

    # --- Multitask targets ---------------------------------------------------
    # The classifier head needs (N,) binary labels.
    # The autoencoder head needs (N, timesteps, features) — the input itself.
    train_targets = {
        "classifier_output":  y_train,
        "autoencoder_output": X_train,   # reconstruct the normalised input
    }
    val_targets = {
        "classifier_output":  y_val,
        "autoencoder_output": X_val,
    }

    # --- Class-weight for imbalanced CICIDS2017 data -------------------------
    # CICIDS2017 is heavily imbalanced (most traffic is benign).
    n_total   = len(y_train)
    n_pos     = int(y_train.sum())
    n_neg     = n_total - n_pos
    pos_weight = n_neg / (n_pos + 1e-9)   # up-weight malicious class
    class_weight = {0: 1.0, 1: float(pos_weight)}
    print(f"[INFO] Class weight → benign: 1.0 | malicious: {pos_weight:.2f}")

    # --- Fit -----------------------------------------------------------------
    history = model.fit(
        x=X_train,
        y=train_targets,
        validation_data=(X_val, val_targets),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(),
        class_weight=class_weight,   # only affects classifier head
        verbose=1,
    )

    return model, history.history


# ═════════════════════════════════════════════════════════════════════════════
# 7. INFERENCE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def predict(model: Model, X: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Run inference on a batch of traffic sequences.

    Returns
    -------
    dict with:
      'prob'            : P(malicious) for each sample, shape (N,)
      'label'           : binary prediction (0/1),    shape (N,)
      'reconstruction'  : reconstructed sequence,     shape (N, T, F)
      'recon_error'     : per-sample MSE,              shape (N,)
    """
    clf_prob, reconstruction = model.predict(X, verbose=0)
    clf_prob = clf_prob.squeeze()                  # (N, 1) → (N,)
    recon_error = np.mean((X - reconstruction) ** 2, axis=(1, 2))  # (N,)

    return {
        "prob":           clf_prob,
        "label":          (clf_prob >= threshold).astype(int),
        "reconstruction": reconstruction,
        "recon_error":    recon_error,
    }


def anomaly_score(recon_error: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Compute a normalised anomaly score from the reconstruction error.
    Samples where score > sigma standard deviations above the mean are flagged.
    Useful for detecting zero-day attacks that fool the classifier.

    Returns a boolean mask: True = potential anomaly.
    """
    mu    = recon_error.mean()
    sigma_ = recon_error.std()
    return (recon_error - mu) / (sigma_ + 1e-9) > sigma


# ═════════════════════════════════════════════════════════════════════════════
# 8. EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(model: Model, X: np.ndarray, y: np.ndarray) -> None:
    """
    Print a classification report using sklearn metrics.
    """
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_auc_score
    )

    results = predict(model, X)
    y_pred  = results["label"]
    y_prob  = results["prob"]

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y, y_pred, target_names=["Benign", "Malicious"]))

    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(f"  TN={cm[0,0]:6d}  FP={cm[0,1]:6d}")
    print(f"  FN={cm[1,0]:6d}  TP={cm[1,1]:6d}")

    auc = roc_auc_score(y, y_prob)
    print(f"\nROC-AUC: {auc:.4f}")

    # Anomaly score summary
    anomalies = anomaly_score(results["recon_error"])
    print(f"\nAnomaly detector flagged: {anomalies.sum()} / {len(anomalies)} samples")
    print("=" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Quick architecture check (no real data needed) ─────────────────────
    print("\n" + "=" * 70)
    print("  STEP 1: Architecture check with dummy data")
    print("=" * 70)
    model, history = train(use_dummy=True, epochs=3)

    # ── Inspect one batch of predictions ───────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 2: Sample prediction")
    print("=" * 70)
    X_test_dummy = np.random.randn(10, TIMESTEPS, N_FEATURES).astype(np.float32)
    y_test_dummy = np.random.randint(0, 2, size=(10,)).astype(np.float32)
    evaluate(model, X_test_dummy, y_test_dummy)

    # ── To train on REAL CICIDS2017 data, uncomment and set your path ──────
    # CSV_PATH = "/your/path/to/cicids2017/merged.csv"
    # model, history = train(csv_path=CSV_PATH, use_dummy=False, epochs=50)
    # X_test, y_test = ...  # load your held-out test set
    # evaluate(model, X_test, y_test)
    # model.save("multitask_ids_model.keras")