"""
Multitask CNN + LSTM architecture for malicious traffic detection.

Pipeline (Keras Functional API)
-------------------------------
Input  (batch, timesteps, features)
  -> LayerNormalization                                       (in-graph preprocessing)
  -> Conv1D (64) -> BN -> Conv1D (128) -> BN -> MaxPool       (LOCAL pattern extraction)
  -> LSTM (128, seq) -> LSTM (64)                             (TEMPORAL dependencies)
  -> classifier_output (Dense -> sigmoid)                     (binary detection)
  -> autoencoder_output (RepeatVector -> LSTM -> TimeDist)    (reconstruction)

Why we use POSITIONAL lists (not dicts) for loss/metrics
--------------------------------------------------------
Keras 3 changed how multi-output models map dict-keyed losses/metrics to
outputs. Using `metrics={"name": [...]}` can route metrics to the wrong
output and produce shape mismatches (e.g. BinaryAccuracy from the
classifier head being applied against the (B, T, F) AE output).

The reliable pattern in Keras 3 for multi-output models is:
  outputs       = [out_0, out_1]
  loss          = [loss_0, loss_1]              # positional
  loss_weights  = [w_0,    w_1]                  # positional
  metrics       = [[m0_a, m0_b], [m1_a]]         # list of lists, positional

This guarantees that loss[i] and metrics[i] are applied to outputs[i].

Class imbalance handling
------------------------
We bake the class weights INTO a custom weighted-BCE loss for the
classifier head. This avoids `class_weight` (unsupported on multi-output)
and `sample_weight` (broadcasting issues with mixed-shape outputs).
"""

from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam

from . import config


def make_weighted_bce(class_weight: dict | None):
    """
    Build a binary-crossentropy loss with embedded class weights.

    Returns a per-sample tensor of shape (B,). Keras 3 averages over the
    batch automatically and applies loss_weights cleanly.
    """
    if class_weight is None:
        return tf.keras.losses.BinaryCrossentropy(name="binary_crossentropy")

    w0 = tf.constant(float(class_weight[0]), dtype=tf.float32)
    w1 = tf.constant(float(class_weight[1]), dtype=tf.float32)

    def weighted_bce(y_true, y_pred):
        # y_true is (B, 1), y_pred is (B, 1)
        y_true_f = tf.cast(y_true, tf.float32)
        # binary_crossentropy reduces last axis -> (B,)
        bce = tf.keras.losses.binary_crossentropy(y_true_f, y_pred)
        # Per-sample weight, flattened to (B,)
        y_flat = tf.reshape(y_true_f, [-1])
        sample_w = y_flat * w1 + (1.0 - y_flat) * w0
        return bce * sample_w

    weighted_bce.__name__ = "weighted_binary_crossentropy"
    return weighted_bce


def build_multitask_model(
    timesteps: int,
    n_features: int,
    cnn_filters: tuple = (64, 128),
    cnn_kernel: int = 3,
    lstm_units: int = 128,
    latent_dim: int = 64,
    dropout: float = 0.3,
    learning_rate: float = config.LEARNING_RATE,
    class_weight: dict | None = None,
    loss_weight_cls: float = config.LOSS_WEIGHT_CLASSIFIER,
    loss_weight_ae:  float = config.LOSS_WEIGHT_AUTOENCODER,
) -> Model:
    """
    Build and compile the multitask CNN+LSTM model.

    Output order (matters for positional list mapping):
      outputs[0] = classifier_output  (B, 1)
      outputs[1] = autoencoder_output (B, T, F)
    """

    # 1. INPUT
    inputs = Input(shape=(timesteps, n_features), name="input_sequence")

    # 2. PREPROCESSING — in-graph LayerNormalization
    x = layers.LayerNormalization(name="preproc_layernorm")(inputs)

    # 3. CNN BLOCK — local pattern extraction along the temporal axis
    x = layers.Conv1D(cnn_filters[0], cnn_kernel, padding="same",
                      activation="relu", name="conv1d_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Conv1D(cnn_filters[1], cnn_kernel, padding="same",
                      activation="relu", name="conv1d_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same", name="maxpool_1")(x)
    x = layers.Dropout(dropout, name="dropout_cnn")(x)

    # 4. LSTM BLOCK — temporal dependencies
    x = layers.LSTM(lstm_units, return_sequences=True, name="lstm_1")(x)
    x = layers.Dropout(dropout, name="dropout_lstm_1")(x)
    encoded = layers.LSTM(latent_dim, return_sequences=False,
                          name="lstm_encoder")(x)
    encoded = layers.Dropout(dropout, name="dropout_lstm_2")(encoded)

    # 5A. CLASSIFIER HEAD
    c = layers.Dense(64, activation="relu", name="cls_dense_1")(encoded)
    c = layers.Dropout(dropout, name="cls_dropout")(c)
    classifier_output = layers.Dense(
        1, activation="sigmoid", name="classifier_output"
    )(c)

    # 5B. AUTOENCODER HEAD — reconstruct (normalized) input sequence
    d = layers.RepeatVector(timesteps, name="ae_repeat")(encoded)
    d = layers.LSTM(lstm_units, return_sequences=True, name="ae_lstm")(d)
    autoencoder_output = layers.TimeDistributed(
        layers.Dense(n_features, activation="linear"),
        name="autoencoder_output",
    )(d)

    # 6. MODEL — positional output list. The order is fixed and used by
    #    the positional loss/metrics lists below.
    model = Model(
        inputs=inputs,
        outputs=[classifier_output, autoencoder_output],
        name="cnn_lstm_multitask_apt",
    )

    # 7. COMPILE — POSITIONAL lists mapped to outputs[0] and outputs[1].
    cls_loss = make_weighted_bce(class_weight)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=[
            cls_loss,                  # for outputs[0] = classifier_output
            "mean_squared_error",      # for outputs[1] = autoencoder_output
        ],
        loss_weights=[
            loss_weight_cls,           # for outputs[0]
            loss_weight_ae,            # for outputs[1]
        ],
        metrics=[
            # Metrics for outputs[0] = classifier_output
            [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
            # Metrics for outputs[1] = autoencoder_output
            [
                tf.keras.metrics.MeanSquaredError(name="mse"),
            ],
        ],
    )

    return model


if __name__ == "__main__":
    m = build_multitask_model(timesteps=10, n_features=70)
    m.summary()
