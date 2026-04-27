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

Class imbalance handling
------------------------
CICIDS2017 is imbalanced (~80% benign). Instead of using sample_weight
(which produces broadcasting issues with multi-output models in Keras 3),
we bake the class weight INTO a custom weighted-BCE loss for the
classifier head. This is fully compatible with Keras 3 and multi-output
models, and avoids any tf.data pipeline complications.

The autoencoder head uses standard MSE; class imbalance is irrelevant
there because the AE is a regression objective.
"""

from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam

from . import config


def make_weighted_bce(class_weight: dict | None):
    """
    Build a weighted binary-crossentropy loss that uses class_weight
    internally. Returns the standard BCE if class_weight is None.

    weight(y) = class_weight[1] * y + class_weight[0] * (1 - y)
    loss(y, p) = weight(y) * BCE(y, p)
    """
    if class_weight is None:
        return tf.keras.losses.BinaryCrossentropy(name="binary_crossentropy")

    w0 = tf.constant(class_weight[0], dtype=tf.float32)
    w1 = tf.constant(class_weight[1], dtype=tf.float32)

    def weighted_bce(y_true, y_pred):
        # y_true and y_pred have shape (batch, 1)
        y_true_f = tf.cast(y_true, tf.float32)
        # Per-sample BCE (no reduction over the batch yet)
        bce = tf.keras.losses.binary_crossentropy(y_true_f, y_pred)
        # Per-sample class weight
        sample_w = y_true_f * w1 + (1.0 - y_true_f) * w0
        # Squeeze to match shapes: bce is (batch,), sample_w is (batch, 1) -> (batch,)
        sample_w = tf.reshape(sample_w, tf.shape(bce))
        return tf.reduce_mean(bce * sample_w)

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
    loss_weights: dict | None = None,
    learning_rate: float = config.LEARNING_RATE,
    class_weight: dict | None = None,
) -> Model:
    """
    Build and compile the multitask CNN+LSTM model.

    Parameters
    ----------
    class_weight : optional dict {0: w0, 1: w1}
        If provided, the classifier head uses a weighted BCE loss that
        bakes the class weights directly into the gradient. This is the
        Keras-3-friendly alternative to sample_weight for multi-output
        models.
    """

    if loss_weights is None:
        loss_weights = {
            "classifier_output":  config.LOSS_WEIGHT_CLASSIFIER,
            "autoencoder_output": config.LOSS_WEIGHT_AUTOENCODER,
        }

    # 1. INPUT
    inputs = Input(shape=(timesteps, n_features), name="input_sequence")

    # 2. PREPROCESSING — in-graph LayerNormalization across the feature axis
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

    # 5B. AUTOENCODER HEAD
    d = layers.RepeatVector(timesteps, name="ae_repeat")(encoded)
    d = layers.LSTM(lstm_units, return_sequences=True, name="ae_lstm")(d)
    autoencoder_output = layers.TimeDistributed(
        layers.Dense(n_features, activation="linear"),
        name="autoencoder_output",
    )(d)

    # 6. MODEL
    model = Model(
        inputs=inputs,
        outputs=[classifier_output, autoencoder_output],
        name="cnn_lstm_multitask_apt",
    )

    # 7. COMPILE — weighted BCE for the classifier, MSE for the AE
    cls_loss = make_weighted_bce(class_weight)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={
            "classifier_output":  cls_loss,
            "autoencoder_output": "mean_squared_error",
        },
        loss_weights=loss_weights,
        metrics={
            "classifier_output": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
            "autoencoder_output": [
                tf.keras.metrics.MeanSquaredError(name="mse"),
            ],
        },
    )

    return model


if __name__ == "__main__":
    m = build_multitask_model(timesteps=10, n_features=70)
    m.summary()
