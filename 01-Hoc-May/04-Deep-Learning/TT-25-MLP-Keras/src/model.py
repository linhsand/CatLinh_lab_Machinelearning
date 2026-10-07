"""Kien truc MLP bang Keras cho TT-25.

* build_mlp():           Sequential, dau vao one-hot. Tuy chon Dropout + BatchNorm sau moi lop an.
* build_embedding_mlp(): Functional API, 3 dau vao (so thuc, ma vung, ma kenh ban). 2 lop Embedding
                         bien ma -> vector hoc duoc, noi (concatenate) voi phan so thuc roi qua cac lop Dense.
* compile_model():       Adam + binary_crossentropy, theo doi PR-AUC va Recall.
* make_callbacks():      EarlyStopping (val_pr_auc) + ReduceLROnPlateau (val_loss) + ModelCheckpoint.

Nhi phan -> 1 neuron sigmoid (khong dung 2 neuron softmax).
"""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models


def _hidden_block(x, units: int, dropout: float, batchnorm: bool, name: str):
    """Dense(ReLU) -> [BatchNorm] -> [Dropout], dung chung cho ca 2 kieu mang."""
    x = layers.Dense(units, activation="relu", name=f"{name}_dense")(x)
    if batchnorm:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return x


def build_mlp(n_features: int, hidden=(128, 64), dropout=(0.3, 0.2), batchnorm: bool = True,
              name: str = "mlp") -> tf.keras.Model:
    """MLP Sequential. dropout la 1 so (dung cho moi lop) hoac tuple dai bang hidden; 0 = khong Dropout."""
    if not isinstance(dropout, (tuple, list)):
        dropout = [dropout] * len(hidden)
    model = models.Sequential(name=name)
    model.add(layers.Input(shape=(n_features,)))
    for units, d in zip(hidden, dropout):
        model.add(layers.Dense(units, activation="relu"))
        if batchnorm:
            model.add(layers.BatchNormalization())
        if d > 0:
            model.add(layers.Dropout(d))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def build_embedding_mlp(n_dense: int, n_region: int, n_channel: int, emb_region: int = 8,
                        emb_channel: int = 12, hidden=(128, 64), dropout=(0.3, 0.2),
                        batchnorm: bool = True, name: str = "mlp_embedding") -> tf.keras.Model:
    """3 dau vao: 'dense' (so thuc da chuan hoa), 'region' va 'channel' (chi so nguyen, 0 = khac)."""
    in_dense = layers.Input(shape=(n_dense,), name="dense")
    in_region = layers.Input(shape=(), dtype="int32", name="region")
    in_channel = layers.Input(shape=(), dtype="int32", name="channel")
    # Embedding(n, d): bang tra cuu n x d. Ma i -> hang thu i. Hang duoc hoc bang backprop nhu trong so Dense.
    e_region = layers.Embedding(n_region, emb_region, name="emb_region")(in_region)
    e_channel = layers.Embedding(n_channel, emb_channel, name="emb_channel")(in_channel)
    x = layers.Concatenate(name="concat")([in_dense, e_region, e_channel])
    if not isinstance(dropout, (tuple, list)):
        dropout = [dropout] * len(hidden)
    for i, (units, d) in enumerate(zip(hidden, dropout)):
        x = _hidden_block(x, units, d, batchnorm, name=f"h{i + 1}")
    out = layers.Dense(1, activation="sigmoid", name="prob")(x)
    return models.Model([in_dense, in_region, in_channel], out, name=name)


def compile_model(model: tf.keras.Model, lr: float = 1e-3) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc", num_thresholds=500),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


def make_callbacks(checkpoint_path: Path, patience: int = 10, lr_patience: int = 5):
    """3 callback bat buoc cua de.

    EarlyStopping theo doi val_pr_auc (metric ta quan tam) va KHOI PHUC trong so tot nhat.
    ReduceLROnPlateau theo doi val_loss: loss muot hon PR-AUC nen hop de quyet dinh giam learning rate.
    ModelCheckpoint luu model tot nhat ra dia (phong khi tien trinh chet giua chung).
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=lr_patience,
                                             min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_pr_auc", mode="max",
                                           save_best_only=True),
    ]
