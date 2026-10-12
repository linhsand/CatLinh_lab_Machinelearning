"""Kien truc CNN bang Keras cho TT-26.

Ca 2 kieu mang co cung "khung": input (224,224,3) 0..255 -> [features] -> GlobalAveragePooling -> Dropout
-> Dense(1, sigmoid). Phan trich xuat dac trung la 1 sub-model (features_of()), nen Grad-CAM dung chung 1 ham.

* build_small_cnn():  baseline train TU DAU. 3 khoi [Conv-BN-ReLU-Conv-BN-ReLU-MaxPool] (32/64/128 bo loc).
* build_transfer():   EfficientNetB0 hoc tren ImageNet, include_top=False. Giai doan 1: dong bang toan bo base.
* unfreeze_top():     giai doan 2 (fine-tuning): mo khoa N lop cuoi cua base, TRU BatchNorm, compile lai lr nho.
* compile_model():    Adam + binary_crossentropy, theo doi Recall, Precision, ROC-AUC.
* make_callbacks():   EarlyStopping (val_auc) + ReduceLROnPlateau (val_loss) + ModelCheckpoint.
* grad_cam():         ban do nhiet "model nhin vao dau" cho tung anh.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

INPUT_SHAPE = (224, 224, 3)
FEATURES = "features"          # ten sub-model cua CNN tu dau; voi EfficientNet la "efficientnetb0"


def features_of(model: tf.keras.Model) -> tf.keras.Model:
    """Sub-model trich xuat dac trung (lop duy nhat la 1 Model ben trong model)."""
    return next(l for l in model.layers if isinstance(l, tf.keras.Model))


def _head(x, dropout: float):
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    return layers.Dense(1, activation="sigmoid", name="prob")(x)


def build_small_cnn(filters=(32, 64, 128), dropout: float = 0.3, name: str = "cnn_tu_dau") -> tf.keras.Model:
    """CNN nho tu xay. Moi khoi: 2 Conv 3x3 (BN + ReLU) roi MaxPool 2x2 -> anh nho 1/2, so bo loc x2.
    224 -(conv buoc 2)-> 112 -(pool)-> 56 -> 28 -> 14: ban do dac trung cuoi 14x14x128."""
    inp = layers.Input(INPUT_SHAPE, name="image")
    x = layers.Rescaling(1 / 255, name="rescale")(inp)
    feat_in = layers.Input(INPUT_SHAPE, name="feat_in")
    h = feat_in
    for i, f in enumerate(filters):
        for j in range(2):
            stride = 2 if (i, j) == (0, 0) else 1    # Conv dau tien buoc 2 (224 -> 112): nhanh gap 5 lan tren CPU
            h = layers.Conv2D(f, 3, strides=stride, padding="same", use_bias=False, name=f"b{i + 1}_conv{j + 1}")(h)
            h = layers.BatchNormalization(name=f"b{i + 1}_bn{j + 1}")(h)
            h = layers.ReLU(name=f"b{i + 1}_relu{j + 1}")(h)
        h = layers.MaxPooling2D(2, name=f"b{i + 1}_pool")(h)
    features = models.Model(feat_in, h, name=FEATURES)
    return models.Model(inp, _head(features(x), dropout), name=name)


def build_transfer(dropout: float = 0.3, name: str = "efficientnetb0_tl") -> tf.keras.Model:
    """EfficientNetB0 (ImageNet) dong bang + dau phan loai moi. Base nhan anh 0..255 (tu chuan hoa ben trong).

    Goi base(x, training=False): BatchNorm cua base LUON chay o che do suy luan (dung mean/var cua ImageNet),
    ke ca khi fine-tune sau nay -> khong bi batch nho (32 anh X-quang) lam lech thong ke.
    """
    # Giu ten mac dinh "efficientnetb0": Keras suy ra ten file trong so ImageNet tu ten model
    base = tf.keras.applications.EfficientNetB0(input_shape=INPUT_SHAPE, include_top=False, weights="imagenet")
    base.trainable = False                              # GIAI DOAN 1: dong bang
    inp = layers.Input(INPUT_SHAPE, name="image")
    return models.Model(inp, _head(base(inp, training=False), dropout), name=name)


def unfreeze_top(model: tf.keras.Model, n_last: int = 30) -> int:
    """GIAI DOAN 2: mo khoa n_last lop cuoi cua base, giu BatchNorm dong bang. Tra ve so tensor trong so duoc mo."""
    base = features_of(model)
    base.trainable = True
    for layer in base.layers[:-n_last]:
        layer.trainable = False
    for layer in base.layers[-n_last:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return len(base.trainable_weights)


def compile_model(model: tf.keras.Model, lr: float = 1e-3) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    return model


def make_callbacks(checkpoint_path: Path, patience: int = 5, lr_patience: int = 2, min_lr: float = 1e-7):
    """EarlyStopping theo val_auc (chat luong XEP HANG, khong phu thuoc nguong; nguong chon rieng o buoc 9)
    va KHOI PHUC trong so tot nhat. ReduceLROnPlateau theo val_loss. ModelCheckpoint luu ban tot nhat ra dia."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=lr_patience, min_lr=min_lr),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_auc", mode="max", save_best_only=True),
    ]


def grad_cam(model: tf.keras.Model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM cho lop PNEUMONIA (dau ra sigmoid).

    A = ban do dac trung cuoi cua 'features' (7x7x1280 voi EfficientNetB0, 14x14x128 voi CNN tu dau).
    Trong so moi kenh = trung binh gradient d(xac suat)/dA theo khong gian. CAM = ReLU(sum_k w_k * A_k),
    chuan hoa ve [0,1] roi phong to ve 224x224. Tra ve (cam (N,224,224), xac suat (N,)).
    """
    feat = features_of(model)
    gap, drop, dense = model.get_layer("gap"), model.get_layer("dropout"), model.get_layer("prob")
    pre = [l for l in model.layers if l.name == "rescale"]   # chi CNN tu dau co
    cams, probs = [], []
    for i in range(0, len(x), 16):
        h = tf.convert_to_tensor(x[i:i + 16])
        for l in pre:
            h = l(h)
        with tf.GradientTape() as tape:
            A = feat(h, training=False)
            tape.watch(A)
            p = dense(drop(gap(A), training=False))[:, 0]
        g = tape.gradient(p, A)
        w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(w * A, axis=-1))
        cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
        cam = tf.image.resize(cam[..., None], INPUT_SHAPE[:2], method="bilinear")[..., 0]
        cams.append(cam.numpy()); probs.append(p.numpy())
    return np.concatenate(cams), np.concatenate(probs)


def export_clean(model: tf.keras.Model, path: Path) -> None:
    """Luu ban sach (khong kem trang thai optimizer) de nop: models/best_model.keras."""
    clean = tf.keras.models.clone_model(model)
    clean.set_weights(model.get_weights())
    clean.save(path)
