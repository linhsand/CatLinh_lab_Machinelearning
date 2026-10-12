"""Transformer Encoder tu xay bang Keras cho TT-28 (nhanh A).

  ids (B, L) -> Embedding (B, L, d) x sqrt(d) [+ PositionalEncoding] -> KhoiTransformer x N -> trung binh co mask -> Dense(4)

* PositionalEncoding: sin/cos co dinh (Vaswani 2017) CONG vao embedding:
      PE[pos, 2i] = sin(pos / 10000^(2i/d)),  PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
  Bo lop nay (use_pos=False) -> attention + trung binh deu KHONG phu thuoc thu tu -> mo hinh thanh "tui tu".
* KhoiTransformer: dung khung cua de (MultiHeadAttention -> Dropout -> residual + LayerNorm -> FFN -> residual +
  LayerNorm), them 2 thu: (1) attention_mask de tu khong nhin vao vi tri padding, (2) luu trong so attention cua lan
  goi gan nhat (self.last_scores) de ve heatmap.
  key_dim = d_model // n_heads (chuan cua bai bao): tong so tham so KHONG doi khi doi so head, nen khao sat
  1/2/4/8 head la so sanh cong bang. (Code mau cua de dung key_dim = d_model -> so tham so tang theo so head.)
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


@tf.keras.utils.register_keras_serializable("tt28")
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len: int, d_model: int, **kw):
        super().__init__(**kw)
        self.max_len, self.d_model = max_len, d_model
        pos = np.arange(max_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
        pe = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
        self.pe = tf.constant(pe[None], dtype=tf.float32)          # (1, L, d)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

    def get_config(self):
        return {**super().get_config(), "max_len": self.max_len, "d_model": self.d_model}


@tf.keras.utils.register_keras_serializable("tt28")
class KhoiTransformer(layers.Layer):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, **kw):
        super().__init__(**kw)
        self.d_model, self.n_heads, self.d_ff, self.dropout = d_model, n_heads, d_ff, dropout
        self.att = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)
        self.ffn = tf.keras.Sequential([layers.Dense(d_ff, activation="relu"), layers.Dense(d_model)])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.do1 = layers.Dropout(dropout)
        self.do2 = layers.Dropout(dropout)
        self.last_scores = None

    def build(self, input_shape):
        self.att.build(input_shape, input_shape)
        self.ffn.build(input_shape)
        self.ln1.build(input_shape)
        self.ln2.build(input_shape)
        super().build(input_shape)

    def call(self, x, pad_mask=None, training=False):
        # pad_mask (B, L) True = tu that -> attention_mask (B, L, L): moi query chi nhin cac key la tu that
        att_mask = None if pad_mask is None else tf.logical_and(pad_mask[:, :, None], pad_mask[:, None, :])
        a, scores = self.att(x, x, attention_mask=att_mask, return_attention_scores=True, training=training)
        self.last_scores = scores                                   # (B, heads, L, L)
        a = self.do1(a, training=training)
        x = self.ln1(x + a)                                         # residual + norm
        f = self.do2(self.ffn(x), training=training)
        return self.ln2(x + f)

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model, "n_heads": self.n_heads, "d_ff": self.d_ff,
                "dropout": self.dropout}


@tf.keras.utils.register_keras_serializable("tt28")
class MaskedMean(layers.Layer):
    """Trung binh theo vi tri, bo qua padding (GlobalAveragePooling1D se tinh ca so 0 cua padding)."""

    def call(self, x, pad_mask):
        m = tf.cast(pad_mask, x.dtype)[:, :, None]
        return tf.reduce_sum(x * m, axis=1) / tf.maximum(tf.reduce_sum(m, axis=1), 1.0)


@tf.keras.utils.register_keras_serializable("tt28")
class ScaleEmbedding(layers.Layer):
    def __init__(self, d_model: int, **kw):
        super().__init__(**kw)
        self.d_model = d_model

    def call(self, x):
        return x * tf.math.sqrt(tf.cast(self.d_model, x.dtype))

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model}


@tf.keras.utils.register_keras_serializable("tt28")
class NotPad(layers.Layer):
    def call(self, ids):
        return tf.not_equal(ids, 0)


def build_transformer(vocab_size: int, max_len: int, n_classes: int = 4, d_model: int = 64, n_heads: int = 4,
                      d_ff: int = 128, n_blocks: int = 2, dropout: float = 0.1, use_pos: bool = True,
                      name: str = "transformer") -> tf.keras.Model:
    ids = layers.Input((max_len,), dtype="int32", name="ids")
    mask = NotPad(name="mask")(ids)
    x = layers.Embedding(vocab_size, d_model, name="embedding")(ids)
    # Nhan sqrt(d_model) nhu bai bao goc: embedding khoi tao ~U(-0,05; 0,05) con PE co bien do 1. Khong nhan thi
    # PE ap dao nghia cua tu va mo hinh ket o muc doan mo (~28% sau 1 epoch khi chay thu).
    x = ScaleEmbedding(d_model, name="scale_sqrt_d")(x)
    if use_pos:
        x = PositionalEncoding(max_len, d_model, name="pos_encoding")(x)
    x = layers.Dropout(dropout)(x)
    for i in range(n_blocks):
        x = KhoiTransformer(d_model, n_heads, d_ff, dropout, name=f"khoi_{i + 1}")(x, pad_mask=mask)
    x = MaskedMean(name="tb_co_mask")(x, mask)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", name="prob")(x)
    return models.Model(ids, out, name=name)


def build_lstm(vocab_size: int, max_len: int, n_classes: int = 4, d_model: int = 64, units: int = 64,
               dropout: float = 0.2) -> tf.keras.Model:
    """Doi chung "LSTM" cho bang so sanh 4 cach: Embedding -> BiLSTM -> Dense. Phan loai ca cau nen dung 2 chieu
    la hop le (khac TT-27: du bao tuong lai thi KHONG duoc nhin ve sau)."""
    ids = layers.Input((max_len,), dtype="int32", name="ids")
    x = layers.Embedding(vocab_size, d_model, mask_zero=True, name="embedding")(ids)
    x = layers.Bidirectional(layers.LSTM(units), name="bilstm")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", name="prob")(x)
    return models.Model(ids, out, name="bilstm")


def attention_maps(model: tf.keras.Model, ids: np.ndarray) -> list[np.ndarray]:
    """Chay model (eager) roi lay trong so attention cua tung khoi: list (B, heads, L, L)."""
    model(tf.constant(ids), training=False)
    return [l.last_scores.numpy() for l in model.layers if isinstance(l, KhoiTransformer)]
