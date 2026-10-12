"""Cua so truot cho chuoi thoi gian co lo hong (TT-27).

Quy uoc: "moc" a = gio cuoi cung DA BIET khi ra du bao. Du bao cho gio a + h (h = 1, 3, 6).
  * Cua so dau vao   = cac gio a-w+1 .. a (chi QUA KHU, ke ca gio a).
  * Thong tin phu    = nhung gi BIET TRUOC ve gio a+h: lich (gio, thu, cuoi tuan, ngay le) va 2 gia tri
                       da xay ra: y[a+h-24] (cung gio hom qua), y[a+h-168] (cung gio tuan truoc). Ca 2 deu <= a
                       vi h <= 6 < 24.
  * Nhan             = y[a+h], chi lay gio CO DU LIEU THAT (khong lay gio noi suy).

valid_anchors() tra ve CUNG MOT tap moc cho moi mo hinh va moi do dai cua so (kiem tra theo cua so dai nhat
48 gio) -> MAE cua baseline, XGBoost, RNN/LSTM/GRU, cua so 6/12/24/48 deu tinh tren dung cung cac gio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import data as D

MAX_WINDOW = 48
SCALE_COLS = [D.TARGET, "temp_c", "rain_log", "clouds"]


def tao_chuoi(X, y, do_dai=24, buoc_du_bao=1):
    """Ham cua de bai (gia dinh chuoi LIEN TUC). Giu lai de doi chieu; bai nay dung make_windows()
    vi chuoi co lo hong: cua so cua tao_chuoi() co the chua 2 thoi diem cach nhau nhieu ngay."""
    Xs, ys = [], []
    for i in range(len(X) - do_dai - buoc_du_bao + 1):
        Xs.append(X[i:i + do_dai])
        ys.append(y[i + do_dai + buoc_du_bao - 1])
    return np.array(Xs), np.array(ys)


class Scaler:
    """StandardScaler cho cac cot lien tuc, fit CHI tren cac gio thuoc tap train. Cot 0/1 va sin/cos giu nguyen."""

    def fit(self, feat: pd.DataFrame, train_range) -> "Scaler":
        tr = feat.loc[train_range[0]:train_range[1], SCALE_COLS].dropna()
        self.sc = StandardScaler().fit(tr)
        self.mu, self.sd = self.sc.mean_[0], self.sc.scale_[0]      # cua traffic_volume
        return self

    def transform(self, feat: pd.DataFrame) -> pd.DataFrame:
        out = feat.copy()
        ok = out[SCALE_COLS].notna().all(axis=1)
        out.loc[ok, SCALE_COLS] = self.sc.transform(out.loc[ok, SCALE_COLS])
        return out

    def y_to_scaled(self, y):
        return (np.asarray(y) - self.mu) / self.sd

    def y_from_scaled(self, z):
        return np.asarray(z) * self.sd + self.mu


def valid_anchors(feat: pd.DataFrame, rng, horizon: int, max_window: int = MAX_WINDOW) -> np.ndarray:
    """Chi so moc a (vi tri trong feat) sao cho: gio a+h nam trong khoang `rng`, y[a+h] la du lieu that,
    max_window gio a-max_window+1..a khong co NaN (khong bac qua lo dai), y[a+h-24] va y[a+h-168] co gia tri."""
    vals = feat.to_numpy(dtype=float)
    row_ok = ~np.isnan(vals).any(axis=1)
    # so gio lien tiep khong NaN ket thuc tai moi vi tri
    run = np.zeros(len(feat), dtype=int)
    c = 0
    for i, ok in enumerate(row_ok):
        c = c + 1 if ok else 0
        run[i] = c
    y = feat[D.TARGET].to_numpy()
    real = (~np.isnan(y)) & (feat["noi_suy"].to_numpy() == 0)
    t = feat.index
    a = np.arange(168, len(feat) - horizon)
    tgt = a + horizon
    m = (t[tgt] >= rng[0]) & (t[tgt] <= rng[1]) & real[tgt] & (run[a] >= max_window)
    m &= ~np.isnan(y[tgt - 24]) & ~np.isnan(y[tgt - 168])
    return a[m]


@dataclass
class Windows:
    X: np.ndarray            # (n, w, so dac trung)  -- da chuan hoa
    side: np.ndarray         # (n, 8)  lich cua gio dich + y[-24], y[-168] (chuan hoa)
    y: np.ndarray            # (n,)    nhan da chuan hoa
    y_raw: np.ndarray        # (n,)    nhan goc (xe/gio)
    naive24: np.ndarray      # (n,)    baseline "gio nay = gio nay hom qua"
    naive168: np.ndarray     # (n,)    baseline "gio nay = gio nay tuan truoc"
    t: pd.DatetimeIndex      # thoi diem cua nhan (a + h)


def make_windows(feat_s: pd.DataFrame, feat_raw: pd.DataFrame, anchors: np.ndarray, window: int, horizon: int,
                 scaler: Scaler) -> Windows:
    """Cat cua so [a-w+1, a] tu bang da chuan hoa. Kiem tra lai: khong gia tri NaN, khong cham tuong lai."""
    vals = feat_s.to_numpy(dtype=np.float32)
    idx = anchors[:, None] + np.arange(-window + 1, 1)[None, :]           # (n, w), phan tu cuoi = a
    tgt = anchors + horizon
    assert (idx.max(axis=1) < tgt).all(), "cua so cham toi gio can du bao"    # chi qua khu
    X = vals[idx]
    assert not np.isnan(X).any(), "cua so bac qua lo hong"
    y_raw = feat_raw[D.TARGET].to_numpy()
    cal = feat_raw[D.CALENDAR_COLS].to_numpy(dtype=np.float32)[tgt]
    lag24, lag168 = y_raw[tgt - 24], y_raw[tgt - 168]
    side = np.column_stack([cal, scaler.y_to_scaled(lag24), scaler.y_to_scaled(lag168)]).astype(np.float32)
    return Windows(X=X, side=side, y=scaler.y_to_scaled(y_raw[tgt]).astype(np.float32), y_raw=y_raw[tgt],
                   naive24=lag24, naive168=lag168, t=feat_raw.index[tgt])


def tree_matrix(feat_raw: pd.DataFrame, anchors: np.ndarray, horizon: int, n_lags: int = 24) -> pd.DataFrame:
    """Dau vao XGBoost: lag y[a-k] (k = 0..23), y[a+h-24], y[a+h-168], thoi tiet tai a, lich cua gio dich (so nguyen)."""
    y = feat_raw[D.TARGET].to_numpy()
    tgt = anchors + horizon
    t = feat_raw.index[tgt]
    cols = {f"lag_{k}": y[anchors - k] for k in range(n_lags)}
    cols["cung_gio_hom_qua"] = y[tgt - 24]
    cols["cung_gio_tuan_truoc"] = y[tgt - 168]
    cols["tb_3_gio"] = np.mean([y[anchors - k] for k in range(3)], axis=0)
    for c in ["temp_c", "rain_log", "snow", "clouds"] + [c for c in feat_raw.columns if c.startswith("w_")]:
        cols[c] = feat_raw[c].to_numpy()[anchors]
    cols["gio_dich"] = t.hour
    cols["thu_dich"] = t.dayofweek
    cols["thang_dich"] = t.month
    cols["le_dich"] = feat_raw["holiday"].to_numpy()[tgt]
    return pd.DataFrame(cols)
