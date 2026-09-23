"""Feature engineering dung chung cho notebook va script train (TT-19).

Tach rieng module nay de notebook (giai thich tung buoc) va train.py (chay
lai nhanh, khong can mo Jupyter) dung CHUNG mot logic, tranh lech ket qua.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# cnt = casual + registered (dung tuyet doi) -> ro ri neu giu lai 2 cot nay
LEAK_COLS = ["casual", "registered"]
# instant la so thu tu dong, dteday da duoc tach thanh yr/mnth (con hr rieng)
ID_COLS = ["instant", "dteday"]
# (ten cot, chu ky) - hr lap lai moi 24h, mnth moi 12 thang, weekday moi 7 ngay
CYCLICAL = [("hr", 24), ("mnth", 12), ("weekday", 7)]


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ma hoa sin/cos cho cac cot mang tinh CHU KY (gio 23 va gio 0 ke nhau)."""
    out = df.copy()
    for col, period in CYCLICAL:
        out[f"{col}_sin"] = np.sin(2 * np.pi * out[col] / period)
        out[f"{col}_cos"] = np.cos(2 * np.pi * out[col] / period)
    return out


def build_features(df: pd.DataFrame, drop_leak: bool = True) -> pd.DataFrame:
    """Tra ve dataframe da: bo cot id/ro ri (tuy chon) + them dac trung chu ky."""
    cols_to_drop = [c for c in ID_COLS if c in df.columns]
    if drop_leak:
        cols_to_drop += [c for c in LEAK_COLS if c in df.columns]
    out = df.drop(columns=cols_to_drop)
    out = add_cyclical_features(out)
    return out


def time_based_split(df: pd.DataFrame):
    """Chia THEO THOI GIAN: nam 1 (yr=0) train - 9 thang dau nam 2 validation
    - 3 thang cuoi nam 2 test. Du lieu la chuoi thoi gian nen KHONG duoc
    shuffle/chia ngau nhien (se nhin thay tuong lai khi train)."""
    train = df[df["yr"] == 0].reset_index(drop=True)
    year2 = df[df["yr"] == 1]
    val = year2[year2["mnth"] <= 9].reset_index(drop=True)
    test = year2[year2["mnth"] > 9].reset_index(drop=True)
    return train, val, test
