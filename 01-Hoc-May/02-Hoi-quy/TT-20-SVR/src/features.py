"""Feature engineering dung chung cho notebook va script train (TT-20).

Ca 3 dac trung deu la KIEN THUC MIEN nganh xay dung, model khong the tu
"nghi ra" duoc chi tu 8 cot nguyen lieu tho.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Strength"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Ty le nuoc/xi mang - yeu to quyet dinh cuong do theo dinh luat Abrams
    out["water_cement_ratio"] = out["Water"] / out["Cement"]
    # Tong chat ket dinh (xi mang + phu gia khoang) - cung gop phan tao cuong do
    out["tong_chat_ket_dinh"] = out["Cement"] + out["BlastFurnaceSlag"] + out["FlyAsh"]
    # Age lech phai manh (nhieu mau dung 28 ngay) - quan he voi cuong do gan voi log(Age)
    out["log_age"] = np.log1p(out["Age"])
    return out


def split_xy(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c != TARGET]
    return df[feature_cols], df[TARGET]
