"""Nap va lam sach du lieu California Housing dung chung cho notebook/train.py (TT-21)."""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_california_housing

TARGET = "MedHouseVal"
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
LOC_IDX = [FEATURES.index("Latitude"), FEATURES.index("Longitude")]


def load_raw() -> pd.DataFrame:
    return fetch_california_housing(as_frame=True).frame


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Bo cac ban ghi loi ky thuat/du lieu ro rang bat thuong cua bo nay:
    AveRooms/AveOccup/AveBedrms co gia tri max lon gap hang chuc-hang tram
    lan trung vi (vi du AveOccup toi 1243 nguoi/ho - khong the co that)."""
    mask = (df["AveRooms"] <= 15) & (df["AveOccup"] <= 10) & (df["AveBedrms"] <= 5)
    return df[mask].reset_index(drop=True)


def load_clean() -> pd.DataFrame:
    return filter_outliers(load_raw())


def split_xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET]


def nhan_vi_tri(X_s, w=1.0):
    """Nhan 2 cot Latitude/Longitude (mang DA chuan hoa) voi he so w de KNN uu tien
    hang xom gan ve dia ly. Dat o module rieng de pipeline pickle/joblib duoc."""
    X_s = X_s.copy()
    X_s[:, LOC_IDX] *= w
    return X_s
