"""Nap du lieu Auto MPG va tien xu ly dung chung cho notebook/train.py (TT-22)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "auto-mpg.data"
TARGET = "mpg"
NUM_COLS = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
CAT_COLS = ["origin"]
COLS = [TARGET] + NUM_COLS + CAT_COLS + ["car_name"]
ORIGIN_NAMES = {1: "My", 2: "Chau_Au", 3: "Nhat"}
MPG_TO_L100KM = 235.215  # L/100km = 235,215 / mpg


def load_raw() -> pd.DataFrame:
    """Doc file goc UCI: 8 cot so cach nhau boi khoang trang, roi TAB + ten xe trong ngoac kep.
    Giu nguyen moi cot o dang chuoi de thay dung van de '?' cua horsepower."""
    rows = []
    for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        so, ten = line.split("\t")
        rows.append(so.split() + [ten.strip().strip('"')])
    return pd.DataFrame(rows, columns=COLS)


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Ep kieu so ('?' -> NaN, KHONG dien o day de tranh ro ri test vao train -
    median se duoc hoc trong pipeline tu tap train), doi ma origin thanh nhan
    phan loai va bo car_name (dinh danh, khong mang tin hieu tong quat)."""
    df = df_raw.drop(columns=["car_name"]).copy()
    for c in [TARGET] + NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["origin"] = df["origin"].astype(int).map(ORIGIN_NAMES)
    return df


def load_clean() -> pd.DataFrame:
    return clean(load_raw())


def split_xy(df: pd.DataFrame):
    return df[NUM_COLS + CAT_COLS], df[TARGET]


def make_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Cot so: dien median (hoc tu train) + chuan hoa (tuy chon).
    origin: one-hot (3 vung -> 3 cot 0/1), tranh model hieu 'Nhat > Chau Au > My'."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])


def mpg_to_l100km(mpg):
    return MPG_TO_L100KM / mpg
