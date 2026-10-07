"""Tai, chia va tien xu ly bo Health Insurance Cross Sell Prediction (Kaggle) cho TT-25.

- Lan dau: tai bang kagglehub (bo cong khai, khong can API key) roi copy train.csv vao data/.
  test.csv cua Kaggle KHONG co nhan -> chi dung train.csv (381.109 dong) va tu chia 70/15/15.
- Chia train/val/test co stratify theo Response (ti le quan tam ~12,3% o ca 3 tap).
- MOI phep bien doi (StandardScaler, OneHotEncoder, bang ma cho Embedding) chi fit tren TRAIN.

Hai cach ma hoa bien phan loai nhieu muc (Region_Code 53 muc, Policy_Sales_Channel 155 muc):
  * "onehot":    1 ma tran so thuc duy nhat (~220 cot) -> MLP Sequential.
  * "embedding": ma tran so thuc nho (~11 cot) + 2 vector chi so nguyen -> MLP co 2 lop Embedding.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_CSV = DATA_DIR / "train.csv"
KAGGLE_ID = "anmolkumar/health-insurance-cross-sell-prediction"

TARGET = "Response"
RANDOM_STATE = 42
# Cot so (Annual_Premium se log1p truoc) va cot nhi phan -> deu StandardScaler
NUM_COLS = ["Age", "Annual_Premium", "Vintage"]
BIN_COLS = ["Gender", "Driving_License", "Previously_Insured", "Vehicle_Damage"]
LOW_CARD = ["Vehicle_Age"]                          # 3 muc -> one-hot o ca 2 cach ma hoa
HIGH_CARD = ["Region_Code", "Policy_Sales_Channel"]  # 53 / 155 muc -> one-hot HOAC Embedding


def download() -> None:
    if DATA_CSV.exists():
        return
    import kagglehub  # chi can khi chua co du lieu
    print(f"[data] Dang tai {KAGGLE_ID} bang kagglehub ...")
    src = Path(kagglehub.dataset_download(KAGGLE_ID))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "train.csv", DATA_CSV)
    print(f"[data] Da copy train.csv vao {DATA_CSV}")


def load_raw() -> pd.DataFrame:
    """381.109 dong x 12 cot. Ma hoa ngay cac cot chu thanh 0/1 de de xu ly ve sau."""
    download()
    df = pd.read_csv(DATA_CSV).drop(columns="id")
    df["Gender"] = (df["Gender"] == "Male").astype(int)
    df["Vehicle_Damage"] = (df["Vehicle_Damage"] == "Yes").astype(int)
    for c in HIGH_CARD:
        df[c] = df[c].astype(int)        # luu bang float trong CSV nhung thuc chat la MA
    return df


def split(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15):
    """70/15/15 co stratify. Tra ve 3 DataFrame (giu nguyen cot goc)."""
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[TARGET],
                                       random_state=RANDOM_STATE)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size),
                                  stratify=train_val[TARGET], random_state=RANDOM_STATE)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


@dataclass
class Prepared:
    """Ket qua tien xu ly cho 1 tap. X_dense luon co; region/channel chi dung cho Embedding."""
    X_dense: np.ndarray
    region: np.ndarray
    channel: np.ndarray
    y: np.ndarray


class Preprocessor:
    """fit tren train -> transform train/val/test.

    log1p(Annual_Premium) -> StandardScaler cho so + nhi phan; one-hot Vehicle_Age;
    bien nhieu muc: one-hot (mode='onehot') hoac bang ma -> chi so nguyen (mode='embedding').
    Ma chua gap o train (hoac qua hiem, < min_count dong) -> chi so 0 = "khac".
    """

    def __init__(self, mode: str = "onehot", min_count: int = 1):
        assert mode in ("onehot", "embedding")
        self.mode, self.min_count = mode, min_count

    def _numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[NUM_COLS + BIN_COLS].astype(float).copy()
        out["Annual_Premium"] = np.log1p(out["Annual_Premium"])
        return out

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.scaler = StandardScaler().fit(self._numeric(df))
        cat_cols = LOW_CARD + (HIGH_CARD if self.mode == "onehot" else [])
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                 dtype=np.float32).fit(df[cat_cols])
        self.cat_cols = cat_cols
        # bang ma cho Embedding: ma -> 1..n, 0 danh cho "khac"
        self.vocab = {}
        for c in HIGH_CARD:
            counts = df[c].value_counts()
            keep = sorted(counts[counts >= self.min_count].index)
            self.vocab[c] = {code: i + 1 for i, code in enumerate(keep)}
        self.feature_names = (NUM_COLS + BIN_COLS + list(self.ohe.get_feature_names_out(cat_cols)))
        return self

    def transform(self, df: pd.DataFrame) -> Prepared:
        num = self.scaler.transform(self._numeric(df)).astype(np.float32)
        X = np.hstack([num, self.ohe.transform(df[self.cat_cols])])
        idx = {c: df[c].map(self.vocab[c]).fillna(0).astype(np.int32).to_numpy() for c in HIGH_CARD}
        return Prepared(X, idx["Region_Code"], idx["Policy_Sales_Channel"],
                        df[TARGET].to_numpy().astype(np.float32))

    def vocab_size(self, col: str) -> int:
        return len(self.vocab[col]) + 1   # +1 cho chi so 0 "khac"


def tree_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Dau vao cho LightGBM: giu nguyen bang (cay khong can chuan hoa/log). 2 bien nhieu muc giu ma so nguyen,
    khi fit truyen categorical_feature=HIGH_CARD de LightGBM chia nhom theo muc. Vehicle_Age -> thu tu 0/1/2."""
    X = df.drop(columns=TARGET).copy()
    X["Vehicle_Age"] = X["Vehicle_Age"].map({"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2})
    return X
