"""Tai va doc bo Human Activity Recognition Using Smartphones (UCI) cho TT-24.

- Lan dau: tai zip tu UCI (~61 MB, ben trong la 1 zip long nhau), giai nen vao data/UCI HAR Dataset/
  roi cache 561 dac trung + nhan + ma nguoi sang data/har.parquet (doc lai < 1 giay).
- GIU NGUYEN cach chia train/test theo NGUOI cua bo goc: 21 nguoi train, 9 nguoi test, khong trung nhau.
- features.txt co 42 ten bi trung (vd 'fBodyAcc-bandsEnergy()-1,8' lap 3 lan cho X/Y/Z) -> them hau to
  '__<so thu tu>' cho cac ten trung de moi cot co ten duy nhat.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "UCI HAR Dataset"
CACHE_PATH = DATA_DIR / "har.parquet"
DATA_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

ACTIVITIES = {1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
              4: "SITTING", 5: "STANDING", 6: "LAYING"}
MOVING = {"WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"}


def download() -> None:
    """Tai va giai nen (zip ngoai chua 'UCI HAR Dataset.zip')."""
    if RAW_DIR.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[data] Dang tai {DATA_URL} ...")
    with urllib.request.urlopen(DATA_URL, timeout=120) as r:
        outer = zipfile.ZipFile(io.BytesIO(r.read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read("UCI HAR Dataset.zip")))
    inner.extractall(DATA_DIR, members=[m for m in inner.namelist() if not m.startswith("__MACOSX")])
    print(f"[data] Da giai nen vao {RAW_DIR}")


def feature_names() -> list[str]:
    raw = pd.read_csv(RAW_DIR / "features.txt", sep=r"\s+", header=None, names=["idx", "name"])["name"]
    dup = raw.duplicated(keep=False)
    return [f"{n}__{i + 1}" if d else n for i, (n, d) in enumerate(zip(raw, dup))]


def _read_split(split: str, names: list[str]) -> pd.DataFrame:
    d = RAW_DIR / split
    X = pd.read_csv(d / f"X_{split}.txt", sep=r"\s+", header=None, dtype=np.float32)
    X.columns = names
    meta = pd.DataFrame({
        "activity": pd.read_csv(d / f"y_{split}.txt", header=None)[0].map(ACTIVITIES).values,
        "subject": pd.read_csv(d / f"subject_{split}.txt", header=None)[0].values,
        "split": split,
    })
    return pd.concat([X, meta], axis=1)


def load_har() -> pd.DataFrame:
    """1 bang 10.299 dong: 561 dac trung (float32) + activity + subject + split ('train'/'test')."""
    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    download()
    names = feature_names()
    df = pd.concat([_read_split("train", names), _read_split("test", names)], ignore_index=True)
    df.to_parquet(CACHE_PATH, index=False)
    return df


def split_by_subject(df: pd.DataFrame):
    """Tra ve X_train, X_test, y_train, y_test, groups_train (ma nguoi) theo cach chia GOC."""
    feats = [c for c in df.columns if c not in ("activity", "subject", "split")]
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    overlap = set(tr["subject"]) & set(te["subject"])
    assert not overlap, f"Nguoi xuat hien o ca train va test: {overlap}"
    return (tr[feats].to_numpy(), te[feats].to_numpy(), tr["activity"].to_numpy(),
            te["activity"].to_numpy(), tr["subject"].to_numpy(), feats)


if __name__ == "__main__":
    data = load_har()
    print(data.shape, data["split"].value_counts().to_dict())
