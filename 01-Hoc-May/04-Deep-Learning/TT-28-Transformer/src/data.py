"""Nap, lam sach va chia bo AG News cho TT-28.

- Lan dau: tai bang `datasets` (repo `fancyzhx/ag_news`; ten cu "ag_news" khong con dung voi datasets >= 4)
  roi luu data/ag_news.parquet. Cac lan sau doc file nay.
- Van ban co rac HTML do qua trinh thu thap:
    * 26% tin co ma ky tu MAT dau '&': "AUSTRALIA #39;S" (= ') , "#36;" (= $), "#151;" (= --)
    * 11% tin dung '\\' thay xuong dong ("dwindling\\band")
    * 4% tin co the HTML da ma hoa: "&lt;b&gt;...&lt;/b&gt;"
  -> clean_text(): bo sung '&', html.unescape, xoa the, doi '\\' thanh khoang trang.
- Chia: test goc (7.600) giu nguyen. Tu train goc tach 10% lam VAL (stratify, seed 42) de EarlyStopping / chon
  so head / chon nguong; TEST chi de bao cao.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE = DATA_DIR / "ag_news.parquet"
HF_ID = "fancyzhx/ag_news"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
RANDOM_STATE = 42
VAL_FRACTION = 0.10


def download() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    from datasets import load_dataset   # chi can lan dau
    print(f"[data] Dang tai {HF_ID} ...", flush=True)
    ds = load_dataset(HF_ID)
    df = pd.concat([ds["train"].to_pandas().assign(tap_goc="train"),
                    ds["test"].to_pandas().assign(tap_goc="test")], ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    print(f"[data] Da luu {CACHE}", flush=True)
    return df


_MISSING_AMP = re.compile(r"(?<!&)#(\d+);")
_MISSING_AMP_NAMED = re.compile(r"(?<!&)\b(quot|amp|lt|gt);")
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = _MISSING_AMP.sub(r"&#\1;", s)              # "#39;" -> "&#39;"
    s = _MISSING_AMP_NAMED.sub(r"&\1;", s)         # "quot;" -> "&quot;"
    s = html.unescape(html.unescape(s))            # 2 lan: co tin bi ma hoa 2 lop (&amp;lt;)
    s = _TAG.sub(" ", s)                           # <b>, <a href=...>
    s = s.replace("\\", " ")
    return _WS.sub(" ", s).strip()


def load() -> dict[str, pd.DataFrame]:
    """{'train', 'val', 'test'}: moi DataFrame co cot text (da lam sach), text_goc, label."""
    df = download()
    df["text_goc"] = df["text"]
    df["text"] = df["text"].map(clean_text)
    train_full = df[df["tap_goc"] == "train"].reset_index(drop=True)
    test = df[df["tap_goc"] == "test"].reset_index(drop=True)
    train, val = train_test_split(train_full, test_size=VAL_FRACTION, stratify=train_full["label"],
                                  random_state=RANDOM_STATE)
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True), "test": test}


def subset(df: pd.DataFrame, n: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """n mau, stratify theo label (moi lop n/4)."""
    return df.groupby("label", group_keys=False).sample(n // len(LABELS), random_state=seed).sample(
        frac=1, random_state=seed).reset_index(drop=True)


def artifact_table(df: pd.DataFrame) -> pd.DataFrame:
    """So tin co tung loai rac, truoc va sau khi lam sach."""
    pats = {"#NN; thieu '&'": r"(?<!&)#\d+;", "the HTML": r"&lt;|<[^>]+>", "dau '\\'": r"\\",
            "quot; / &amp;": r"quot;|&amp;"}
    rows = []
    for k, p in pats.items():
        rows.append({"loai_rac": k, "truoc": int(df["text_goc"].str.contains(p, regex=True).sum()),
                     "sau": int(df["text"].str.contains(p, regex=True).sum())})
    t = pd.DataFrame(rows).set_index("loai_rac")
    t["ti_le_truoc"] = t["truoc"] / len(df)
    return t
