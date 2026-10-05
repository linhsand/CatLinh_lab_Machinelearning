"""Doc, lam sach giao dich Online Retail II va tong hop len cap SAN PHAM (TT-23).

Don vi phan tich la StockCode (san pham), khong phai khach hang.
File xlsx goc ~1,07 trieu dong, doc bang openpyxl mat vai phut -> cache sang parquet
o lan doc dau tien (data/online_retail_ii.parquet).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
XLSX_PATH = DATA_DIR / "online_retail_II.xlsx"
CACHE_PATH = DATA_DIR / "online_retail_ii.parquet"

# Ma khong phai san pham: phi buu dien, dieu chinh thu cong, phi ngan hang, phi van chuyen...
NON_PRODUCT_CODES = {"POST", "M", "BANK CHARGES", "DOT", "ADJUST", "ADJUST2",
                     "C2", "D", "CRUK", "PADS", "B", "S", "AMAZONFEE", "TEST001", "TEST002"}
MIN_ORDERS = 5

FEATURES = ["tong_so_luong", "tong_doanh_thu", "gia_trung_binh", "so_don_hang",
            "so_khach_mua", "do_lech_sl", "sl_moi_don", "ty_le_mua_lai"]


def load_raw() -> pd.DataFrame:
    """Gop 2 sheet (2009-2010 va 2010-2011) thanh 1 bang; cache parquet cho lan sau."""
    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    if not XLSX_PATH.exists():
        raise FileNotFoundError(
            f"Khong thay {XLSX_PATH}. Tai tai https://archive.ics.uci.edu/dataset/502/online+retail+ii "
            "va giai nen vao thu muc data/")
    sheets = pd.read_excel(XLSX_PATH, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Description"] = df["Description"].astype("string")
    df.to_parquet(CACHE_PATH, index=False)
    return df


def is_non_product(code: pd.Series) -> pd.Series:
    """Ma phi/dieu chinh: nam trong danh sach, hoac khong chua chu so nao
    (san pham that luon co phan so, vd 85123A; ma phi kieu 'POST', 'DOT', 'BANK CHARGES')."""
    return code.isin(NON_PRODUCT_CODES) | ~code.str.contains(r"\d", regex=True)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """4 buoc lam sach bat buoc; tra ve (giao dich sach, nhat ky so dong loai moi buoc).
    Buoc 4 (san pham < 5 don) loc o cap san pham nen ghi nhan o build_product_features."""
    log = [("Ban dau", len(df), 0)]

    def step(name, mask_keep):
        nonlocal df
        before = len(df)
        df = df[mask_keep]
        log.append((name, len(df), before - len(df)))

    step("1. Loai hoa don huy (Invoice bat dau 'C')", ~df["Invoice"].str.startswith("C"))
    step("2. Loai Quantity <= 0 va Price <= 0", (df["Quantity"] > 0) & (df["Price"] > 0))
    step("3. Loai StockCode khong phai san pham", ~is_non_product(df["StockCode"]))
    step("   Loai dong trung lap hoan toan", ~df.duplicated())

    df = df.copy()
    df["ThanhTien"] = df["Quantity"] * df["Price"]
    return df, pd.DataFrame(log, columns=["buoc", "so_dong_con_lai", "so_dong_bi_loai"])


def build_product_features(tx: pd.DataFrame, min_orders: int = MIN_ORDERS) -> pd.DataFrame:
    """Tong hop giao dich -> 1 dong / san pham voi 8 dac trung hanh vi mua."""
    sp = tx.groupby("StockCode").agg(
        tong_so_luong=("Quantity", "sum"),
        tong_doanh_thu=("ThanhTien", "sum"),
        gia_trung_binh=("Price", "mean"),
        so_don_hang=("Invoice", "nunique"),
        so_khach_mua=("Customer ID", "nunique"),
        do_lech_sl=("Quantity", "std"),
    ).reset_index()

    # Mo ta: lay ten xuat hien nhieu nhat cua moi ma (1 ma co the co vai cach viet)
    desc = (tx.dropna(subset=["Description"])
              .groupby("StockCode")["Description"]
              .agg(lambda s: s.value_counts().index[0]))
    sp = sp.merge(desc.rename("Description"), on="StockCode", how="left")

    # Buoc 4: san pham ban qua it don -> khong du du lieu de mo ta hanh vi
    sp = sp[sp["so_don_hang"] >= min_orders].copy()

    sp["do_lech_sl"] = sp["do_lech_sl"].fillna(0)
    sp["sl_moi_don"] = sp["tong_so_luong"] / sp["so_don_hang"]
    # Khach vang lai (khong co Customer ID) khong duoc dem -> so_khach_mua co the = 0;
    # chan duoi 1 de tranh chia cho 0
    sp["ty_le_mua_lai"] = sp["so_don_hang"] / sp["so_khach_mua"].clip(lower=1)
    return sp.reset_index(drop=True)


def weekend_share(tx: pd.DataFrame) -> pd.Series:
    """Ty le doanh thu ban vao thu 6-7-CN cua moi san pham (dung de MO TA cum, khong dua vao K-Means)."""
    dow = tx["InvoiceDate"].dt.dayofweek
    rev_we = tx["ThanhTien"].where(dow >= 4, 0).groupby(tx["StockCode"]).sum()
    rev = tx.groupby("StockCode")["ThanhTien"].sum()
    return (rev_we / rev).rename("ty_le_dt_cuoi_tuan")


def load_products() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tien ich: (giao dich sach, bang san pham, nhat ky lam sach)."""
    tx, log = clean(load_raw())
    sp = build_product_features(tx)
    return tx, sp, log


def log1p_safe(X):
    return np.log1p(np.asarray(X, dtype=float))
