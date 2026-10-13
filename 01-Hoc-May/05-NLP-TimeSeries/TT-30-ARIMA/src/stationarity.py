"""Nap du lieu + cac buoc "NHIN" va "LAM DUNG" cua Box-Jenkins cho TT-30 (AirPassengers).

* load():            144 thang (1949-01 -> 1960-12), so khach quoc te (nghin nguoi). Lan dau tai qua
                     statsmodels.datasets.get_rdataset (can mang) roi luu data/AirPassengers.csv.
* amplitude_table(): bien do mua vu moi nam (max - min) o thang goc va thang log -> co phai mo hinh NHAN TINH?
* decompose_compare(): seasonal_decompose additive vs multiplicative, do phan du con phu thuoc muc chuoi khong.
* adf_table():       ADF (+ KPSS doi chung) tren nhieu phien ban chuoi -> chon d, D.
* plot_acf_pacf():   ACF / PACF cua chuoi da dung -> de xuat (p, q), (P, Q).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = BASE_DIR / "data" / "AirPassengers.csv"


def load() -> pd.Series:
    if not DATA_CSV.exists():
        import statsmodels.api as sm
        df = sm.datasets.get_rdataset("AirPassengers").data
        DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_CSV, index=False)
    df = pd.read_csv(DATA_CSV)
    # cot `time` la nam thap phan (1949.083 = thang 2) -> dung chi muc thang ro rang
    return pd.Series(df["value"].to_numpy(dtype=float), name="so_khach",
                     index=pd.date_range("1949-01-01", periods=len(df), freq="MS"))


def amplitude_table(y: pd.Series) -> pd.DataFrame:
    g, lg = y.groupby(y.index.year), np.log(y).groupby(y.index.year)
    t = pd.DataFrame({"trung_binh_nam": g.mean(), "bien_do_goc": g.max() - g.min(),
                      "bien_do_tuong_doi": (g.max() - g.min()) / g.mean(), "bien_do_log": lg.max() - lg.min()})
    t.index.name = "nam"
    return t


def decompose_compare(y: pd.Series, out: Path) -> pd.DataFrame:
    """Phan ra 2 kieu. Mo hinh dung thi phan du KHONG con phu thuoc muc chuoi:
    do tuong quan giua do lech chuan phan du tung nam va trung binh nam (gan 0 = tot)."""
    rows, res = [], {}
    for kind in ["additive", "multiplicative"]:
        d = seasonal_decompose(y, model=kind, period=12)
        r = d.resid.dropna()
        if kind == "multiplicative":
            r = r - 1                                   # phan du nhan tinh dao dong quanh 1
        sd = r.groupby(r.index.year).std()
        lvl = y.groupby(y.index.year).mean().loc[sd.index]
        rows.append({"kieu": kind, "tuong_quan_do_lech_phan_du_vs_muc": float(np.corrcoef(sd, lvl)[0, 1]),
                     "do_lech_phan_du_nam_dau": float(sd.iloc[0]), "do_lech_phan_du_nam_cuoi": float(sd.iloc[-1])})
        res[kind] = d
    fig, axes = plt.subplots(4, 2, figsize=(15, 10), sharex=True)
    for c, kind in enumerate(["additive", "multiplicative"]):
        d = res[kind]
        for r_i, (part, title) in enumerate([(d.observed, "chuoi goc"), (d.trend, "trend"), (d.seasonal, "mua vu"),
                                             (d.resid, "phan du")]):
            ax = axes[r_i, c]
            if part is d.resid:
                ax.scatter(part.index, part, s=8, c="#C44E52")
                ax.axhline(0 if kind == "additive" else 1, c="gray", lw=0.8)
            else:
                ax.plot(part.index, part, lw=1.2)
            ax.set_title(f"{kind} - {title}", fontsize=10)
    save(fig, out)
    return pd.DataFrame(rows)


def adf_table(y: pd.Series) -> pd.DataFrame:
    """ADF: H0 = co nghiem don vi (KHONG dung) -> p < 0,05 moi ket luan dung.
    KPSS (doi chung): H0 = DUNG -> p < 0,05 la khong dung. statsmodels gioi han p cua KPSS trong [0,01; 0,1]."""
    ly = np.log(y)
    versions = {"goc": y, "log": ly, "log + diff(1)": ly.diff(), "log + diff(12)": ly.diff(12),
                "log + diff(1) + diff(12)": ly.diff().diff(12), "goc + diff(1) + diff(12)": y.diff().diff(12)}
    rows = []
    for name, s in versions.items():
        s = s.dropna()
        a = adfuller(s, autolag="AIC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = kpss(s, regression="c", nlags="auto")
        rows.append({"phien_ban": name, "so_diem": len(s), "ADF_p": a[1], "ADF_ket_luan": "DUNG" if a[1] < 0.05 else "KHONG DUNG",
                     "so_lag_ADF": a[2], "KPSS_p": k[1], "KPSS_ket_luan": "DUNG" if k[1] >= 0.05 else "KHONG DUNG"})
    return pd.DataFrame(rows)


def plot_acf_pacf(s: pd.Series, out: Path, lags: int = 36) -> pd.DataFrame:
    from statsmodels.tsa.stattools import acf, pacf
    s = s.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    plot_acf(s, lags=lags, ax=axes[0], title="ACF - log + diff(1) + diff(12)")
    plot_pacf(s, lags=lags, ax=axes[1], method="ywm", title="PACF - log + diff(1) + diff(12)")
    for ax in axes:
        for k in (12, 24, 36):
            ax.axvline(k, ls=":", c="gray", lw=0.8)
    save(fig, out)
    bound = 1.96 / np.sqrt(len(s))
    a, p = acf(s, nlags=lags)[1:], pacf(s, nlags=lags, method="ywm")[1:]
    t = pd.DataFrame({"lag": np.arange(1, lags + 1), "ACF": a, "PACF": p})
    t["ACF_vuot_nguong"] = t["ACF"].abs() > bound
    t["PACF_vuot_nguong"] = t["PACF"].abs() > bound
    t.attrs["nguong_95"] = bound
    return t


def save(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
