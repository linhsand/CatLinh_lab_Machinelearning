"""Tai, lam sach va tao dac trung cho bo Metro Interstate Traffic Volume (UCI) cho TT-27.

Bon van de du lieu (README de bai) va cach xu ly:
  1. Lo hong thoi gian: 52.551 gio tu 2012-10-02 den 2018-09-30 nhung chi 40.575 gio co du lieu.
     -> reindex theo gio day du. Lo ngan (<= 3 gio lien tiep) noi suy tuyen tinh + cot co `noi_suy`.
        Lo dai hon (136 lo, ke ca lo ~10 thang 2014-08 -> 2015-06) de NaN -> chuoi bi cat thanh nhieu doan,
        khong cua so nao duoc bac qua lo (sequences.py kiem tra).
  2. Ngoai lai phi ly: temp = 0 K (10 dong), rain_1h = 9.831 mm/gio (1 dong) -> NaN roi noi suy.
  3. Trung date_time (7.629 dong, cung traffic_volume, chi khac mo ta thoi tiet) -> gop 1 dong/gio,
     one-hot weather_main lay MAX (giu ca 2 dieu kien thoi tiet).
  4. `holiday` chi ghi o dong 0h cua ngay le -> lan ra ca 24 gio cua ngay do.
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
DATA_CSV = DATA_DIR / "Metro_Interstate_Traffic_Volume.csv.gz"
URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"

TARGET = "traffic_volume"
MAX_INTERP = 3                       # so gio thieu lien tiep toi da duoc noi suy
WEATHER = ["Clear", "Clouds", "Mist", "Rain", "Snow", "Drizzle", "Haze", "Thunderstorm", "Fog"]  # con lai -> Other
SPLIT = (0.70, 0.15, 0.15)


def download() -> None:
    if DATA_CSV.exists():
        return
    print(f"[data] Dang tai {URL} ...", flush=True)
    raw = urllib.request.urlopen(URL, timeout=120).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_CSV.write_bytes(z.read("Metro_Interstate_Traffic_Volume.csv.gz"))
    print(f"[data] Da luu {DATA_CSV}", flush=True)


def load_raw() -> pd.DataFrame:
    """48.204 dong goc. keep_default_na=False: chuoi 'None' o cot holiday la 'khong phai ngay le', khong phai NaN."""
    download()
    df = pd.read_csv(DATA_CSV, keep_default_na=False)
    df["date_time"] = pd.to_datetime(df["date_time"])
    return df


def _runs_of_nan(s: pd.Series) -> pd.Series:
    """Do dai cua doan NaN lien tiep ma moi phan tu thuoc vao (0 neu khong NaN)."""
    isna = s.isna()
    grp = (isna != isna.shift()).cumsum()
    return isna.groupby(grp).transform("sum").where(isna, 0).astype(int)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Xu ly 4 van de -> bang 1 dong / gio (ke ca gio thieu = NaN) + bang tom tat so lieu tung van de."""
    rep = {}
    df = df.copy()
    # (2) ngoai lai phi ly
    rep["temp = 0 K (dong)"] = int((df["temp"] == 0).sum())
    rep["rain_1h > 1000 mm (dong)"] = int((df["rain_1h"] > 1000).sum())
    df.loc[df["temp"] == 0, "temp"] = np.nan
    df.loc[df["rain_1h"] > 1000, "rain_1h"] = np.nan
    # (3) trung date_time: gop, one-hot thoi tiet lay max
    rep["dong trung date_time"] = int(df["date_time"].duplicated().sum())
    rep["gio co > 1 gia tri traffic_volume"] = int((df.groupby("date_time")[TARGET].nunique() > 1).sum())
    w = df["weather_main"].where(df["weather_main"].isin(WEATHER), "Other")
    oh = pd.get_dummies(w, prefix="w").astype(float).reindex(columns=[f"w_{c}" for c in WEATHER + ["Other"]],
                                                            fill_value=0.0)
    df = pd.concat([df, oh], axis=1)
    agg = {"holiday": "first", "temp": "mean", "rain_1h": "mean", "snow_1h": "mean", "clouds_all": "mean",
           TARGET: "first", **{c: "max" for c in oh.columns}}
    df = df.groupby("date_time").agg(agg)
    # (1) reindex theo gio day du
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", name="date_time")
    rep["gio trong khoang 2012-10 -> 2018-09"] = len(full)
    rep["gio co du lieu"] = len(df)
    df = df.reindex(full)
    run = _runs_of_nan(df[TARGET])
    rep["gio thieu"] = int((run > 0).sum())
    rep[f"gio thieu trong lo <= {MAX_INTERP} gio (noi suy)"] = int(((run > 0) & (run <= MAX_INTERP)).sum())
    rep[f"gio thieu trong lo > {MAX_INTERP} gio (de NaN)"] = int((run > MAX_INTERP).sum())
    rep[f"so lo > {MAX_INTERP} gio"] = int(((run > MAX_INTERP) & (run.shift(fill_value=0) == 0)).sum())
    df["noi_suy"] = ((run > 0) & (run <= MAX_INTERP)).astype(float)
    short = run <= MAX_INTERP                       # vi tri duoc phep dien (co du lieu hoac lo ngan)
    num = [TARGET, "temp", "rain_1h", "snow_1h", "clouds_all"] + list(oh.columns)
    filled = df[num].interpolate(method="linear", limit_area="inside")
    df[num] = filled.where(short)
    # (4) holiday: lan ra ca ngay. Lo dai van giu NaN o cac cot so; holiday la lich nen dien duoc cho moi gio.
    hol = df["holiday"].where(df["holiday"].notna() & (df["holiday"] != "None"))
    day_name = hol.groupby(df.index.date).transform(lambda s: s.dropna().iloc[0] if s.notna().any() else None)
    rep["dong ghi ngay le (goc)"] = int(hol.notna().sum())
    df["holiday"] = day_name.fillna("None")
    rep["gio thuoc ngay le (sau khi lan)"] = int((df["holiday"] != "None").sum())
    return df, pd.Series(rep, name="so_luong").to_frame()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dac trung theo gio. Lich (gio, thu, ngay le) biet truoc nen dien cho ca gio thieu."""
    out = pd.DataFrame(index=df.index)
    out[TARGET] = df[TARGET]
    out["temp_c"] = df["temp"] - 273.15
    out["rain_log"] = np.log1p(df["rain_1h"])
    out["snow"] = (df["snow_1h"] > 0).astype(float).where(df["snow_1h"].notna())
    out["clouds"] = df["clouds_all"] / 100
    for c in [c for c in df.columns if c.startswith("w_")]:
        out[c] = df[c]
    out["noi_suy"] = df["noi_suy"]
    out = out.join(calendar(df.index, df["holiday"]))
    return out


def calendar(idx: pd.DatetimeIndex, holiday: pd.Series) -> pd.DataFrame:
    """sin/cos cho gio (chu ky 24) va thu (chu ky 7): 23h va 0h gan nhau, CN va T2 gan nhau. + co ngay le, cuoi tuan."""
    h, d = idx.hour.to_numpy(), idx.dayofweek.to_numpy()
    return pd.DataFrame({"hour_sin": np.sin(2 * np.pi * h / 24), "hour_cos": np.cos(2 * np.pi * h / 24),
                         "dow_sin": np.sin(2 * np.pi * d / 7), "dow_cos": np.cos(2 * np.pi * d / 7),
                         "weekend": (d >= 5).astype(float), "holiday": (holiday.to_numpy() != "None").astype(float)},
                        index=idx)


CALENDAR_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend", "holiday"]


def split_times(feat: pd.DataFrame, frac=SPLIT) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Chia THEO THOI GIAN 70/15/15 tren cac gio CO nhan (khong shuffle). Tra ve {tap: (tu, den)}."""
    t = feat.index[feat[TARGET].notna()]
    n = len(t)
    i1, i2 = int(n * frac[0]), int(n * (frac[0] + frac[1]))
    return {"train": (t[0], t[i1 - 1]), "val": (t[i1], t[i2 - 1]), "test": (t[i2], t[-1])}
