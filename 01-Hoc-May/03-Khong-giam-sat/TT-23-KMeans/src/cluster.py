"""Gom cum san pham bang K-Means va xuat file ban giao cho bo phan trung bay (TT-23).

Chay doc lap: python src/cluster.py
Cau hinh lay tu notebook (notebooks/kmeans_san_pham.ipynb):
  8 dac trung -> log1p -> StandardScaler -> PCA giu 90% phuong sai (3 thanh phan)
  -> KMeans(K=5, n_init=10, random_state=42).
Ghi:
  models/kmeans_pipeline.joblib
  outputs/san_pham_theo_cum.csv   (StockCode | Description | cum | ten_cum | ...)
  outputs/mo_ta_cum.csv           (bang mo ta + de xuat trung bay)
  reports/cluster_metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from features import FEATURES, load_products, weekend_share  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
RANDOM_STATE = 42
K = 5
PCA_VARIANCE = 0.90

# Ten + mo ta + de xuat trung bay cho tung "kieu" cum. Gan kieu cho cum dua tren
# ho so trung vi (assign_names) chu khong dua tren so thu tu cum - vi so thu tu
# do K-Means dat ngau nhien, doi seed la doi.
CLUSTER_CATALOG = {
    "chu_luc": (
        "Hàng chủ lực bán chạy",
        "Giá rẻ, rất nhiều đơn và nhiều khách mua; mang về phần lớn doanh thu.",
        "Kệ đầu dãy / lối đi chính, tầm mắt; luôn đủ tồn kho, không để trống kệ."),
    "si_lo": (
        "Hàng mua sỉ theo lô",
        "Đơn giá thấp nhất nhưng mỗi đơn mua số lượng lớn, số lượng dao động mạnh giữa các đơn.",
        "Kệ dưới / pallet cuối dãy, đóng gói theo thùng-lốc; gắn biển giá theo số lượng."),
    "ban_cham": (
        "Hàng đuôi dài bán chậm",
        "Ít đơn nhất (trung vị ~21 đơn/2 năm), ít khách, mỗi đơn 2–3 cái; giá trải rộng, gồm cả nội thất giá cao.",
        "Không chiếm kệ chính: gom vào khu 'mẫu trưng bày' hoặc chỉ bán online/đặt trước; rà soát loại bớt mã."),
    "khach_quen": (
        "Hàng ngách khách quen",
        "Rất ít khách mua nhưng mỗi khách quay lại mua nhiều lần (tỷ lệ mua lại cao nhất).",
        "Kệ cuối cửa hàng / khu chuyên biệt; ưu tiên gợi ý cá nhân hoá cho đúng nhóm khách cũ."),
    "ban_deu": (
        "Hàng giá khá bán đều",
        "Giá cao hơn hàng chủ lực (trung vị ~4), lượng đơn ổn định, mua lẻ vài cái mỗi đơn; ~1/5 doanh thu.",
        "Kệ giữa dãy theo chủ đề (bếp, quà tặng, trang trí nhà), tầm tay; xoay vòng theo mùa."),
}


def build_preprocessor() -> Pipeline:
    """log1p (giam lech phai) -> chuan hoa -> PCA bo cac huong nhieu/trung lap."""
    return Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)),
    ])


def build_pipeline(k: int = K, n_init: int = 10, random_state: int = RANDOM_STATE) -> Pipeline:
    pipe = build_preprocessor()
    pipe.steps.append(("km", KMeans(n_clusters=k, n_init=n_init, random_state=random_state)))
    return pipe


def profile_clusters(sp: pd.DataFrame, label_col: str = "cum") -> pd.DataFrame:
    """Bang mo ta cum: so ma hang, % doanh thu, gia TB, tan suat mua (trung vi de khong bi ca voi keo)."""
    g = sp.groupby(label_col).agg(
        so_ma_hang=("StockCode", "size"),
        doanh_thu=("tong_doanh_thu", "sum"),
        gia_tb_median=("gia_trung_binh", "median"),
        so_don_median=("so_don_hang", "median"),
        so_khach_median=("so_khach_mua", "median"),
        sl_moi_don_median=("sl_moi_don", "median"),
        do_lech_sl_median=("do_lech_sl", "median"),
        ty_le_mua_lai_median=("ty_le_mua_lai", "median"),
        dt_cuoi_tuan_median=("ty_le_dt_cuoi_tuan", "median"),
    )
    g.insert(1, "pct_ma_hang", g["so_ma_hang"] / g["so_ma_hang"].sum() * 100)
    g.insert(3, "pct_doanh_thu", g["doanh_thu"] / g["doanh_thu"].sum() * 100)
    return g


def assign_names(prof: pd.DataFrame) -> dict[int, str]:
    """Gan kieu cum theo luat tham lam tren ho so trung vi (moi kieu 1 cum):
    nhieu don nhat -> chu luc; mua lai cao nhat -> khach quen; SL/don cao nhat -> si lo;
    it don nhat trong so con lai -> duoi dai ban cham; cum con lai -> ban deu."""
    left = list(prof.index)
    keys = {}

    def take(key, col, largest=True):
        s = prof.loc[left, col]
        c = s.idxmax() if largest else s.idxmin()
        keys[c] = key
        left.remove(c)

    take("chu_luc", "so_don_median")
    take("khach_quen", "ty_le_mua_lai_median")
    take("si_lo", "sl_moi_don_median")
    take("ban_cham", "so_don_median", largest=False)
    for c in left:
        keys[c] = "ban_deu"
    return keys


def fit_and_export(k: int = K) -> dict:
    tx, sp, clean_log = load_products()
    sp = sp.merge(weekend_share(tx), left_on="StockCode", right_index=True, how="left")

    pipe = build_pipeline(k)
    sp["cum"] = pipe.fit_predict(sp[FEATURES])
    Z = pipe[:-1].transform(sp[FEATURES])
    sil = float(silhouette_score(Z, sp["cum"]))

    prof = profile_clusters(sp)
    keys = assign_names(prof)
    prof["ten_cum"] = [CLUSTER_CATALOG[keys[c]][0] for c in prof.index]
    prof["mo_ta"] = [CLUSTER_CATALOG[keys[c]][1] for c in prof.index]
    prof["de_xuat_trung_bay"] = [CLUSTER_CATALOG[keys[c]][2] for c in prof.index]
    sp["ten_cum"] = sp["cum"].map(prof["ten_cum"])
    sp["de_xuat_trung_bay"] = sp["cum"].map(prof["de_xuat_trung_bay"])

    for d in (MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "kmeans_pipeline.joblib")

    handoff = sp[["StockCode", "Description", "cum", "ten_cum", "de_xuat_trung_bay",
                  "tong_doanh_thu", "so_don_hang", "gia_trung_binh"]]
    handoff = handoff.sort_values(["cum", "tong_doanh_thu"], ascending=[True, False])
    handoff.to_csv(OUTPUTS_DIR / "san_pham_theo_cum.csv", index=False, encoding="utf-8-sig")
    prof.round(3).to_csv(OUTPUTS_DIR / "mo_ta_cum.csv", encoding="utf-8-sig")

    metrics = {
        "so_giao_dich_sach": int(len(tx)),
        "so_san_pham": int(len(sp)),
        "K": k,
        "so_thanh_phan_pca": int(pipe["pca"].n_components_),
        "phuong_sai_giu_lai": float(pipe["pca"].explained_variance_ratio_.sum()),
        "inertia": float(pipe["km"].inertia_),
        "silhouette": sil,
        "kich_thuoc_cum": {prof.loc[c, "ten_cum"]: int(prof.loc[c, "so_ma_hang"]) for c in prof.index},
    }
    (REPORTS_DIR / "cluster_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(fit_and_export(), ensure_ascii=False, indent=2))
