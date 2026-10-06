"""
TT-16 - Decision Tree Regressor: Bang gia cuoc taxi NYC cho tong dai dieu xe.

Pipeline day du (theo 12 buoc trong README.md cap de):
 1. Tai 1 thang du lieu NYC Yellow Taxi, lay mau 200.000 dong
 2. Lam sach theo 4 buoc (lay mau, loai chuyen vo ly, tranh ro ri, tao dac trung thoi gian),
    ghi lai so dong bi loai moi buoc
 3. Tao dac trung thoi gian (gio, thu, cao diem)
 4. EDA: scatter quang duong vs cuoc, gia trung binh theo gio trong ngay
 5. Baseline: DummyRegressor(mean) va cong thuc tuyen tinh thu cong -> MAE bao nhieu?
 6. Cay KHONG gioi han do sau -> so MAE train vs test -> CHUNG MINH overfit
 7. Ve MAE train/CV theo max_depth = 1..20 -> chon diem toi uu bang 5-fold CV TREN TRAIN
    (test KHONG duoc dung de chon - chi bao cao 1 lan o buoc 12)
 8. Ve ham du doan theo quang duong -> NHIN THAY hinh BAC THANG
 9. Cay max_depth=5 -> xuat export_text va ve cay
10. Chuyen cay thanh BANG TRA CUOC (CSV) cho tong dai
11. Kiem tra: bao nhieu % chuyen co sai so trong +-15%?
12. So sanh voi Random Forest Regressor (max_depth chon bang OOB) va Linear Regression

Mo rong (section 8 cua README):
13. criterion='absolute_error' -> it nhay outlier hon, nhung cham hon nhieu
14. Chung minh tinh KHONG ON DINH: train 10 cay voi 10 mau con khac nhau
15. Du doan KHOANG GIA bang GradientBoostingRegressor(loss='quantile')

+ Giai thich vi sao cay KHONG NGOAI SUY duoc (chuyen 200 km se ra sao?)

Bo sung (dien giai danh doi):
16. Duong bien so la <-> MAE-CV (min_samples_leaf=500, depth 2..12): them 1 tang cay mua duoc bao nhieu?
17. Cay sau / RF thang cay tong dai O DAU: MAE va % trong +-15% theo dai quang duong (test)

Chay: python src/train.py
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
OUTPUTS_DIR = ROOT / "outputs"
for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR):
    d.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
DATA_PARQUET = DATA_DIR / "yellow_tripdata_2023-01.parquet"
SAMPLE_SIZE = 200_000

RAW_COLS = ["tpep_pickup_datetime", "passenger_count", "trip_distance",
            "PULocationID", "DOLocationID", "payment_type", "fare_amount"]
FEATURES = ["trip_distance", "passenger_count", "PULocationID", "DOLocationID",
            "payment_type", "hour", "dow", "is_peak"]
TARGET = "fare_amount"

MAX_DEPTHS = list(range(1, 21))
DISPATCH_DEPTH = 5
DISPATCH_MIN_LEAF = 500
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
FRONTIER_DEPTHS = list(range(2, 13))
RF_DEPTHS = [8, 12, 16, None]
DIST_BANDS = [0, 1, 2, 5, 10, 20, 100]  # miles


def log(msg: str) -> None:
    print(f"[TT-16] {msg}")


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pct_within_band(y_true, y_pred, band: float = 0.15) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    err_pct = np.abs(y_pred - y_true) / y_true
    return float((err_pct <= band).mean() * 100)


def evaluate(y_true, y_pred) -> dict:
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred),
            "R2": float(r2_score(y_true, y_pred)), "trong_15%_%": pct_within_band(y_true, y_pred)}


CV_SCORING = {"mae": "neg_mean_absolute_error", "trong15": make_scorer(pct_within_band)}


# ----------------------------------------------------------------------------
# 1. Tai du lieu NYC Yellow Taxi (1 thang), lay mau 200.000 dong
# ----------------------------------------------------------------------------
def download_data() -> Path:
    if DATA_PARQUET.exists():
        return DATA_PARQUET
    log(f"  Dang tai du lieu tu NYC TLC: {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, DATA_PARQUET)
    log(f"  Da luu du lieu vao {DATA_PARQUET.relative_to(ROOT)}")
    return DATA_PARQUET


def load_and_sample() -> pd.DataFrame:
    path = download_data()
    df_full = pd.read_parquet(path, columns=RAW_COLS)
    n_full = len(df_full)
    df = df_full.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    log(f"  Du lieu goc: {n_full:,} dong -> lay mau ngau nhien {len(df):,} dong (random_state={RANDOM_STATE})")
    return df


# ----------------------------------------------------------------------------
# 2. Lam sach: loai chuyen vo ly, ghi lai so dong bi loai moi buoc
# ----------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    log_steps = []
    n0 = len(df)
    log_steps.append({"buoc": "0. Sau khi lay mau", "so_dong_con_lai": n0, "so_dong_bi_loai": 0})

    mask_fare = df["fare_amount"] > 0
    n_removed = int((~mask_fare).sum())
    df = df[mask_fare]
    log_steps.append({"buoc": "1. Loai fare_amount <= 0 (huy/hoan tien)",
                       "so_dong_con_lai": len(df), "so_dong_bi_loai": n_removed})

    mask_dist = (df["trip_distance"] > 0) & (df["trip_distance"] <= 100)
    n_removed = int((~mask_dist).sum())
    df = df[mask_dist]
    log_steps.append({"buoc": "2. Loai trip_distance <= 0 hoac > 100 miles",
                       "so_dong_con_lai": len(df), "so_dong_bi_loai": n_removed})

    mask_pax = df["passenger_count"].notna() & (df["passenger_count"] > 0)
    n_removed = int((~mask_pax).sum())
    df = df[mask_pax]
    log_steps.append({"buoc": "3. Loai passenger_count == 0 hoac thieu",
                       "so_dong_con_lai": len(df), "so_dong_bi_loai": n_removed})

    mask_date = (df["tpep_pickup_datetime"] >= "2023-01-01") & (df["tpep_pickup_datetime"] < "2023-02-01")
    n_removed = int((~mask_date).sum())
    df = df[mask_date]
    log_steps.append({"buoc": "4. Loai pickup ngoai thang 1/2023 (loi nhap lieu)",
                       "so_dong_con_lai": len(df), "so_dong_bi_loai": n_removed})

    df = df.reset_index(drop=True)
    log(f"  Con lai {len(df):,} / {n0:,} dong sau khi lam sach ({len(df) / n0 * 100:.1f}%)")
    for s in log_steps[1:]:
        log(f"    {s['buoc']}: loai {s['so_dong_bi_loai']:,} dong -> con {s['so_dong_con_lai']:,}")
    return df, log_steps


# ----------------------------------------------------------------------------
# 3. Tao dac trung thoi gian (gio, thu, cao diem)
# ----------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    df["dow"] = df["tpep_pickup_datetime"].dt.dayofweek  # 0=Thu Hai ... 6=Chu Nhat
    df["is_peak"] = (
        (df["dow"] < 5) & (((df["hour"] >= 7) & (df["hour"] <= 9)) | ((df["hour"] >= 16) & (df["hour"] <= 19)))
    ).astype(int)
    log(f"  Da tao hour, dow, is_peak. Ty le chuyen gio cao diem: {df['is_peak'].mean() * 100:.1f}%")
    return df


# ----------------------------------------------------------------------------
# 4. EDA
# ----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    sample_plot = df.sample(n=min(20_000, len(df)), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sample_plot["trip_distance"], sample_plot[TARGET], s=5, alpha=0.15, color="#2980b9")
    ax.set_xlabel("Quang duong (miles)"); ax.set_ylabel("Cuoc (USD)")
    ax.set_title("EDA: quang duong vs cuoc")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_scatter_distance_fare.png", dpi=130)
    plt.close(fig)

    by_hour = df.groupby("hour")[TARGET].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(by_hour.index, by_hour.values, "o-", color="#c0392b")
    ax.set_xlabel("Gio trong ngay"); ax.set_ylabel("Cuoc trung binh (USD)")
    ax.set_title("Cuoc trung binh theo gio trong ngay")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_avg_fare_by_hour.png", dpi=130)
    plt.close(fig)

    log(f"  Cuoc TB thap nhat luc {by_hour.idxmin()}h (${by_hour.min():.2f}), "
        f"cao nhat luc {by_hour.idxmax()}h (${by_hour.max():.2f})")
    log("  Da luu eda_scatter_distance_fare.png, eda_avg_fare_by_hour.png")
    return by_hour


# ----------------------------------------------------------------------------
# 5. Baseline: Dummy(mean) va cong thuc tuyen tinh thu cong
# ----------------------------------------------------------------------------
def run_baselines(X_train, y_train, X_test, y_test) -> dict:
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    dummy_metrics = evaluate(y_test, dummy.predict(X_test))
    log(f"  Dummy (du doan = trung binh ${y_train.mean():.2f}): "
        f"MAE=${dummy_metrics['MAE']:.2f}  trong+-15%={dummy_metrics['trong_15%_%']:.1f}%")

    manual_pred = 3.00 + 3.50 * X_test["trip_distance"]
    manual_metrics = evaluate(y_test, manual_pred)
    log(f"  Cong thuc thu cong (fare = $3.00 + $3.50 x km): "
        f"MAE=${manual_metrics['MAE']:.2f}  trong+-15%={manual_metrics['trong_15%_%']:.1f}%")
    return {"Dummy": dummy_metrics, "CongThucThuCong": manual_metrics}


# ----------------------------------------------------------------------------
# 6. Cay KHONG gioi han do sau -> chung minh overfit
# ----------------------------------------------------------------------------
def demo_overfit(X_train, y_train, X_test, y_test) -> dict:
    tree = DecisionTreeRegressor(random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    train_mae = mae(y_train, tree.predict(X_train))
    test_mae = mae(y_test, tree.predict(X_test))
    log(f"  Cay KHONG gioi han do sau: {tree.get_n_leaves():,} la, do sau={tree.get_depth()}")
    log(f"    MAE train=${train_mae:.4f}  MAE test=${test_mae:.4f}  "
        f"(chenh lech {test_mae / max(train_mae, 1e-9):.1f}x -> OVERFIT ro ret)")
    return {"n_leaves": int(tree.get_n_leaves()), "depth": int(tree.get_depth()),
            "MAE_train": train_mae, "MAE_test": test_mae}


# ----------------------------------------------------------------------------
# 7. MAE train/CV theo max_depth = 1..20 - chon depth bang 5-fold CV TREN TRAIN
# ----------------------------------------------------------------------------
def run_depth_sweep(X_train, y_train) -> tuple[pd.DataFrame, int]:
    rows = []
    for d in MAX_DEPTHS:
        cv = cross_validate(DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE), X_train, y_train,
                            cv=CV, scoring=CV_SCORING, return_train_score=True, n_jobs=-1)
        n_leaves = DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE).fit(X_train, y_train).get_n_leaves()
        rows.append({"max_depth": d, "MAE_train": -cv["train_mae"].mean(),
                     "MAE_cv": -cv["test_mae"].mean(), "std_MAE_cv": cv["test_mae"].std(),
                     "trong_15%_cv_%": cv["test_trong15"].mean(), "so_la": int(n_leaves)})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "mae_theo_depth.csv", index=False)
    best_depth = int(df.loc[df["MAE_cv"].idxmin(), "max_depth"])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df["max_depth"], df["MAE_train"], "o-", color="#2980b9", label="MAE train")
    ax.errorbar(df["max_depth"], df["MAE_cv"], yerr=df["std_MAE_cv"], fmt="o-", color="#c0392b", capsize=3,
                label="MAE validation (5-fold CV tren train)")
    ax.axvline(best_depth, color="gray", ls="--", label=f"toi uu theo CV = {best_depth}")
    ax.axvline(DISPATCH_DEPTH, color="green", ls=":", label=f"depth dung cho tong dai = {DISPATCH_DEPTH}")
    ax.set_xlabel("max_depth"); ax.set_ylabel("MAE (USD)")
    ax.set_title("MAE train/CV theo do sau cay (test KHONG dung de chon)")
    ax.set_xticks(MAX_DEPTHS[::2])
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "mae_theo_depth.png", dpi=130)
    plt.close(fig)

    best = df.loc[df["max_depth"] == best_depth].iloc[0]
    log(f"  MAE-CV thap nhat tai max_depth={best_depth} (MAE_cv=${best['MAE_cv']:.4f} +- {best['std_MAE_cv']:.3f}, "
        f"{int(best['so_la']):,} la)")
    log(f"  Tu do sau ~{best_depth} tro di, MAE train tiep tuc giam nhung MAE-CV tang lai -> overfit")
    log("  Da luu mae_theo_depth.csv/.png")
    return df, best_depth


# ----------------------------------------------------------------------------
# 8. Ham du doan theo quang duong -> hinh BAC THANG
# ----------------------------------------------------------------------------
def plot_staircase(dispatch_tree: DecisionTreeRegressor, df: pd.DataFrame) -> None:
    mode_row = {
        "passenger_count": int(df["passenger_count"].mode()[0]),
        "PULocationID": int(df["PULocationID"].mode()[0]),
        "DOLocationID": int(df["DOLocationID"].mode()[0]),
        "payment_type": int(df["payment_type"].mode()[0]),
        "hour": int(df["hour"].median()),
        "dow": int(df["dow"].median()),
        "is_peak": 0,
    }
    dist_grid = np.linspace(0.1, 20, 400)
    grid_df = pd.DataFrame([{**mode_row, "trip_distance": d} for d in dist_grid])[FEATURES]
    pred_grid = dispatch_tree.predict(grid_df)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(dist_grid, pred_grid, color="#8e44ad", lw=2)
    ax.set_xlabel("Quang duong (miles)"); ax.set_ylabel("Gia du doan (USD)")
    ax.set_title(f"Ham du doan theo quang duong - cay depth={DISPATCH_DEPTH} (hinh BAC THANG)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "ham_bac_thang.png", dpi=130)
    plt.close(fig)

    n_distinct = len(np.unique(pred_grid))
    log(f"  Ham du doan chi co {n_distinct} muc gia khac nhau tren {len(dist_grid)} diem quet quang duong")
    log("  Da luu ham_bac_thang.png")


# ----------------------------------------------------------------------------
# 9. Cay max_depth=5 -> export_text + plot_tree
# ----------------------------------------------------------------------------
def build_dispatch_tree(X_train, y_train, X_test, y_test) -> tuple[DecisionTreeRegressor, dict]:
    tree = DecisionTreeRegressor(max_depth=DISPATCH_DEPTH, min_samples_leaf=DISPATCH_MIN_LEAF,
                                  criterion="squared_error", random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    metrics = evaluate(y_test, tree.predict(X_test))
    log(f"  Cay tong dai (depth={DISPATCH_DEPTH}, min_samples_leaf={DISPATCH_MIN_LEAF}): "
        f"{tree.get_n_leaves()} la  MAE=${metrics['MAE']:.4f}  trong+-15%={metrics['trong_15%_%']:.1f}%")

    text = export_text(tree, feature_names=FEATURES)
    (REPORTS_DIR / "cay_quyet_dinh_text.txt").write_text(text, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(26, 12))
    plot_tree(tree, feature_names=FEATURES, filled=True, rounded=True, fontsize=8,
              precision=1, ax=ax)
    ax.set_title(f"Cay quyet dinh - Bang gia cuoc (depth={DISPATCH_DEPTH}, min_samples_leaf={DISPATCH_MIN_LEAF})")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "cay_quyet_dinh.png", dpi=130)
    plt.close(fig)

    log("  Da luu cay_quyet_dinh_text.txt, cay_quyet_dinh.png")
    return tree, metrics


# ----------------------------------------------------------------------------
# 10. Chuyen cay thanh BANG TRA CUOC (CSV) cho tong dai
# ----------------------------------------------------------------------------
def tree_to_price_table(tree: DecisionTreeRegressor, feature_names: list[str]) -> pd.DataFrame:
    t = tree.tree_
    rows = []

    def recurse(node: int, conditions: list[str]) -> None:
        if t.children_left[node] == t.children_right[node] == -1:
            rows.append({
                "leaf_id": node,
                "dieu_kien": " VA ".join(conditions) if conditions else "(goc)",
                "gia_du_doan_usd": round(float(t.value[node][0][0]), 2),
                "so_chuyen_thuc_te": int(t.n_node_samples[node]),
            })
            return
        feat = feature_names[t.feature[node]]
        thr = t.threshold[node]
        recurse(t.children_left[node], conditions + [f"{feat} <= {thr:.2f}"])
        recurse(t.children_right[node], conditions + [f"{feat} > {thr:.2f}"])

    recurse(0, [])
    table = pd.DataFrame(rows).sort_values("gia_du_doan_usd").reset_index(drop=True)
    table.to_csv(OUTPUTS_DIR / "bang_tra_cuoc.csv", index=False, encoding="utf-8-sig")
    log(f"  Bang tra cuoc: {len(table)} muc gia (= so la cua cay), "
        f"tu ${table['gia_du_doan_usd'].min():.2f} den ${table['gia_du_doan_usd'].max():.2f}")
    log(f"  So chuyen thuc te it nhat trong 1 la: {table['so_chuyen_thuc_te'].min()} "
        f"(>= {DISPATCH_MIN_LEAF} theo yeu cau min_samples_leaf)")
    log("  Da luu outputs/bang_tra_cuoc.csv")
    return table


# ----------------------------------------------------------------------------
# 12. So sanh voi Random Forest va Linear Regression
#     Moi sieu tham so deu chon TREN TRAIN (cay: CV o buoc 7, RF: OOB); test chi bao cao 1 lan
# ----------------------------------------------------------------------------
def tune_random_forest(X_train, y_train) -> tuple[RandomForestRegressor, pd.DataFrame]:
    rows, models = [], {}
    for d in RF_DEPTHS:
        rf = RandomForestRegressor(n_estimators=100, max_depth=d, oob_score=True,
                                   random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
        oob = rf.oob_prediction_
        rows.append({"max_depth": "None" if d is None else d, "MAE_oob": mae(y_train, oob),
                     "trong_15%_oob_%": pct_within_band(y_train, oob),
                     "tong_so_la": int(sum(e.get_n_leaves() for e in rf.estimators_))})
        models[rows[-1]["max_depth"]] = rf
        log(f"    RF max_depth={rows[-1]['max_depth']!s:5s} MAE_oob=${rows[-1]['MAE_oob']:.4f}  "
            f"tong so la={rows[-1]['tong_so_la']:,}")
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "rf_chon_depth_oob.csv", index=False)
    best = df.loc[df["MAE_oob"].idxmin(), "max_depth"]
    log(f"  RF chon max_depth={best} theo MAE OOB (tren train). Da luu rf_chon_depth_oob.csv")
    return models[best], df


def compare_models(X_train, y_train, X_test, y_test, dispatch_tree, best_depth: int, rf, lr_pipe) -> tuple[pd.DataFrame, dict]:
    best_tree = DecisionTreeRegressor(max_depth=best_depth, random_state=RANDOM_STATE).fit(X_train, y_train)
    models = {
        f"Cay tong dai (depth={DISPATCH_DEPTH}, leaf>={DISPATCH_MIN_LEAF})": (dispatch_tree, dispatch_tree.get_n_leaves()),
        f"Cay sau (depth={best_depth}, chon bang CV)": (best_tree, best_tree.get_n_leaves()),
        f"Random Forest ({len(rf.estimators_)} cay, max_depth={rf.max_depth}, chon bang OOB)":
            (rf, sum(e.get_n_leaves() for e in rf.estimators_)),
        "Linear Regression": (lr_pipe, None),
    }
    rows, preds = [], {}
    for name, (m, n_leaves) in models.items():
        preds[name] = m.predict(X_test)
        rows.append({"model": name, **evaluate(y_test, preds[name]),
                     "so_la_hoac_quy_tac": None if n_leaves is None else int(n_leaves)})
    df = pd.DataFrame(rows)
    df["so_la_hoac_quy_tac"] = df["so_la_hoac_quy_tac"].astype("Int64")
    df.to_csv(REPORTS_DIR / "so_sanh_models.csv", index=False)
    for r in rows:
        log(f"  {r['model']:62s} MAE=${r['MAE']:.4f}  RMSE=${r['RMSE']:.4f}  "
            f"R2={r['R2']:.4f}  trong+-15%={r['trong_15%_%']:.1f}%")
    log("  Da luu so_sanh_models.csv")
    return df, preds


# ----------------------------------------------------------------------------
# 16. Duong bien so la <-> MAE-CV: them 1 tang cay mua duoc bao nhieu? (chi dung TRAIN)
# ----------------------------------------------------------------------------
def run_leaves_frontier(X_train, y_train, depth_sweep_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in FRONTIER_DEPTHS:
        est = DecisionTreeRegressor(max_depth=d, min_samples_leaf=DISPATCH_MIN_LEAF, random_state=RANDOM_STATE)
        cv = cross_validate(est, X_train, y_train, cv=CV, scoring=CV_SCORING, n_jobs=-1)
        rows.append({"max_depth": d, "so_la": int(est.fit(X_train, y_train).get_n_leaves()),
                     "MAE_cv": -cv["test_mae"].mean(), "trong_15%_cv_%": cv["test_trong15"].mean()})
    df = pd.DataFrame(rows)
    df["MAE_giam_so_voi_depth_truoc"] = -df["MAE_cv"].diff()
    df["MAE_giam_moi_la_them_cent"] = df["MAE_giam_so_voi_depth_truoc"] / df["so_la"].diff() * 100
    df.to_csv(REPORTS_DIR / "duong_bien_so_la_mae.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df["so_la"], df["MAE_cv"], "o-", color="#27ae60", label=f"min_samples_leaf={DISPATCH_MIN_LEAF}")
    for r in df[df["max_depth"].isin([2, 3, 4, 5, 6, 8, 12])].itertuples():
        ax.annotate(f"d={r.max_depth}", (r.so_la, r.MAE_cv), textcoords="offset points", xytext=(4, 4), fontsize=8)
    free = depth_sweep_df[depth_sweep_df["max_depth"] <= 14]
    ax.plot(free["so_la"], free["MAE_cv"], "s--", color="#7f8c8d", alpha=0.7, label="khong gioi han la (buoc 7)")
    ax.axvspan(1, 40, color="#f1c40f", alpha=0.15, label="tra tay duoc (<= ~1 trang A4)")
    ax.set_xscale("log")
    ax.set_xlabel("So la = so dong bang tra (log)"); ax.set_ylabel("MAE 5-fold CV tren train (USD)")
    ax.set_title("Danh doi: kich thuoc bang tra <-> sai so")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "duong_bien_so_la_mae.png", dpi=130)
    plt.close(fig)
    for r in df.itertuples():
        log(f"    depth={r.max_depth:2d}  {r.so_la:4d} la  MAE_cv=${r.MAE_cv:.3f}  trong+-15%={r[4]:.1f}%")
    log("  Da luu duong_bien_so_la_mae.csv/.png")
    return df


# ----------------------------------------------------------------------------
# 17. Cay sau / RF thang cay tong dai O DAU? (test, theo dai quang duong)
# ----------------------------------------------------------------------------
def analyze_gap_by_distance(X_test, y_test, preds: dict) -> pd.DataFrame:
    y = np.asarray(y_test)
    band = pd.cut(X_test["trip_distance"].values, DIST_BANDS)
    names = list(preds)
    rows = []
    for b in band.categories:
        m = np.asarray(band == b)
        row = {"dai_mi": f"{b.left:g}-{b.right:g}", "so_chuyen": int(m.sum()), "ty_trong_%": m.mean() * 100,
               "cuoc_TB": float(y[m].mean())}
        for k in names:
            row[f"MAE|{k}"] = mae(y[m], preds[k][m])
            row[f"trong15|{k}"] = pct_within_band(y[m], preds[k][m])
        rows.append(row)
    df = pd.DataFrame(rows)
    # Phan chenh MAE (cay tong dai - RF) dong gop boi tung dai, tinh tren toan test
    disp, rf_name = names[0], names[2]
    df["dong_gop_chenh_MAE_vs_RF_%"] = ((df[f"MAE|{disp}"] - df[f"MAE|{rf_name}"]) * df["so_chuyen"])
    df["dong_gop_chenh_MAE_vs_RF_%"] = df["dong_gop_chenh_MAE_vs_RF_%"] / df["dong_gop_chenh_MAE_vs_RF_%"].sum() * 100
    df.to_csv(REPORTS_DIR / "chenh_lech_theo_quang_duong.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(df))
    show = names[:3]
    width = 0.8 / len(show)
    colors = ["#27ae60", "#7f8c8d", "#2980b9"]
    labels = ["Cay tong dai d=5", "Cay sau (CV)", "Random Forest"]
    for i, k in enumerate(show):
        off = (i - (len(show) - 1) / 2) * width
        axes[0].bar(x + off, df[f"MAE|{k}"], width, color=colors[i], label=labels[i])
        axes[1].bar(x + off, df[f"trong15|{k}"], width, color=colors[i], label=labels[i])
    for ax, yl in zip(axes, ["MAE test (USD)", "% chuyen trong +-15% (test)"]):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r.dai_mi}\n({r[3]:.0f}%)" for r in df.itertuples()])
        ax.set_xlabel("Dai quang duong (miles) - (ty trong chuyen)")
        ax.set_ylabel(yl)
        ax.legend()
    axes[0].set_title("Sai so tuyet doi tang theo quang duong")
    axes[1].set_title("Chuyen ngan: cay tong dai thua xa ve %")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "chenh_lech_theo_quang_duong.png", dpi=130)
    plt.close(fig)

    for _, r in df.iterrows():
        log(f"    {r['dai_mi']:>7s} mi ({r['ty_trong_%']:4.1f}%): "
            + "  ".join(f"{lab}: MAE ${r[f'MAE|{k}']:.2f} / {r[f'trong15|{k}']:.0f}%" for lab, k in zip(labels, show))
            + f"  | dong gop chenh vs RF {r['dong_gop_chenh_MAE_vs_RF_%']:.0f}%")
    log("  Da luu chenh_lech_theo_quang_duong.csv/.png")
    return df


# ----------------------------------------------------------------------------
# 13. Mo rong: criterion='absolute_error'
# ----------------------------------------------------------------------------
def compare_criterion(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    rows = []
    for crit in ["squared_error", "absolute_error"]:
        t0 = time.time()
        t = DecisionTreeRegressor(max_depth=DISPATCH_DEPTH, min_samples_leaf=DISPATCH_MIN_LEAF,
                                   criterion=crit, random_state=RANDOM_STATE)
        t.fit(X_train, y_train)
        fit_time = time.time() - t0
        m = evaluate(y_test, t.predict(X_test))
        rows.append({"criterion": crit, "thoi_gian_fit_s": fit_time, **m})
        log(f"  criterion={crit:15s} thoi_gian_fit={fit_time:.2f}s  MAE=${m['MAE']:.4f}  "
            f"trong+-15%={m['trong_15%_%']:.1f}%")
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "so_sanh_criterion.csv", index=False)
    log("  Da luu so_sanh_criterion.csv")
    return df


# ----------------------------------------------------------------------------
# 14. Mo rong: tinh KHONG ON DINH - 10 cay tren 10 mau con
# ----------------------------------------------------------------------------
def demo_instability(X_train: pd.DataFrame, y_train: pd.Series, df: pd.DataFrame, n_trees: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    mode_row = {
        "passenger_count": int(df["passenger_count"].mode()[0]),
        "PULocationID": int(df["PULocationID"].mode()[0]),
        "DOLocationID": int(df["DOLocationID"].mode()[0]),
        "payment_type": int(df["payment_type"].mode()[0]),
        "hour": int(df["hour"].median()), "dow": int(df["dow"].median()), "is_peak": 0,
    }
    query_distances = [1.0, 2.0, 5.0, 10.0, 20.0]
    query_df = pd.DataFrame([{**mode_row, "trip_distance": d} for d in query_distances])[FEATURES]

    preds = np.zeros((n_trees, len(query_distances)))
    n_leaves_list = []
    for i in range(n_trees):
        idx = rng.choice(len(X_train), size=int(0.8 * len(X_train)), replace=False)
        t = DecisionTreeRegressor(max_depth=DISPATCH_DEPTH, min_samples_leaf=DISPATCH_MIN_LEAF,
                                   random_state=RANDOM_STATE + i)
        t.fit(X_train.iloc[idx], y_train.iloc[idx])
        preds[i] = t.predict(query_df)
        n_leaves_list.append(t.get_n_leaves())

    table = pd.DataFrame(preds, columns=[f"gia_tai_{d}mi" for d in query_distances])
    table.insert(0, "cay_so", range(1, n_trees + 1))
    table["so_la"] = n_leaves_list
    table.to_csv(REPORTS_DIR / "instability_10_cay.csv", index=False)

    std_row = table[[c for c in table.columns if c.startswith("gia_tai_")]].std()
    mean_row = table[[c for c in table.columns if c.startswith("gia_tai_")]].mean()
    cv_pct = (std_row / mean_row * 100)
    log("  Do bien thien gia du doan qua 10 cay (tren 10 mau con 80%):")
    for d, cv in zip(query_distances, cv_pct):
        log(f"    Quang duong {d} mi: gia trung binh=${mean_row[f'gia_tai_{d}mi']:.2f}  "
            f"do lech chuan=${std_row[f'gia_tai_{d}mi']:.2f}  he_so_bien_thien={cv:.1f}%")
    log(f"  So la dao dong tu {min(n_leaves_list)} den {max(n_leaves_list)} (cung tham so max_depth/min_samples_leaf)")
    log("  Da luu instability_10_cay.csv")
    return table


# ----------------------------------------------------------------------------
# 15. Mo rong: du doan KHOANG GIA (quantile regression)
# ----------------------------------------------------------------------------
def demo_quantile_range(X_train, y_train, df: pd.DataFrame) -> pd.DataFrame:
    gbr_low = GradientBoostingRegressor(loss="quantile", alpha=0.1, n_estimators=150,
                                         max_depth=3, random_state=RANDOM_STATE)
    gbr_mid = GradientBoostingRegressor(loss="quantile", alpha=0.5, n_estimators=150,
                                         max_depth=3, random_state=RANDOM_STATE)
    gbr_high = GradientBoostingRegressor(loss="quantile", alpha=0.9, n_estimators=150,
                                          max_depth=3, random_state=RANDOM_STATE)
    for m in (gbr_low, gbr_mid, gbr_high):
        m.fit(X_train, y_train)

    mode_row = {
        "passenger_count": int(df["passenger_count"].mode()[0]),
        "PULocationID": int(df["PULocationID"].mode()[0]),
        "DOLocationID": int(df["DOLocationID"].mode()[0]),
        "payment_type": int(df["payment_type"].mode()[0]),
        "hour": int(df["hour"].median()), "dow": int(df["dow"].median()), "is_peak": 0,
    }
    query_distances = [1.0, 2.0, 5.0, 10.0, 20.0]
    query_df = pd.DataFrame([{**mode_row, "trip_distance": d} for d in query_distances])[FEATURES]

    table = pd.DataFrame({
        "quang_duong_mi": query_distances,
        "gia_10%": gbr_low.predict(query_df),
        "gia_trung_vi_50%": gbr_mid.predict(query_df),
        "gia_90%": gbr_high.predict(query_df),
    })
    table.to_csv(REPORTS_DIR / "khoang_gia_quantile.csv", index=False)
    log("  Khoang gia du doan (phan vi 10%-90%) theo quang duong:")
    for _, r in table.iterrows():
        log(f"    {r['quang_duong_mi']:.0f} mi: ${r['gia_10%']:.2f} - ${r['gia_trung_vi_50%']:.2f} - ${r['gia_90%']:.2f}")
    log("  Da luu khoang_gia_quantile.csv")
    return table


# ----------------------------------------------------------------------------
# Vi sao cay KHONG ngoai suy duoc? (chuyen 200 km ~ 124 mi)
# ----------------------------------------------------------------------------
def demo_no_extrapolation(dispatch_tree, lr_pipe: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    mode_row = {
        "passenger_count": int(df["passenger_count"].mode()[0]),
        "PULocationID": int(df["PULocationID"].mode()[0]),
        "DOLocationID": int(df["DOLocationID"].mode()[0]),
        "payment_type": int(df["payment_type"].mode()[0]),
        "hour": int(df["hour"].median()), "dow": int(df["dow"].median()), "is_peak": 0,
    }
    max_dist_train = df["trip_distance"].max()
    query_distances = [max_dist_train, 60.0, 100.0, 124.0]  # 124 mi ~ 200 km
    rows = []
    for d in query_distances:
        x_row = pd.DataFrame([{**mode_row, "trip_distance": d}])[FEATURES]
        rows.append({
            "quang_duong_mi": d, "trong_dai_du_lieu": bool(d <= max_dist_train),
            "gia_cay_quyet_dinh": float(dispatch_tree.predict(x_row)[0]),
            "gia_linear_regression": float(lr_pipe.predict(x_row)[0]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / "ngoai_suy_quang_duong.csv", index=False)
    log(f"  Quang duong xa nhat trong du lieu train: {max_dist_train:.1f} mi")
    for _, r in table.iterrows():
        flag = "" if r["trong_dai_du_lieu"] else "  <-- NGOAI DAI DU LIEU"
        log(f"    {r['quang_duong_mi']:6.1f} mi: cay=${r['gia_cay_quyet_dinh']:.2f}  "
            f"linear=${r['gia_linear_regression']:.2f}{flag}")
    log("  Da luu ngoai_suy_quang_duong.csv")
    return table


def main() -> None:
    log("1. Tai du lieu NYC Yellow Taxi, lay mau 200.000 dong...")
    df_raw = load_and_sample()

    log("2. Lam sach du lieu (4 buoc bat buoc)...")
    df_clean, clean_log = clean_data(df_raw)

    log("3. Tao dac trung thoi gian...")
    df = add_time_features(df_clean)

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    log(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    log("4. EDA...")
    by_hour = run_eda(df)

    log("5. Baseline Dummy + cong thuc tuyen tinh thu cong...")
    baseline_metrics = run_baselines(X_train, y_train, X_test, y_test)

    log("6. Cay KHONG gioi han do sau -> chung minh overfit...")
    overfit_info = demo_overfit(X_train, y_train, X_test, y_test)

    log("7. Quet max_depth 1..20 (5-fold CV tren train)...")
    depth_sweep_df, sweep_best_depth = run_depth_sweep(X_train, y_train)

    log("9. Xay cay tong dai (depth=5, min_samples_leaf=500)...")
    dispatch_tree, dispatch_metrics = build_dispatch_tree(X_train, y_train, X_test, y_test)

    log("8. Ve ham du doan theo quang duong (hinh bac thang)...")
    plot_staircase(dispatch_tree, df)

    log("10. Chuyen cay thanh bang tra cuoc...")
    price_table = tree_to_price_table(dispatch_tree, FEATURES)

    log("11. Kiem tra % chuyen trong +-15%...")
    log(f"  Cay tong dai: {dispatch_metrics['trong_15%_%']:.1f}% chuyen co sai so trong +-15%")

    log("12. So sanh voi Random Forest (chon max_depth bang OOB) va Linear Regression...")
    rf, rf_oob_df = tune_random_forest(X_train, y_train)
    lr_pipe = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())]).fit(X_train, y_train)
    model_comparison, test_preds = compare_models(X_train, y_train, X_test, y_test, dispatch_tree,
                                                  sweep_best_depth, rf, lr_pipe)

    log("13. Mo rong: criterion='absolute_error'...")
    criterion_comparison = compare_criterion(X_train, y_train, X_test, y_test)

    log("14. Mo rong: kiem chung tinh khong on dinh (10 cay)...")
    instability_table = demo_instability(X_train, y_train, df)

    log("15. Mo rong: du doan khoang gia (quantile regression)...")
    quantile_table = demo_quantile_range(X_train, y_train, df)

    log("Giai thich vi sao cay KHONG ngoai suy duoc...")
    extrapolation_table = demo_no_extrapolation(dispatch_tree, lr_pipe, df)

    log("16. Duong bien so la <-> MAE-CV (min_samples_leaf=500)...")
    frontier_df = run_leaves_frontier(X_train, y_train, depth_sweep_df)

    log("17. Cay sau / RF thang cay tong dai o dau (theo dai quang duong, test)...")
    gap_df = analyze_gap_by_distance(X_test, y_test, test_preds)

    joblib.dump(dispatch_tree, MODELS_DIR / "tree_reg.joblib")
    log("  Da luu model -> models/tree_reg.joblib")

    summary = {
        "n_rows_raw_sample": int(len(df_raw)), "n_rows_clean": int(len(df)),
        "clean_log": clean_log,
        "avg_fare_by_hour": by_hour.to_dict(),
        "baseline": baseline_metrics,
        "overfit_demo": overfit_info,
        "depth_sweep_cv": depth_sweep_df.to_dict(orient="records"),
        "depth_best_cv": sweep_best_depth,
        "rf_chon_depth_oob": rf_oob_df.to_dict(orient="records"),
        "dispatch_tree": {"depth": DISPATCH_DEPTH, "min_samples_leaf": DISPATCH_MIN_LEAF,
                           "n_leaves": int(dispatch_tree.get_n_leaves()), "metrics": dispatch_metrics},
        "model_comparison": json.loads(model_comparison.to_json(orient="records")),
        "criterion_comparison": criterion_comparison.to_dict(orient="records"),
        "instability_10_cay": instability_table.to_dict(orient="records"),
        "khoang_gia_quantile": quantile_table.to_dict(orient="records"),
        "extrapolation_demo": extrapolation_table.to_dict(orient="records"),
        "duong_bien_so_la_mae_cv": frontier_df.to_dict(orient="records"),
        "chenh_lech_theo_quang_duong_test": gap_df.to_dict(orient="records"),
    }
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("HOAN THANH. Xem ket qua chi tiet trong thu muc reports/ va outputs/.")


if __name__ == "__main__":
    main()
