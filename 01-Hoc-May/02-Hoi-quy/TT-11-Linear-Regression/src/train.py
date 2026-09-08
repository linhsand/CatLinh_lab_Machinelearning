"""
TT-11 - Linear Regression: Dinh gia nha o (California Housing).

Pipeline day du (theo 12 buoc trong README.md cap de):
 1. Nap du lieu, describe() -> phat hien outlier AveOccup, AveRooms
 2. Phat hien nhan bi cat ngon o 5.0 (dem so dong)
 3. EDA: scatter MedInc vs gia, heatmap tuong quan, ban do gia theo toa do
 4. Baseline: DummyRegressor(strategy='mean') -> RMSE
 5. Linear Regression co ban (sau khi clip outlier) -> RMSE, MAE, R2
 6. Residual plot (phan du vs gia du doan) -> kiem tra hinh pheu
 7. Q-Q plot kiem tra phan phoi phan du
 8. Thu du doan log(gia) thay vi gia -> so sanh residual/RMSE
 9. Kiem tra da cong tuyen bang VIF (AveRooms <-> AveBedrms)
10. Feature engineering: ty le phong ngu/phong, khoang cach toi SF/LA
11. Bang he so da chuan hoa -> dien giai 3 yeu to anh huong manh nhat
12. So sanh voi Ridge (TT-12) va Random Forest Regressor (TT-17)

Chay: python src/train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42
CLIP_PERCENTILE = 0.99
FEATURE_COLUMNS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
SF_COORD = (37.7749, -122.4194)
LA_COORD = (34.0522, -118.2437)


def log(msg: str) -> None:
    print(f"[TT-11] {msg}")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    return float(np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred)) / np.asarray(y_true))) * 100)


def evaluate(y_true, y_pred) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": mape(y_true, y_pred),
    }


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2r - lat1r, lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------------
# Transformer: clip outlier (dung chung cho model co ban va model mo rong)
# ----------------------------------------------------------------------------
class ClipOutliers(BaseEstimator, TransformerMixin):
    """Clip AveRooms/AveOccup o phan vi CLIP_PERCENTILE hoc tu TRAIN (tranh leakage)."""

    def __init__(self, percentile: float = CLIP_PERCENTILE):
        self.percentile = percentile

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X, columns=FEATURE_COLUMNS)
        self.averooms_clip_ = float(X["AveRooms"].quantile(self.percentile))
        self.aveoccup_clip_ = float(X["AveOccup"].quantile(self.percentile))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=FEATURE_COLUMNS).copy()
        X["AveRooms"] = X["AveRooms"].clip(upper=self.averooms_clip_)
        X["AveOccup"] = X["AveOccup"].clip(upper=self.aveoccup_clip_)
        self.feature_names_out_ = list(X.columns)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


# ----------------------------------------------------------------------------
# Transformer: FeatureEngineer = ClipOutliers + ty le phong ngu + khoang cach
# ----------------------------------------------------------------------------
class FeatureEngineer(ClipOutliers):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)
        X["BedrmsRatio"] = X["AveBedrms"] / X["AveRooms"]
        X["DistSF_km"] = haversine_km(X["Latitude"], X["Longitude"], *SF_COORD)
        X["DistLA_km"] = haversine_km(X["Latitude"], X["Longitude"], *LA_COORD)
        X["DistNearestCity_km"] = np.minimum(X["DistSF_km"], X["DistLA_km"])
        self.feature_names_out_ = list(X.columns)
        return X


# ----------------------------------------------------------------------------
# 1. Nap du lieu, describe() -> phat hien outlier
# ----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.describe().to_csv(REPORTS_DIR / "data_describe.csv")
    log(f"  Nap {len(df)} dong x {df.shape[1] - 1} dac trung.")
    log(f"  AveOccup: max={df['AveOccup'].max():.1f}  p99={df['AveOccup'].quantile(0.99):.2f}  "
        f"-> outlier cuc doan ro ret")
    log(f"  AveRooms: max={df['AveRooms'].max():.1f}  p99={df['AveRooms'].quantile(0.99):.2f}")
    log(f"  Tuong quan AveRooms<->AveBedrms: {df['AveRooms'].corr(df['AveBedrms']):.3f} -> nghi ngo da cong tuyen")
    return df


# ----------------------------------------------------------------------------
# 2. Phat hien nhan bi cat ngon o 5.0
# ----------------------------------------------------------------------------
def detect_capped_label(df: pd.DataFrame) -> int:
    capped = df["MedHouseVal"] >= 4.999
    n_capped = int(capped.sum())
    log(f"  So dong bi cat ngon o gia 5.0: {n_capped} / {len(df)} ({n_capped / len(df):.2%})")
    return n_capped


# ----------------------------------------------------------------------------
# 3. EDA: scatter MedInc, heatmap tuong quan, ban do gia
# ----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["MedInc"], df["MedHouseVal"], s=4, alpha=0.2, color="#2980b9")
    ax.axhline(5.0, color="#c0392b", ls="--", lw=1, label="nguong cat ngon 5.0")
    ax.set_xlabel("MedInc (thu nhap trung vi, chuc nghin USD)")
    ax.set_ylabel("MedHouseVal (gia, 100k USD)")
    ax.set_title("MedInc vs Gia nha")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_medinc_scatter.png", dpi=130)
    plt.close(fig)

    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Ma tran tuong quan")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_correlation_heatmap.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    sc = ax.scatter(df["Longitude"], df["Latitude"], c=df["MedHouseVal"], cmap="viridis", s=5, alpha=0.5)
    fig.colorbar(sc, ax=ax, label="Gia (100k USD)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Ban do gia nha theo toa do (sang = dat, toi = re)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "ban_do_gia.png", dpi=130)
    plt.close(fig)
    log("  Da luu eda_medinc_scatter.png, eda_correlation_heatmap.png, ban_do_gia.png")


# ----------------------------------------------------------------------------
# 4. Baseline DummyRegressor
# ----------------------------------------------------------------------------
def run_baseline(X_train, y_train, X_test, y_test) -> dict:
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    metrics = evaluate(y_test, dummy.predict(X_test))
    log(f"  Baseline (mean): RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}")
    return metrics


# ----------------------------------------------------------------------------
# 5. Linear Regression co ban (sau khi clip outlier)
# ----------------------------------------------------------------------------
def run_basic_lr(X_train, y_train, X_test, y_test) -> tuple[Pipeline, np.ndarray, dict]:
    pipe = Pipeline([
        ("clip", ClipOutliers()),
        ("scale", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = evaluate(y_test, pred)
    log(f"  LR co ban: RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
        f"R2={metrics['R2']:.4f}  MAPE={metrics['MAPE_%']:.2f}%")
    return pipe, pred, metrics


# ----------------------------------------------------------------------------
# 6-7. Residual plot + Q-Q plot
# ----------------------------------------------------------------------------
def run_residual_qq(y_test, pred, title_suffix: str = "", file_suffix: str = "") -> np.ndarray:
    residuals = np.asarray(y_test) - pred
    residual_name = f"residual_plot{file_suffix}.png"
    qq_name = f"qq_plot{file_suffix}.png"

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(pred, residuals, s=6, alpha=0.25, color="#2980b9")
    ax.axhline(0, color="#c0392b", ls="--", lw=1)
    ax.set_xlabel("Gia du doan"); ax.set_ylabel("Phan du (thuc te - du doan)")
    ax.set_title(f"Residual plot{title_suffix}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / residual_name, dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f"Q-Q plot phan du{title_suffix}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / qq_name, dpi=130)
    plt.close(fig)

    log(f"  Da luu {residual_name}, {qq_name}  (std phan du={residuals.std():.4f})")
    return residuals


# ----------------------------------------------------------------------------
# 8. Du doan log(gia) thay vi gia
# ----------------------------------------------------------------------------
def run_log_target(X_train, y_train, X_test, y_test) -> dict:
    pipe = Pipeline([
        ("clip", ClipOutliers()),
        ("scale", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    pipe.fit(X_train, np.log1p(y_train))
    pred_log = pipe.predict(X_test)
    pred = np.expm1(pred_log)
    metrics = evaluate(y_test, pred)
    log(f"  LR tren log1p(gia): RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}")
    run_residual_qq(y_test, pred, title_suffix=" (log-target, quy doi ve gia goc)", file_suffix="_log")
    return metrics


# ----------------------------------------------------------------------------
# 9. VIF - kiem tra da cong tuyen
# ----------------------------------------------------------------------------
def run_vif(X_train_clipped: pd.DataFrame) -> pd.DataFrame:
    X_const = add_constant(X_train_clipped[FEATURE_COLUMNS])
    vif_rows = [
        {"feature": col, "VIF": float(variance_inflation_factor(X_const.values, i))}
        for i, col in enumerate(X_const.columns) if col != "const"
    ]
    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
    vif_df.to_csv(REPORTS_DIR / "vif.csv", index=False)
    log("  VIF (>10 la dau hieu da cong tuyen manh):")
    for _, row in vif_df.iterrows():
        flag = " <-- CAO" if row["VIF"] > 10 else ""
        log(f"    {row['feature']:14s} VIF={row['VIF']:.2f}{flag}")
    return vif_df


# ----------------------------------------------------------------------------
# 10-11. Feature engineering + bang he so chuan hoa
# ----------------------------------------------------------------------------
def run_feature_engineering(X_train, y_train, X_test, y_test, basic_metrics: dict) -> tuple[Pipeline, dict, pd.DataFrame]:
    pipe = Pipeline([
        ("fe", FeatureEngineer()),
        ("scale", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = evaluate(y_test, pred)
    log(f"  LR + feature engineering: RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}")
    log(f"  Cai thien RMSE so voi LR co ban: {basic_metrics['RMSE'] - metrics['RMSE']:+.4f}")

    feature_names = pipe.named_steps["fe"].feature_names_out_
    coefs = pipe.named_steps["lr"].coef_
    he_so = pd.Series(coefs, index=feature_names).sort_values(key=abs, ascending=False)
    he_so_df = he_so.reset_index()
    he_so_df.columns = ["dac_trung", "he_so_chuan_hoa"]
    he_so_df.to_csv(REPORTS_DIR / "he_so.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#c0392b" if v < 0 else "#2980b9" for v in he_so.values]
    ax.barh(he_so.index[::-1], he_so.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("He so hoi quy (da chuan hoa)")
    ax.set_title("Muc do anh huong cua tung dac trung toi gia nha")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "he_so.png", dpi=130)
    plt.close(fig)

    top3 = he_so.head(3)
    log("  Top 3 yeu to anh huong manh nhat (theo |he so| da chuan hoa):")
    for name, val in top3.items():
        chieu = "tang" if val > 0 else "giam"
        log(f"    {name:20s} he_so={val:+.4f}  -> {chieu} gia khi tang 1 do lech chuan")
    log("  Da luu he_so.csv, he_so.png")
    return pipe, metrics, he_so_df


# ----------------------------------------------------------------------------
# 12. So sanh voi Ridge (TT-12) va Random Forest (TT-17)
# ----------------------------------------------------------------------------
def run_model_comparison(X_train, y_train, X_test, y_test, results_so_far: dict) -> pd.DataFrame:
    rows = [{"model": name, **metrics} for name, metrics in results_so_far.items()]

    ridge_pipe = Pipeline([
        ("fe", FeatureEngineer()),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])
    ridge_pipe.fit(X_train, y_train)
    ridge_metrics = evaluate(y_test, ridge_pipe.predict(X_test))
    rows.append({"model": "Ridge (TT-12, alpha=1.0)", **ridge_metrics})

    rf_pipe = Pipeline([
        ("fe", FeatureEngineer()),
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE)),
    ])
    rf_pipe.fit(X_train, y_train)
    rf_metrics = evaluate(y_test, rf_pipe.predict(X_test))
    rows.append({"model": "Random Forest (TT-17)", **rf_metrics})

    comp = pd.DataFrame(rows)
    comp.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    log("  So sanh model (RMSE cang thap cang tot):")
    for _, row in comp.iterrows():
        log(f"    {row['model']:32s} RMSE={row['RMSE']:.4f}  R2={row['R2']:.4f}")
    lr_fe_metrics = results_so_far.get("LR + feature engineering")
    if lr_fe_metrics is not None:
        log(f"  Chenh lech R2 giua Random Forest va LR (feature engineering) = "
            f"{rf_metrics['R2'] - lr_fe_metrics['R2']:+.4f} -> cai gia cua tinh giai thich duoc")
    log("  Da luu model_comparison.csv")
    return comp


def main() -> None:
    log("1. Nap du lieu + describe()...")
    df = load_data()

    log("2. Phat hien nhan bi cat ngon...")
    n_capped = detect_capped_label(df)

    log("3. EDA...")
    run_eda(df)

    X = df[FEATURE_COLUMNS].copy()
    y = df["MedHouseVal"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    log(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    log("4. Baseline DummyRegressor...")
    baseline_metrics = run_baseline(X_train, y_train, X_test, y_test)

    log("5. Linear Regression co ban...")
    basic_pipe, basic_pred, basic_metrics = run_basic_lr(X_train, y_train, X_test, y_test)

    log("6-7. Residual plot + Q-Q plot...")
    run_residual_qq(y_test, basic_pred)

    log("8. Thu du doan log(gia)...")
    log_metrics = run_log_target(X_train, y_train, X_test, y_test)

    log("9. Kiem tra VIF...")
    X_train_clipped = ClipOutliers().fit(X_train).transform(X_train)
    vif_df = run_vif(X_train_clipped)

    log("10-11. Feature engineering + bang he so...")
    fe_pipe, fe_metrics, he_so_df = run_feature_engineering(X_train, y_train, X_test, y_test, basic_metrics)

    log("12. So sanh voi Ridge va Random Forest...")
    results_so_far = {
        "Baseline (mean)": baseline_metrics,
        "LR co ban": basic_metrics,
        "LR log-target": log_metrics,
        "LR + feature engineering": fe_metrics,
    }
    comparison = run_model_comparison(X_train, y_train, X_test, y_test, results_so_far)

    joblib.dump(fe_pipe, MODELS_DIR / "lr_pipeline.joblib")
    log("Da luu model -> models/lr_pipeline.joblib")

    summary = {
        "n_rows": int(len(df)),
        "n_capped_at_5": n_capped,
        "pct_capped": n_capped / len(df),
        "vif": vif_df.to_dict(orient="records"),
        "he_so_chuan_hoa": he_so_df.to_dict(orient="records"),
        "model_comparison": comparison.to_dict(orient="records"),
        "final_model": "LR + feature engineering (clip outlier + ty le phong ngu + khoang cach SF/LA)",
    }
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("HOAN THANH. Xem ket qua chi tiet trong thu muc reports/.")


if __name__ == "__main__":
    main()
