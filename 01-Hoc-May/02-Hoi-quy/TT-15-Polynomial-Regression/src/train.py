"""
TT-15 - Polynomial Regression: Du bao cong suat phat (PE) cua nha may nhiet dien
chu trinh hon hop (Combined Cycle Power Plant - UCI) tu 4 dieu kien moi truong.

Pipeline day du (theo 10 buoc trong README.md cap de):
 1. EDA: scatter AT vs PE -> nhin thay duong cong bang mat
 2. Baseline: Linear Regression bac 1 -> ghi RMSE
 3. Ve residual plot cua model bac 1 -> kiem tra hinh chu U
 4. Chay bac 1 -> 5, ghi RMSE train va validation (dung validation_curve)
 5. Ve duong cong xac thuc -> chi ra bac toi uu va diem bat dau overfit
 6. Dem so cot sinh ra o moi bac -> lap bang (bac | so cot | RMSE train | RMSE test)
 7. So sanh Linear vs Ridge o bac cao (bac 4-5) -> Ridge cuu duoc bao nhieu?
 8. Ve lai residual plot cua model bac toi uu -> chu U da bien mat chua?
 9. So sanh voi Random Forest Regressor (TT-17) -> cay bat phi tuyen tu nhien
10. Dien giai: o dai nhiet do nao cong suat giam nhanh nhat?

Mo rong (section 8 cua README):
11. interaction_only=True -> chi tao tich cheo, khong tao bac cao. So sanh.
12. SplineTransformer -> linh hoat hon da thuc, khong "phat dien" khi ngoai suy
13. Chung minh hien tuong ngoai suy: du doan tai AT=50C (ngoai dai du lieu) bang bac 5

Chay: python src/train.py
"""
from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42
DATA_URL = "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip"
DATA_XLSX = DATA_DIR / "Folds5x2_pp.xlsx"

FEATURES = ["AT", "V", "AP", "RH"]
TARGET = "PE"
FEATURE_DESC = {
    "AT": "Nhiet do moi truong (C)", "V": "Ap suat chan khong (cm Hg)",
    "AP": "Ap suat khi quyen (mbar)", "RH": "Do am tuong doi (%)",
}
DEGREES = [1, 2, 3, 4, 5]
ALPHA_GRID = np.logspace(-3, 4, 100)


def log(msg: str) -> None:
    print(f"[TT-15] {msg}")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ----------------------------------------------------------------------------
# 0. Tai du lieu Combined Cycle Power Plant (UCI), cache vao data/
# ----------------------------------------------------------------------------
def download_data() -> Path:
    if DATA_XLSX.exists():
        return DATA_XLSX
    log(f"  Dang tai du lieu tu UCI: {DATA_URL}")
    with urllib.request.urlopen(DATA_URL, timeout=30) as r:
        raw = r.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        DATA_XLSX.write_bytes(z.read("CCPP/Folds5x2_pp.xlsx"))
    log(f"  Da luu du lieu vao {DATA_XLSX.relative_to(ROOT)}")
    return DATA_XLSX


def load_data() -> pd.DataFrame:
    path = download_data()
    df = pd.read_excel(path)[FEATURES + [TARGET]].copy()
    log(f"  Du lieu: {df.shape[0]} dong x {df.shape[1]} cot")
    return df


# ----------------------------------------------------------------------------
# 1. EDA: scatter AT vs PE -> nhin thay duong cong bang mat
# ----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    corr = df.corr()
    corr.to_csv(REPORTS_DIR / "correlation_matrix.csv")
    log("  Tuong quan voi PE: " + ", ".join(f"{c}={corr.loc[c, TARGET]:.3f}" for c in FEATURES))
    r_at_v = corr.loc["AT", "V"]
    log(f"  Tuong quan AT-V = {r_at_v:.3f} (>0.8 => can dung Ridge khi tao dac trung da thuc)")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["AT"], df[TARGET], s=6, alpha=0.25, color="#2980b9")
    ax.set_xlabel("AT - Nhiet do moi truong (C)")
    ax.set_ylabel("PE - Cong suat phat (MW)")
    ax.set_title("EDA: quan he AT vs PE - ro rang la duong CONG, khong phai duong thang")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "scatter_AT_PE.png", dpi=130)
    plt.close(fig)
    log("  Da luu scatter_AT_PE.png")
    return corr


# ----------------------------------------------------------------------------
# Helper: dung pipeline poly + scale + model
# ----------------------------------------------------------------------------
def make_pipe(degree: int, model, interaction_only: bool = False) -> Pipeline:
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False, interaction_only=interaction_only)),
        ("scale", StandardScaler()),
        ("model", model),
    ])


# ----------------------------------------------------------------------------
# 2. Baseline: Linear Regression bac 1
# ----------------------------------------------------------------------------
def run_baseline_degree1(X_train, y_train, X_test, y_test) -> tuple[Pipeline, np.ndarray, dict]:
    pipe = make_pipe(1, LinearRegression())
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = evaluate(y_test, pred)
    log(f"  Baseline bac 1 (duong thang): RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}")
    return pipe, pred, metrics


# ----------------------------------------------------------------------------
# 3 & 8. Residual plot (truoc/sau) - kiem tra hinh chu U
# ----------------------------------------------------------------------------
def curvature_score(pred: np.ndarray, residuals: np.ndarray) -> float:
    """R2 cua duong bac 2 fit vao (pred, residuals) - cang cao cang 'chu U' ro."""
    coeffs = np.polyfit(pred, residuals, deg=2)
    fitted = np.polyval(coeffs, pred)
    ss_res = np.sum((residuals - fitted) ** 2)
    ss_tot = np.sum((residuals - residuals.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def run_residual_before_after(pred_before, y_test, pred_after, degree_before: int, degree_after: int) -> dict:
    resid_before = np.asarray(y_test) - pred_before
    resid_after = np.asarray(y_test) - pred_after
    curve_before = curvature_score(pred_before, resid_before)
    curve_after = curvature_score(pred_after, resid_after)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    ax1.scatter(pred_before, resid_before, s=6, alpha=0.25, color="#c0392b")
    ax1.axhline(0, color="black", ls="--", lw=1)
    ax1.set_xlabel("Gia du doan"); ax1.set_ylabel("Phan du (thuc te - du doan)")
    ax1.set_title(f"TRUOC - bac {degree_before}\ndo cong (R2 duong bac 2 fit phan du) = {curve_before:.3f}")

    ax2.scatter(pred_after, resid_after, s=6, alpha=0.25, color="#27ae60")
    ax2.axhline(0, color="black", ls="--", lw=1)
    ax2.set_xlabel("Gia du doan")
    ax2.set_title(f"SAU - bac {degree_after} (Ridge)\ndo cong (R2 duong bac 2 fit phan du) = {curve_after:.3f}")
    fig.suptitle("Residual plot TRUOC / SAU khi them dac trung da thuc")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "residual_truoc_sau.png", dpi=130)
    plt.close(fig)

    log(f"  Do cong phan du TRUOC (bac {degree_before}) = {curve_before:.4f}")
    log(f"  Do cong phan du SAU   (bac {degree_after}) = {curve_after:.4f}")
    log("  Da luu residual_truoc_sau.png")
    return {"curvature_before": curve_before, "curvature_after": curve_after,
            "std_resid_before": float(resid_before.std()), "std_resid_after": float(resid_after.std())}


# ----------------------------------------------------------------------------
# 4-5. Duong cong xac thuc theo bac (validation_curve) -> chon bac toi uu
# ----------------------------------------------------------------------------
def run_validation_curve(X_train, y_train) -> tuple[pd.DataFrame, int]:
    pipe = make_pipe(1, Ridge(alpha=1.0))
    train_scores, val_scores = validation_curve(
        pipe, X_train, y_train, param_name="poly__degree",
        param_range=DEGREES, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    train_rmse = -train_scores.mean(axis=1)
    val_rmse = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    df = pd.DataFrame({"bac": DEGREES, "RMSE_train_cv": train_rmse, "RMSE_val_cv": val_rmse, "std_val_cv": val_std})
    df.to_csv(REPORTS_DIR / "validation_curve.csv", index=False)

    best_degree = int(df.loc[df["RMSE_val_cv"].idxmin(), "bac"])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(DEGREES, train_rmse, "o-", color="#2980b9", label="RMSE train (CV)")
    ax.plot(DEGREES, val_rmse, "o-", color="#c0392b", label="RMSE validation (CV)")
    ax.fill_between(DEGREES, val_rmse - val_std, val_rmse + val_std, color="#c0392b", alpha=0.15)
    ax.axvline(best_degree, color="gray", ls="--", label=f"bac toi uu = {best_degree}")
    ax.set_xlabel("Bac da thuc (poly__degree)")
    ax.set_ylabel("RMSE (5-fold CV, Ridge alpha=1.0)")
    ax.set_title("Duong cong xac thuc theo bac da thuc")
    ax.set_xticks(DEGREES)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "validation_curve.png", dpi=130)
    plt.close(fig)

    log(f"  RMSE (val CV) theo bac: " + ", ".join(f"bac{d}={v:.3f}" for d, v in zip(DEGREES, val_rmse)))
    log(f"  Bac toi uu (RMSE validation thap nhat) = {best_degree}")
    log("  Da luu validation_curve.csv/.png")
    return df, best_degree


# ----------------------------------------------------------------------------
# 6. Bang: bac | so cot sinh ra | RMSE train | RMSE test (Linear thuan)
# ----------------------------------------------------------------------------
def build_degree_table(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    rows = []
    for d in DEGREES:
        pipe = make_pipe(d, LinearRegression())
        pipe.fit(X_train, y_train)
        n_cols = pipe.named_steps["poly"].n_output_features_
        rmse_train = rmse(y_train, pipe.predict(X_train))
        rmse_test = rmse(y_test, pipe.predict(X_test))
        max_abs_coef = float(np.abs(pipe.named_steps["model"].coef_).max())
        rows.append({"bac": d, "so_cot": n_cols, "RMSE_train": rmse_train, "RMSE_test": rmse_test,
                     "he_so_tuyet_doi_lon_nhat": max_abs_coef})
        log(f"  [Linear thuan] bac {d}: {n_cols:3d} cot  RMSE_train={rmse_train:.4f}  "
            f"RMSE_test={rmse_test:.4f}  |he_so|_max={max_abs_coef:.2e}")
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "bang_bac_so_cot_rmse.csv", index=False)
    log("  Da luu bang_bac_so_cot_rmse.csv")
    return df


# ----------------------------------------------------------------------------
# 7. So sanh Linear vs Ridge o bac cao (bac 4-5)
# ----------------------------------------------------------------------------
def compare_linear_ridge_high_degree(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    rows = []
    for d in [4, 5]:
        lin_pipe = make_pipe(d, LinearRegression())
        lin_pipe.fit(X_train, y_train)
        lin_rmse = rmse(y_test, lin_pipe.predict(X_test))
        lin_max_coef = float(np.abs(lin_pipe.named_steps["model"].coef_).max())

        ridge_pipe = make_pipe(d, RidgeCV(alphas=ALPHA_GRID, cv=5))
        ridge_pipe.fit(X_train, y_train)
        ridge_rmse = rmse(y_test, ridge_pipe.predict(X_test))
        ridge_alpha = float(ridge_pipe.named_steps["model"].alpha_)
        ridge_max_coef = float(np.abs(ridge_pipe.named_steps["model"].coef_).max())

        rows.append({"bac": d, "RMSE_linear": lin_rmse, "he_so_max_linear": lin_max_coef,
                      "RMSE_ridge": ridge_rmse, "ridge_alpha": ridge_alpha, "he_so_max_ridge": ridge_max_coef,
                      "RMSE_giam_%": (lin_rmse - ridge_rmse) / lin_rmse * 100,
                      "he_so_giam_lan": lin_max_coef / ridge_max_coef if ridge_max_coef > 0 else np.nan})
        log(f"  Bac {d}: Linear RMSE={lin_rmse:.4f} (|he_so|_max={lin_max_coef:.2e})  vs  "
            f"Ridge(alpha={ridge_alpha:.3g}) RMSE={ridge_rmse:.4f} (|he_so|_max={ridge_max_coef:.2e})")
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "so_sanh_linear_ridge_bac_cao.csv", index=False)
    log("  Da luu so_sanh_linear_ridge_bac_cao.csv")
    return df


# ----------------------------------------------------------------------------
# 9. So sanh voi Random Forest Regressor (TT-17)
# ----------------------------------------------------------------------------
def compare_with_random_forest(X_train, y_train, X_test, y_test, poly_metrics: dict) -> pd.DataFrame:
    rf = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate(y_test, rf.predict(X_test))

    rows = [
        {"model": "Polynomial Ridge (bac toi uu)", **poly_metrics},
        {"model": "Random Forest Regressor", **rf_metrics},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "so_sanh_random_forest.csv", index=False)
    log(f"  Random Forest: RMSE={rf_metrics['RMSE']:.4f}  MAE={rf_metrics['MAE']:.4f}  R2={rf_metrics['R2']:.4f}")
    log("  Da luu so_sanh_random_forest.csv")
    return df


# ----------------------------------------------------------------------------
# 10. O dai nhiet do nao cong suat giam nhanh nhat?
# ----------------------------------------------------------------------------
def analyze_temperature_slope(df: pd.DataFrame, degree: int) -> pd.DataFrame:
    at_sorted = df["AT"].values
    pe_sorted = df["PE"].values
    coeffs = np.polyfit(at_sorted, pe_sorted, deg=degree)
    deriv = np.polyder(coeffs)

    at_grid = np.linspace(df["AT"].min(), df["AT"].max(), 200)
    slope_grid = np.polyval(deriv, at_grid)
    steepest_idx = int(np.argmin(slope_grid))
    steepest_at = float(at_grid[steepest_idx])
    steepest_slope = float(slope_grid[steepest_idx])

    table = pd.DataFrame({"AT": at_grid, "dPE_dAT": slope_grid})
    table.to_csv(REPORTS_DIR / "do_doc_theo_nhiet_do.csv", index=False)

    log(f"  Fit da thuc 1 bien (AT->PE) bac {degree} tren toan bo du lieu de phan tich do doc")
    log(f"  Cong suat giam NHANH NHAT quanh AT ~= {steepest_at:.1f} C (dPE/dAT = {steepest_slope:.3f} MW/C)")
    log(f"  dPE/dAT tai AT=5C: {np.polyval(deriv, 5):.3f} MW/C | tai AT=20C: {np.polyval(deriv, 20):.3f} MW/C "
        f"| tai AT=35C: {np.polyval(deriv, 35):.3f} MW/C")
    log("  Da luu do_doc_theo_nhiet_do.csv")
    return table


# ----------------------------------------------------------------------------
# 11. interaction_only=True: chi tao tich cheo, khong tao bac cao
# ----------------------------------------------------------------------------
def compare_interaction_only(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    rows = []
    for interaction_only in (False, True):
        pipe = make_pipe(2, RidgeCV(alphas=ALPHA_GRID, cv=5), interaction_only=interaction_only)
        pipe.fit(X_train, y_train)
        n_cols = pipe.named_steps["poly"].n_output_features_
        m = evaluate(y_test, pipe.predict(X_test))
        rows.append({"interaction_only": interaction_only, "so_cot": n_cols, **m})
        log(f"  degree=2 interaction_only={interaction_only!s:5s}: {n_cols} cot  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "interaction_only_vs_full.csv", index=False)
    log("  Da luu interaction_only_vs_full.csv")
    return df


# ----------------------------------------------------------------------------
# 12. SplineTransformer - linh hoat hon da thuc, khong "phat dien" khi ngoai suy
# ----------------------------------------------------------------------------
def compare_spline(X_train, y_train, X_test, y_test, best_degree: int) -> tuple[Pipeline, dict]:
    spline_pipe = Pipeline([
        ("spline", SplineTransformer(n_knots=6, degree=3, extrapolation="constant")),
        ("scale", StandardScaler()),
        ("model", RidgeCV(alphas=ALPHA_GRID, cv=5)),
    ])
    spline_pipe.fit(X_train, y_train)
    spline_metrics = evaluate(y_test, spline_pipe.predict(X_test))

    poly_pipe = make_pipe(best_degree, RidgeCV(alphas=ALPHA_GRID, cv=5))
    poly_pipe.fit(X_train, y_train)
    poly_metrics = evaluate(y_test, poly_pipe.predict(X_test))

    df = pd.DataFrame([
        {"model": f"Polynomial Ridge (bac {best_degree})", **poly_metrics},
        {"model": "Spline Ridge (n_knots=6, degree=3)", **spline_metrics},
    ])
    df.to_csv(REPORTS_DIR / "spline_vs_polynomial.csv", index=False)
    log(f"  Spline (n_knots=6): RMSE={spline_metrics['RMSE']:.4f}  R2={spline_metrics['R2']:.4f}")
    log(f"  Polynomial bac {best_degree}: RMSE={poly_metrics['RMSE']:.4f}  R2={poly_metrics['R2']:.4f}")
    log("  Da luu spline_vs_polynomial.csv")
    return spline_pipe, spline_metrics


# ----------------------------------------------------------------------------
# 13. Chung minh hien tuong ngoai suy: du doan tai AT=50C (ngoai dai du lieu)
# ----------------------------------------------------------------------------
def demo_extrapolation(df: pd.DataFrame, degree1_pipe: Pipeline, best_pipe: Pipeline,
                        degree5_pipe: Pipeline, spline_pipe: Pipeline) -> pd.DataFrame:
    means = df[["V", "AP", "RH"]].mean()
    at_min, at_max = df["AT"].min(), df["AT"].max()
    at_points = [at_min, at_max, 45.0, 50.0, 60.0]

    rows = []
    for at in at_points:
        x_row = pd.DataFrame([{"AT": at, "V": means["V"], "AP": means["AP"], "RH": means["RH"]}])
        rows.append({
            "AT": at, "trong_dai_du_lieu": bool(at_min <= at <= at_max),
            "du_doan_bac1": float(degree1_pipe.predict(x_row)[0]),
            "du_doan_bac_toi_uu_ridge": float(best_pipe.predict(x_row)[0]),
            "du_doan_bac5_ridge": float(degree5_pipe.predict(x_row)[0]),
            "du_doan_spline": float(spline_pipe.predict(x_row)[0]),
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(REPORTS_DIR / "ngoai_suy_AT.csv", index=False)

    log(f"  Du lieu train chi co AT trong [{at_min:.1f}, {at_max:.1f}] C. PE thuc te chi trong khoang "
        f"[{df['PE'].min():.1f}, {df['PE'].max():.1f}] MW.")
    for _, r in df_out.iterrows():
        flag = "" if r["trong_dai_du_lieu"] else "  <-- NGOAI DAI DU LIEU"
        log(f"    AT={r['AT']:5.1f}C: bac1={r['du_doan_bac1']:7.1f}  "
            f"bac_toi_uu={r['du_doan_bac_toi_uu_ridge']:7.1f}  bac5={r['du_doan_bac5_ridge']:7.1f}  "
            f"spline={r['du_doan_spline']:7.1f}{flag}")
    log("  Da luu ngoai_suy_AT.csv")
    return df_out


def main() -> None:
    log("0. Nap du lieu Combined Cycle Power Plant...")
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    log(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    log("1. EDA: scatter AT vs PE, ma tran tuong quan...")
    corr = run_eda(df)

    log("2. Baseline: Linear Regression bac 1...")
    pipe1, pred1, metrics1 = run_baseline_degree1(X_train, y_train, X_test, y_test)

    log("4-5. Duong cong xac thuc theo bac (validation_curve)...")
    val_curve_df, best_degree = run_validation_curve(X_train, y_train)

    log("6. Bang bac | so cot | RMSE (Linear thuan)...")
    degree_table = build_degree_table(X_train, y_train, X_test, y_test)

    log("7. So sanh Linear vs Ridge o bac cao (4-5)...")
    linear_vs_ridge = compare_linear_ridge_high_degree(X_train, y_train, X_test, y_test)

    log(f"   Refit pipeline toi uu: bac={best_degree}, Ridge (RidgeCV chon alpha)...")
    best_pipe = make_pipe(best_degree, RidgeCV(alphas=ALPHA_GRID, cv=5))
    best_pipe.fit(X_train, y_train)
    pred_best = best_pipe.predict(X_test)
    best_metrics = evaluate(y_test, pred_best)
    best_alpha = float(best_pipe.named_steps["model"].alpha_)
    log(f"   Model toi uu: bac={best_degree}  alpha={best_alpha:.4g}  "
        f"RMSE={best_metrics['RMSE']:.4f}  MAE={best_metrics['MAE']:.4f}  R2={best_metrics['R2']:.4f}")

    log("3&8. Residual plot TRUOC (bac 1) / SAU (bac toi uu)...")
    residual_info = run_residual_before_after(pred1, y_test, pred_best, 1, best_degree)

    log("9. So sanh voi Random Forest Regressor...")
    rf_comparison = compare_with_random_forest(X_train, y_train, X_test, y_test, best_metrics)

    log("10. Phan tich do doc cong suat theo nhiet do...")
    slope_table = analyze_temperature_slope(df, best_degree)

    log("11. interaction_only=True vs day du (mo rong)...")
    interaction_df = compare_interaction_only(X_train, y_train, X_test, y_test)

    log("12. SplineTransformer vs Polynomial (mo rong)...")
    spline_pipe, spline_metrics = compare_spline(X_train, y_train, X_test, y_test, best_degree)

    log("   Fit them pipeline bac 5 Ridge de minh hoa ngoai suy...")
    degree5_pipe = make_pipe(5, RidgeCV(alphas=ALPHA_GRID, cv=5))
    degree5_pipe.fit(X_train, y_train)

    log("13. Minh hoa hien tuong ngoai suy tai AT=50C (mo rong)...")
    extrapolation_df = demo_extrapolation(df, pipe1, best_pipe, degree5_pipe, spline_pipe)

    joblib.dump(best_pipe, MODELS_DIR / "poly_pipeline.joblib")
    log(f"  Da luu model -> models/poly_pipeline.joblib")

    summary = {
        "n_rows": int(len(df)), "features": FEATURES, "target": TARGET,
        "correlation_AT_V": float(corr.loc["AT", "V"]),
        "baseline_degree1": metrics1,
        "validation_curve": val_curve_df.to_dict(orient="records"),
        "best_degree": best_degree,
        "best_alpha": best_alpha,
        "best_metrics": best_metrics,
        "degree_table_linear": degree_table.to_dict(orient="records"),
        "linear_vs_ridge_high_degree": linear_vs_ridge.to_dict(orient="records"),
        "residual_curvature": residual_info,
        "random_forest_comparison": rf_comparison.to_dict(orient="records"),
        "interaction_only_vs_full": interaction_df.to_dict(orient="records"),
        "spline_vs_polynomial_rmse": spline_metrics,
        "extrapolation_demo": extrapolation_df.to_dict(orient="records"),
    }
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("HOAN THANH. Xem ket qua chi tiet trong thu muc reports/.")


if __name__ == "__main__":
    main()
