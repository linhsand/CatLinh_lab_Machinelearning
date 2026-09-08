"""
TT-12 - Ridge Regression (L2): Phan bo ngan sach quang cao da kenh.

Pipeline day du (theo 10 buoc trong README.md cap de):
 1. Sinh du lieu tong hop co da cong tuyen ro ret (TV <-> Facebook, r ~ 0.95)
 2. Ma tran tuong quan + VIF -> chung minh da cong tuyen (VIF > 10)
 3. Linear Regression co ban -> ghi lai he so (ky vong bat on dinh)
 4. Thi nghiem on dinh: bootstrap 100 lan, moi lan lay 80% du lieu
    -> so sanh do dao dong he so Linear vs Ridge
 5. RidgeCV do alpha toi uu bang cross-validation
 6. Ve coefficient path (he so co dan ve 0 khi alpha tang)
 7. Ve duong RMSE train/test theo alpha -> diem can bang bias-variance
 8. So sanh RMSE: Linear vs Ridge tren tap test
 9. So sanh voi Lasso (TT-13) va ElasticNet (TT-14)
10. De xuat phan bo ngan sach dua tren he so Ridge

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
from sklearn.linear_model import ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
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
N_SAMPLES = 500
FEATURE_COLUMNS = ["TV", "Facebook", "Google"]
TARGET_COLUMN = "DoanhThu"
TOTAL_BUDGET_VND = 2_000_000_000
ALPHA_GRID = np.logspace(-3, 3, 50)
ALPHA_GRID_WIDE = np.logspace(-3, 4, 100)


def log(msg: str) -> None:
    print(f"[TT-12] {msg}")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ----------------------------------------------------------------------------
# 1. Sinh du lieu tong hop co da cong tuyen ro ret
# ----------------------------------------------------------------------------
def generate_data(n: int = N_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    # So voi ban goc trong README (fb_noise=15, target_noise=50): da tang do tuong quan
    # TV<->Facebook (r~0.999, VIF~600) va tang nhieu cua DoanhThu (std=400) de tao dieu
    # kien da cong tuyen + ty le tin hieu/nhieu du "kho" -> he so OLS moi that su dao dong
    # manh (kem theo doi dau) qua bootstrap, dung nhu README mo ta ("TikTok am 300 trieu").
    rng = np.random.default_rng(seed)
    tv = rng.uniform(50, 500, n)
    fb = tv * 0.6 + rng.normal(0, 3, n)  # tuong quan rat cao voi TV (r ~ 0.999)
    gg = rng.uniform(20, 300, n)
    doanh_thu = 3.2 * tv + 1.8 * fb + 2.5 * gg + rng.normal(0, 400, n)
    df = pd.DataFrame({"TV": tv, "Facebook": fb, "Google": gg, "DoanhThu": doanh_thu})
    log(f"  Sinh {len(df)} dong. Tuong quan TV<->Facebook: {df['TV'].corr(df['Facebook']):.3f}")
    return df


# ----------------------------------------------------------------------------
# 2. Ma tran tuong quan + VIF
# ----------------------------------------------------------------------------
def run_correlation_vif(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = df[FEATURE_COLUMNS].corr()
    corr.to_csv(REPORTS_DIR / "correlation_matrix.csv")

    X_const = add_constant(df[FEATURE_COLUMNS])
    vif_rows = [
        {"feature": col, "VIF": float(variance_inflation_factor(X_const.values, i))}
        for i, col in enumerate(X_const.columns) if col != "const"
    ]
    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
    vif_df.to_csv(REPORTS_DIR / "vif_table.csv", index=False)

    log("  VIF (>10 la dau hieu da cong tuyen manh):")
    for _, row in vif_df.iterrows():
        flag = " <-- CAO" if row["VIF"] > 10 else ""
        log(f"    {row['feature']:10s} VIF={row['VIF']:.2f}{flag}")
    return corr, vif_df


# ----------------------------------------------------------------------------
# 3. Linear Regression co ban -> he so
# ----------------------------------------------------------------------------
def run_linear_baseline(X_train, y_train, X_test, y_test) -> tuple[Pipeline, dict, pd.Series]:
    pipe = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())])
    pipe.fit(X_train, y_train)
    metrics = evaluate(y_test, pipe.predict(X_test))
    he_so = pd.Series(pipe.named_steps["lr"].coef_, index=FEATURE_COLUMNS)
    log(f"  Linear Regression: RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.4f}")
    log(f"  He so (da chuan hoa): {he_so.round(3).to_dict()}")
    return pipe, metrics, he_so


# ----------------------------------------------------------------------------
# 4. Thi nghiem on dinh: bootstrap Linear vs Ridge
# ----------------------------------------------------------------------------
def run_bootstrap_stability(X: pd.DataFrame, y: pd.Series, ridge_alpha: float,
                             n_boot: int = 100, frac: float = 0.8) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X)
    size = int(frac * n)
    linear_coefs, ridge_coefs = [], []

    for _ in range(n_boot):
        idx = rng.choice(n, size=size, replace=False)
        X_sample, y_sample = X.iloc[idx], y.iloc[idx]
        X_scaled = StandardScaler().fit_transform(X_sample)
        linear_coefs.append(LinearRegression().fit(X_scaled, y_sample).coef_)
        ridge_coefs.append(Ridge(alpha=ridge_alpha).fit(X_scaled, y_sample).coef_)

    linear_coefs = np.array(linear_coefs)
    ridge_coefs = np.array(ridge_coefs)

    std_table = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "std_linear": linear_coefs.std(axis=0),
        "std_ridge": ridge_coefs.std(axis=0),
    })
    std_table["ty_le_giam_do_lech_%"] = (1 - std_table["std_ridge"] / std_table["std_linear"]) * 100
    std_table.to_csv(REPORTS_DIR / "bootstrap_he_so.csv", index=False)

    n_features = len(FEATURE_COLUMNS)
    pos_linear = np.arange(n_features) * 2.5
    pos_ridge = pos_linear + 1.0

    fig, ax = plt.subplots(figsize=(9, 6))
    bp1 = ax.boxplot([linear_coefs[:, i] for i in range(n_features)], positions=pos_linear,
                      widths=0.8, patch_artist=True, boxprops=dict(facecolor="#c0392b", alpha=0.6))
    bp2 = ax.boxplot([ridge_coefs[:, i] for i in range(n_features)], positions=pos_ridge,
                      widths=0.8, patch_artist=True, boxprops=dict(facecolor="#2980b9", alpha=0.6))
    ax.set_xticks(pos_linear + 0.5)
    ax.set_xticklabels(FEATURE_COLUMNS)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("He so hoi quy (tren du lieu da chuan hoa)")
    ax.set_title(f"Do on dinh he so qua {n_boot} lan bootstrap (moi lan lay {int(frac*100)}% du lieu)")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
              ["Linear Regression", f"Ridge (alpha={ridge_alpha:.3g})"])
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "bootstrap_he_so.png", dpi=130)
    plt.close(fig)

    log("  Do lech chuan he so qua bootstrap (Linear vs Ridge):")
    for _, row in std_table.iterrows():
        log(f"    {row['feature']:10s} std_linear={row['std_linear']:.4f}  "
            f"std_ridge={row['std_ridge']:.4f}  giam {row['ty_le_giam_do_lech_%']:.1f}%")
    log("  Da luu bootstrap_he_so.csv, bootstrap_he_so.png")
    return std_table


# ----------------------------------------------------------------------------
# 5. RidgeCV do alpha
# ----------------------------------------------------------------------------
def run_ridgecv(X_train, y_train) -> tuple[Pipeline, float]:
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=ALPHA_GRID, cv=5)),
    ])
    pipe.fit(X_train, y_train)
    best_alpha = float(pipe.named_steps["ridge"].alpha_)
    log(f"  RidgeCV (cv=5): alpha toi uu = {best_alpha:.4f}")
    return pipe, best_alpha


# ----------------------------------------------------------------------------
# 6. Coefficient path
# ----------------------------------------------------------------------------
def run_coefficient_path(X_train, y_train) -> None:
    X_scaled = StandardScaler().fit_transform(X_train)
    paths = np.array([Ridge(alpha=a).fit(X_scaled, y_train).coef_ for a in ALPHA_GRID_WIDE])

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(FEATURE_COLUMNS):
        ax.plot(ALPHA_GRID_WIDE, paths[:, i], label=name, lw=2)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("alpha (thang log)")
    ax.set_ylabel("He so Ridge (tren du lieu da chuan hoa)")
    ax.set_title("Coefficient path: he so co dan ve 0 nhung khong bao gio dung 0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "coefficient_path.png", dpi=130)
    plt.close(fig)
    log("  Da luu coefficient_path.png")


# ----------------------------------------------------------------------------
# 7. RMSE train/test theo alpha
# ----------------------------------------------------------------------------
def run_rmse_vs_alpha(X_train, y_train, X_test, y_test) -> float:
    train_rmse, test_rmse = [], []
    for a in ALPHA_GRID_WIDE:
        pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=a))])
        pipe.fit(X_train, y_train)
        train_rmse.append(rmse(y_train, pipe.predict(X_train)))
        test_rmse.append(rmse(y_test, pipe.predict(X_test)))

    best_idx = int(np.argmin(test_rmse))
    best_alpha_by_test = float(ALPHA_GRID_WIDE[best_idx])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ALPHA_GRID_WIDE, train_rmse, label="RMSE train", color="#2980b9")
    ax.plot(ALPHA_GRID_WIDE, test_rmse, label="RMSE test", color="#c0392b")
    ax.axvline(best_alpha_by_test, color="#27ae60", ls="--", lw=1,
               label=f"alpha tot nhat tren test = {best_alpha_by_test:.3g}")
    ax.set_xscale("log")
    ax.set_xlabel("alpha (thang log)")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE train/test theo alpha - can bang bias-variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "rmse_theo_alpha.png", dpi=130)
    plt.close(fig)
    log(f"  Da luu rmse_theo_alpha.png (alpha tot nhat theo test = {best_alpha_by_test:.4f})")
    return best_alpha_by_test


# ----------------------------------------------------------------------------
# 8-9. So sanh Linear vs Ridge vs Lasso (TT-13) vs ElasticNet (TT-14)
# ----------------------------------------------------------------------------
def run_model_comparison(X_train, y_train, X_test, y_test, ridge_alpha: float) -> pd.DataFrame:
    models = {
        "Linear Regression": LinearRegression(),
        f"Ridge (alpha={ridge_alpha:.3g})": Ridge(alpha=ridge_alpha),
        "Lasso (TT-13, LassoCV)": LassoCV(alphas=np.logspace(-3, 2, 50), cv=5,
                                           random_state=RANDOM_STATE, max_iter=20000),
        "ElasticNet (TT-14, ElasticNetCV)": ElasticNetCV(
            alphas=np.logspace(-3, 2, 50), l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
            cv=5, random_state=RANDOM_STATE, max_iter=20000),
    }
    rows = []
    for name, model in models.items():
        pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
        pipe.fit(X_train, y_train)
        metrics = evaluate(y_test, pipe.predict(X_test))
        rows.append({"model": name, **metrics})

    comp = pd.DataFrame(rows)
    comp.to_csv(REPORTS_DIR / "so_sanh_mo_hinh.csv", index=False)
    log("  So sanh model (RMSE cang thap cang tot):")
    for _, row in comp.iterrows():
        log(f"    {row['model']:34s} RMSE={row['RMSE']:.3f}  R2={row['R2']:.4f}")
    log("  Da luu so_sanh_mo_hinh.csv")
    return comp


# ----------------------------------------------------------------------------
# 10. De xuat phan bo ngan sach dua tren he so Ridge
# ----------------------------------------------------------------------------
def propose_budget_allocation(ridge_pipe: Pipeline, total_budget: float = TOTAL_BUDGET_VND) -> pd.DataFrame:
    coefs = ridge_pipe.named_steps["ridge"].coef_
    positive = np.clip(coefs, 0, None)
    weights = positive / positive.sum() if positive.sum() > 0 else np.ones_like(positive) / len(positive)
    allocation = weights * total_budget

    df = pd.DataFrame({
        "kenh": FEATURE_COLUMNS,
        "he_so_ridge_chuan_hoa": coefs,
        "trong_so_%": weights * 100,
        "ngan_sach_de_xuat_vnd": allocation,
    }).sort_values("ngan_sach_de_xuat_vnd", ascending=False)
    df.to_csv(REPORTS_DIR / "phan_bo_ngan_sach.csv", index=False)

    log(f"  De xuat phan bo ngan sach {total_budget:,.0f} VND theo he so Ridge:")
    for _, row in df.iterrows():
        log(f"    {row['kenh']:10s} he_so={row['he_so_ridge_chuan_hoa']:.3f}  "
            f"trong_so={row['trong_so_%']:.1f}%  ngan_sach={row['ngan_sach_de_xuat_vnd']:,.0f} VND")
    log("  Da luu phan_bo_ngan_sach.csv")
    return df


def main() -> None:
    log("1. Sinh du lieu tong hop co da cong tuyen...")
    df = generate_data()

    log("2. Ma tran tuong quan + VIF...")
    corr, vif_df = run_correlation_vif(df)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    log(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    log("3. Linear Regression co ban...")
    linear_pipe, linear_metrics, linear_he_so = run_linear_baseline(X_train, y_train, X_test, y_test)

    log("5. RidgeCV do alpha (chay truoc de dung trong buoc 4)...")
    ridge_pipe, ridge_alpha = run_ridgecv(X_train, y_train)
    ridge_metrics = evaluate(y_test, ridge_pipe.predict(X_test))
    log(f"  Ridge (alpha={ridge_alpha:.4f}): RMSE={ridge_metrics['RMSE']:.3f}  R2={ridge_metrics['R2']:.4f}")

    log("4. Thi nghiem on dinh (bootstrap 100 lan, 80% du lieu)...")
    stability_table = run_bootstrap_stability(X, y, ridge_alpha=ridge_alpha)

    log("6. Coefficient path...")
    run_coefficient_path(X_train, y_train)

    log("7. RMSE train/test theo alpha...")
    best_alpha_by_test = run_rmse_vs_alpha(X_train, y_train, X_test, y_test)

    log("8-9. So sanh Linear / Ridge / Lasso / ElasticNet...")
    comparison = run_model_comparison(X_train, y_train, X_test, y_test, ridge_alpha)

    log("10. De xuat phan bo ngan sach theo he so Ridge...")
    budget_df = propose_budget_allocation(ridge_pipe)

    joblib.dump(ridge_pipe, MODELS_DIR / "ridge_pipeline.joblib")
    log("Da luu model -> models/ridge_pipeline.joblib")

    ridge_he_so = pd.Series(ridge_pipe.named_steps["ridge"].coef_, index=FEATURE_COLUMNS)
    summary = {
        "n_rows": int(len(df)),
        "corr_TV_Facebook": float(df["TV"].corr(df["Facebook"])),
        "vif": vif_df.to_dict(orient="records"),
        "linear_he_so_chuan_hoa": linear_he_so.round(4).to_dict(),
        "linear_metrics": linear_metrics,
        "ridge_alpha_ridgecv": ridge_alpha,
        "ridge_alpha_best_on_test": best_alpha_by_test,
        "ridge_he_so_chuan_hoa": ridge_he_so.round(4).to_dict(),
        "ridge_metrics": ridge_metrics,
        "bootstrap_stability": stability_table.to_dict(orient="records"),
        "model_comparison": comparison.to_dict(orient="records"),
        "de_xuat_ngan_sach": budget_df.to_dict(orient="records"),
    }
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("HOAN THANH. Xem ket qua chi tiet trong thu muc reports/.")


if __name__ == "__main__":
    main()
