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
 7. Ve duong RMSE train / CROSS-VALIDATION theo alpha (KHONG dung tap test de chon alpha)
 8. So sanh RMSE tren test: Dummy (mean) / Linear / Ridge
 9. So sanh voi Lasso (TT-13) va ElasticNet (TT-14)
10. ROI tren thang goc (doanh thu / 1 don vi chi) + khoang tin cay bootstrap, doi chieu he so THAT
    -> de xuat phan bo ngan sach chi dua tren nhung gi du lieu xac dinh duoc

Test (20%) chi dung de bao cao o buoc 8-9; moi lua chon (alpha) lam bang CV tren train.

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
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
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
CV_FOLDS = 5
# He so THAT dung de sinh du lieu (doanh thu / 1 don vi chi) - chi biet duoc vi du lieu tu sinh
TRUE_COEFS = {"TV": 3.2, "Facebook": 1.8, "Google": 2.5}
SHIFT_POINTS = 10  # dich toi da 10 diem % ngan sach giua 2 nhom neu bang chung du manh


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
    doanh_thu = (TRUE_COEFS["TV"] * tv + TRUE_COEFS["Facebook"] * fb + TRUE_COEFS["Google"] * gg
                 + rng.normal(0, 400, n))
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
    linear_coefs, ridge_coefs, sample_std = [], [], []

    for _ in range(n_boot):
        idx = rng.choice(n, size=size, replace=False)
        X_sample, y_sample = X.iloc[idx], y.iloc[idx]
        scaler = StandardScaler().fit(X_sample)
        X_scaled = scaler.transform(X_sample)
        linear_coefs.append(LinearRegression().fit(X_scaled, y_sample).coef_)
        ridge_coefs.append(Ridge(alpha=ridge_alpha).fit(X_scaled, y_sample).coef_)
        sample_std.append(scaler.scale_)

    linear_coefs = np.array(linear_coefs)
    ridge_coefs = np.array(ridge_coefs)
    # He so tren thang goc = he so chuan hoa / do lech chuan cua dac trung (doanh thu / 1 don vi chi)
    raw = {"Linear": linear_coefs / np.array(sample_std), "Ridge": ridge_coefs / np.array(sample_std)}

    std_table = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "std_linear": linear_coefs.std(axis=0),
        "std_ridge": ridge_coefs.std(axis=0),
    })
    std_table["ty_le_giam_do_lech_%"] = (1 - std_table["std_ridge"] / std_table["std_linear"]) * 100
    std_table["pct_lan_am_linear"] = (linear_coefs < 0).mean(axis=0) * 100
    std_table["pct_lan_am_ridge"] = (ridge_coefs < 0).mean(axis=0) * 100
    std_table["min_linear"], std_table["max_linear"] = linear_coefs.min(axis=0), linear_coefs.max(axis=0)
    std_table["min_ridge"], std_table["max_ridge"] = ridge_coefs.min(axis=0), ridge_coefs.max(axis=0)
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
    return std_table, raw


# ----------------------------------------------------------------------------
# 5. RidgeCV do alpha
# ----------------------------------------------------------------------------
def run_ridgecv(X_train, y_train) -> tuple[Pipeline, float]:
    pipe = Pipeline([
        ("scale", StandardScaler()),
        # scoring MSE: cung thuoc do voi duong RMSE-CV o buoc 7 (mac dinh RidgeCV cham bang R2)
        ("ridge", RidgeCV(alphas=ALPHA_GRID, cv=CV_FOLDS, scoring="neg_mean_squared_error")),
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
# 7. RMSE train / cross-validation theo alpha (chi dung TRAIN)
# ----------------------------------------------------------------------------
def run_rmse_vs_alpha(X_train, y_train, ridge_alpha: float) -> pd.DataFrame:
    """Duong bias-variance tren TRAIN: RMSE fit tren train va RMSE 5-fold CV. Tap test khong duoc dung."""
    # Cung cach RidgeCV nhin du lieu: scaler fit tren toan train, CV 5 fold khong xao tron
    X_scaled = StandardScaler().fit_transform(X_train)
    rows = []
    for a in ALPHA_GRID:
        model = Ridge(alpha=a)
        cv_mse = -cross_val_score(model, X_scaled, y_train, cv=CV_FOLDS, scoring="neg_mean_squared_error").mean()
        train_rmse = rmse(y_train, model.fit(X_scaled, y_train).predict(X_scaled))
        rows.append({"alpha": a, "rmse_train": train_rmse, "rmse_cv": float(np.sqrt(cv_mse))})
    curve = pd.DataFrame(rows)
    curve.to_csv(REPORTS_DIR / "rmse_theo_alpha.csv", index=False)
    best_cv_alpha = float(curve.loc[curve["rmse_cv"].idxmin(), "alpha"])
    assert np.isclose(best_cv_alpha, ridge_alpha), "duong CV phai co cuc tieu dung tai alpha cua RidgeCV"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(curve["alpha"], curve["rmse_train"], label="RMSE train", color="#2980b9")
    ax.plot(curve["alpha"], curve["rmse_cv"], label="RMSE cross-validation (5 fold, tren train)", color="#c0392b")
    ax.axvline(ridge_alpha, color="#27ae60", ls="--", lw=1, label=f"alpha chon bang CV = {ridge_alpha:.3g}")
    ax.set_xscale("log")
    ax.set_xlabel("alpha (thang log)")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE train / CV theo alpha - can bang bias-variance (khong dung tap test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "rmse_theo_alpha.png", dpi=130)
    plt.close(fig)
    ols_like = curve.iloc[0]
    best = curve.loc[curve["rmse_cv"].idxmin()]
    log(f"  RMSE-CV: alpha={ols_like['alpha']:.3g} -> {ols_like['rmse_cv']:.2f} | "
        f"alpha={best['alpha']:.3g} (chon) -> {best['rmse_cv']:.2f} | "
        f"alpha=1000 -> {curve.iloc[-1]['rmse_cv']:.2f}")
    log("  Da luu rmse_theo_alpha.png, rmse_theo_alpha.csv")
    return curve


# ----------------------------------------------------------------------------
# 8-9. So sanh Linear vs Ridge vs Lasso (TT-13) vs ElasticNet (TT-14)
# ----------------------------------------------------------------------------
def run_model_comparison(X_train, y_train, X_test, y_test, ridge_alpha: float) -> pd.DataFrame:
    models = {
        "Baseline Dummy (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        f"Ridge (alpha={ridge_alpha:.3g})": Ridge(alpha=ridge_alpha),
        "Lasso (TT-13, LassoCV)": LassoCV(alphas=np.logspace(-3, 2, 50), cv=CV_FOLDS,
                                           random_state=RANDOM_STATE, max_iter=20000),
        "ElasticNet (TT-14, ElasticNetCV)": ElasticNetCV(
            alphas=np.logspace(-3, 2, 50), l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
            cv=CV_FOLDS, random_state=RANDOM_STATE, max_iter=20000),
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
# 10a. ROI tren thang goc + khoang tin cay bootstrap, doi chieu he so THAT
# ----------------------------------------------------------------------------
def ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def run_roi_analysis(df: pd.DataFrame, linear_pipe: Pipeline, ridge_pipe: Pipeline,
                     raw_boot: dict) -> tuple[pd.DataFrame, dict]:
    """ROI = doanh thu tang them khi chi them 1 don vi o 1 kenh (he so tren thang goc).

    TV va Facebook gan nhu luon di cung nhau (Facebook ~ 0,6 x TV) nen du lieu KHONG tach duoc
    rieng ROI tung kenh, nhung xac dinh tot ROI cua ca "goi TV+Facebook" chia theo ty le lich su.
    """
    point = {
        "Linear": linear_pipe.named_steps["lr"].coef_ / linear_pipe.named_steps["scale"].scale_,
        "Ridge": ridge_pipe.named_steps["ridge"].coef_ / ridge_pipe.named_steps["scale"].scale_,
    }
    tv_share = float(df["TV"].mean() / (df["TV"].mean() + df["Facebook"].mean()))

    def package(c):  # ROI cua 1 don vi chi vao goi, chia TV:Facebook theo ty le lich su
        c = np.atleast_2d(c)
        return tv_share * c[:, 0] + (1 - tv_share) * c[:, 1]

    true = np.array([TRUE_COEFS[k] for k in FEATURE_COLUMNS])
    rows = []
    for i, name in enumerate(FEATURE_COLUMNS):
        lo_l, hi_l = ci(raw_boot["Linear"][:, i])
        lo_r, hi_r = ci(raw_boot["Ridge"][:, i])
        rows.append({"muc": name, "roi_that": true[i],
                     "roi_linear": point["Linear"][i], "ci95_linear": f"[{lo_l:.2f}; {hi_l:.2f}]",
                     "roi_ridge": point["Ridge"][i], "ci95_ridge": f"[{lo_r:.2f}; {hi_r:.2f}]",
                     "ridge_ci_chua_gia_tri_that": bool(lo_r <= true[i] <= hi_r)})
    pkg_boot = {k: package(v) for k, v in raw_boot.items()}
    pkg_true = float(package(true)[0])
    lo_l, hi_l = ci(pkg_boot["Linear"])
    lo_r, hi_r = ci(pkg_boot["Ridge"])
    rows.append({"muc": f"Goi TV+Facebook ({tv_share:.1%} TV)", "roi_that": pkg_true,
                 "roi_linear": float(package(point["Linear"])[0]), "ci95_linear": f"[{lo_l:.2f}; {hi_l:.2f}]",
                 "roi_ridge": float(package(point["Ridge"])[0]), "ci95_ridge": f"[{lo_r:.2f}; {hi_r:.2f}]",
                 "ridge_ci_chua_gia_tri_that": bool(lo_r <= pkg_true <= hi_r)})
    diff_boot = pkg_boot["Ridge"] - raw_boot["Ridge"][:, 2]
    diff_point = float(package(point["Ridge"])[0] - point["Ridge"][2])
    lo_d, hi_d = ci(diff_boot)
    rows.append({"muc": "Chenh lech ROI: goi - Google", "roi_that": pkg_true - true[2],
                 "roi_linear": float(package(point["Linear"])[0] - point["Linear"][2]), "ci95_linear": "",
                 "roi_ridge": diff_point, "ci95_ridge": f"[{lo_d:.2f}; {hi_d:.2f}]",
                 "ridge_ci_chua_gia_tri_that": bool(lo_d <= pkg_true - true[2] <= hi_d)})
    roi = pd.DataFrame(rows)
    roi.to_csv(REPORTS_DIR / "roi_kenh.csv", index=False)
    for _, r in roi.iterrows():
        log(f"    {r['muc']:28s} that={r['roi_that']:.2f}  linear={r['roi_linear']:.2f} {r['ci95_linear']:16s} "
            f"ridge={r['roi_ridge']:.2f} {r['ci95_ridge']}")
    log("  Da luu roi_kenh.csv")
    return roi, {"tv_share": tv_share, "diff_ci": (lo_d, hi_d), "diff_point": diff_point}


# ----------------------------------------------------------------------------
# 10b. De xuat phan bo ngan sach
# ----------------------------------------------------------------------------
def propose_budget_allocation(df: pd.DataFrame, ridge_pipe: Pipeline, roi_info: dict,
                              total_budget: float = TOTAL_BUDGET_VND) -> pd.DataFrame:
    """So sanh 3 phuong an chia ngan sach; doanh thu ky vong tinh bang he so Ridge VA he so that.

    Mo hinh tuyen tinh co ROI khong doi -> toi uu ly thuyet la don het vao 1 kenh (vo ly vi thuc te
    co bao hoa). Vi vay chi DICH TOI DA SHIFT_POINTS diem % giua 2 nhom ma du lieu phan biet duoc
    (goi TV+Facebook vs Google), va CHI khi khoang tin cay cua chenh lech ROI khong chua 0.
    """
    hist = df[FEATURE_COLUMNS].mean()
    hist_share = (hist / hist.sum()).values
    tv_share = roi_info["tv_share"]
    lo, hi = roi_info["diff_ci"]

    coefs_std = ridge_pipe.named_steps["ridge"].coef_
    old_share = np.clip(coefs_std, 0, None) / np.clip(coefs_std, 0, None).sum()

    pkg_share, gg_share = hist_share[0] + hist_share[1], hist_share[2]
    if lo > 0:
        shift, ly_do = SHIFT_POINTS / 100, "CI chenh lech ROI > 0 -> dich sang goi TV+Facebook"
    elif hi < 0:
        shift, ly_do = -SHIFT_POINTS / 100, "CI chenh lech ROI < 0 -> dich sang Google"
    else:
        shift, ly_do = 0.0, "CI chenh lech ROI chua 0 -> chua du bang chung de dich, giu co cau"
    pkg_new = pkg_share + shift
    new_share = np.array([pkg_new * tv_share, pkg_new * (1 - tv_share), gg_share - shift])

    roi_ridge = coefs_std / ridge_pipe.named_steps["scale"].scale_
    roi_true = np.array([TRUE_COEFS[k] for k in FEATURE_COLUMNS])
    plans = {
        "Co cau hien tai (trung binh lich su)": hist_share,
        "Cach cu: ty le theo he so Ridge CHUAN HOA": old_share,
        f"De xuat ({ly_do})": new_share,
    }
    rows = []
    for name, share in plans.items():
        units = share * total_budget / 1e6  # 1 don vi chi = 1 trieu VND
        rows.append({"phuong_an": name,
                     **{f"{k}_%": share[i] * 100 for i, k in enumerate(FEATURE_COLUMNS)},
                     **{f"{k}_trieu_vnd": units[i] for i, k in enumerate(FEATURE_COLUMNS)},
                     "doanh_thu_tang_them_theo_ridge": float(units @ roi_ridge),
                     "doanh_thu_tang_them_theo_he_so_that": float(units @ roi_true)})
    plan_df = pd.DataFrame(rows)
    plan_df.to_csv(REPORTS_DIR / "phan_bo_ngan_sach.csv", index=False)
    log(f"  Chenh lech ROI goi-Google (Ridge) = {roi_info['diff_point']:.2f}, CI95 [{lo:.2f}; {hi:.2f}] -> {ly_do}")
    for _, r in plan_df.iterrows():
        log(f"    {r['phuong_an'][:44]:44s} TV={r['TV_%']:.1f}% FB={r['Facebook_%']:.1f}% GG={r['Google_%']:.1f}%  "
            f"DT(ridge)={r['doanh_thu_tang_them_theo_ridge']:,.0f}  DT(that)={r['doanh_thu_tang_them_theo_he_so_that']:,.0f}")
    log("  Da luu phan_bo_ngan_sach.csv")
    return plan_df


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
    stability_table, raw_boot = run_bootstrap_stability(X, y, ridge_alpha=ridge_alpha)

    log("6. Coefficient path...")
    run_coefficient_path(X_train, y_train)

    log("7. RMSE train / cross-validation theo alpha (chi dung train)...")
    rmse_curve = run_rmse_vs_alpha(X_train, y_train, ridge_alpha)

    log("8-9. So sanh Linear / Ridge / Lasso / ElasticNet...")
    comparison = run_model_comparison(X_train, y_train, X_test, y_test, ridge_alpha)

    log("10a. ROI tren thang goc + CI bootstrap, doi chieu he so that...")
    roi_df, roi_info = run_roi_analysis(df, linear_pipe, ridge_pipe, raw_boot)

    log("10b. De xuat phan bo ngan sach...")
    budget_df = propose_budget_allocation(df, ridge_pipe, roi_info)

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
        "rmse_cv_tai_alpha_chon": float(rmse_curve["rmse_cv"].min()),
        "ridge_he_so_chuan_hoa": ridge_he_so.round(4).to_dict(),
        "ridge_metrics": ridge_metrics,
        "bootstrap_stability": stability_table.to_dict(orient="records"),
        "model_comparison": comparison.to_dict(orient="records"),
        "roi_kenh": roi_df.to_dict(orient="records"),
        "de_xuat_ngan_sach": budget_df.to_dict(orient="records"),
    }
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("HOAN THANH. Xem ket qua chi tiet trong thu muc reports/.")


if __name__ == "__main__":
    main()
