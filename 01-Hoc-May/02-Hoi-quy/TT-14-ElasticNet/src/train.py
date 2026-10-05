"""
TT-14 - ElasticNet (L1 + L2): Du bao tai suoi/lam mat toa nha khi cac bien thiet ke dinh chat nhau.

Pipeline day du (theo 10 buoc trong README.md cap de):
 1. Nap du lieu Energy Efficiency (UCI), tinh ma tran tuong quan + VIF -> xac nhan da cong
    tuyen nang giua X1, X2, X4, X5
 2. One-hot X6 (huong nha), X8 (phan bo kinh); chuan hoa cac bien so
 3. Baseline: DummyRegressor + Linear Regression
 4. Chay 3 model tren CUNG du lieu: Ridge - Lasso - ElasticNet (moi model tu do alpha/l1_ratio
    bang CV)
 5. Bang so sanh: moi model giu bao nhieu bien, RMSE bao nhieu
 6. Kiem chung HIEU UNG GOM NHOM: Lasso giu bien nao trong nhom tuong quan {X1,X2,X4,X5}?
    ElasticNet giu may bien trong nhom do?
 7. Ve heatmap RMSE CROSS-VALIDATION theo luoi (alpha x l1_ratio) - lay tu mse_path_ cua ElasticNetCV,
    khong dung tap test
 8. Lam ca hai nhan Y1 (tai suoi) va Y2 (tai lam mat) -> so sanh bien nao quan trong cho tung nhan
 9. Kiem tra on dinh: bootstrap 100 lan -> he so ElasticNet dao dong bao nhieu
10. Xep hang bien theo do quan trong (|he so| chuan hoa)
11. DON BAY THIET KE: du lieu la mo phong giai thua can bang (12 hinh khoi x 4 huong x 16 to hop kinh)
    -> trung binh theo tung muc la uoc luong sach tac dong cua tung quyet dinh thiet ke
    -> quy ra kWh/nam va tien dien; doi chieu cho mo hinh tuyen tinh sai lech (phan du theo nhom)

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
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42
DATA_URL = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"
DATA_XLSX = DATA_DIR / "ENB2012_data.xlsx"

FEATURE_NUMERIC = ["X1", "X2", "X3", "X4", "X5", "X7"]
FEATURE_CATEGORICAL = ["X6", "X8"]
TARGETS = [("Y1", "tai_suoi"), ("Y2", "tai_lam_mat")]
GROUP_TUONG_QUAN = ["X1", "X2", "X4", "X5"]  # nhom bien dinh chat nhau ve hinh hoc

FEATURE_DESC = {
    "X1": "Do gon tuong doi", "X2": "Dien tich be mat", "X3": "Dien tich tuong",
    "X4": "Dien tich mai", "X5": "Chieu cao tong", "X7": "Dien tich kinh",
}

ALPHA_GRID = np.logspace(-4, 1, 100)
ALPHA_GRID_HEATMAP = np.logspace(-4, 1, 25)
L1_RATIO_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

# Quy doi nghiep vu (gia dinh - ghi ro trong README)
VOLUME_M3 = 771.75                    # moi toa nha trong bo du lieu co cung the tich
FLOOR_AREA_M2 = VOLUME_M3 / 3.5       # = 220,5 m2 san (1 tang 3,5 m hoac 2 tang)
COP = 3.0                             # he so hieu qua may dieu hoa / bom nhiet: 1 kWh dien -> 3 kWh nhiet
ELECTRICITY_VND_PER_KWH = 2_500       # gia dien tham khao


def log(msg: str) -> None:
    print(f"[TT-14] {msg}")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ----------------------------------------------------------------------------
# 1a. Tai du lieu Energy Efficiency (UCI), cache vao data/
# ----------------------------------------------------------------------------
def download_data() -> Path:
    if DATA_XLSX.exists():
        return DATA_XLSX
    log(f"  Dang tai du lieu tu UCI: {DATA_URL}")
    with urllib.request.urlopen(DATA_URL, timeout=30) as r:
        raw = r.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        DATA_XLSX.write_bytes(z.read("ENB2012_data.xlsx"))
    log(f"  Da luu du lieu vao {DATA_XLSX.relative_to(ROOT)}")
    return DATA_XLSX


def load_data() -> pd.DataFrame:
    path = download_data()
    df = pd.read_excel(path)
    df = df[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "Y1", "Y2"]].copy()
    log(f"  Du lieu: {df.shape[0]} dong x {df.shape[1]} cot")
    return df


# ----------------------------------------------------------------------------
# 1b. Ma tran tuong quan + VIF -> xac nhan da cong tuyen
# ----------------------------------------------------------------------------
def run_correlation_vif(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = df[FEATURE_NUMERIC].corr()
    corr.to_csv(REPORTS_DIR / "correlation_matrix.csv")

    X_const = add_constant(df[FEATURE_NUMERIC])
    vif_rows = [
        {"feature": col, "mo_ta": FEATURE_DESC.get(col, col),
         "VIF": float(variance_inflation_factor(X_const.values, i))}
        for i, col in enumerate(X_const.columns) if col != "const"
    ]
    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
    vif_df.to_csv(REPORTS_DIR / "vif_table.csv", index=False)

    log("  VIF (>10 la dau hieu da cong tuyen manh):")
    for _, row in vif_df.iterrows():
        flag = " <-- RAT CAO" if row["VIF"] > 10 else ""
        log(f"    {row['feature']:4s} ({row['mo_ta']:20s}) VIF={row['VIF']:>10.2f}{flag}")

    high_corr_pairs = []
    for i, a in enumerate(FEATURE_NUMERIC):
        for b in FEATURE_NUMERIC[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) > 0.8:
                high_corr_pairs.append((a, b, float(r)))
    log(f"  Cac cap |r| > 0.8: {high_corr_pairs}")
    return corr, vif_df


# ----------------------------------------------------------------------------
# 2. One-hot X6, X8 (bien phan loai ma hoa bang so)
# ----------------------------------------------------------------------------
def make_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(df[FEATURE_NUMERIC + FEATURE_CATEGORICAL],
                        columns=FEATURE_CATEGORICAL, drop_first=True)
    return X.astype(float)


# ----------------------------------------------------------------------------
# 3. Baseline: DummyRegressor + Linear Regression
# ----------------------------------------------------------------------------
def run_baseline(X_train, y_train, X_test, y_test, label: str) -> dict:
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    dummy_metrics = evaluate(y_test, dummy.predict(X_test))

    lr_pipe = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())])
    lr_pipe.fit(X_train, y_train)
    lr_metrics = evaluate(y_test, lr_pipe.predict(X_test))

    log(f"  [{label}] Baseline Dummy (du doan = trung binh): RMSE={dummy_metrics['RMSE']:.3f}  R2={dummy_metrics['R2']:.4f}")
    log(f"  [{label}] Baseline Linear Regression:            RMSE={lr_metrics['RMSE']:.3f}  R2={lr_metrics['R2']:.4f}")
    return {"Dummy": dummy_metrics, "Linear": lr_metrics}


# ----------------------------------------------------------------------------
# 4. Ridge - Lasso - ElasticNet tren cung du lieu
# ----------------------------------------------------------------------------
def run_three_models(X_train, y_train, X_test, y_test, feature_names: list[str], label: str) -> dict:
    results = {}

    ridge_pipe = Pipeline([("scale", StandardScaler()),
                            ("ridge", RidgeCV(alphas=ALPHA_GRID, cv=5))])
    ridge_pipe.fit(X_train, y_train)
    ridge_coefs = pd.Series(ridge_pipe.named_steps["ridge"].coef_, index=feature_names)
    results["Ridge"] = {
        "alpha": float(ridge_pipe.named_steps["ridge"].alpha_), "l1_ratio": None,
        "coefs": ridge_coefs, "metrics": evaluate(y_test, ridge_pipe.predict(X_test)),
        "n_kept": int((ridge_coefs.abs() > 1e-8).sum()), "pipe": ridge_pipe,
    }

    lasso_pipe = Pipeline([("scale", StandardScaler()),
                            ("lasso", LassoCV(alphas=ALPHA_GRID, cv=5, max_iter=50_000, random_state=RANDOM_STATE))])
    lasso_pipe.fit(X_train, y_train)
    lasso_coefs = pd.Series(lasso_pipe.named_steps["lasso"].coef_, index=feature_names)
    results["Lasso"] = {
        "alpha": float(lasso_pipe.named_steps["lasso"].alpha_), "l1_ratio": 1.0,
        "coefs": lasso_coefs, "metrics": evaluate(y_test, lasso_pipe.predict(X_test)),
        "n_kept": int((lasso_coefs != 0).sum()), "pipe": lasso_pipe,
    }

    en_pipe = Pipeline([("scale", StandardScaler()),
                         ("en", ElasticNetCV(l1_ratio=L1_RATIO_GRID, alphas=ALPHA_GRID, cv=5,
                                              max_iter=50_000, random_state=RANDOM_STATE))])
    en_pipe.fit(X_train, y_train)
    en_coefs = pd.Series(en_pipe.named_steps["en"].coef_, index=feature_names)
    results["ElasticNet"] = {
        "alpha": float(en_pipe.named_steps["en"].alpha_), "l1_ratio": float(en_pipe.named_steps["en"].l1_ratio_),
        "coefs": en_coefs, "metrics": evaluate(y_test, en_pipe.predict(X_test)),
        "n_kept": int((en_coefs != 0).sum()), "pipe": en_pipe,
    }

    for name, r in results.items():
        l1_str = f" l1_ratio={r['l1_ratio']:.2f}" if r["l1_ratio"] is not None else ""
        log(f"  [{label}] {name:10s} alpha={r['alpha']:.5f}{l1_str}  n_kept={r['n_kept']}/{len(feature_names)}"
            f"  RMSE={r['metrics']['RMSE']:.3f}  R2={r['metrics']['R2']:.4f}")
    return results


# ----------------------------------------------------------------------------
# 5. Bang so sanh 3 model (+ baseline de doi chieu)
# ----------------------------------------------------------------------------
def build_comparison_table(results: dict, baseline: dict, feature_names: list[str], label: str) -> pd.DataFrame:
    rows = []
    for name, m in baseline.items():
        rows.append({"model": name, "alpha": None, "l1_ratio": None, "so_bien_giu": len(feature_names), **m})
    for name, r in results.items():
        rows.append({"model": name, "alpha": r["alpha"], "l1_ratio": r["l1_ratio"],
                      "so_bien_giu": r["n_kept"], **r["metrics"]})
    comp = pd.DataFrame(rows)
    comp.to_csv(REPORTS_DIR / f"so_sanh_3_model_{label}.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#95a5a6", "#7f8c8d", "#2980b9", "#c0392b", "#8e44ad"]
    ax1.bar(comp["model"], comp["RMSE"], color=colors[:len(comp)])
    ax1.set_ylabel("RMSE (test)")
    ax1.set_title(f"RMSE theo model - nhan {label}")
    ax1.tick_params(axis="x", rotation=25)
    ax2.bar(comp["model"], comp["so_bien_giu"], color=colors[:len(comp)])
    ax2.set_ylabel(f"So bien giu lai (/ {len(feature_names)})")
    ax2.set_title(f"So bien duoc giu - nhan {label}")
    ax2.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"so_sanh_3_model_{label}.png", dpi=130)
    plt.close(fig)

    log(f"  [{label}] Da luu so_sanh_3_model_{label}.csv/.png")
    return comp


# ----------------------------------------------------------------------------
# 6. Kiem chung hieu ung gom nhom
# ----------------------------------------------------------------------------
def check_grouping_effect(results: dict, label: str) -> pd.DataFrame:
    lasso_coefs, en_coefs, ridge_coefs = results["Lasso"]["coefs"], results["ElasticNet"]["coefs"], results["Ridge"]["coefs"]
    rows = []
    for f in GROUP_TUONG_QUAN:
        rows.append({
            "bien": f, "mo_ta": FEATURE_DESC.get(f, f),
            "he_so_ridge": ridge_coefs[f], "he_so_lasso": lasso_coefs[f], "he_so_elasticnet": en_coefs[f],
            "lasso_giu": bool(lasso_coefs[f] != 0), "elasticnet_giu": bool(en_coefs[f] != 0),
        })
    df = pd.DataFrame(rows)

    lasso_kept = df.loc[df["lasso_giu"], "bien"].tolist()
    en_kept = df.loc[df["elasticnet_giu"], "bien"].tolist()
    log(f"  [{label}] Nhom tuong quan {GROUP_TUONG_QUAN}:")
    log(f"    Lasso giu:      {lasso_kept}  ({len(lasso_kept)}/{len(GROUP_TUONG_QUAN)})")
    log(f"    ElasticNet giu: {en_kept}  ({len(en_kept)}/{len(GROUP_TUONG_QUAN)})")
    if len(en_kept) > len(lasso_kept):
        log(f"    -> XAC NHAN hieu ung gom nhom: ElasticNet giu nhieu bien tuong quan hon Lasso.")
    else:
        log(f"    -> Tai alpha toi uu rieng cua tung model (rat nho, vi p=14 << n=614), CA HAI giu du "
            f"ca 4 bien -> chua thay ro hieu ung gom nhom. Can EP alpha tang len (xem thi nghiem ben duoi).")
    return df


# ----------------------------------------------------------------------------
# 6b. EP alpha tang dan (vuot alpha toi uu) -> buoc lo hieu ung gom nhom
# ----------------------------------------------------------------------------
def run_grouping_effect_forced(X_train, y_train, feature_names: list[str], label: str) -> tuple[pd.DataFrame, float]:
    alphas = np.logspace(-3, 2, 60)
    X_scaled = StandardScaler().fit_transform(X_train)
    group_idx = [feature_names.index(f) for f in GROUP_TUONG_QUAN]

    lasso_group_kept, en_group_kept, lasso_total, en_total = [], [], [], []
    for a in alphas:
        lasso = Lasso(alpha=a, max_iter=50_000, random_state=RANDOM_STATE).fit(X_scaled, y_train)
        en = ElasticNet(alpha=a, l1_ratio=0.5, max_iter=50_000, random_state=RANDOM_STATE).fit(X_scaled, y_train)
        lasso_group_kept.append(int((np.abs(lasso.coef_[group_idx]) > 1e-8).sum()))
        en_group_kept.append(int((np.abs(en.coef_[group_idx]) > 1e-8).sum()))
        lasso_total.append(int((np.abs(lasso.coef_) > 1e-8).sum()))
        en_total.append(int((np.abs(en.coef_) > 1e-8).sum()))

    df = pd.DataFrame({
        "alpha": alphas, "lasso_so_bien_trong_nhom": lasso_group_kept,
        "elasticnet_so_bien_trong_nhom": en_group_kept,
        "lasso_tong_so_bien": lasso_total, "elasticnet_tong_so_bien": en_total,
    })
    df.to_csv(REPORTS_DIR / f"hieu_ung_gom_nhom_theo_alpha_{label}.csv", index=False)

    mask = df["lasso_so_bien_trong_nhom"] < len(GROUP_TUONG_QUAN)
    alpha_star = float(df.loc[mask, "alpha"].iloc[0]) if mask.any() else float(alphas[-1])
    row_star = df.loc[df["alpha"] == alpha_star].iloc[0]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, lasso_group_kept, label="Lasso (l1_ratio=1.0)", color="#c0392b", lw=2)
    ax.plot(alphas, en_group_kept, label="ElasticNet (l1_ratio=0.5)", color="#27ae60", lw=2)
    ax.axvline(alpha_star, color="gray", ls="--",
               label=f"alpha={alpha_star:.3g} (Lasso vua bat dau loai bien trong nhom)")
    ax.set_xscale("log")
    ax.set_ylim(-0.3, len(GROUP_TUONG_QUAN) + 0.3)
    ax.set_yticks(range(len(GROUP_TUONG_QUAN) + 1))
    ax.set_xlabel("alpha (thang log)")
    ax.set_ylabel(f"So bien con lai trong nhom {GROUP_TUONG_QUAN}")
    ax.set_title(f"Ep alpha tang de buoc lo hieu ung gom nhom - nhan {label}\n"
                 f"(alpha CV-toi-uu rat nho nen o do ca 2 model deu giu du 4 bien)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"hieu_ung_gom_nhom_theo_alpha_{label}.png", dpi=130)
    plt.close(fig)

    log(f"  [{label}] Tai alpha={alpha_star:.4g} (Lasso vua bat dau loai bien trong nhom):")
    log(f"    Lasso giu {int(row_star['lasso_so_bien_trong_nhom'])}/4 bien trong nhom "
        f"(tong {int(row_star['lasso_tong_so_bien'])}/{len(feature_names)} bien)")
    log(f"    ElasticNet(l1_ratio=0.5) giu {int(row_star['elasticnet_so_bien_trong_nhom'])}/4 bien trong nhom "
        f"(tong {int(row_star['elasticnet_tong_so_bien'])}/{len(feature_names)} bien)")
    log(f"  [{label}] Da luu hieu_ung_gom_nhom_theo_alpha_{label}.csv/.png")
    return df, alpha_star


# ----------------------------------------------------------------------------
# 7. Heatmap RMSE cross-validation theo luoi alpha x l1_ratio (chi dung TRAIN)
# ----------------------------------------------------------------------------
def run_heatmap_cv(en_pipe: Pipeline, label: str) -> tuple[pd.DataFrame, tuple]:
    """RMSE 5-fold CV tren train cho moi (l1_ratio, alpha) - lay thang tu mse_path_ cua ElasticNetCV,
    tuc chinh cac diem CV da dung de chon sieu tham so. Tap test khong duoc dung."""
    en = en_pipe.named_steps["en"]
    l1_ratios = np.atleast_1d(en.l1_ratio)
    alphas = np.atleast_2d(en.alphas_)[0]                      # giam dan, giong nhau cho moi l1_ratio
    rmse_grid = np.sqrt(en.mse_path_.mean(axis=-1))            # (n_l1, n_alpha)
    order = np.argsort(alphas)
    alphas, rmse_grid = alphas[order], rmse_grid[:, order]

    bi, bj = np.unravel_index(int(np.argmin(rmse_grid)), rmse_grid.shape)
    best = (float(alphas[bj]), float(l1_ratios[bi]), float(rmse_grid[bi, bj]))
    assert np.isclose(best[0], en.alpha_) and np.isclose(best[1], en.l1_ratio_), "o tot nhat phai trung lua chon CV"

    grid_df = pd.DataFrame(rmse_grid, index=[f"l1_ratio={l:.2f}" for l in l1_ratios],
                           columns=[f"{a:.4g}" for a in alphas])
    grid_df.to_csv(REPORTS_DIR / f"heatmap_cv_{label}.csv")

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(rmse_grid, aspect="auto", origin="lower", cmap="viridis_r")
    ticks = list(range(0, len(alphas), 11))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{alphas[t]:.3g}" for t in ticks], rotation=45, ha="right")
    ax.set_yticks(range(len(l1_ratios)))
    ax.set_yticklabels([f"{l:.2f}" for l in l1_ratios])
    ax.set_xlabel("alpha")
    ax.set_ylabel("l1_ratio")
    ax.plot(bj, bi, marker="*", color="red", markersize=18, markeredgecolor="white")
    ax.set_title(f"RMSE cross-validation (5 fold, tren train) theo alpha x l1_ratio - nhan {label}\n"
                 f"Chon: alpha={best[0]:.3g}, l1_ratio={best[1]:.2f}, RMSE-CV={best[2]:.3f} (sao do)")
    fig.colorbar(im, ax=ax, label="RMSE CV")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"heatmap_alpha_l1ratio_{label}.png", dpi=130)
    plt.close(fig)

    flat_alpha = int(np.searchsorted(alphas, 0.1))
    log(f"  [{label}] RMSE-CV tot nhat={best[2]:.3f} tai alpha={best[0]:.3g}, l1_ratio={best[1]:.2f}; "
        f"tai alpha~0.1: {rmse_grid[:, flat_alpha].min():.3f}-{rmse_grid[:, flat_alpha].max():.3f}; "
        f"tai alpha=10: l1=0.1 -> {rmse_grid[0, -1]:.3f}, l1=1.0 -> {rmse_grid[-1, -1]:.3f}")
    log(f"  [{label}] Da luu heatmap_alpha_l1ratio_{label}.png, heatmap_cv_{label}.csv")
    return grid_df, best


# ----------------------------------------------------------------------------
# 9. Bootstrap 100 lan -> do on dinh he so ElasticNet
# ----------------------------------------------------------------------------
def run_bootstrap_stability(X: pd.DataFrame, y: pd.Series, alpha: float, l1_ratio: float,
                             feature_names: list[str], label: str, n_boot: int = 100, frac: float = 0.8) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X)
    size = int(frac * n)
    coefs_list = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=size, replace=False)
        X_s, y_s = X.iloc[idx], y.iloc[idx]
        pipe = Pipeline([("scale", StandardScaler()),
                          ("en", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50_000, random_state=RANDOM_STATE))])
        pipe.fit(X_s, y_s)
        coefs_list.append(pipe.named_steps["en"].coef_)
    coefs_arr = np.array(coefs_list)

    mean_abs = np.abs(coefs_arr.mean(axis=0))
    std = coefs_arr.std(axis=0)
    cv_pct = np.where(mean_abs > 1e-6, std / mean_abs * 100, np.nan)
    table = pd.DataFrame({
        "bien": feature_names, "he_so_trung_binh": coefs_arr.mean(axis=0),
        "do_lech_chuan": std, "he_so_bien_thien_%": cv_pct,
    }).sort_values("he_so_trung_binh", key=np.abs, ascending=False)
    table.to_csv(REPORTS_DIR / f"bootstrap_elasticnet_{label}.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    order = table["bien"].tolist()
    idx_order = [feature_names.index(f) for f in order]
    ax.boxplot([coefs_arr[:, i] for i in idx_order], vert=False, labels=order)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("He so ElasticNet (tren du lieu da chuan hoa)")
    ax.set_title(f"Do on dinh he so qua {n_boot} lan bootstrap (80% du lieu) - nhan {label}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"bootstrap_elasticnet_{label}.png", dpi=130)
    plt.close(fig)

    log(f"  [{label}] He so bien thien (%) cao nhat: "
        f"{table.iloc[0]['bien']}={table.iloc[0]['he_so_bien_thien_%']:.1f}%, "
        f"thap nhat on dinh nhat: {table.iloc[-1]['bien']}={table.iloc[-1]['he_so_bien_thien_%']:.1f}%")
    log(f"  [{label}] Da luu bootstrap_elasticnet_{label}.csv/.png")
    return table


# ----------------------------------------------------------------------------
# 8. So sanh bien quan trong giua Y1 va Y2
# ----------------------------------------------------------------------------
def compare_y1_y2(results_y1: dict, results_y2: dict, feature_names: list[str]) -> pd.DataFrame:
    en1, en2 = results_y1["ElasticNet"]["coefs"], results_y2["ElasticNet"]["coefs"]
    df = pd.DataFrame({
        "bien": feature_names,
        "he_so_Y1_tai_suoi": en1.values,
        "he_so_Y2_tai_lam_mat": en2.values,
    })
    df["chenh_lech_tuyet_doi"] = (df["he_so_Y1_tai_suoi"] - df["he_so_Y2_tai_lam_mat"]).abs()
    df = df.sort_values("chenh_lech_tuyet_doi", ascending=False).reset_index(drop=True)
    df.to_csv(REPORTS_DIR / "so_sanh_Y1_Y2_tam_quan_trong.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(feature_names))
    width = 0.38
    order = df["bien"].tolist()
    y1_vals = [df.loc[df["bien"] == f, "he_so_Y1_tai_suoi"].values[0] for f in order]
    y2_vals = [df.loc[df["bien"] == f, "he_so_Y2_tai_lam_mat"].values[0] for f in order]
    ax.bar(x - width / 2, y1_vals, width, label="Y1 - tai suoi", color="#c0392b")
    ax.bar(x + width / 2, y2_vals, width, label="Y2 - tai lam mat", color="#2980b9")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("He so ElasticNet (tren du lieu da chuan hoa)")
    ax.set_title("So sanh he so ElasticNet: Y1 (tai suoi) vs Y2 (tai lam mat)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "so_sanh_Y1_Y2_tam_quan_trong.png", dpi=130)
    plt.close(fig)

    log("  Chenh lech he so lon nhat giua Y1 va Y2 (bien co tac dung khac nhau ro ret giua suoi/lam mat):")
    for _, row in df.head(3).iterrows():
        log(f"    {row['bien']:10s} Y1={row['he_so_Y1_tai_suoi']:+.4f}  Y2={row['he_so_Y2_tai_lam_mat']:+.4f}")
    log("  Da luu so_sanh_Y1_Y2_tam_quan_trong.csv/.png")
    return df


# ----------------------------------------------------------------------------
# 10. Xep hang bien theo do quan trong (lam co so de xuat thiet ke)
# ----------------------------------------------------------------------------
def rank_feature_importance(results: dict, feature_names: list[str], label: str) -> pd.DataFrame:
    en_coefs = results["ElasticNet"]["coefs"]
    df = pd.DataFrame({"bien": feature_names, "he_so_elasticnet": en_coefs.values})
    df["do_quan_trong_tuyet_doi"] = df["he_so_elasticnet"].abs()
    df = df.sort_values("do_quan_trong_tuyet_doi", ascending=False).reset_index(drop=True)
    df.to_csv(REPORTS_DIR / f"tam_quan_trong_bien_{label}.csv", index=False)
    log(f"  [{label}] Top 3 bien quan trong nhat: "
        + ", ".join(f"{r.bien}({r.he_so_elasticnet:+.3f})" for r in df.head(3).itertuples()))
    return df


# ----------------------------------------------------------------------------
# 11. Don bay thiet ke: tac dong cua tung quyet dinh, quy ra kWh/nam va tien dien
# ----------------------------------------------------------------------------
def _lever_row(name, change, sub_from: pd.DataFrame, sub_to: pd.DataFrame) -> dict:
    y1a, y2a, y1b, y2b = sub_from["Y1"].mean(), sub_from["Y2"].mean(), sub_to["Y1"].mean(), sub_to["Y2"].mean()
    d_total = (y1a + y2a) - (y1b + y2b)                       # giam tai (kWh/m2) khi doi tu "from" -> "to"
    kwh_heat = d_total * FLOOR_AREA_M2
    return {
        "don_bay": name, "thay_doi": change,
        "Y1_truoc": y1a, "Y1_sau": y1b, "giam_Y1_%": (1 - y1b / y1a) * 100,
        "Y2_truoc": y2a, "Y2_sau": y2b, "giam_Y2_%": (1 - y2b / y2a) * 100,
        "giam_tong_tai_kwh_m2": d_total,
        "giam_tai_nhiet_kwh_moi_toa": kwh_heat,
        "tiet_kiem_dien_kwh_moi_toa": kwh_heat / COP,
        "tiet_kiem_vnd_moi_toa": kwh_heat / COP * ELECTRICITY_VND_PER_KWH,
    }


def run_design_levers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bo du lieu la thi nghiem mo phong GIAI THUA CAN BANG: moi hinh khoi duoc mo phong voi moi huong
    va moi to hop kinh. Vi vay so sanh trung binh theo tung muc cua 1 quyet dinh (giu nguyen phan bo cac
    quyet dinh khac) la uoc luong sach tac dong cua quyet dinh do - khong bi da cong tuyen lam nhieu
    nhu he so hoi quy. Rieng nhom hinh hoc (X1-X5) chi co 12 hinh khoi, nen so sanh o muc HINH KHOI."""
    glazed = df[df["X7"] > 0]
    rows = [
        _lever_row("Chieu cao (X5)", "7 m (2 tang) -> 3,5 m (1 tang), cung the tich",
                   df[df["X5"] == 7.0], df[df["X5"] == 3.5]),
        _lever_row("Dien tich kinh (X7)", "40% -> 25% dien tich san", df[df["X7"] == 0.40], df[df["X7"] == 0.25]),
        _lever_row("Dien tich kinh (X7)", "25% -> 10% dien tich san", df[df["X7"] == 0.25], df[df["X7"] == 0.10]),
        _lever_row("Dien tich kinh (X7)", "40% -> 10% dien tich san", df[df["X7"] == 0.40], df[df["X7"] == 0.10]),
    ]
    for height, ten in [(3.5, "1 tang"), (7.0, "2 tang")]:
        sub = df[df["X5"] == height]
        shape_mean = sub.groupby("X1")[["Y1", "Y2"]].mean().sum(axis=1)
        worst, best = shape_mean.idxmax(), shape_mean.idxmin()
        rows.append(_lever_row(f"Hinh khoi trong nhom {ten} (X1)", f"X1={worst:.2f} -> X1={best:.2f}",
                               sub[sub["X1"] == worst], sub[sub["X1"] == best]))
    dist_mean = glazed.groupby("X8")[["Y1", "Y2"]].mean().sum(axis=1)
    rows.append(_lever_row("Phan bo kinh (X8, khi da co kinh)",
                           f"X8={dist_mean.idxmax()} -> X8={dist_mean.idxmin()} (te nhat -> tot nhat)",
                           glazed[glazed["X8"] == dist_mean.idxmax()], glazed[glazed["X8"] == dist_mean.idxmin()]))
    ori_mean = df.groupby("X6")[["Y1", "Y2"]].mean().sum(axis=1)
    rows.append(_lever_row("Huong nha (X6)", f"X6={ori_mean.idxmax()} -> X6={ori_mean.idxmin()} (te nhat -> tot nhat)",
                           df[df["X6"] == ori_mean.idxmax()], df[df["X6"] == ori_mean.idxmin()]))
    levers = pd.DataFrame(rows)
    levers.to_csv(REPORTS_DIR / "don_bay_thiet_ke.csv", index=False)

    # Tuong tac chieu cao x kinh: tac dong cua kinh co phu thuoc chieu cao khong?
    inter = df.groupby(["X5", "X7"])[["Y1", "Y2"]].mean().unstack("X7")
    inter.to_csv(REPORTS_DIR / "tuong_tac_chieu_cao_x_kinh.csv")

    # X8 = 0 <=> X7 = 0: cac cot one-hot X8_k thuc chat ma hoa "co kinh"
    x8_zero_is_no_glass = bool(((df["X8"] == 0) == (df["X7"] == 0)).all())

    fig, ax = plt.subplots(figsize=(10, 5.5))
    lab = [f"{r.don_bay}\n{r.thay_doi}" for r in levers.itertuples()]
    ax.barh(lab[::-1], levers["giam_tong_tai_kwh_m2"][::-1], color="#27ae60")
    for i, v in enumerate(levers["giam_tong_tai_kwh_m2"][::-1]):
        ax.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set_xlabel("Giam tong tai suoi + lam mat (kWh/m2)")
    ax.set_title("Don bay thiet ke: moi quyet dinh giam bao nhieu tai nang luong\n"
                 "(trung binh theo muc tren thiet ke giai thua can bang)")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "don_bay_thiet_ke.png", dpi=130)
    plt.close(fig)

    log(f"  X8=0 trung khop X7=0 (khong kinh) tren moi dong: {x8_zero_is_no_glass} "
        f"-> he so X8_1..X8_5 = tac dong cua viec CO KINH, khong phai cua cach phan bo")
    for r in levers.itertuples():
        log(f"    {r.don_bay:34s} {r.thay_doi:48s} Y1 {r.Y1_truoc:5.2f}->{r.Y1_sau:5.2f} "
            f"Y2 {r.Y2_truoc:5.2f}->{r.Y2_sau:5.2f}  giam {r.giam_tong_tai_kwh_m2:5.2f} kWh/m2  "
            f"~{r.tiet_kiem_vnd_moi_toa / 1e6:,.2f} trieu VND/toa/nam")
    log("  Da luu don_bay_thiet_ke.csv/.png, tuong_tac_chieu_cao_x_kinh.csv")
    return levers, inter


def run_model_misfit(en_pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, label: str) -> pd.DataFrame:
    """Phan du trung binh cua ElasticNet tren TEST theo (chieu cao x dien tich kinh) - chi de chan doan
    (khong chon gi): mo hinh tuyen tinh co bat duoc tuong tac va phi tuyen cua kinh khong?"""
    res = pd.DataFrame({"X5": X_test["X5"].values, "X7": X_test["X7"].values,
                        "phan_du": y_test.values - en_pipe.predict(X_test)})
    table = res.groupby(["X5", "X7"])["phan_du"].agg(["mean", "count"]).reset_index()
    table.columns = ["X5", "X7", "phan_du_trung_binh", "so_mau"]
    table.to_csv(REPORTS_DIR / f"phan_du_theo_nhom_{label}.csv", index=False)
    log(f"  [{label}] Phan du TB tren test theo (X5, X7): "
        + ", ".join(f"({r.X5:g},{r.X7:g})={r.phan_du_trung_binh:+.2f}" for r in table.itertuples()))
    return table


def main() -> None:
    log("1. Nap du lieu Energy Efficiency, tinh ma tran tuong quan + VIF...")
    df = load_data()
    run_correlation_vif(df)

    log("2. One-hot X6, X8...")
    X_full = make_feature_matrix(df)
    feature_names = list(X_full.columns)
    log(f"  Sau one-hot: {len(feature_names)} bien -> {feature_names}")

    all_results, all_baseline, all_comparison = {}, {}, {}
    all_grouping, all_heatmap_best, all_bootstrap, all_ranking, all_misfit = {}, {}, {}, {}, {}

    for label, ten_day_du in TARGETS:
        log(f"=== NHAN {label} ({ten_day_du}) ===")
        y = df[label]
        X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=RANDOM_STATE)
        log(f"  Train: {len(X_train)}  Test: {len(X_test)}")

        log(f"3. [{label}] Baseline Dummy + Linear Regression...")
        baseline_metrics = run_baseline(X_train, y_train, X_test, y_test, label)
        all_baseline[label] = baseline_metrics

        log(f"4. [{label}] Ridge / Lasso / ElasticNet (CV do sieu tham so)...")
        results = run_three_models(X_train, y_train, X_test, y_test, feature_names, label)
        all_results[label] = results

        log(f"5. [{label}] Bang so sanh 3 model...")
        all_comparison[label] = build_comparison_table(results, baseline_metrics, feature_names, label)

        log(f"6. [{label}] Kiem chung hieu ung gom nhom...")
        all_grouping[label] = check_grouping_effect(results, label)
        _, alpha_star = run_grouping_effect_forced(X_train, y_train, feature_names, label)
        all_grouping[f"{label}_alpha_star_forced"] = alpha_star

        log(f"7. [{label}] Heatmap RMSE cross-validation theo alpha x l1_ratio (chi dung train)...")
        _, best = run_heatmap_cv(results["ElasticNet"]["pipe"], label)
        all_heatmap_best[label] = best

        log(f"11b. [{label}] Mo hinh tuyen tinh sai lech o dau (phan du tren test theo nhom)...")
        all_misfit[label] = run_model_misfit(results["ElasticNet"]["pipe"], X_test, y_test, label)

        log(f"9. [{label}] Bootstrap 100 lan kiem tra on dinh he so ElasticNet...")
        en = results["ElasticNet"]
        all_bootstrap[label] = run_bootstrap_stability(X_full, y, en["alpha"], en["l1_ratio"], feature_names, label)

        log(f"10. [{label}] Xep hang bien theo do quan trong...")
        all_ranking[label] = rank_feature_importance(results, feature_names, label)

        joblib.dump(en["pipe"], MODELS_DIR / f"elasticnet_{label}.joblib")
        log(f"  [{label}] Da luu model -> models/elasticnet_{label}.joblib")

    log("8. So sanh bien quan trong giua Y1 va Y2...")
    y1y2_comparison = compare_y1_y2(all_results["Y1"], all_results["Y2"], feature_names)

    log("11. Don bay thiet ke (trung binh tren thiet ke giai thua) + quy doi kWh/tien dien...")
    levers, interaction = run_design_levers(df)

    summary = {
        "n_rows": int(len(df)),
        "n_features_after_onehot": len(feature_names),
        "feature_names": feature_names,
        "group_tuong_quan": GROUP_TUONG_QUAN,
    }
    for label, _ in TARGETS:
        en = all_results[label]["ElasticNet"]
        summary[label] = {
            "baseline": all_baseline[label],
            "model_comparison": all_comparison[label].to_dict(orient="records"),
            "grouping_effect": all_grouping[label].to_dict(orient="records"),
            "grouping_effect_alpha_star_forced": all_grouping[f"{label}_alpha_star_forced"],
            "heatmap_cv_best": {"alpha": all_heatmap_best[label][0], "l1_ratio": all_heatmap_best[label][1],
                                 "rmse_cv": all_heatmap_best[label][2]},
            "phan_du_test_theo_nhom": all_misfit[label].to_dict(orient="records"),
            "elasticnet_alpha": en["alpha"], "elasticnet_l1_ratio": en["l1_ratio"],
            "elasticnet_metrics": en["metrics"],
            "bootstrap_stability": all_bootstrap[label].to_dict(orient="records"),
            "feature_ranking": all_ranking[label].to_dict(orient="records"),
        }
    summary["y1_vs_y2_importance"] = y1y2_comparison.to_dict(orient="records")
    summary["don_bay_thiet_ke"] = levers.to_dict(orient="records")
    summary["tuong_tac_chieu_cao_x_kinh"] = {f"{k[0]}_X7={k[1]}": {f"X5={i}": float(v) for i, v in col.items()}
                                             for k, col in interaction.items()}
    summary["gia_dinh_quy_doi"] = {"dien_tich_san_m2": FLOOR_AREA_M2, "COP": COP,
                                   "gia_dien_vnd_kwh": ELECTRICITY_VND_PER_KWH}

    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("HOAN THANH. Xem ket qua chi tiet trong thu muc reports/.")


if __name__ == "__main__":
    main()
