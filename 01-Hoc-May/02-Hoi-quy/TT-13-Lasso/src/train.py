"""
TT-13 - Lasso Regression (L1): Chon 10 chi so xet nghiem quan trong nhat trong 200 chi so.

Pipeline day du (theo 10 buoc trong README.md cap de):
 1. Nap load_diabetes, mo rong thanh 200 cot (10 that + 190 nhieu)
 2. Baseline: Linear Regression tren 200 cot -> quan sat overfit nang
 3. LassoCV do alpha
 4. Cham diem chon bien: bao nhieu/10 bien that duoc giu, bao nhieu bien nhieu bi giu nham
 5. Ve coefficient path cua Lasso -> he so lan luot "roi" ve 0
 6. Ve RMSE train/test theo alpha
 7. So sanh Ridge vs Lasso tren cung du lieu (so bien giu + RMSE)
 8. Thi nghiem bien tuong quan: nhan doi 1 cot that (them nhieu nho) -> chay lai voi seed
    khac -> Lasso co doi lua chon khong?
 9. Train lai Linear Regression CHI tren cac bien Lasso chon -> so sanh RMSE (debiased lasso)
10. De xuat bo xet nghiem cuoi cung + uoc tinh chi phi tiet kiem

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
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV, lasso_path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42
N_NOISE = 190
ALPHA_GRID = np.logspace(-4, 1, 100)
ALPHA_GRID_PATH = np.logspace(-4, 1, 150)

# Chi phi uoc tinh (VND) cho 10 chi so that cua bo diabetes (theo README: "moi chi so
# ton 30.000-200.000d"). age/sex la thong tin nhan khau hoc (khong ton chi phi xet
# nghiem); bmi/bp la do luong lam sang co ban (re); s1-s6 la xet nghiem sinh hoa mau
# (dat hon, tuong ung cholesterol toan phan, LDL, HDL, TC/HDL, triglyceride, duong huyet).
REAL_FEATURE_COST_VND = {
    "age": 0,
    "sex": 0,
    "bmi": 20_000,
    "bp": 30_000,
    "s1": 50_000,
    "s2": 60_000,
    "s3": 60_000,
    "s4": 40_000,
    "s5": 70_000,
    "s6": 40_000,
}


def log(msg: str) -> None:
    print(f"[TT-13] {msg}")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ----------------------------------------------------------------------------
# 1. Nap load_diabetes, mo rong thanh 200 cot (10 that + 190 nhieu)
# ----------------------------------------------------------------------------
def generate_data(n_noise: int = N_NOISE, seed: int = RANDOM_STATE):
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    real_cols = list(X.columns)

    rng = np.random.default_rng(seed)
    noise_cols = [f"chi_so_nhieu_{i:03d}" for i in range(n_noise)]
    noise_df = pd.DataFrame(rng.normal(size=(len(X), n_noise)), columns=noise_cols)
    X_full = pd.concat([X, noise_df], axis=1)

    log(f"  X_full: {X_full.shape[0]} dong x {X_full.shape[1]} cot "
        f"({len(real_cols)} that + {len(noise_cols)} nhieu)")
    return X_full, y, real_cols, noise_cols


# ----------------------------------------------------------------------------
# 2. Baseline: Linear Regression tren 200 cot -> overfit
# ----------------------------------------------------------------------------
def run_linear_overfit(X_train, y_train, X_test, y_test) -> dict:
    pipe = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())])
    pipe.fit(X_train, y_train)
    train_metrics = evaluate(y_train, pipe.predict(X_train))
    test_metrics = evaluate(y_test, pipe.predict(X_test))
    log(f"  Linear (200 bien): RMSE train={train_metrics['RMSE']:.2f} R2 train={train_metrics['R2']:.4f}")
    log(f"  Linear (200 bien): RMSE test ={test_metrics['RMSE']:.2f} R2 test ={test_metrics['R2']:.4f}")
    log("  -> Chenh lech train/test lon = overfit (p=200 gan bang n_train, du dat vua khop nhieu).")
    return {"train": train_metrics, "test": test_metrics}


# ----------------------------------------------------------------------------
# 3. LassoCV do alpha
# ----------------------------------------------------------------------------
def run_lassocv(X_train, y_train) -> tuple[Pipeline, float]:
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lasso", LassoCV(alphas=ALPHA_GRID, cv=5, max_iter=50_000, random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)
    best_alpha = float(pipe.named_steps["lasso"].alpha_)
    n_kept = int((pipe.named_steps["lasso"].coef_ != 0).sum())
    log(f"  LassoCV (cv=5): alpha toi uu = {best_alpha:.5f}  -> giu {n_kept}/{X_train.shape[1]} bien")
    return pipe, best_alpha


# ----------------------------------------------------------------------------
# 4. Cham diem chon bien
# ----------------------------------------------------------------------------
def score_feature_selection(lasso_pipe: Pipeline, feature_names: list[str],
                             real_cols: list[str], noise_cols: list[str]) -> pd.DataFrame:
    coefs = pd.Series(lasso_pipe.named_steps["lasso"].coef_, index=feature_names)
    kept = coefs[coefs != 0]

    real_kept = [c for c in real_cols if c in kept.index]
    noise_kept = [c for c in noise_cols if c in kept.index]

    rows = [{
        "loai": "bien_that",
        "tong_so": len(real_cols),
        "so_duoc_giu": len(real_kept),
        "ty_le_%": len(real_kept) / len(real_cols) * 100,
        "danh_sach_giu": ", ".join(real_kept),
    }, {
        "loai": "bien_nhieu",
        "tong_so": len(noise_cols),
        "so_duoc_giu": len(noise_kept),
        "ty_le_%": len(noise_kept) / len(noise_cols) * 100,
        "danh_sach_giu": ", ".join(noise_kept) if noise_kept else "(khong co)",
    }]
    score_df = pd.DataFrame(rows)
    score_df.to_csv(REPORTS_DIR / "chon_bien_score.csv", index=False)

    log(f"  CHAM DIEM CHON BIEN: giu {len(real_kept)}/{len(real_cols)} bien THAT "
        f"(recall = {len(real_kept)/len(real_cols)*100:.0f}%)")
    log(f"    Bien that duoc giu: {real_kept}")
    log(f"  GIU NHAM {len(noise_kept)}/{len(noise_cols)} bien NHIEU (false positive)")
    if noise_kept:
        log(f"    Bien nhieu bi giu nham: {noise_kept}")
    log(f"  Tong so bien Lasso giu: {len(kept)}/{len(feature_names)}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Bien that\n(10 bien)", "Bien nhieu\n(190 bien)"],
           [len(real_kept) / len(real_cols) * 100, len(noise_kept) / len(noise_cols) * 100],
           color=["#27ae60", "#c0392b"])
    ax.set_ylabel("Ty le duoc Lasso giu lai (%)")
    ax.set_title(f"Cham diem chon bien: giu {len(real_kept)}/10 bien that, "
                 f"giu nham {len(noise_kept)}/190 bien nhieu")
    ax.set_ylim(0, 100)
    for i, v in enumerate([len(real_kept) / len(real_cols) * 100, len(noise_kept) / len(noise_cols) * 100]):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "chon_bien_score.png", dpi=130)
    plt.close(fig)

    log("  Da luu chon_bien_score.csv, chon_bien_score.png")
    return score_df


# ----------------------------------------------------------------------------
# 5. Coefficient path
# ----------------------------------------------------------------------------
def run_coefficient_path(X_train, y_train, feature_names: list[str], real_cols: list[str]) -> None:
    X_scaled = StandardScaler().fit_transform(X_train)
    alphas_lp, coefs_path, _ = lasso_path(X_scaled, y_train, alphas=ALPHA_GRID_PATH[::-1],
                                           max_iter=50_000)
    # lasso_path tra ve coefs_path shape (n_features, n_alphas); alphas_lp giam dan
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, name in enumerate(feature_names):
        if name in real_cols:
            ax.plot(alphas_lp, coefs_path[i], label=name, lw=2)
        else:
            ax.plot(alphas_lp, coefs_path[i], color="lightgray", lw=0.5, zorder=0)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("alpha (thang log)")
    ax.set_ylabel("He so Lasso (tren du lieu da chuan hoa)")
    ax.set_title("Coefficient path: he so cac bien nhieu (xam) roi ve 0 truoc,\n"
                 "he so 10 bien that (mau) roi sau hoac con lai")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "lasso_path.png", dpi=130)
    plt.close(fig)
    log("  Da luu lasso_path.png")


# ----------------------------------------------------------------------------
# 6. RMSE train/test theo alpha
# ----------------------------------------------------------------------------
def run_rmse_vs_alpha(X_train, y_train, X_test, y_test) -> float:
    train_rmse, test_rmse, n_kept_list = [], [], []
    for a in ALPHA_GRID_PATH:
        pipe = Pipeline([("scale", StandardScaler()), ("lasso", Lasso(alpha=a, max_iter=50_000))])
        pipe.fit(X_train, y_train)
        train_rmse.append(rmse(y_train, pipe.predict(X_train)))
        test_rmse.append(rmse(y_test, pipe.predict(X_test)))
        n_kept_list.append(int((pipe.named_steps["lasso"].coef_ != 0).sum()))

    best_idx = int(np.argmin(test_rmse))
    best_alpha_by_test = float(ALPHA_GRID_PATH[best_idx])

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(ALPHA_GRID_PATH, train_rmse, label="RMSE train", color="#2980b9")
    ax1.plot(ALPHA_GRID_PATH, test_rmse, label="RMSE test", color="#c0392b")
    ax1.axvline(best_alpha_by_test, color="#27ae60", ls="--", lw=1,
                label=f"alpha tot nhat tren test = {best_alpha_by_test:.4g}")
    ax1.set_xscale("log")
    ax1.set_xlabel("alpha (thang log)")
    ax1.set_ylabel("RMSE")
    ax1.set_title("RMSE train/test theo alpha (Lasso)")
    ax2 = ax1.twinx()
    ax2.plot(ALPHA_GRID_PATH, n_kept_list, color="gray", lw=1, ls=":", label="So bien con lai")
    ax2.set_ylabel("So bien co he so khac 0")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "rmse_theo_alpha.png", dpi=130)
    plt.close(fig)
    log(f"  Da luu rmse_theo_alpha.png (alpha tot nhat theo test = {best_alpha_by_test:.5f})")
    return best_alpha_by_test


# ----------------------------------------------------------------------------
# 7. So sanh Ridge vs Lasso
# ----------------------------------------------------------------------------
def compare_ridge_vs_lasso(X_train, y_train, X_test, y_test, lasso_alpha: float) -> pd.DataFrame:
    ridge_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-2, 4, 100), cv=5)),
    ])
    ridge_pipe.fit(X_train, y_train)
    ridge_coefs = ridge_pipe.named_steps["ridge"].coef_
    ridge_alpha = float(ridge_pipe.named_steps["ridge"].alpha_)
    ridge_n_kept = int((np.abs(ridge_coefs) > 1e-10).sum())
    ridge_metrics = evaluate(y_test, ridge_pipe.predict(X_test))

    lasso_pipe = Pipeline([("scale", StandardScaler()),
                            ("lasso", Lasso(alpha=lasso_alpha, max_iter=50_000))])
    lasso_pipe.fit(X_train, y_train)
    lasso_n_kept = int((lasso_pipe.named_steps["lasso"].coef_ != 0).sum())
    lasso_metrics = evaluate(y_test, lasso_pipe.predict(X_test))

    comp = pd.DataFrame([
        {"model": f"Ridge (alpha={ridge_alpha:.3g})", "so_bien_giu": ridge_n_kept,
         "tong_so_bien": X_train.shape[1], **ridge_metrics},
        {"model": f"Lasso (alpha={lasso_alpha:.4g})", "so_bien_giu": lasso_n_kept,
         "tong_so_bien": X_train.shape[1], **lasso_metrics},
    ])
    comp.to_csv(REPORTS_DIR / "ridge_vs_lasso.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Ridge", "Lasso"], [ridge_n_kept, lasso_n_kept], color=["#2980b9", "#c0392b"])
    ax.axhline(10, color="black", ls="--", lw=1, label="10 bien that")
    ax.set_ylabel(f"So bien con lai (tren tong {X_train.shape[1]} bien)")
    ax.set_title("Ridge giu gan het bien, Lasso chi giu so it")
    ax.legend()
    for i, v in enumerate([ridge_n_kept, lasso_n_kept]):
        ax.text(i, v + 2, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "ridge_vs_lasso.png", dpi=130)
    plt.close(fig)

    log("  So sanh Ridge vs Lasso:")
    for _, row in comp.iterrows():
        log(f"    {row['model']:22s} so_bien_giu={row['so_bien_giu']:.0f}/{row['tong_so_bien']:.0f}"
            f"  RMSE={row['RMSE']:.3f}  R2={row['R2']:.4f}")
    log("  Da luu ridge_vs_lasso.csv, ridge_vs_lasso.png")
    return comp


# ----------------------------------------------------------------------------
# 8. Thi nghiem bien tuong quan
# ----------------------------------------------------------------------------
def correlated_feature_experiment(X_full: pd.DataFrame, y: pd.Series, feature_names: list[str],
                                   dup_col: str = "bmi", noise_std: float = 0.01,
                                   seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    X_dup = X_full.copy()
    dup_name = f"{dup_col}_dup"
    X_dup[dup_name] = X_dup[dup_col] + rng.normal(0, noise_std, size=len(X_dup))
    corr = X_dup[dup_col].corr(X_dup[dup_name])
    log(f"  Tuong quan {dup_col} <-> {dup_name}: {corr:.4f}")

    rows = []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(X_dup, y, test_size=0.2, random_state=seed)
        pipe = Pipeline([("scale", StandardScaler()),
                          ("lasso", LassoCV(alphas=ALPHA_GRID, cv=5, max_iter=50_000,
                                             random_state=RANDOM_STATE))])
        pipe.fit(X_tr, y_tr)
        coefs = pd.Series(pipe.named_steps["lasso"].coef_, index=X_dup.columns)
        rows.append({
            "seed_split": seed,
            f"he_so_{dup_col}": coefs[dup_col],
            f"he_so_{dup_name}": coefs[dup_name],
            "bien_duoc_giu": (
                f"ca hai" if coefs[dup_col] != 0 and coefs[dup_name] != 0 else
                dup_col if coefs[dup_col] != 0 else
                dup_name if coefs[dup_name] != 0 else "khong bien nao"
            ),
        })
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "thi_nghiem_tuong_quan.csv", index=False)

    log(f"  Thi nghiem tuong quan ({dup_col} vs {dup_name}, r={corr:.4f}) qua {len(seeds)} seed khac nhau:")
    for _, row in result.iterrows():
        log(f"    seed={row['seed_split']}: giu -> {row['bien_duoc_giu']}")
    n_unique_choices = result["bien_duoc_giu"].nunique()
    if n_unique_choices > 1:
        log("  -> KHONG ON DINH: lua chon doi giua cac seed -> dung Lasso mot minh de 'khang dinh "
            "bien nao quan trong hon' la RUI RO.")
    else:
        log("  -> On dinh trong cac seed da thu (khong co nghia luon on dinh voi moi seed khac).")
    log("  Da luu thi_nghiem_tuong_quan.csv")
    return result


# ----------------------------------------------------------------------------
# 9. Debiased lasso: Linear Regression chi tren bien Lasso chon
# ----------------------------------------------------------------------------
def run_debiased_lasso(X_train, y_train, X_test, y_test, lasso_pipe: Pipeline,
                        feature_names: list[str], full_lasso_metrics: dict) -> dict:
    coefs = pd.Series(lasso_pipe.named_steps["lasso"].coef_, index=feature_names)
    selected = list(coefs[coefs != 0].index)

    pipe = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())])
    pipe.fit(X_train[selected], y_train)
    metrics = evaluate(y_test, pipe.predict(X_test[selected]))

    log(f"  Debiased Lasso: Linear Regression tren {len(selected)} bien Lasso chon")
    log(f"    RMSE Lasso day du (200 bien) = {full_lasso_metrics['RMSE']:.3f}")
    log(f"    RMSE debiased (chi {len(selected)} bien) = {metrics['RMSE']:.3f}")
    return {"selected_features": selected, "metrics": metrics}


# ----------------------------------------------------------------------------
# 10. De xuat bo xet nghiem + uoc tinh chi phi
# ----------------------------------------------------------------------------
def propose_test_panel(real_cols: list[str], noise_cols: list[str], selected_features: list[str],
                        seed: int = RANDOM_STATE) -> dict:
    rng = np.random.default_rng(seed)
    noise_costs = {c: float(rng.uniform(30_000, 200_000)) for c in noise_cols}
    all_costs = {**REAL_FEATURE_COST_VND, **noise_costs}

    total_cost_full_panel = sum(all_costs.values())
    selected_costs = {f: all_costs[f] for f in selected_features}
    total_cost_selected = sum(selected_costs.values())
    savings_pct = (1 - total_cost_selected / total_cost_full_panel) * 100

    panel_df = pd.DataFrame([
        {"chi_so": f, "chi_phi_vnd": all_costs[f], "loai": "that" if f in real_cols else "nhieu"}
        for f in selected_features
    ]).sort_values("chi_phi_vnd", ascending=False)
    panel_df.to_csv(REPORTS_DIR / "de_xuat_bo_xet_nghiem.csv", index=False)

    log(f"  Bo xet nghiem day du: {len(all_costs)} chi so, tong chi phi = {total_cost_full_panel:,.0f} VND/benh nhan")
    log(f"  Bo xet nghiem de xuat: {len(selected_features)} chi so, tong chi phi = {total_cost_selected:,.0f} VND/benh nhan")
    log(f"  Tiet kiem: {savings_pct:.1f}% ({total_cost_full_panel - total_cost_selected:,.0f} VND/benh nhan)")
    log("  Da luu de_xuat_bo_xet_nghiem.csv")

    return {
        "total_cost_full_panel_vnd": total_cost_full_panel,
        "total_cost_selected_vnd": total_cost_selected,
        "savings_pct": savings_pct,
        "panel": panel_df.to_dict(orient="records"),
    }


def main() -> None:
    log("1. Nap load_diabetes, mo rong thanh 200 cot...")
    X_full, y, real_cols, noise_cols = generate_data()
    feature_names = list(X_full.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=RANDOM_STATE)
    log(f"  Train: {len(X_train)}  Test: {len(X_test)}  (n_train={len(X_train)}, p=200 -> p gan bang n)")

    log("2. Linear Regression tren 200 cot (baseline overfit)...")
    overfit_metrics = run_linear_overfit(X_train, y_train, X_test, y_test)

    log("3. LassoCV do alpha...")
    lasso_pipe, lasso_alpha = run_lassocv(X_train, y_train)
    lasso_metrics = evaluate(y_test, lasso_pipe.predict(X_test))
    log(f"  Lasso (alpha={lasso_alpha:.5f}): RMSE test={lasso_metrics['RMSE']:.3f}  R2 test={lasso_metrics['R2']:.4f}")

    log("4. Cham diem chon bien...")
    score_df = score_feature_selection(lasso_pipe, feature_names, real_cols, noise_cols)

    log("5. Coefficient path...")
    run_coefficient_path(X_train, y_train, feature_names, real_cols)

    log("6. RMSE train/test theo alpha...")
    best_alpha_by_test = run_rmse_vs_alpha(X_train, y_train, X_test, y_test)

    log("7. So sanh Ridge vs Lasso...")
    ridge_lasso_comp = compare_ridge_vs_lasso(X_train, y_train, X_test, y_test, lasso_alpha)

    log("8. Thi nghiem bien tuong quan (nhan doi 1 cot that)...")
    corr_experiment = correlated_feature_experiment(X_full, y, feature_names)

    log("9. Debiased lasso (Linear Regression chi tren bien Lasso chon)...")
    debiased = run_debiased_lasso(X_train, y_train, X_test, y_test, lasso_pipe, feature_names, lasso_metrics)

    log("10. De xuat bo xet nghiem + uoc tinh chi phi...")
    panel_result = propose_test_panel(real_cols, noise_cols, debiased["selected_features"])

    joblib.dump(lasso_pipe, MODELS_DIR / "lasso_pipeline.joblib")
    log("Da luu model -> models/lasso_pipeline.joblib")

    summary = {
        "n_features_total": len(feature_names),
        "n_real_features": len(real_cols),
        "n_noise_features": len(noise_cols),
        "overfit_baseline": overfit_metrics,
        "lasso_alpha_lassocv": lasso_alpha,
        "lasso_alpha_best_on_test": best_alpha_by_test,
        "lasso_metrics_test": lasso_metrics,
        "feature_selection_score": score_df.to_dict(orient="records"),
        "ridge_vs_lasso": ridge_lasso_comp.to_dict(orient="records"),
        "correlated_feature_experiment": corr_experiment.to_dict(orient="records"),
        "debiased_lasso": {"n_selected": len(debiased["selected_features"]),
                            "selected_features": debiased["selected_features"],
                            "metrics_test": debiased["metrics"]},
        "test_panel_proposal": panel_result,
    }
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("HOAN THANH. Xem ket qua chi tiet trong thu muc reports/.")


if __name__ == "__main__":
    main()
