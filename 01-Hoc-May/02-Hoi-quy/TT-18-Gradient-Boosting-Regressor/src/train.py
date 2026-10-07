"""
TT-18 - Gradient Boosting Regressor: Tham dinh gia nha tu dong (AVM) tren Ames Housing.

Pipeline (theo 13 buoc cua de):
 1. Thong ke gia tri thieu, PHAN LOAI: thieu that vs "khong co tien ich"
 2. Xu ly rieng 2 nhom ('None'/0 vs median theo Neighborhood / mode - thong ke tinh TREN TRAIN)
 3. OrdinalEncoder dung thu tu (18 cot)
 4. One-hot bien danh muc thuan
 5. Skew SalePrice -> log1p
 6. Baseline: Dummy + Linear Regression + Ridge (scale + alpha chon bang 5-fold CV)
 7. Gradient Boosting: learning_rate x max_depth chon bang 5-fold CV tren train (early stopping trong moi fold)
 8. Duong train/validation loss theo so cay - TRUNG BINH 5 fold -> chon so cay
 9. Feature engineering TotalSF, TuoiNha, DaSuaChua -> do bang CV + test
10. Hoi quy phan vi 10/50/90 + split-conformal tren tap calib rieng
11. Median APE
12. Human-in-the-loop: nguong do rong khoang -> % tu dong (+ quet nguong)
13. GradientBoosting vs HistGradientBoosting (thoi gian)

Bo sung theo feedback:
14. So sanh CONG BANG moi model: RMSE_log 5-fold CV tren train (mean +- std, so fold thang) + test.
    GB co thang Linear khong? Chenh lech co y nghia khong? Sai so lon nam o dau?
15. Blend GB + Ridge (trung binh tren thang log)

Moi sieu tham so deu chon tren TRAIN. Test chi dung de bao cao.

Chay: python src/train.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(exist_ok=True, parents=True)
DATA_CSV = DATA_DIR / "train.csv"

RANDOM_STATE = 42
N_JOBS = 4
TARGET = "SalePrice"
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

NONE_CAT_COLS = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
                 "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType"]
NONE_NUM_COLS = ["GarageYrBlt", "MasVnrArea"]
TRUE_MISSING = ["LotFrontage", "Electrical"]

QUALITY_SCALE = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
QUALITY_COLS = ["ExterQual", "ExterCond", "HeatingQC", "KitchenQual", "BsmtQual", "BsmtCond", "FireplaceQu",
                "GarageQual", "GarageCond", "PoolQC"]
ORDINAL_SPECS = {
    **{c: QUALITY_SCALE for c in QUALITY_COLS},
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "PavedDrive": ["N", "P", "Y"],
    "Utilities": ["NoSeWa", "AllPub"],
}
ORDINAL_COLS = list(ORDINAL_SPECS)
FE_COLS = ["TotalSF", "TuoiNha", "DaSuaChua"]

GB_LRS = [0.01, 0.03, 0.05, 0.1]
GB_DEPTHS = [2, 3, 4, 5]
GB_MAX_TREES = 3000
RIDGE_ALPHAS = np.logspace(-1, 3, 17)
HITL_THRESHOLD = 0.25
HITL_SWEEP = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def log(msg: str) -> None:
    print(f"[TT-18] {msg}", flush=True)


def rmse(a, b) -> float:
    return float(mean_squared_error(a, b) ** 0.5)


def median_ape(y_usd, p_usd) -> float:
    return float(np.median(np.abs(y_usd - p_usd) / y_usd) * 100)


def evaluate(y_log, p_log) -> dict:
    y_usd, p_usd = np.expm1(y_log), np.expm1(p_log)
    ape = np.abs(y_usd - p_usd) / y_usd * 100
    return {"RMSE_log": rmse(y_log, p_log), "MAE_log": float(mean_absolute_error(y_log, p_log)),
            "R2_log": float(r2_score(y_log, p_log)), "MAE_USD": float(mean_absolute_error(y_usd, p_usd)),
            "Median_APE_%": float(np.median(ape)), "P90_APE_%": float(np.percentile(ape, 90)),
            "Max_APE_%": float(ape.max())}


# ----------------------------------------------------------------------------
# 1-2. Nap du lieu, phan loai & xu ly gia tri thieu
# ----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    if not DATA_CSV.exists():
        from sklearn.datasets import fetch_openml  # mirror cong khai cua Kaggle House Prices (data id 42165)
        log("  Khong thay data/train.csv -> tai tu OpenML (data id 42165)...")
        fetch_openml(data_id=42165, as_frame=True, parser="auto").frame.to_csv(DATA_CSV, index=False)
    df_raw = pd.read_csv(DATA_CSV)
    log(f"  Du lieu goc: {df_raw.shape[0]:,} dong x {df_raw.shape[1]} cot")
    return df_raw


def analyze_missing(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    rows = []
    for c, n in miss.items():
        kind = ("khong co tien ich -> 'None'" if c in NONE_CAT_COLS else
                "khong co tien ich -> 0" if c in NONE_NUM_COLS else
                "thieu that -> median theo Neighborhood (train)" if c == "LotFrontage" else
                "thieu that -> mode (train)" if c == "Electrical" else "CHUA PHAN LOAI")
        rows.append({"cot": c, "so_dong_thieu": int(n), "phan_tram": n / len(df) * 100, "phan_loai": kind})
    table = pd.DataFrame(rows)
    assert not (table["phan_loai"] == "CHUA PHAN LOAI").any(), "Co cot thieu chua duoc phan loai"
    table.to_csv(REPORTS_DIR / "phan_loai_gia_tri_thieu.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#c0392b" if c in TRUE_MISSING else "#2980b9" for c in table["cot"]]
    ax.barh(table["cot"][::-1], table["phan_tram"][::-1], color=colors[::-1])
    ax.set_xlabel("% dong bi thieu")
    ax.set_title("Gia tri thieu: xanh = 'khong co tien ich', do = thieu that")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "missing_analysis.png", dpi=130); plt.close(fig)
    log(f"  {len(table)} cot thieu: {len(NONE_CAT_COLS) + len(NONE_NUM_COLS)} 'khong co tien ich', "
        f"{len(TRUE_MISSING)} thieu that. Da luu phan_loai_gia_tri_thieu.csv, missing_analysis.png")
    return table


def fill_no_amenity(df: pd.DataFrame) -> pd.DataFrame:
    # Khong dung thong ke nao -> an toan lam truoc khi chia
    df = df.copy()
    df[NONE_CAT_COLS] = df[NONE_CAT_COLS].fillna("None")
    df[NONE_NUM_COLS] = df[NONE_NUM_COLS].fillna(0)
    return df


class TrueMissingFiller:
    """Dien 2 cot thieu that bang thong ke hoc tu TRAIN (tranh ro ri tu test)."""

    def fit(self, X: pd.DataFrame):
        self.lot_by_nb_ = X.groupby("Neighborhood")["LotFrontage"].median()
        self.lot_global_ = float(X["LotFrontage"].median())
        self.elec_mode_ = X["Electrical"].mode()[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        lot = X["Neighborhood"].map(self.lot_by_nb_).fillna(self.lot_global_)
        X["LotFrontage"] = X["LotFrontage"].fillna(lot)
        X["Electrical"] = X["Electrical"].fillna(self.elec_mode_)
        return X


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
    X["TuoiNha"] = X["YrSold"] - X["YearBuilt"]
    X["DaSuaChua"] = (X["YearRemodAdd"] != X["YearBuilt"]).astype(int)
    return X


def column_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    onehot = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and c not in ORDINAL_COLS]
    numeric = [c for c in X.columns if c not in ORDINAL_COLS + onehot]
    return onehot, numeric


def make_preprocessor(X: pd.DataFrame, scale: bool = False) -> ColumnTransformer:
    onehot, numeric = column_groups(X)
    num_step = StandardScaler() if scale else "passthrough"
    ord_enc = OrdinalEncoder(categories=[ORDINAL_SPECS[c] for c in ORDINAL_COLS],
                             handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer([
        ("ord", Pipeline([("enc", ord_enc), ("sc", StandardScaler())]) if scale else ord_enc, ORDINAL_COLS),
        ("oh", OneHotEncoder(handle_unknown="ignore"), onehot),
        ("num", num_step, numeric),
    ])


def gb(**kw) -> GradientBoostingRegressor:
    base = dict(subsample=0.8, max_features="sqrt", random_state=RANDOM_STATE)
    base.update(kw)
    return GradientBoostingRegressor(**base)


# ----------------------------------------------------------------------------
# 5. Skew
# ----------------------------------------------------------------------------
def run_skew(y: pd.Series) -> dict:
    y_log = np.log1p(y)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(y, bins=50, color="#e74c3c"); axes[0].set_title(f"SalePrice goc (skew={skew(y):.2f})")
    axes[1].hist(y_log, bins=50, color="#2980b9"); axes[1].set_title(f"log1p(SalePrice) (skew={skew(y_log):.2f})")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "skew_saleprice.png", dpi=130); plt.close(fig)
    out = {"skew_goc": float(skew(y)), "skew_log1p": float(skew(y_log))}
    log(f"  Skew SalePrice {out['skew_goc']:.3f} -> sau log1p {out['skew_log1p']:.3f}")
    return out


# ----------------------------------------------------------------------------
# 6. Ridge: scale + alpha bang 5-fold CV
# ----------------------------------------------------------------------------
def tune_ridge(X_train, y_train) -> tuple[Pipeline, pd.DataFrame]:
    gs = GridSearchCV(Pipeline([("prep", make_preprocessor(X_train, scale=True)), ("model", Ridge())]),
                      {"model__alpha": RIDGE_ALPHAS}, cv=CV, scoring="neg_root_mean_squared_error", n_jobs=N_JOBS)
    gs.fit(X_train, y_train)
    df = pd.DataFrame({"alpha": RIDGE_ALPHAS, "RMSE_log_cv": -gs.cv_results_["mean_test_score"],
                       "std": gs.cv_results_["std_test_score"]})
    df.to_csv(REPORTS_DIR / "ridge_chon_alpha_cv.csv", index=False)
    a = gs.best_params_["model__alpha"]
    log(f"  Ridge: alpha={a:.3g} theo 5-fold CV (RMSE_log_cv={-gs.best_score_:.4f}); "
        f"alpha=10 khong scale (ban cu) khong con dung")
    return gs.best_estimator_, df


# ----------------------------------------------------------------------------
# 7. Tuning GB: learning_rate x max_depth bang 5-fold CV (early stopping trong moi fold)
# ----------------------------------------------------------------------------
def tune_gb(X_train, y_train) -> tuple[pd.DataFrame, dict]:
    pipe = Pipeline([("prep", make_preprocessor(X_train)),
                     ("model", gb(n_estimators=GB_MAX_TREES, validation_fraction=0.1, n_iter_no_change=50))])
    gs = GridSearchCV(pipe, {"model__learning_rate": GB_LRS, "model__max_depth": GB_DEPTHS}, cv=CV,
                      scoring="neg_root_mean_squared_error", n_jobs=N_JOBS, return_train_score=True)
    t0 = time.time()
    gs.fit(X_train, y_train)
    res = pd.DataFrame({"learning_rate": gs.cv_results_["param_model__learning_rate"].astype(float),
                        "max_depth": gs.cv_results_["param_model__max_depth"].astype(int),
                        "RMSE_log_train": -gs.cv_results_["mean_train_score"],
                        "RMSE_log_cv": -gs.cv_results_["mean_test_score"],
                        "std_cv": gs.cv_results_["std_test_score"],
                        "fit_s_TB": gs.cv_results_["mean_fit_time"]})
    res.to_csv(REPORTS_DIR / "gb_tuning_cv.csv", index=False)

    pivot = res.pivot(index="max_depth", columns="learning_rate", values="RMSE_log_cv")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.4f}", ha="center", va="center", color="white", fontsize=9)
    ax.set_xlabel("learning_rate"); ax.set_ylabel("max_depth")
    ax.set_title("GB: RMSE_log 5-fold CV tren train (early stopping trong fold)")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(REPORTS_DIR / "gb_tuning_cv.png", dpi=130); plt.close(fig)

    params = {"learning_rate": float(gs.best_params_["model__learning_rate"]),
              "max_depth": int(gs.best_params_["model__max_depth"])}
    for r in res.sort_values("RMSE_log_cv").head(6).itertuples():
        log(f"    lr={r.learning_rate:<5} depth={r.max_depth}  CV={r.RMSE_log_cv:.4f} +- {r.std_cv:.4f}  "
            f"train={r.RMSE_log_train:.4f}")
    old = res[(res["learning_rate"] == 0.03) & (res["max_depth"] == 3)].iloc[0]
    log(f"  Chon {params} (CV={-gs.best_score_:.4f}). Cau hinh co dinh cu (0.03, 3): CV={old['RMSE_log_cv']:.4f}. "
        f"({time.time() - t0:.0f}s) Da luu gb_tuning_cv.csv/.png")
    return res, params


# ----------------------------------------------------------------------------
# 8. Duong loss theo so cay - trung binh 5 fold -> chon so cay
# ----------------------------------------------------------------------------
def loss_curve_cv(X_train, y_train, params: dict) -> tuple[pd.DataFrame, int]:
    tr_curves, val_curves = [], []
    for tr_idx, va_idx in CV.split(X_train):
        Xa, Xv = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        ya, yv = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        prep = make_preprocessor(Xa).fit(Xa)
        m = gb(n_estimators=GB_MAX_TREES, **params).fit(prep.transform(Xa), ya)
        tr_curves.append(m.train_score_)
        val_curves.append([mean_squared_error(yv, p) for p in m.staged_predict(prep.transform(Xv))])
    tr, va = np.mean(tr_curves, axis=0), np.mean(val_curves, axis=0)
    va_std = np.std(val_curves, axis=0)
    best_n = int(np.argmin(va)) + 1
    df = pd.DataFrame({"so_cay": np.arange(1, GB_MAX_TREES + 1), "MSE_log_train": tr, "MSE_log_val": va,
                       "std_val": va_std})
    df.iloc[::10].to_csv(REPORTS_DIR / "loss_theo_so_cay.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["so_cay"], df["MSE_log_train"], label="Train loss (TB 5 fold)", color="#2980b9")
    ax.plot(df["so_cay"], df["MSE_log_val"], label="Validation loss (TB 5 fold)", color="#e74c3c")
    ax.fill_between(df["so_cay"], va - va_std, va + va_std, color="#e74c3c", alpha=0.12)
    ax.axvline(best_n, color="gray", ls="--", label=f"so cay toi uu = {best_n}")
    ax.set_ylim(0, max(0.05, float(va[50]) * 1.5))
    ax.set_xlabel("So cay (boosting stage)"); ax.set_ylabel("MSE tren thang log")
    ax.set_title(f"Loss theo so cay (lr={params['learning_rate']}, depth={params['max_depth']})"); ax.legend()
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "loss_theo_so_cay.png", dpi=130); plt.close(fig)
    log(f"  So cay toi uu (TB 5 fold) = {best_n}: val MSE {va[best_n - 1]:.5f}; tai {GB_MAX_TREES} cay {va[-1]:.5f} "
        f"(+{(va[-1] / va[best_n - 1] - 1) * 100:.1f}%). Da luu loss_theo_so_cay.csv/.png")
    return df, best_n


# ----------------------------------------------------------------------------
# 9. Feature engineering: truoc/sau (CV + test)
# ----------------------------------------------------------------------------
def cv_scores(pipe: Pipeline, X, y) -> np.ndarray:
    out = []
    for tr_idx, va_idx in CV.split(X):
        p = clone(pipe).fit(X.iloc[tr_idx], y.iloc[tr_idx])
        out.append(rmse(y.iloc[va_idx], p.predict(X.iloc[va_idx])))
    return np.array(out)


def run_fe(X_train, X_test, y_train, y_test, params: dict, n_trees: int) -> pd.DataFrame:
    rows = []
    for name, Xa, Xb in [("Truoc FE", X_train, X_test), ("Sau FE", add_features(X_train), add_features(X_test))]:
        for model_name, scale, model in [("GB", False, gb(n_estimators=n_trees, **params)),
                                         ("Ridge", True, None)]:
            if model is None:
                model = Ridge(alpha=RIDGE_ALPHA_CHOSEN)
            pipe = Pipeline([("prep", make_preprocessor(Xa, scale=scale)), ("model", model)])
            cv = cv_scores(pipe, Xa, y_train)
            te = evaluate(y_test, pipe.fit(Xa, y_train).predict(Xb))
            rows.append({"bo_dac_trung": name, "model": model_name, "RMSE_log_cv": cv.mean(), "std_cv": cv.std(),
                         "RMSE_log_test": te["RMSE_log"], "Median_APE_test_%": te["Median_APE_%"]})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "so_sanh_feature_engineering.csv", index=False)
    for r in rows:
        log(f"    {r['model']:5s} {r['bo_dac_trung']:8s} CV={r['RMSE_log_cv']:.4f} +- {r['std_cv']:.4f}  "
            f"test={r['RMSE_log_test']:.4f}  MedianAPE={r['Median_APE_test_%']:.2f}%")
    log("  Da luu so_sanh_feature_engineering.csv")
    return df


# ----------------------------------------------------------------------------
# 14-15. So sanh cong bang: CV tren train + test, kem blend GB + Ridge
# ----------------------------------------------------------------------------
class Blend:
    def __init__(self, a: Pipeline, b: Pipeline):
        self.a, self.b = a, b

    def fit(self, X, y):
        self.a.fit(X, y); self.b.fit(X, y)
        return self

    def predict(self, X):
        return (self.a.predict(X) + self.b.predict(X)) / 2

    def get_params(self, deep=False):
        return {"a": self.a, "b": self.b}


def clone_any(m):
    return Blend(clone(m.a), clone(m.b)) if isinstance(m, Blend) else clone(m)


def compare_models(models: dict, X_train, X_test, y_train, y_test) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, fold_scores, preds = [], {}, {}
    for name, m in models.items():
        scores = []
        for tr_idx, va_idx in CV.split(X_train):
            p = clone_any(m).fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            scores.append(rmse(y_train.iloc[va_idx], p.predict(X_train.iloc[va_idx])))
        fold_scores[name] = np.array(scores)
        t0 = time.time()
        fitted = clone_any(m).fit(X_train, y_train)
        fit_s = time.time() - t0
        preds[name] = fitted.predict(X_test)
        rows.append({"model": name, "RMSE_log_cv": np.mean(scores), "std_cv": np.std(scores),
                     **{f"test_{k}": v for k, v in evaluate(y_test, preds[name]).items()}, "fit_s": fit_s})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "so_sanh_models.csv", index=False)
    folds = pd.DataFrame(fold_scores)
    folds.index = [f"fold_{i + 1}" for i in range(len(folds))]
    folds.to_csv(REPORTS_DIR / "so_sanh_models_theo_fold.csv")
    for r in rows:
        log(f"    {r['model']:28s} CV={r['RMSE_log_cv']:.4f} +- {r['std_cv']:.4f} | test RMSE_log={r['test_RMSE_log']:.4f} "
            f"MAE={r['test_MAE_USD']:,.0f}$ MedAPE={r['test_Median_APE_%']:.2f}% P90APE={r['test_P90_APE_%']:.1f}% "
            f"MaxAPE={r['test_Max_APE_%']:.0f}%")
    log("  Da luu so_sanh_models.csv, so_sanh_models_theo_fold.csv")
    return df, folds, preds


def analyze_gb_vs_linear(y_test, preds: dict, gb_name: str, lin_name: str, X_test) -> pd.DataFrame:
    """Sai so lon nam o dau: chia test theo 5 nhom gia, + top sai so lon nhat."""
    y_usd = np.expm1(y_test.values)
    ape = {k: np.abs(np.expm1(v) - y_usd) / y_usd * 100 for k, v in preds.items()}
    bins = pd.qcut(y_usd, 5)
    rows = []
    for b in bins.categories:
        m = np.asarray(bins == b)
        row = {"nhom_gia_USD": f"{b.left:,.0f}-{b.right:,.0f}", "so_can": int(m.sum())}
        for k in preds:
            row[f"RMSE_log|{k}"] = rmse(y_test.values[m], preds[k][m])
            row[f"MedAPE|{k}"] = float(np.median(ape[k][m]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "sai_so_theo_nhom_gia.csv", index=False)

    sq_gb = (preds[gb_name] - y_test.values) ** 2
    sq_lin = (preds[lin_name] - y_test.values) ** 2
    order = np.argsort(-np.maximum(sq_gb, sq_lin))[:5]
    top = pd.DataFrame({"Id": X_test.index[order], "gia_that": y_usd[order],
                        f"du_doan_{gb_name}": np.expm1(preds[gb_name][order]),
                        f"du_doan_{lin_name}": np.expm1(preds[lin_name][order]),
                        "GrLivArea": X_test["GrLivArea"].values[order],
                        "SaleCondition": X_test["SaleCondition"].values[order]})
    top.to_csv(REPORTS_DIR / "top5_sai_so_lon.csv", index=False)
    # Ty trong cua 5 can sai nhat trong tong binh phuong sai so
    share_gb = sq_gb[np.argsort(-sq_gb)[:5]].sum() / sq_gb.sum() * 100
    share_lin = sq_lin[np.argsort(-sq_lin)[:5]].sum() / sq_lin.sum() * 100
    log(f"  5 can sai nhat chiem {share_gb:.0f}% tong binh phuong sai so cua GB, {share_lin:.0f}% cua Linear")
    for _, r in df.iterrows():
        log(f"    {r['nhom_gia_USD']:>17s}: " + "  ".join(
            f"{k[:14]} {r[f'RMSE_log|{k}']:.3f}/{r[f'MedAPE|{k}']:.1f}%" for k in preds))
    log("  Da luu sai_so_theo_nhom_gia.csv, top5_sai_so_lon.csv")
    return df, {"ty_trong_top5_GB_%": float(share_gb), "ty_trong_top5_Linear_%": float(share_lin)}


# ----------------------------------------------------------------------------
# 10-12. Quantile + conformal, Median APE, human-in-the-loop
# ----------------------------------------------------------------------------
def run_quantile(X_train, X_test, y_train, y_test, params: dict, n_trees: int) -> tuple[dict, dict]:
    X_pr, X_cal, y_pr, y_cal = train_test_split(X_train, y_train, test_size=0.15, random_state=RANDOM_STATE)
    prep = make_preprocessor(X_pr).fit(X_pr)
    A, C, T_ = prep.transform(X_pr), prep.transform(X_cal), prep.transform(X_test)
    qm = {q: gb(loss="quantile", alpha=q, n_estimators=n_trees, **params).fit(A, y_pr) for q in (0.1, 0.5, 0.9)}
    lo_raw, hi_raw = qm[0.1].predict(T_), qm[0.9].predict(T_)
    cov_raw = float(np.mean((y_test.values >= lo_raw) & (y_test.values <= hi_raw)) * 100)
    cov_cal_raw = float(np.mean((y_cal.values >= qm[0.1].predict(C)) & (y_cal.values <= qm[0.9].predict(C))) * 100)

    scores = np.maximum(qm[0.1].predict(C) - y_cal.values, y_cal.values - qm[0.9].predict(C))
    n_cal = len(scores)
    level = min(1.0, np.ceil((n_cal + 1) * 0.80) / n_cal)
    corr = float(np.quantile(scores, level))
    band = np.sort(np.vstack([np.expm1(lo_raw - corr), np.expm1(qm[0.5].predict(T_)), np.expm1(hi_raw + corr)]).T, axis=1)
    lo, mid, hi = band[:, 0], band[:, 1], band[:, 2]
    y_usd = np.expm1(y_test.values)
    cov = float(np.mean((y_usd >= lo) & (y_usd <= hi)) * 100)
    width = (hi - lo) / mid

    idx = np.argsort(y_usd)[::5]
    fig, ax = plt.subplots(figsize=(11, 5))
    xs = np.arange(len(idx))
    ax.fill_between(xs, lo[idx], hi[idx], color="#3498db", alpha=0.3, label="Khoang 10-90% (sau conformal)")
    ax.plot(xs, mid[idx], color="#2980b9", label="Trung vi q50")
    ax.scatter(xs, y_usd[idx], color="#e74c3c", s=14, label="Gia that", zorder=5)
    ax.set_xlabel("Can nha test (1/5, sap theo gia)"); ax.set_ylabel("Gia (USD)")
    ax.set_title(f"Khoang gia 10-90%: do phu tho {cov_raw:.1f}% -> sau conformal {cov:.1f}%"); ax.legend()
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "khoang_gia.png", dpi=130); plt.close(fig)

    out = {"n_calib": n_cal, "do_phu_tho_test_%": cov_raw, "do_phu_tho_calib_%": cov_cal_raw,
           "hieu_chinh_log": corr, "do_phu_sau_conformal_test_%": cov,
           "do_rong_TB_%_gia": float(width.mean() * 100), "median_APE_q50_%": median_ape(y_usd, mid)}
    log(f"  Phu tho: test {cov_raw:.1f}% / calib {cov_cal_raw:.1f}% -> conformal (+{corr:.4f} log) test {cov:.1f}%; "
        f"do rong TB {out['do_rong_TB_%_gia']:.1f}% gia; Median APE q50 {out['median_APE_q50_%']:.2f}%")
    return out, {"lo": lo, "mid": mid, "hi": hi, "width": width, "y_usd": y_usd, "models": qm, "prep": prep,
                 "correction": corr}


def run_hitl(q: dict, point_pred_usd: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    y, w = q["y_usd"], q["width"]
    ape = np.abs(point_pred_usd - y) / y * 100
    rows = []
    for t in HITL_SWEEP:
        auto = w <= t
        rows.append({"nguong_do_rong": t, "ty_le_tu_dong_%": auto.mean() * 100,
                     "MedAPE_tu_dong_%": float(np.median(ape[auto])) if auto.any() else np.nan,
                     "MAE_tu_dong_USD": float(np.mean(np.abs(point_pred_usd - y)[auto])) if auto.any() else np.nan,
                     "ty_le_APE>20%_trong_tu_dong_%": float((ape[auto] > 20).mean() * 100) if auto.any() else np.nan,
                     "MedAPE_chuyen_nguoi_%": float(np.median(ape[~auto])) if (~auto).any() else np.nan})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(REPORTS_DIR / "human_in_the_loop_quet_nguong.csv", index=False)

    auto = w <= HITL_THRESHOLD
    table = pd.DataFrame({"nhom": ["Tu dong duyet (AVM)", "Chuyen tham dinh vien"],
                          "so_luong": [int(auto.sum()), int((~auto).sum())],
                          "ty_le_%": [auto.mean() * 100, (~auto).mean() * 100],
                          "MAE_USD": [np.mean(np.abs(point_pred_usd - y)[auto]), np.mean(np.abs(point_pred_usd - y)[~auto])],
                          "Median_APE_%": [np.median(ape[auto]), np.median(ape[~auto])]})
    table.to_csv(REPORTS_DIR / "human_in_the_loop.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ape, bins=40, color="#16a085")
    ax.axvline(np.median(ape), color="#e74c3c", ls="--", label=f"Median APE = {np.median(ape):.2f}%")
    ax.set_xlabel("APE (%)"); ax.set_ylabel("So can nha"); ax.set_title("Phan phoi APE tren test (model chinh)")
    ax.legend(); fig.tight_layout(); fig.savefig(REPORTS_DIR / "ape_distribution.png", dpi=130); plt.close(fig)

    for r in rows:
        log(f"    nguong {r['nguong_do_rong']:.2f}: tu dong {r['ty_le_tu_dong_%']:5.1f}%  MedAPE tu dong "
            f"{r['MedAPE_tu_dong_%']:.2f}%  APE>20% trong nhom tu dong {r['ty_le_APE>20%_trong_tu_dong_%']:.1f}%")
    log("  Da luu human_in_the_loop.csv, human_in_the_loop_quet_nguong.csv, ape_distribution.png")
    summary = {"ty_le_APE<10%": float((ape < 10).mean() * 100), "ty_le_APE<20%": float((ape < 20).mean() * 100)}
    return table, sweep, summary


def compare_hgb(X_train, X_test, y_train, y_test, params: dict, n_trees: int) -> pd.DataFrame:
    rows = []
    for name, model in [("GradientBoostingRegressor", gb(n_estimators=n_trees, **params)),
                        ("HistGradientBoostingRegressor",
                         HistGradientBoostingRegressor(max_iter=n_trees, learning_rate=params["learning_rate"],
                                                       max_depth=params["max_depth"], random_state=RANDOM_STATE))]:
        pipe = Pipeline([("prep", make_preprocessor(X_train)), ("model", model)])
        if name.startswith("Hist"):
            pipe.set_params(prep__oh__sparse_output=False)
        t0 = time.time()
        pipe.fit(X_train, y_train)
        rows.append({"model": name, "fit_s": time.time() - t0, **evaluate(y_test, pipe.predict(X_test))})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "gb_vs_hist_gb.csv", index=False)
    for r in rows:
        log(f"    {r['model']:30s} fit={r['fit_s']:.2f}s  RMSE_log={r['RMSE_log']:.4f}  MedAPE={r['Median_APE_%']:.2f}%")
    return df


RIDGE_ALPHA_CHOSEN = 10.0  # ghi de trong main() bang alpha chon tu CV


def main() -> None:
    global RIDGE_ALPHA_CHOSEN
    log("1. Nap du lieu, phan loai gia tri thieu...")
    df_raw = load_data()
    df = df_raw.drop(columns=["Id"]).set_index(df_raw["Id"])
    missing_table = analyze_missing(df)
    df = fill_no_amenity(df)

    log("5. Skew SalePrice...")
    skew_info = run_skew(df[TARGET])

    log("   Chia 80/20; dien 2 cot thieu that bang thong ke TRAIN...")
    X, y = df.drop(columns=[TARGET]), np.log1p(df[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    filler = TrueMissingFiller().fit(X_train)
    X_train, X_test = filler.transform(X_train), filler.transform(X_test)
    assert X_train.isna().sum().sum() == 0 and X_test.isna().sum().sum() == 0
    onehot, numeric = column_groups(X_train)
    log(f"  Train {X_train.shape}, Test {X_test.shape}. Ordinal {len(ORDINAL_COLS)} | one-hot {len(onehot)} | "
        f"so {len(numeric)}")

    log("6. Ridge: chon alpha bang CV...")
    ridge_pipe, ridge_df = tune_ridge(X_train, y_train)
    RIDGE_ALPHA_CHOSEN = float(ridge_pipe.named_steps["model"].alpha)

    log("7. Tuning GB (learning_rate x max_depth, 5-fold CV)...")
    gb_tune_df, gb_params = tune_gb(X_train, y_train)

    log("8. Duong loss theo so cay (TB 5 fold)...")
    curve_df, n_trees = loss_curve_cv(X_train, y_train, gb_params)

    log("9. Feature engineering truoc/sau...")
    fe_df = run_fe(X_train, X_test, y_train, y_test, gb_params, n_trees)
    gb_fe = fe_df[fe_df["model"] == "GB"].set_index("bo_dac_trung")["RMSE_log_cv"]
    use_fe = bool(gb_fe["Sau FE"] < gb_fe["Truoc FE"])
    if use_fe:
        X_train, X_test = add_features(X_train), add_features(X_test)
    log(f"  Dung FE cho cac buoc sau: {use_fe} (quyet dinh theo CV cua GB)")

    log("14-15. So sanh cong bang (5-fold CV tren train + test)...")
    gb_name = f"Gradient Boosting (lr={gb_params['learning_rate']}, d={gb_params['max_depth']}, {n_trees} cay)"
    ridge_name = f"Ridge (alpha={RIDGE_ALPHA_CHOSEN:.3g}, scale)"
    models = {
        "Dummy (trung binh)": Pipeline([("prep", make_preprocessor(X_train)), ("model", DummyRegressor())]),
        "Linear Regression": Pipeline([("prep", make_preprocessor(X_train)), ("model", LinearRegression())]),
        ridge_name: Pipeline([("prep", make_preprocessor(X_train, scale=True)), ("model", Ridge(alpha=RIDGE_ALPHA_CHOSEN))]),
        "GB cau hinh cu (lr=0.03, d=3, early stop)": Pipeline([("prep", make_preprocessor(X_train)), (
            "model", gb(n_estimators=1000, learning_rate=0.03, max_depth=3, validation_fraction=0.1, n_iter_no_change=50))]),
        gb_name: Pipeline([("prep", make_preprocessor(X_train)), ("model", gb(n_estimators=n_trees, **gb_params))]),
    }
    models["Blend: (GB + Ridge) / 2"] = Blend(models[gb_name], models[ridge_name])
    comparison, folds, preds = compare_models(models, X_train, X_test, y_train, y_test)
    wins = {k: int((folds[gb_name] < folds[k]).sum()) for k in folds.columns if k != gb_name}
    log(f"  So fold GB thang: {wins}")
    gap_df, gap_info = analyze_gb_vs_linear(y_test, {k: preds[k] for k in ["Linear Regression", ridge_name, gb_name,
                                                                         "Blend: (GB + Ridge) / 2"]},
                                            gb_name, "Linear Regression", X_test)

    log("10. Quantile 10/50/90 + split-conformal...")
    q_info, q = run_quantile(X_train, X_test, y_train, y_test, gb_params, n_trees)

    log("11-12. Median APE + human-in-the-loop...")
    final_gb = models[gb_name].fit(X_train, y_train)
    hitl_table, hitl_sweep, ape_info = run_hitl(q, np.expm1(final_gb.predict(X_test)))
    log(f"    Nguong {HITL_THRESHOLD}: " + " | ".join(
        f"{r.nhom}: {r.so_luong} ({r[3]:.1f}%), MAE {r.MAE_USD:,.0f}$, MedAPE {r.Median_APE_:.2f}%"
        for r in hitl_table.rename(columns={"Median_APE_%": "Median_APE_"}).itertuples()))

    log("13. GB vs HistGB...")
    hgb_df = compare_hgb(X_train, X_test, y_train, y_test, gb_params, n_trees)

    joblib.dump({"gbr_pipeline": final_gb, "true_missing_filler": filler, "dung_feature_engineering": use_fe,
                 "quantile_models": q["models"], "quantile_preprocessor": q["prep"],
                 "conformal_correction_log": q["correction"], "hitl_threshold": HITL_THRESHOLD,
                 "none_cat_cols": NONE_CAT_COLS, "none_num_cols": NONE_NUM_COLS, "ordinal_specs": ORDINAL_SPECS},
                MODELS_DIR / "gbr_pipeline.joblib", compress=3)
    log(f"  Da luu models/gbr_pipeline.joblib ({(MODELS_DIR / 'gbr_pipeline.joblib').stat().st_size / 1e6:.1f} MB)")

    summary = {
        "shape_raw": list(df_raw.shape),
        "gia_tri_thieu": missing_table.to_dict(orient="records"),
        "so_cot_thieu_khong_co_tien_ich": len(NONE_CAT_COLS) + len(NONE_NUM_COLS), "so_cot_thieu_that": len(TRUE_MISSING),
        "so_cot_ordinal": len(ORDINAL_COLS), "skew": skew_info,
        "ridge_alpha_cv": ridge_df.to_dict(orient="records"), "ridge_alpha_chon": RIDGE_ALPHA_CHOSEN,
        "gb_tuning_cv": gb_tune_df.to_dict(orient="records"), "gb_params_chon": gb_params, "gb_so_cay": n_trees,
        "feature_engineering": fe_df.to_dict(orient="records"), "dung_fe": use_fe,
        "so_sanh_models": comparison.to_dict(orient="records"),
        "so_sanh_models_theo_fold": folds.to_dict(), "so_fold_gb_thang": wins,
        "sai_so_theo_nhom_gia": gap_df.to_dict(orient="records"), **gap_info,
        "khoang_du_bao": q_info, "ape": ape_info,
        "human_in_the_loop": hitl_table.to_dict(orient="records"),
        "human_in_the_loop_quet_nguong": hitl_sweep.to_dict(orient="records"),
        "gb_vs_hist_gb": hgb_df.to_dict(orient="records"),
    }
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    log("HOAN THANH. Xem reports/ va models/.")


if __name__ == "__main__":
    main()
