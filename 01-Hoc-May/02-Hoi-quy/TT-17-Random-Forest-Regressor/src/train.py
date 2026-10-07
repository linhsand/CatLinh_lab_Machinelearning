"""
TT-17 - Random Forest Regressor: Du doan gia ve may bay de tu van "nen mua bay gio hay cho".

Pipeline (theo 12 buoc cua de):
 1. Nap du lieu, bo cot `flight` va cot index thua
 2. EDA: gia trung binh theo days_left
 3. EDA: boxplot gia theo class va airline
 4. Pipeline OneHotEncoder (cay KHONG can scale), chia 80/20
 5. Baseline: Dummy + Linear Regression + 1 cay don (train vs test -> overfit; them 1 cay chon do sau bang CV)
 6. Random Forest: TUNING max_features x min_samples_leaf bang sai so OOB (chi dung TRAIN) -> bao cao oob_score_
    (chon cau hinh gon nhat trong 1% so voi RMSE-OOB tot nhat: model phai phuc vu web va luu duoc vao repo)
 7. RMSE-OOB theo n_estimators -> diem bao hoa, chon so cay (KHONG dung test)
 8. Permutation importance gop theo bien goc (so voi feature_importances_ mac dinh)
 9. PDP cho days_left -> mua som tiet kiem bao nhieu tien
10. Khoang du bao 10-90% tu cac cay + ty le phu thuc te
11. Thi nghiem ngoai suy days_left = 60/100/365
12. So sanh voi XGBoost (TUNING max_depth x learning_rate, early stopping tren tap validation tach tu train)

Mo rong:
13. Tach 2 model Economy / Business
14. ExtraTreesRegressor (cung tham so)

Moi sieu tham so deu chon tren TRAIN (OOB / CV / validation). Test chi dung de bao cao.

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
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
for d in (MODELS_DIR, REPORTS_DIR):
    d.mkdir(exist_ok=True, parents=True)
DATA_CSV = DATA_DIR / "Clean_Dataset.csv"

RANDOM_STATE = 42
CAT_COLS = ["airline", "source_city", "departure_time", "stops", "arrival_time", "destination_city", "class"]
NUM_COLS = ["duration", "days_left"]
FEATURES = CAT_COLS + NUM_COLS
TARGET = "price"

TREE_DEPTHS = [10, 15, 20, 25, None]
RF_MAX_FEATURES = [0.3, 0.5, 0.7, 1.0]
RF_MIN_LEAF = [1, 2, 5, 10]
RF_TUNE_TREES = 100
RF_PARAM_TOLERANCE = 0.01  # chon cau hinh it nut nhat co RMSE-OOB trong 1% so voi tot nhat
RF_N_LIST = [10, 20, 30, 50, 75, 100, 150, 200, 300]
RF_N_TOLERANCE = 0.005  # chon so cay nho nhat co RMSE-OOB trong 0,5% so voi muc tot nhat
XGB_DEPTHS = [6, 8, 10, 12]
XGB_LRS = [0.1, 0.05]


def log(msg: str) -> None:
    print(f"[TT-17] {msg}", flush=True)


def evaluate(y_true, y_pred) -> dict:
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "R2": float(r2_score(y_true, y_pred))}


def make_preprocess() -> ColumnTransformer:
    # dense (sparse_output=False): bo xay cay cua sklearn nhanh hon ~12 lan so voi sparse tren bo nay
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS)],
                             remainder="passthrough")


def model_size_mb(obj, name: str) -> float:
    tmp = MODELS_DIR / f"_tmp_{name}.joblib"
    joblib.dump(obj, tmp, compress=3)
    size = tmp.stat().st_size / 1e6
    tmp.unlink()
    return size


# ----------------------------------------------------------------------------
# 1. Nap du lieu
# ----------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Khong thay {DATA_CSV}. Tai Clean_Dataset.csv theo huong dan trong data/DATA_SOURCE.md")
    df_raw = pd.read_csv(DATA_CSV)
    df = df_raw.drop(columns=[c for c in df_raw.columns if c.startswith("Unnamed")] + ["flight"])
    log(f"  Du lieu goc {df_raw.shape[0]:,} x {df_raw.shape[1]} -> bo 'flight' ({df_raw['flight'].nunique():,} ma "
        f"chuyen) va cot index -> {df.shape[0]:,} x {df.shape[1]}. Gia tri thieu: {int(df.isna().sum().sum())}")
    return df_raw, df


# ----------------------------------------------------------------------------
# 2-3. EDA
# ----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> dict:
    by_days = df.groupby("days_left")[TARGET].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(by_days.index, by_days.values, "o-", color="#2980b9", ms=3)
    ax.set_xlabel("So ngay con lai truoc chuyen bay (days_left)"); ax.set_ylabel("Gia ve trung binh (Rupee)")
    ax.set_title("Gia ve trung binh theo days_left")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "gia_theo_days_left.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    df.boxplot(column=TARGET, by="class", ax=axes[0])
    axes[0].set_title("Gia ve theo hang ve (class)"); axes[0].set_ylabel("Gia (Rupee)")
    df.boxplot(column=TARGET, by="airline", ax=axes[1])
    axes[1].set_title("Gia ve theo hang bay (airline)"); axes[1].tick_params(axis="x", rotation=30)
    plt.suptitle(""); fig.tight_layout(); fig.savefig(REPORTS_DIR / "boxplot_class_airline.png", dpi=130); plt.close(fig)

    class_mean = df.groupby("class")[TARGET].mean()
    eda = {"gia_TB_theo_days_left": {int(k): float(v) for k, v in by_days.items()},
           "gia_TB_theo_class": class_mean.to_dict(),
           "ty_le_business_economy": float(class_mean["Business"] / class_mean["Economy"]),
           "gia_trung_vi_theo_airline": df.groupby("airline")[TARGET].median().sort_values(ascending=False).to_dict()}
    log(f"  Gia TB: days_left=1 {by_days.loc[1]:,.0f} | =2 {by_days.loc[2]:,.0f} | =20 {by_days.loc[20]:,.0f} | "
        f"=49 {by_days.loc[49]:,.0f}. Business/Economy = {eda['ty_le_business_economy']:.2f} lan")
    return eda


# ----------------------------------------------------------------------------
# 5. Baseline: Dummy, Linear, cay don (khong gioi han + chon do sau bang CV)
# ----------------------------------------------------------------------------
def run_baselines(X_train, y_train, X_test, y_test) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, fitted = [], {}

    def add(name, model):
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_s = time.time() - t0
        fitted[name] = model
        tr, te = evaluate(y_train, model.predict(X_train)), evaluate(y_test, model.predict(X_test))
        rows.append({"model": name, "MAE_train": tr["MAE"], "RMSE_train": tr["RMSE"], "R2_train": tr["R2"],
                     **te, "fit_s": fit_s})

    add("Dummy (trung binh)", DummyRegressor(strategy="mean"))
    add("Linear Regression", Pipeline([("pre", make_preprocess()), ("m", LinearRegression())]))
    add("1 cay don (khong gioi han)", Pipeline([("pre", make_preprocess()),
                                                ("m", DecisionTreeRegressor(random_state=RANDOM_STATE))]))

    # Cay don chon max_depth bang 5-fold CV tren train (de so sanh cong bang voi RF da tuning)
    gs = GridSearchCV(Pipeline([("pre", make_preprocess()), ("m", DecisionTreeRegressor(random_state=RANDOM_STATE))]),
                      {"m__max_depth": TREE_DEPTHS}, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
                      scoring="neg_root_mean_squared_error", n_jobs=-1, return_train_score=True)
    gs.fit(X_train, y_train)
    cv_df = pd.DataFrame({"max_depth": [str(d) for d in TREE_DEPTHS],
                          "RMSE_train": -gs.cv_results_["mean_train_score"],
                          "RMSE_cv": -gs.cv_results_["mean_test_score"],
                          "std_RMSE_cv": gs.cv_results_["std_test_score"]})
    cv_df.to_csv(REPORTS_DIR / "cay_don_chon_depth_cv.csv", index=False)
    best_depth = gs.best_params_["m__max_depth"]
    add(f"1 cay don (max_depth={best_depth}, chon bang CV)", gs.best_estimator_)

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "baseline_train_test.csv", index=False)
    for r in rows:
        log(f"    {r['model']:40s} MAE train={r['MAE_train']:8,.0f} test={r['MAE']:8,.0f}  "
            f"RMSE train={r['RMSE_train']:7,.0f} test={r['RMSE']:7,.0f}  R2 test={r['R2']:.4f}")
    for r in cv_df.itertuples():
        log(f"    cay don max_depth={r.max_depth:>4s}: RMSE train={r.RMSE_train:,.0f}  CV={r.RMSE_cv:,.0f} +- {r.std_RMSE_cv:.0f}")
    log("  Da luu baseline_train_test.csv, cay_don_chon_depth_cv.csv")
    return df, cv_df, fitted


# ----------------------------------------------------------------------------
# 6. Tuning RF bang OOB (chi dung TRAIN)
# ----------------------------------------------------------------------------
def tune_rf_oob(Xt_train, y_train) -> tuple[pd.DataFrame, dict]:
    rows = []
    for mf in RF_MAX_FEATURES:
        for leaf in RF_MIN_LEAF:
            t0 = time.time()
            rf = RandomForestRegressor(n_estimators=RF_TUNE_TREES, max_features=mf, min_samples_leaf=leaf,
                                       oob_score=True, n_jobs=-1, random_state=RANDOM_STATE).fit(Xt_train, y_train)
            fit_s = time.time() - t0
            oob_rmse = float(mean_squared_error(y_train, rf.oob_prediction_) ** 0.5)
            nodes = int(np.mean([e.tree_.node_count for e in rf.estimators_]))
            rows.append({"max_features": mf, "min_samples_leaf": leaf, "RMSE_oob": oob_rmse,
                         "R2_oob": float(rf.oob_score_), "fit_s_100_cay": fit_s, "so_nut_TB_moi_cay": nodes})
            log(f"    max_features={mf:<4} min_samples_leaf={leaf:<3} RMSE_oob={oob_rmse:8.1f}  "
                f"R2_oob={rf.oob_score_:.5f}  {fit_s:5.1f}s  {nodes:,} nut/cay")
            del rf
    df = pd.DataFrame(rows)

    pivot = df.pivot(index="min_samples_leaf", columns="max_features", values="RMSE_oob")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:,.0f}", ha="center", va="center", color="white", fontsize=9)
    ax.set_xlabel("max_features"); ax.set_ylabel("min_samples_leaf")
    ax.set_title(f"RMSE OOB tren train ({RF_TUNE_TREES} cay) - thap hon la tot hon")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(REPORTS_DIR / "rf_tuning_oob.png", dpi=130); plt.close(fig)

    # Quy tac chon: cau hinh GON NHAT (it nut nhat) trong RF_PARAM_TOLERANCE so voi RMSE_oob tot nhat.
    # Ly do: model phuc vu tren web OTA (bo nho, toc do) va phai luu duoc vao repo (< 100 MB/file).
    best_rmse = df["RMSE_oob"].min()
    df["trong_nguong_1%"] = df["RMSE_oob"] <= best_rmse * (1 + RF_PARAM_TOLERANCE)
    chosen = df[df["trong_nguong_1%"]].sort_values("so_nut_TB_moi_cay").iloc[0]
    df.to_csv(REPORTS_DIR / "rf_tuning_oob.csv", index=False)
    params = {"max_features": float(chosen["max_features"]), "min_samples_leaf": int(chosen["min_samples_leaf"])}
    best = df.loc[df["RMSE_oob"].idxmin()]
    log(f"  Tot nhat tuyet doi: max_features={best['max_features']}, leaf={int(best['min_samples_leaf'])} "
        f"(RMSE_oob={best_rmse:.1f}, {int(best['so_nut_TB_moi_cay']):,} nut/cay)")
    log(f"  Chon {params}: gon nhat trong {RF_PARAM_TOLERANCE:.0%} (RMSE_oob={chosen['RMSE_oob']:.1f}, "
        f"+{(chosen['RMSE_oob'] / best_rmse - 1) * 100:.2f}%, {int(chosen['so_nut_TB_moi_cay']):,} nut/cay). "
        f"Da luu rf_tuning_oob.csv/.png")
    return df, params


# ----------------------------------------------------------------------------
# 7. RMSE-OOB theo so cay -> chon n_estimators (KHONG dung test)
# ----------------------------------------------------------------------------
def oob_curve(Xt_train, y_train, params: dict) -> tuple[pd.DataFrame, int]:
    rf = RandomForestRegressor(n_estimators=RF_N_LIST[0], oob_score=True, warm_start=True, n_jobs=-1,
                               random_state=RANDOM_STATE, **params)
    rows = []
    for n in RF_N_LIST:
        rf.n_estimators = n
        rf.fit(Xt_train, y_train)  # warm_start: chi xay them cay moi
        rows.append({"n_estimators": n, "RMSE_oob": float(mean_squared_error(y_train, rf.oob_prediction_) ** 0.5),
                     "R2_oob": float(rf.oob_score_)})
    del rf
    df = pd.DataFrame(rows)
    best = df["RMSE_oob"].min()
    n_chosen = int(df.loc[df["RMSE_oob"] <= best * (1 + RF_N_TOLERANCE), "n_estimators"].min())
    df.to_csv(REPORTS_DIR / "rmse_theo_so_cay.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df["n_estimators"], df["RMSE_oob"], "o-", color="#c0392b", label="RMSE OOB (tren train)")
    ax.axvline(n_chosen, color="gray", ls="--", label=f"chon n={n_chosen} (trong {RF_N_TOLERANCE:.1%} so voi tot nhat)")
    ax.set_xlabel("So cay (n_estimators)"); ax.set_ylabel("RMSE OOB (Rupee)")
    ax.set_title("RMSE-OOB theo so cay - test KHONG dung de chon"); ax.legend()
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "rmse_theo_so_cay.png", dpi=130); plt.close(fig)
    for r in rows:
        log(f"    n={r['n_estimators']:3d}  RMSE_oob={r['RMSE_oob']:.1f}")
    log(f"  Chon n_estimators={n_chosen}. Da luu rmse_theo_so_cay.csv/.png")
    return df, n_chosen


# ----------------------------------------------------------------------------
# 8. Permutation importance (gop theo bien goc) vs feature_importances_ mac dinh
# ----------------------------------------------------------------------------
def run_importance(rf_pipe: Pipeline, X_test, y_test) -> pd.DataFrame:
    pre, rf = rf_pipe.named_steps["pre"], rf_pipe.named_steps["rf"]
    names = list(pre.named_transformers_["cat"].get_feature_names_out(CAT_COLS)) + NUM_COLS
    agg = {}
    for name, imp in zip(names, rf.feature_importances_):
        base = next((c for c in CAT_COLS if name.startswith(c + "_")), name)
        agg[base] = agg.get(base, 0.0) + float(imp)

    idx = X_test.sample(n=20_000, random_state=RANDOM_STATE).index
    perm = permutation_importance(rf_pipe, X_test.loc[idx], y_test.loc[idx], n_repeats=5,
                                  random_state=RANDOM_STATE, n_jobs=-1)  # xao tron CA cot goc -> da gop san
    df = pd.DataFrame({"bien": FEATURES, "permutation_giam_R2": perm.importances_mean,
                       "permutation_std": perm.importances_std})
    df["impurity_mac_dinh_gop"] = df["bien"].map(agg)
    df["hang_permutation"] = df["permutation_giam_R2"].rank(ascending=False).astype(int)
    df["hang_impurity"] = df["impurity_mac_dinh_gop"].rank(ascending=False).astype(int)
    df = df.sort_values("permutation_giam_R2", ascending=False).reset_index(drop=True)
    df.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(df["bien"][::-1], df["permutation_giam_R2"][::-1], xerr=df["permutation_std"][::-1], color="#16a085")
    ax.set_xscale("log"); ax.set_xlabel("R2 giam khi xao tron (log) - 20.000 dong test")
    ax.set_title("Permutation importance (gop theo bien goc)")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "permutation_importance.png", dpi=130); plt.close(fig)
    for r in df.itertuples():
        log(f"    {r.bien:17s} permutation={r.permutation_giam_R2:.4f} (hang {r.hang_permutation})  "
            f"impurity={r.impurity_mac_dinh_gop:.4f} (hang {r.hang_impurity})")
    log("  Da luu permutation_importance.csv/.png")
    return df


# ----------------------------------------------------------------------------
# 9. PDP cho days_left
# ----------------------------------------------------------------------------
def run_pdp(rf_pipe: Pipeline, X_test) -> tuple[pd.DataFrame, dict]:
    Xp = X_test.sample(n=20_000, random_state=RANDOM_STATE).copy()
    Xp["days_left"] = Xp["days_left"].astype(float)
    res = partial_dependence(rf_pipe, Xp, features=["days_left"], kind="average",
                             custom_values={"days_left": np.arange(1, 50, dtype=float)})
    df = pd.DataFrame({"days_left": res["grid_values"][0].astype(int), "gia_du_doan_trung_binh": res["average"][0]})
    # PDP rieng tung hang ve: so tien tuyet doi khac han nhau
    for cls in ["Economy", "Business"]:
        sub = Xp[Xp["class"] == cls]
        r = partial_dependence(rf_pipe, sub, features=["days_left"], kind="average",
                               custom_values={"days_left": np.arange(1, 50, dtype=float)})
        df[f"PDP_{cls}"] = r["average"][0]
    df.to_csv(REPORTS_DIR / "pdp_days_left.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(df["days_left"], df["gia_du_doan_trung_binh"], color="#8e44ad", lw=2)
    axes[0].set_title("PDP days_left - tat ca chuyen"); axes[0].set_ylabel("Gia du doan TB (Rupee)")
    axes[1].plot(df["days_left"], df["PDP_Economy"], color="#27ae60", lw=2, label="Economy")
    ax2 = axes[1].twinx()
    ax2.plot(df["days_left"], df["PDP_Business"], color="#c0392b", lw=2, label="Business")
    axes[1].set_ylabel("Economy (Rupee)", color="#27ae60"); ax2.set_ylabel("Business (Rupee)", color="#c0392b")
    axes[1].set_title("PDP days_left theo hang ve")
    for ax in axes:
        ax.set_xlabel("days_left")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "pdp_days_left.png", dpi=130); plt.close(fig)

    at = df.set_index("days_left")
    savings = {}
    for col in ["gia_du_doan_trung_binh", "PDP_Economy", "PDP_Business"]:
        savings[col] = {"gia_1": float(at.loc[1, col]), "gia_7": float(at.loc[7, col]), "gia_14": float(at.loc[14, col]),
                        "gia_30": float(at.loc[30, col]), "gia_49": float(at.loc[49, col]),
                        "tiet_kiem_7_vs_1": float(at.loc[1, col] - at.loc[7, col]),
                        "tiet_kiem_14_vs_7": float(at.loc[7, col] - at.loc[14, col]),
                        "tiet_kiem_30_vs_7": float(at.loc[7, col] - at.loc[30, col])}
        s = savings[col]
        log(f"    {col:24s} d=1 {s['gia_1']:,.0f} | d=7 {s['gia_7']:,.0f} | d=14 {s['gia_14']:,.0f} | "
            f"d=30 {s['gia_30']:,.0f} | d=49 {s['gia_49']:,.0f}   7vs1 -{s['tiet_kiem_7_vs_1']:,.0f}  "
            f"14vs7 -{s['tiet_kiem_14_vs_7']:,.0f}  30vs7 -{s['tiet_kiem_30_vs_7']:,.0f}")
    log("  Da luu pdp_days_left.csv/.png")
    return df, savings


# ----------------------------------------------------------------------------
# 10. Khoang du bao 10-90% tu cac cay
# ----------------------------------------------------------------------------
def run_intervals(rf_pipe: Pipeline, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    Xt = rf_pipe.named_steps["pre"].transform(X_test)
    tree_preds = np.stack([est.predict(Xt) for est in rf_pipe.named_steps["rf"].estimators_])
    lo, hi = np.percentile(tree_preds, 10, axis=0), np.percentile(tree_preds, 90, axis=0)
    y = y_test.values
    inside = (y >= lo) & (y <= hi)
    summary = {"ty_le_phu_%": float(inside.mean() * 100), "do_rong_TB": float((hi - lo).mean()),
               "do_rong_trung_vi": float(np.median(hi - lo))}
    by_class = {}
    for cls in ["Economy", "Business"]:
        m = (X_test["class"] == cls).values
        by_class[cls] = {"ty_le_phu_%": float(inside[m].mean() * 100), "do_rong_TB": float((hi - lo)[m].mean()),
                         "do_rong_%_gia": float(((hi - lo)[m] / y[m]).mean() * 100)}
    summary["theo_class"] = by_class
    sample = pd.DataFrame({"class": X_test["class"].values[:10], "gia_that": y[:10], "gia_thap_10%": lo[:10],
                           "gia_du_doan_TB": tree_preds.mean(axis=0)[:10], "gia_cao_90%": hi[:10]})
    sample.to_csv(REPORTS_DIR / "khoang_du_bao.csv", index=False)
    log(f"  Ty le phu [10%-90%] = {summary['ty_le_phu_%']:.1f}% (ky vong 80%), do rong TB {summary['do_rong_TB']:,.0f}")
    for cls, v in by_class.items():
        log(f"    {cls:8s}: phu {v['ty_le_phu_%']:.1f}%  do rong TB {v['do_rong_TB']:,.0f} (~{v['do_rong_%_gia']:.0f}% gia)")
    log("  Da luu khoang_du_bao.csv")
    return sample, summary


# ----------------------------------------------------------------------------
# 11. Ngoai suy days_left
# ----------------------------------------------------------------------------
def run_extrapolation(rf_pipe: Pipeline, lr_pipe: Pipeline, X_test) -> pd.DataFrame:
    row = X_test.iloc[[0]].copy()
    rows = []
    for d in [1, 20, 49, 60, 100, 365]:
        r = row.copy(); r["days_left"] = d
        rows.append({"days_left": d, "gia_RF": float(rf_pipe.predict(r)[0]), "gia_Linear": float(lr_pipe.predict(r)[0]),
                     "trong_dai_train (<=49)": d <= 49})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "ngoai_suy_days_left.csv", index=False)
    log(f"  Chuyen mau: {row[['airline', 'source_city', 'destination_city', 'class', 'stops']].values.tolist()[0]}")
    for r in rows:
        log(f"    days_left={r['days_left']:3d}: RF={r['gia_RF']:,.0f}  Linear={r['gia_Linear']:,.0f}")
    log("  Da luu ngoai_suy_days_left.csv")
    return df


# ----------------------------------------------------------------------------
# 12. XGBoost: tuning max_depth x learning_rate, early stopping tren validation tach tu TRAIN
# ----------------------------------------------------------------------------
def tune_xgb(Xt_train, y_train) -> tuple[pd.DataFrame, dict]:
    Xa, Xv, ya, yv = train_test_split(Xt_train, y_train, test_size=0.15, random_state=RANDOM_STATE)
    rows = []
    for depth in XGB_DEPTHS:
        for lr in XGB_LRS:
            t0 = time.time()
            m = XGBRegressor(n_estimators=4000, max_depth=depth, learning_rate=lr, subsample=0.8,
                             colsample_bytree=0.8, early_stopping_rounds=100, eval_metric="rmse",
                             random_state=RANDOM_STATE, n_jobs=-1).fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
            rows.append({"max_depth": depth, "learning_rate": lr, "best_iteration": int(m.best_iteration) + 1,
                         "RMSE_val": float(m.best_score), "fit_s": time.time() - t0})
            log(f"    max_depth={depth:2d} lr={lr:<5} vong tot nhat={rows[-1]['best_iteration']:4d}  "
                f"RMSE_val={rows[-1]['RMSE_val']:.1f}  {rows[-1]['fit_s']:.0f}s")
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "xgb_tuning_val.csv", index=False)
    best = df.loc[df["RMSE_val"].idxmin()]
    params = {"max_depth": int(best["max_depth"]), "learning_rate": float(best["learning_rate"]),
              "n_estimators": int(best["best_iteration"])}
    log(f"  XGB chon {params}. Da luu xgb_tuning_val.csv")
    return df, params


def compare_models(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "so_sanh_models.csv", index=False)
    for r in rows:
        log(f"    {r['model']:48s} MAE={r['MAE']:8,.0f}  RMSE={r['RMSE']:7,.0f}  R2={r['R2']:.4f}  "
            f"fit={r.get('fit_s', float('nan')):6.1f}s  model={r.get('kich_thuoc_MB', float('nan')):7.1f} MB")
    log("  Da luu so_sanh_models.csv")
    return df


# ----------------------------------------------------------------------------
# 13-14. Mo rong: tach Economy/Business, ExtraTrees
# ----------------------------------------------------------------------------
def run_extensions(X_train, y_train, X_test, y_test, params: dict, n_trees: int) -> pd.DataFrame:
    rows = []
    pred = np.zeros(len(X_test))
    t0 = time.time()
    for cls in ["Economy", "Business"]:
        mtr, mte = (X_train["class"] == cls).values, (X_test["class"] == cls).values
        p = Pipeline([("pre", make_preprocess()),
                      ("rf", RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=RANDOM_STATE, **params))])
        p.fit(X_train[mtr], y_train[mtr])
        pred[mte] = p.predict(X_test[mte])
    rows.append({"model": "RF tach 2 model Economy / Business", **evaluate(y_test, pred), "fit_s": time.time() - t0})
    for cls in ["Economy", "Business"]:
        m = (X_test["class"] == cls).values
        rows[-1][f"RMSE_{cls}"] = evaluate(y_test[m], pred[m])["RMSE"]

    t0 = time.time()
    et = Pipeline([("pre", make_preprocess()),
                   ("rf", ExtraTreesRegressor(n_estimators=n_trees, n_jobs=-1, random_state=RANDOM_STATE, **params))])
    et.fit(X_train, y_train)
    fit_s = time.time() - t0
    pred_et = et.predict(X_test)
    rows.append({"model": "ExtraTrees (cung tham so)", **evaluate(y_test, pred_et), "fit_s": fit_s})
    for cls in ["Economy", "Business"]:
        m = (X_test["class"] == cls).values
        rows[-1][f"RMSE_{cls}"] = evaluate(y_test[m], pred_et[m])["RMSE"]
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "mo_rong.csv", index=False)
    for r in rows:
        log(f"    {r['model']:36s} MAE={r['MAE']:,.0f}  RMSE={r['RMSE']:,.0f}  R2={r['R2']:.4f}  {r['fit_s']:.0f}s  "
            f"(RMSE Eco={r['RMSE_Economy']:,.0f}, Bus={r['RMSE_Business']:,.0f})")
    log("  Da luu mo_rong.csv")
    return df


def main() -> None:
    log("1. Nap du lieu...")
    df_raw, df = load_data()

    log("2-3. EDA...")
    eda = run_eda(df)

    log("4. Chia 80/20, OneHotEncoder...")
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    pre = make_preprocess().fit(X_train)
    Xt_train = pre.transform(X_train)
    log(f"  Train {len(X_train):,}  Test {len(X_test):,}  -> {Xt_train.shape[1]} cot sau one-hot")

    log("5. Baseline (train vs test)...")
    baseline_df, tree_cv_df, baselines = run_baselines(X_train, y_train, X_test, y_test)

    log(f"6. Tuning RF: max_features x min_samples_leaf bang OOB ({RF_TUNE_TREES} cay, chi train)...")
    rf_tune_df, rf_params = tune_rf_oob(Xt_train, y_train)

    log("7. RMSE-OOB theo so cay...")
    oob_df, n_trees = oob_curve(Xt_train, y_train, rf_params)

    log(f"   Huan luyen RF cuoi: n_estimators={n_trees}, {rf_params}...")
    t0 = time.time()
    rf_pipe = Pipeline([("pre", make_preprocess()),
                        ("rf", RandomForestRegressor(n_estimators=n_trees, oob_score=True, n_jobs=-1,
                                                     random_state=RANDOM_STATE, **rf_params))]).fit(X_train, y_train)
    rf_fit_s = time.time() - t0
    rf = rf_pipe.named_steps["rf"]
    rf_test = evaluate(y_test, rf_pipe.predict(X_test))
    rf_train = evaluate(y_train, rf_pipe.predict(X_train))
    log(f"  oob_score_={rf.oob_score_:.5f}  R2 test={rf_test['R2']:.5f}  MAE train={rf_train['MAE']:,.0f} "
        f"test={rf_test['MAE']:,.0f}  ({rf_fit_s:.0f}s)")

    log("8. Permutation importance...")
    imp_df = run_importance(rf_pipe, X_test, y_test)

    log("9. PDP days_left...")
    pdp_df, savings = run_pdp(rf_pipe, X_test)

    log("10. Khoang du bao 10-90%...")
    _, interval = run_intervals(rf_pipe, X_test, y_test)

    log("11. Ngoai suy days_left...")
    extra_df = run_extrapolation(rf_pipe, baselines["Linear Regression"], X_test)

    log("12. XGBoost: tuning tren validation tach tu train...")
    xgb_tune_df, xgb_params = tune_xgb(Xt_train, y_train)
    t0 = time.time()
    xgb_pipe = Pipeline([("pre", make_preprocess()),
                         ("xgb", XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                                              n_jobs=-1, **xgb_params))]).fit(X_train, y_train)
    xgb_fit_s = time.time() - t0

    log("   Bang so sanh tren test...")
    rows = [{"model": r["model"], "MAE": r["MAE"], "RMSE": r["RMSE"], "R2": r["R2"], "fit_s": r["fit_s"]}
            for r in baseline_df.to_dict(orient="records")]
    rows.append({"model": f"Random Forest ({n_trees} cay, mf={rf_params['max_features']}, "
                          f"leaf={rf_params['min_samples_leaf']}, chon bang OOB)",
                 **rf_test, "fit_s": rf_fit_s, "kich_thuoc_MB": model_size_mb(rf_pipe, "rf")})
    best_cfg = rf_tune_df.loc[rf_tune_df["RMSE_oob"].idxmin()]
    best_params = {"max_features": float(best_cfg["max_features"]), "min_samples_leaf": int(best_cfg["min_samples_leaf"])}
    if best_params != rf_params:
        t0 = time.time()
        rf_best_pipe = Pipeline([("pre", make_preprocess()),
                                 ("rf", RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=RANDOM_STATE,
                                                              **best_params))]).fit(X_train, y_train)
        fit_s = time.time() - t0
        rows.append({"model": f"Random Forest ({n_trees} cay, mf={best_params['max_features']}, "
                              f"leaf={best_params['min_samples_leaf']}, RMSE-OOB tot nhat tuyet doi)",
                     **evaluate(y_test, rf_best_pipe.predict(X_test)), "fit_s": fit_s,
                     "kich_thuoc_MB": model_size_mb(rf_best_pipe, "rf_best")})
        del rf_best_pipe
    rows.append({"model": f"XGBoost ({xgb_params['n_estimators']} vong, depth={xgb_params['max_depth']}, "
                          f"lr={xgb_params['learning_rate']}, chon bang validation)",
                 **evaluate(y_test, xgb_pipe.predict(X_test)), "fit_s": xgb_fit_s,
                 "kich_thuoc_MB": model_size_mb(xgb_pipe, "xgb")})
    comparison = compare_models(rows)

    log("13-14. Mo rong: tach Economy/Business, ExtraTrees...")
    ext_df = run_extensions(X_train, y_train, X_test, y_test, rf_params, n_trees)

    joblib.dump(rf_pipe, MODELS_DIR / "rf_reg.joblib", compress=3)
    log(f"  Da luu models/rf_reg.joblib ({(MODELS_DIR / 'rf_reg.joblib').stat().st_size / 1e6:.0f} MB, nen zlib-3)")

    summary = {
        "shape_raw": list(df_raw.shape), "shape_clean": list(df.shape), "eda": eda,
        "baseline_train_test": baseline_df.to_dict(orient="records"),
        "cay_don_chon_depth_cv": tree_cv_df.to_dict(orient="records"),
        "rf_tuning_oob": rf_tune_df.to_dict(orient="records"),
        "rf_params_chon": {**rf_params, "n_estimators": n_trees},
        "rmse_oob_theo_so_cay": oob_df.to_dict(orient="records"),
        "random_forest": {"test": rf_test, "train": rf_train, "oob_score_": float(rf.oob_score_)},
        "xgb_tuning_val": xgb_tune_df.to_dict(orient="records"), "xgb_params_chon": xgb_params,
        "so_sanh_models_test": comparison.to_dict(orient="records"),
        "permutation_importance": imp_df.to_dict(orient="records"),
        "pdp_tiet_kiem": savings,
        "khoang_du_bao_10_90": interval,
        "ngoai_suy": extra_df.to_dict(orient="records"),
        "mo_rong": ext_df.to_dict(orient="records"),
    }
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    log("HOAN THANH. Xem reports/ va models/.")


if __name__ == "__main__":
    main()
