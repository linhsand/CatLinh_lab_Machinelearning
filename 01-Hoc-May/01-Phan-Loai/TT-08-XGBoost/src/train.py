"""
TT-08 - XGBoost: Phat hien gian lan the tin dung theo thoi gian thuc.

Pipeline day du:
 1. Nap du lieu, xac nhan ty le lech
 2. Feature engineering: Time -> Hour, Amount -> log1p + StandardScaler
 3. EDA: ty le gian lan theo gio, phan phoi Amount theo lop
 4. Chia du lieu THEO THOI GIAN (khong shuffle): 70% train / 15% val / 15% test
 5. Baseline: DummyClassifier
 6. Logistic Regression (class_weight='balanced')
 7. XGBoost voi scale_pos_weight + early stopping
 8. So sanh ROC-AUC vs PR-AUC
 9. Duong Precision-Recall, chon nguong Precision >= 0.90 TREN VALIDATION,
    chi ap dung 1 lan len TEST de bao cao (khong quet nguong tren TEST)
10. Toi uu nguong theo chi phi TREN VALIDATION (chan nham vs bo lot), bao cao
    ket qua tren TEST + do nhay theo gia dinh chi phi chan nham
10b. Kiem tra calibration (Brier score + duong calibration) tren validation
11. Do thoi gian du doan 1 giao dich
12. So sanh voi Random Forest va LightGBM

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
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42
# Gia dinh quy doi don gian de tinh chi phi bang VND (Amount goc la EUR).
EUR_TO_VND = 27_000
COST_CHAN_NHAM = 200_000  # VND - cham soc khach hang khi chan nham giao dich that
PRECISION_TARGET = 0.90


def log(msg: str) -> None:
    print(f"[TT-08] {msg}")


# ----------------------------------------------------------------------------
# 1-2. Nap du lieu + feature engineering
# ----------------------------------------------------------------------------
def load_and_engineer(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    fraud_rate = df["Class"].mean()
    log(f"Nap {len(df):,} giao dich | ty le gian lan = {fraud_rate:.4%} "
        f"({df['Class'].sum()} / {len(df)})")

    df["Hour"] = (df["Time"] // 3600 % 24).astype(int)
    df["Amount_log"] = np.log1p(df["Amount"])
    return df


# ----------------------------------------------------------------------------
# 3. EDA
# ----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> None:
    fraud_by_hour = df.groupby("Hour")["Class"].agg(["mean", "count", "sum"])
    fraud_by_hour.columns = ["ty_le_gian_lan", "so_giao_dich", "so_gian_lan"]
    fraud_by_hour.to_csv(REPORTS_DIR / "fraud_theo_gio.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(fraud_by_hour.index, fraud_by_hour["ty_le_gian_lan"] * 100, color="#c0392b")
    axes[0].set_xlabel("Gio trong ngay")
    axes[0].set_ylabel("Ty le gian lan (%)")
    axes[0].set_title("Ty le gian lan theo gio trong ngay")

    axes[1].hist(df.loc[df.Class == 0, "Amount_log"], bins=50, alpha=0.6, density=True, label="Hop le")
    axes[1].hist(df.loc[df.Class == 1, "Amount_log"], bins=50, alpha=0.6, density=True, label="Gian lan")
    axes[1].set_xlabel("log1p(Amount)")
    axes[1].set_ylabel("Mat do")
    axes[1].set_title("Phan phoi Amount (log) theo lop")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_hour_amount.png", dpi=130)
    plt.close(fig)
    log("Da luu EDA -> reports/eda_hour_amount.png, reports/fraud_theo_gio.csv")


# ----------------------------------------------------------------------------
# 4. Chia du lieu THEO THOI GIAN
# ----------------------------------------------------------------------------
def time_based_split(df: pd.DataFrame):
    df_sorted = df.sort_values("Time").reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train = df_sorted.iloc[:n_train]
    val = df_sorted.iloc[n_train:n_train + n_val]
    test = df_sorted.iloc[n_train + n_val:]

    for name, part in [("train", train), ("val", val), ("test", test)]:
        log(f"  {name}: {len(part):,} giao dich, {part['Class'].sum()} gian lan "
            f"({part['Class'].mean():.4%})")

    feature_cols = [c for c in df.columns if c.startswith("V")] + ["Amount_scaled", "Hour"]

    scaler = StandardScaler()
    train = train.copy()
    val = val.copy()
    test = test.copy()
    train["Amount_scaled"] = scaler.fit_transform(train[["Amount_log"]])
    val["Amount_scaled"] = scaler.transform(val[["Amount_log"]])
    test["Amount_scaled"] = scaler.transform(test[["Amount_log"]])

    joblib.dump(scaler, MODELS_DIR / "amount_scaler.joblib")

    X_train, y_train = train[feature_cols], train["Class"].to_numpy()
    X_val, y_val = val[feature_cols], val["Class"].to_numpy()
    X_test, y_test = test[feature_cols], test["Class"].to_numpy()
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols, val, test


# ----------------------------------------------------------------------------
# Danh gia
# ----------------------------------------------------------------------------
def evaluate(name: str, y_true, y_score) -> dict:
    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    log(f"  {name:32s} PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}")
    return {"model": name, "pr_auc": pr_auc, "roc_auc": roc_auc}


def measure_predict_latency(predict_fn, X_sample, n_runs: int = 200) -> float:
    row = X_sample.iloc[[0]]
    predict_fn(row)  # warm-up
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_fn(row)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def main() -> None:
    df = load_and_engineer(DATA_PATH)
    run_eda(df)

    log("Chia du lieu theo THOI GIAN (khong shuffle) 70/15/15:")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols, val_raw, test_raw = time_based_split(df)

    results = []

    # 5. Baseline
    log("Huan luyen Baseline (DummyClassifier)...")
    dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    dummy_score = dummy.predict_proba(X_test)[:, 1]
    results.append(evaluate("Baseline (Dummy)", y_test, dummy_score))

    # 6. Logistic Regression
    log("Huan luyen Logistic Regression (class_weight=balanced)...")
    t0 = time.perf_counter()
    logreg = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    logreg.fit(X_train, y_train)
    logreg_train_time = time.perf_counter() - t0
    logreg_score = logreg.predict_proba(X_test)[:, 1]
    res = evaluate("Logistic Regression", y_test, logreg_score)
    res["train_time_s"] = logreg_train_time
    res["predict_ms"] = measure_predict_latency(lambda r: logreg.predict_proba(r), X_test)
    results.append(res)

    # 7. XGBoost
    log("Huan luyen XGBoost (scale_pos_weight + early stopping)...")
    ty_le = (y_train == 0).sum() / (y_train == 1).sum()
    log(f"  scale_pos_weight = {ty_le:.2f}")

    t0 = time.perf_counter()
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ty_le,
        reg_lambda=1.0,
        reg_alpha=0.1,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    xgb_train_time = time.perf_counter() - t0
    log(f"  So cay thuc te sau early stopping: {xgb_model.best_iteration + 1}")

    xgb_score = xgb_model.predict_proba(X_test)[:, 1]
    res = evaluate("XGBoost", y_test, xgb_score)
    res["train_time_s"] = xgb_train_time
    res["best_iteration"] = int(xgb_model.best_iteration + 1)
    res["predict_ms"] = measure_predict_latency(lambda r: xgb_model.predict_proba(r), X_test)
    results.append(res)

    xgb_model.save_model(MODELS_DIR / "xgb_fraud.json")
    log(f"Da luu model -> models/xgb_fraud.json")

    # 12. So sanh voi Random Forest va LightGBM
    log("Huan luyen Random Forest (so sanh)...")
    t0 = time.perf_counter()
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced_subsample",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    rf_train_time = time.perf_counter() - t0
    rf_score = rf.predict_proba(X_test)[:, 1]
    res = evaluate("Random Forest", y_test, rf_score)
    res["train_time_s"] = rf_train_time
    res["predict_ms"] = measure_predict_latency(lambda r: rf.predict_proba(r), X_test)
    results.append(res)

    log("Huan luyen LightGBM (so sanh)...")
    # LUU Y: scale_pos_weight / is_unbalance ket hop voi early stopping o muc
    # do lech 518:1 lam LightGBM (leaf-wise) sup do chi sau 2 vong (ROC-AUC ~0.19,
    # te hon ngau nhien) - khac han XGBoost (depth-wise) van on dinh voi cung
    # scale_pos_weight. Day la phat hien thuc nghiem, khong phai loi cai dat:
    # khong bat class-reweighting cho LightGBM van cho ket qua on dinh va canh
    # tranh duoc voi XGBoost tren bo du lieu nay.
    t0 = time.perf_counter()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.1,
        n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1,
    )
    lgb_model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    lgb_train_time = time.perf_counter() - t0
    lgb_score = lgb_model.predict_proba(X_test)[:, 1]
    res = evaluate("LightGBM", y_test, lgb_score)
    res["train_time_s"] = lgb_train_time
    res["predict_ms"] = measure_predict_latency(lambda r: lgb_model.predict_proba(r), X_test)
    results.append(res)

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORTS_DIR / "so_sanh_mo_hinh.csv", index=False)
    log("Bang so sanh mo hinh:\n" + results_df.to_string(index=False))

    # 8-9. PR-AUC vs ROC-AUC + duong Precision-Recall / ROC cua XGBoost
    fpr, tpr, _ = roc_curve(y_test, xgb_score)
    prec, rec, pr_thresholds = precision_recall_curve(y_test, xgb_score)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(fpr, tpr, color="#2980b9", label=f"XGBoost (ROC-AUC={results[2]['roc_auc']:.4f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve - trong dep, gay hieu lam voi du lieu lech")
    axes[0].legend()

    axes[1].plot(rec, prec, color="#c0392b", label=f"XGBoost (PR-AUC={results[2]['pr_auc']:.4f})")
    axes[1].axhline(PRECISION_TARGET, ls="--", color="gray", label=f"Precision muc tieu = {PRECISION_TARGET:.0%}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve (TEST) - phan anh dung nang luc that")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "pr_vs_roc.png", dpi=130)
    plt.close(fig)
    log("Da luu -> reports/pr_vs_roc.png")

    # ------------------------------------------------------------------------
    # 9-10. CHON NGUONG TREN VALIDATION, chi bao cao ket qua tren TEST 1 LAN
    # ------------------------------------------------------------------------
    # LUU Y QUAN TRONG: ca nguong theo Precision (muc 9) va nguong theo chi
    # phi (muc 10) BAT BUOC phai duoc do/quet tren VALIDATION, roi chi ap
    # dung 1 LAN DUY NHAT len TEST de bao cao so cuoi cung. Neu quet nguong
    # truc tiep tren TEST roi bao cao Precision/Recall/chi phi cung tren
    # chinh TEST do, ket qua se bi lac quan ao vi da "nhin thay" nhan that
    # cua tap dung de bao cao - cung mot dang ro ri nhu da xu ly o TT-04/TT-05,
    # chi khac la ro ri qua buoc chon nguong thay vi qua buoc huan luyen.
    log("Chon nguong tren VALIDATION (khong dung TEST de chon nguong):")
    xgb_score_val = xgb_model.predict_proba(X_val)[:, 1]
    prec_val, rec_val, pr_thresholds_val = precision_recall_curve(y_val, xgb_score_val)

    valid_idx = np.where(prec_val[:-1] >= PRECISION_TARGET)[0]
    if len(valid_idx) > 0:
        idx = valid_idx[np.argmax(rec_val[valid_idx])]
        thr_precision = float(pr_thresholds_val[idx])
        log(f"  [VAL] Nguong dat Precision >= {PRECISION_TARGET:.0%}: threshold={thr_precision:.4f}, "
            f"Precision={prec_val[idx]:.4f}, Recall={rec_val[idx]:.4f}")
    else:
        thr_precision = None
        log(f"  [VAL] Khong tim thay nguong nao dat Precision >= {PRECISION_TARGET:.0%}")

    def cost_sweep(y_true, y_score, amounts, thresholds, cost_fp=COST_CHAN_NHAM):
        costs = []
        for thr in thresholds:
            pred = (y_score >= thr).astype(int)
            fp_mask = (pred == 1) & (y_true == 0)
            fn_mask = (pred == 0) & (y_true == 1)
            costs.append(fp_mask.sum() * cost_fp + amounts[fn_mask].sum())
        return np.array(costs)

    def threshold_report(y_true, y_score, threshold, amounts, cost_fp=COST_CHAN_NHAM):
        pred = (y_score >= threshold).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        cost = fp * cost_fp + amounts[(pred == 0) & (y_true == 1)].sum()
        return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "total_cost_vnd": float(cost)}

    amounts_val = val_raw["Amount"].to_numpy() * EUR_TO_VND
    amounts_test = test_raw["Amount"].to_numpy() * EUR_TO_VND
    thresholds = np.linspace(0.01, 0.99, 99)

    costs_val = cost_sweep(y_val, xgb_score_val, amounts_val, thresholds)
    best_idx = np.argmin(costs_val)
    best_threshold = float(thresholds[best_idx])
    best_cost_val = float(costs_val[best_idx])
    log(f"  [VAL] Nguong toi uu chi phi: threshold={best_threshold:.2f}, "
        f"tong chi phi uoc tinh tren VAL={best_cost_val:,.0f} VND")

    # Duong chi phi tren TEST duoi day chi de THAM KHAO/DOI CHIEU (khong
    # dung de chon nguong) - kiem tra nguong chon tu VAL co gan voi nguong
    # toi uu that tren TEST hay khong, tuc do "do on dinh" cua lua chon.
    costs_test = cost_sweep(y_test, xgb_score, amounts_test, thresholds)
    thr_test_only = float(thresholds[np.argmin(costs_test)])
    log(f"  [TEST - chi de doi chieu] Neu do truc tiep tren TEST, nguong toi uu se la "
        f"{thr_test_only:.2f} (KHONG dung so nay de chon nguong san xuat)")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(thresholds, costs_val / 1e6, color="#8e44ad", label="Chi phi tren VALIDATION (dung de chon nguong)")
    ax.plot(thresholds, costs_test / 1e6, color="#7f8c8d", ls=":", label="Chi phi tren TEST (chi de doi chieu)")
    ax.axvline(best_threshold, ls="--", color="#c0392b",
               label=f"Nguong toi uu (chon tren VAL) = {best_threshold:.2f}")
    ax.set_xlabel("Nguong xac suat")
    ax.set_ylabel("Tong chi phi uoc tinh (trieu VND)")
    ax.set_title("Chi phi theo nguong: chan nham (200.000d) vs bo lot (so tien giao dich)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "chi_phi_theo_nguong.png", dpi=130)
    plt.close(fig)
    log("Da luu -> reports/chi_phi_theo_nguong.png")

    threshold_summary = pd.DataFrame({
        "threshold": thresholds,
        "total_cost_val_vnd": costs_val,
        "total_cost_test_vnd_doi_chieu": costs_test,
    })
    threshold_summary.to_csv(REPORTS_DIR / "chi_phi_theo_nguong.csv", index=False)

    # Ap dung ca hai nguong (da chon xong tren VAL) len TEST DUY NHAT 1 LAN
    # de bao cao ket qua cuoi cung - day la Precision/Recall/chi phi THAT,
    # khong bi ro ri qua buoc chon nguong.
    report_precision_thr = None
    if thr_precision is not None:
        report_precision_thr = threshold_report(y_test, xgb_score, thr_precision, amounts_test)
        log(f"  [TEST - BAO CAO CUOI] Tai nguong Precision>={PRECISION_TARGET:.0%} (chon tren VAL): "
            f"Precision={report_precision_thr['precision']:.4f}, Recall={report_precision_thr['recall']:.4f}")

    report_cost_thr = threshold_report(y_test, xgb_score, best_threshold, amounts_test)
    log(f"  [TEST - BAO CAO CUOI] Tai nguong toi uu chi phi (chon tren VAL): "
        f"tong chi phi tren TEST={report_cost_thr['total_cost_vnd']:,.0f} VND, "
        f"Precision={report_cost_thr['precision']:.4f}, Recall={report_cost_thr['recall']:.4f}")

    # ------------------------------------------------------------------------
    # Do nhay theo gia dinh chi phi chan nham (COST_CHAN_NHAM co the sai)
    # ------------------------------------------------------------------------
    # Chi phi 200.000d/FP la mot GIA DINH, khong phai so do luong duoc. Quet
    # lai toan bo quy trinh chon-nguong-tren-VAL / bao-cao-tren-TEST voi vai
    # muc chi phi khac nhau de xem nguong toi uu va ket qua nhay den dau khi
    # gia dinh thay doi - tranh bao cao "mot con so duy nhat" nhu the no
    # chac chan dung.
    cost_fp_candidates = [100_000, 200_000, 300_000, 500_000]
    sensitivity_rows = []
    for cost_fp in cost_fp_candidates:
        costs_v = cost_sweep(y_val, xgb_score_val, amounts_val, thresholds, cost_fp=cost_fp)
        thr = float(thresholds[np.argmin(costs_v)])
        rep = threshold_report(y_test, xgb_score, thr, amounts_test, cost_fp=cost_fp)
        sensitivity_rows.append({
            "cost_fp_vnd": cost_fp,
            "threshold_chon_tren_val": thr,
            "test_precision": rep["precision"],
            "test_recall": rep["recall"],
            "test_total_cost_vnd": rep["total_cost_vnd"],
        })
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(REPORTS_DIR / "do_nhay_chi_phi.csv", index=False)
    log("Do nhay theo gia dinh chi phi chan nham (nguong chon tren VAL, bao cao tren TEST):\n"
        + sensitivity_df.to_string(index=False))

    # ------------------------------------------------------------------------
    # Kiem tra calibration (xac suat xgb_score co dang la xac suat that khong)
    # ------------------------------------------------------------------------
    # Ca hai nguong van hanh (~0,97-0,98) deu duoc doc nhu the xgb_score la
    # mot xac suat that ("tin 97%"). XGBoost toi uu log-loss nen thuong kha
    # calibrated, nhung khong co gi dam bao - phai kiem tra bang duong
    # calibration + Brier score tren VALIDATION (khong dung TEST) truoc khi
    # tin ngưỡng theo nghia xac suat.
    frac_pos, mean_pred = calibration_curve(y_val, xgb_score_val, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_val, xgb_score_val)
    calib_df = pd.DataFrame({"mean_predicted_prob": mean_pred, "fraction_of_positives": frac_pos})
    calib_df.to_csv(REPORTS_DIR / "calibration.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(mean_pred, frac_pos, "o-", color="#2c3e50", label="XGBoost (VAL)")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Calibration hoan hao")
    ax.set_xlabel("Xac suat du doan trung binh (moi bin)")
    ax.set_ylabel("Ty le thuc te la gian lan")
    ax.set_title(f"Duong calibration (VAL) - Brier score={brier:.5f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "calibration.png", dpi=130)
    plt.close(fig)
    log(f"Da luu -> reports/calibration.png (Brier score tren VAL={brier:.5f})")

    # Feature importance
    importances = pd.Series(xgb_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    importances.iloc[::-1].plot.barh(ax=ax, color="#16a085")
    ax.set_title("Top 15 dac trung quan trong nhat (XGBoost)")
    ax.set_xlabel("Feature importance (gain)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=130)
    plt.close(fig)
    log("Da luu -> reports/feature_importance.png")

    # 11. Tom tat thoi gian du doan + tong hop threshold
    summary = {
        "fraud_rate": float(df["Class"].mean()),
        "n_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "scale_pos_weight": float(ty_le),
        "xgb_best_iteration": int(xgb_model.best_iteration + 1),
        "threshold_selection_note": (
            "Ca hai nguong duoi day duoc CHON tren VALIDATION (khong dung TEST), "
            "chi ap dung 1 lan len TEST de bao cao Precision/Recall/chi phi cuoi cung."
        ),
        "threshold_precision_90": None if report_precision_thr is None else {
            "threshold": report_precision_thr["threshold"],
            "test_precision": report_precision_thr["precision"],
            "test_recall": report_precision_thr["recall"],
        },
        "threshold_cost_optimal": {
            "threshold": report_cost_thr["threshold"],
            "val_total_cost_vnd": best_cost_val,
            "test_precision": report_cost_thr["precision"],
            "test_recall": report_cost_thr["recall"],
            "test_total_cost_vnd": report_cost_thr["total_cost_vnd"],
        },
        "cost_sensitivity_note": "Xem reports/do_nhay_chi_phi.csv cho cac gia dinh COST_CHAN_NHAM khac nhau.",
        "calibration_brier_score_val": float(brier),
        "predict_latency_ms": {r["model"]: r.get("predict_ms") for r in results if "predict_ms" in r},
    }
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("HOAN THANH. Xem ket qua chi tiet trong thu muc reports/.")


if __name__ == "__main__":
    main()
