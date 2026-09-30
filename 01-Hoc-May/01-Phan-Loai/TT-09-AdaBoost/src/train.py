"""
TT-09 - AdaBoost: Phat hien xam nhap mang trong he thong giam sat an ninh (SOC).

Pipeline day du:
 1. Nap du lieu NSL-KDD (train + test goc), gop nhan thanh nhi phan normal(0)/attack(1)
 2. Tien xu ly: one-hot 3 cot phan loai, scale cac cot so (fit CHI tren phan du lieu dung de hoc)
 3. EDA: phan bo loai tan cong (DoS/Probe/R2L/U2R), ty le normal/attack
 4. Baseline: DummyClassifier + 1 stump don le (depth=1) - 5-fold CV tren train
 5. TUNE AdaBoost: learning_rate x n_estimators bang 5-fold CV (staged_decision_function)
    + DO NGUONG quyet dinh tren diem out-of-fold (toi da F2) - KHONG dung test
 6. Duong Accuracy/F1 theo n_estimators (staged_predict tren tap validation)
 7. THI NGHIEM NHIEU: dao nguoc 5% nhan train -> AdaBoost vs Random Forest
 8. So sanh AdaBoost vs Gradient Boosting (TT-07) vs Random Forest (TT-03)
 9. Danh gia model AdaBoost cuoi cung (fit tren toan bo train) tren tap test NSL-KDD
    goc (co chua 17 loai tan cong CHUA TUNG THAY - mo phong zero-day), nguong 0 vs nguong da do
10. Ma tran nham lan + uoc tinh so bao dong gia/ngay (alert fatigue)
11. Phan tich FN theo tung loai tan cong: da thay (seen) vs chua thay (unseen)

Buoc 6-8 tach train_sub/val tu DataFrame THO truoc, roi moi fit preprocessor tren
train_sub -> scaler khong "thay" val.

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
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, fbeta_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "KDDTrain+.txt"
TEST_PATH = DATA_DIR / "KDDTest+.txt"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42
N_FOLDS = 5
# Luoi tune: moi (fold, learning_rate) chi fit 1 lan voi N_MAX cay, roi doc diem
# tai tung moc n_estimators bang staged_decision_function.
LR_GRID = [0.1, 0.5, 1.0]
N_GRID = [50, 100, 200, 300, 400, 500]
N_MAX = max(N_GRID)
# Cau hinh cu (truoc khi tune) - giu lai de so sanh
OLD_LR, OLD_N = 0.5, 300
# F2: Recall nang gap 2 lan Precision - bo lot tan cong dat hon bao dong gia (README muc 2)
THRESHOLD_BETA = 2.0
NOISE_FRACTION = 0.05
# Gia dinh quy mo SOC de uoc tinh bao dong gia/ngay (khong phai so lieu thuc te
# cua mot to chuc cu the, chi dung minh hoa phuong phap tinh).
ASSUMED_DAILY_CONNECTIONS = 2_000_000

FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
ALL_COLUMNS = FEATURE_COLUMNS + ["label", "difficulty_level"]
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
NUMERIC_COLS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLS]

ATTACK_CATEGORY = {
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS",
    "worm": "DoS", "mailbomb": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


def log(msg: str) -> None:
    print(f"[TT-09] {msg}")


def records(df: pd.DataFrame) -> list[dict]:
    """to_dict an toan cho JSON: NaN -> None (json.dump se ghi NaN khong hop le)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


# ----------------------------------------------------------------------------
# 1-2. Nap du lieu + gop nhan nhi phan + tien xu ly
# ----------------------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Thieu {path.name} - chay: python data/download_data.py")
    df = pd.read_csv(path, names=ALL_COLUMNS, header=None)
    df["binary_label"] = (df["label"] != "normal").astype(int)
    df["attack_category"] = np.where(
        df["label"] == "normal", "Normal", df["label"].map(ATTACK_CATEGORY).fillna("Unknown")
    )
    return df


def build_preprocessor() -> ColumnTransformer:
    # Dau ra dense: cay quyet dinh chay nhanh hon nhieu tren ma tran dense
    # (126k x 122 float64 ~ 120MB, du nho de giu trong RAM).
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ],
        sparse_threshold=0.0,
    )


def make_adaboost(learning_rate: float, n_estimators: int) -> AdaBoostClassifier:
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n_estimators, learning_rate=learning_rate, random_state=RANDOM_STATE,
    )


# ----------------------------------------------------------------------------
# 3. EDA
# ----------------------------------------------------------------------------
def run_eda(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    cat_train = train_df["attack_category"].value_counts()
    cat_test = test_df["attack_category"].value_counts()
    cat_table = pd.DataFrame({"train": cat_train, "test": cat_test}).fillna(0).astype(int)
    cat_table.to_csv(REPORTS_DIR / "phan_bo_loai_tan_cong.csv")

    unseen_labels = sorted(set(test_df["label"]) - set(train_df["label"]))
    n_unseen_rows = int(test_df["label"].isin(unseen_labels).sum())
    log(f"  So loai tan cong CHI co trong test (zero-day mo phong): {len(unseen_labels)} "
        f"-> {unseen_labels}")
    log(f"  So dong test thuoc cac loai la: {n_unseen_rows:,} / {len(test_df):,}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    cat_table["train"].sort_values(ascending=False).plot.bar(ax=axes[0], color="#2980b9")
    axes[0].set_title("Phan bo nhom tan cong - tap TRAIN")
    axes[0].set_ylabel("So dong")
    axes[0].set_yscale("log")

    ratio = train_df["binary_label"].value_counts(normalize=True).rename({0: "normal", 1: "attack"})
    axes[1].bar(ratio.index.astype(str), ratio.values * 100, color=["#27ae60", "#c0392b"])
    axes[1].set_title("Ty le normal vs attack - tap TRAIN")
    axes[1].set_ylabel("Ty le (%)")
    for i, v in enumerate(ratio.values * 100):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha="center")

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_phan_bo_tan_cong.png", dpi=130)
    plt.close(fig)
    log("Da luu EDA -> reports/eda_phan_bo_tan_cong.png, reports/phan_bo_loai_tan_cong.csv")
    return unseen_labels


# ----------------------------------------------------------------------------
# 4. Baseline (Dummy + 1 stump) - 5-fold CV
# ----------------------------------------------------------------------------
def run_cv_baseline(X_train_full: pd.DataFrame, y_train_full: np.ndarray) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "Baseline (Dummy)": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "1 Stump (depth=1)": DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
    }

    rows = []
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        scores = cross_validate(pipe, X_train_full, y_train_full, cv=skf,
                                scoring=("accuracy", "f1"), n_jobs=-1)
        row = {
            "model": name,
            "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
            "cv_f1_mean": float(np.mean(scores["test_f1"])),
            "cv_f1_std": float(np.std(scores["test_f1"])),
        }
        rows.append(row)
        log(f"  {name:26s} CV Accuracy={row['cv_accuracy_mean']:.4f} (+-{row['cv_accuracy_std']:.4f})  "
            f"CV F1={row['cv_f1_mean']:.4f} (+-{row['cv_f1_std']:.4f})")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 5a. Tune learning_rate x n_estimators (5-fold CV, staged_decision_function)
# ----------------------------------------------------------------------------
def _fit_fold_staged(X: pd.DataFrame, y: np.ndarray, tr_idx: np.ndarray, va_idx: np.ndarray,
                     lr: float) -> dict[int, np.ndarray]:
    """Fit 1 fold voi N_MAX stump; tra ve diem decision_function tai tung moc N_GRID."""
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", make_adaboost(lr, N_MAX))])
    pipe.fit(X.iloc[tr_idx], y[tr_idx])
    Xva = pipe[:-1].transform(X.iloc[va_idx])
    out = {}
    for n, dec in enumerate(pipe[-1].staged_decision_function(Xva), start=1):
        if n in N_GRID:
            out[n] = dec
    return out


def run_tuning(X_train_full: pd.DataFrame, y_train_full: np.ndarray):
    """Tra ve (bang ket qua CV, dict out-of-fold decision score theo (lr, n))."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X_train_full, y_train_full))
    jobs = [(k, lr) for k in range(N_FOLDS) for lr in LR_GRID]

    t0 = time.perf_counter()
    results = Parallel(n_jobs=-1)(
        delayed(_fit_fold_staged)(X_train_full, y_train_full, folds[k][0], folds[k][1], lr)
        for k, lr in jobs
    )
    log(f"  Da fit {len(jobs)} mo hinh ({N_FOLDS} fold x {len(LR_GRID)} learning_rate, "
        f"{N_MAX} stump moi mo hinh) trong {time.perf_counter() - t0:.0f}s")

    oof = {(lr, n): np.zeros(len(y_train_full)) for lr in LR_GRID for n in N_GRID}
    fold_f1 = {(lr, n): [] for lr in LR_GRID for n in N_GRID}
    fold_acc = {(lr, n): [] for lr in LR_GRID for n in N_GRID}
    for (k, lr), staged in zip(jobs, results):
        va_idx = folds[k][1]
        for n, dec in staged.items():
            oof[(lr, n)][va_idx] = dec
            pred = (dec > 0).astype(int)   # nguong mac dinh = predict()
            fold_f1[(lr, n)].append(f1_score(y_train_full[va_idx], pred))
            fold_acc[(lr, n)].append(accuracy_score(y_train_full[va_idx], pred))

    rows = [{
        "learning_rate": lr, "n_estimators": n,
        "cv_accuracy_mean": float(np.mean(fold_acc[(lr, n)])),
        "cv_accuracy_std": float(np.std(fold_acc[(lr, n)])),
        "cv_f1_mean": float(np.mean(fold_f1[(lr, n)])),
        "cv_f1_std": float(np.std(fold_f1[(lr, n)])),
    } for lr in LR_GRID for n in N_GRID]
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / "tune_adaboost_cv.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for lr, color in zip(LR_GRID, ["#8e44ad", "#c0392b", "#16a085"]):
        sub = table[table["learning_rate"] == lr]
        ax.errorbar(sub["n_estimators"], sub["cv_f1_mean"], yerr=sub["cv_f1_std"],
                    marker="o", capsize=3, label=f"learning_rate={lr}", color=color)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel(f"F1 trung binh {N_FOLDS}-fold CV")
    ax.set_title("Tune AdaBoost (stump): learning_rate x n_estimators")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "tune_adaboost_cv.png", dpi=130)
    plt.close(fig)

    best = table.loc[table["cv_f1_mean"].idxmax()]
    log(f"  Tot nhat: learning_rate={best['learning_rate']}, n_estimators={int(best['n_estimators'])} "
        f"-> CV F1={best['cv_f1_mean']:.4f} (+-{best['cv_f1_std']:.4f})")
    log("Da luu -> reports/tune_adaboost_cv.png, reports/tune_adaboost_cv.csv")
    return table, oof


# ----------------------------------------------------------------------------
# 5b. Do nguong tren diem out-of-fold (chi dung train)
# ----------------------------------------------------------------------------
def choose_threshold(y: np.ndarray, oof_score: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y, oof_score)
    p, r = precision[:-1], recall[:-1]
    b2 = THRESHOLD_BETA ** 2
    fbeta = np.where(p + r > 0, (1 + b2) * p * r / (b2 * p + r + 1e-12), 0.0)
    i = int(np.argmax(fbeta))
    thr = float(thresholds[i])

    def metrics_at(t: float, name: str) -> dict:
        pred = (oof_score >= t).astype(int) if name != "Mac dinh (0)" else (oof_score > 0).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {
            "nguong": name, "gia_tri": t,
            "precision": precision_score(y, pred), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "f2": fbeta_score(y, pred, beta=THRESHOLD_BETA),
            "fpr": fp / (fp + tn), "fnr": fn / (fn + tp),
        }

    table = pd.DataFrame([metrics_at(0.0, "Mac dinh (0)"), metrics_at(thr, f"Toi da F{THRESHOLD_BETA:g} (OOF)")])
    table.to_csv(REPORTS_DIR / "nguong_oof.csv", index=False)
    for _, row in table.iterrows():
        log(f"  OOF {row['nguong']:18s} t={row['gia_tri']:+.4f}  P={row['precision']:.4f}  "
            f"R={row['recall']:.4f}  F1={row['f1']:.4f}  F2={row['f2']:.4f}  FPR={row['fpr']:.4%}")
    log("Da luu -> reports/nguong_oof.csv")
    return {"threshold": thr, "table": table}


# ----------------------------------------------------------------------------
# 6. Duong F1/Accuracy theo n_estimators (staged_predict)
# ----------------------------------------------------------------------------
def run_staged_curve(Xtr, ytr, Xval, yval, lr: float, n_best: int):
    ada = make_adaboost(lr, N_MAX)
    t0 = time.perf_counter()
    ada.fit(Xtr, ytr)
    train_time = time.perf_counter() - t0

    rows = []
    for i, pred in enumerate(ada.staged_predict(Xval), start=1):
        rows.append({"n_estimators": i, "accuracy": accuracy_score(yval, pred), "f1": f1_score(yval, pred)})
    curve = pd.DataFrame(rows)
    curve.to_csv(REPORTS_DIR / "f1_theo_vong_lap.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve["n_estimators"], curve["f1"], label="F1-score", color="#c0392b")
    ax.plot(curve["n_estimators"], curve["accuracy"], label="Accuracy", color="#2980b9", ls="--")
    ax.axvline(n_best, color="grey", ls=":", label=f"n chon qua CV = {n_best}")
    ax.set_xlabel("So vong lap (n_estimators)")
    ax.set_ylabel("Diem so tren tap validation")
    ax.set_title(f"AdaBoost (learning_rate={lr}): F1/Accuracy theo so vong lap (1..{N_MAX})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "f1_theo_vong_lap.png", dpi=130)
    plt.close(fig)
    log(f"  F1 tai n=1: {curve['f1'].iloc[0]:.4f}  |  n={n_best} (chon qua CV): {curve['f1'].iloc[n_best - 1]:.4f}")
    log("Da luu -> reports/f1_theo_vong_lap.png, reports/f1_theo_vong_lap.csv")

    # Model dung cho buoc 7-8: dung n da chon, khong phai N_MAX
    ada_best = make_adaboost(lr, n_best)
    t0 = time.perf_counter()
    ada_best.fit(Xtr, ytr)
    return ada_best, time.perf_counter() - t0, curve


# ----------------------------------------------------------------------------
# 7. THI NGHIEM NHIEU NHAN
# ----------------------------------------------------------------------------
def run_noise_experiment(Xtr, ytr, Xval, yval, ada_clean: AdaBoostClassifier,
                         rf_clean: RandomForestClassifier) -> pd.DataFrame:
    rng = np.random.RandomState(RANDOM_STATE)
    n_flip = int(NOISE_FRACTION * len(ytr))
    flip_idx = rng.choice(len(ytr), size=n_flip, replace=False)
    ytr_noisy = ytr.copy()
    ytr_noisy[flip_idx] = 1 - ytr_noisy[flip_idx]
    log(f"  Da dao nguoc {n_flip:,} / {len(ytr):,} nhan train ({NOISE_FRACTION:.0%})")

    ada_noisy = make_adaboost(ada_clean.learning_rate, ada_clean.n_estimators)
    ada_noisy.fit(Xtr, ytr_noisy)
    rf_noisy = RandomForestClassifier(n_estimators=rf_clean.n_estimators, max_depth=None,
                                      random_state=RANDOM_STATE, n_jobs=-1)
    rf_noisy.fit(Xtr, ytr_noisy)

    rows = []
    for model_name, clean_model, noisy_model in [
        ("AdaBoost", ada_clean, ada_noisy),
        ("Random Forest", rf_clean, rf_noisy),
    ]:
        f1_clean = f1_score(yval, clean_model.predict(Xval))
        f1_noisy = f1_score(yval, noisy_model.predict(Xval))
        rows.append({"model": model_name, "f1_clean": f1_clean, "f1_noisy_5pct": f1_noisy,
                     "sut_giam_f1": f1_clean - f1_noisy})
        log(f"  {model_name:16s} F1 sach={f1_clean:.4f}  F1 nhieu 5%={f1_noisy:.4f}  "
            f"sut giam={f1_clean - f1_noisy:.4f}")

    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "thi_nghiem_nhieu.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(result))
    width = 0.35
    ax.bar(x - width / 2, result["f1_clean"], width, label="Nhan sach", color="#2980b9")
    ax.bar(x + width / 2, result["f1_noisy_5pct"], width, label=f"Nhieu {NOISE_FRACTION:.0%} nhan", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(result["model"])
    ax.set_ylabel("F1-score (tap validation sach)")
    ax.set_title("Thi nghiem nhieu nhan: AdaBoost vs Random Forest")
    ax.legend()
    for i, (c, n) in enumerate(zip(result["f1_clean"], result["f1_noisy_5pct"])):
        ax.text(i - width / 2, c + 0.005, f"{c:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, n + 0.005, f"{n:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "thi_nghiem_nhieu.png", dpi=130)
    plt.close(fig)
    log("Da luu -> reports/thi_nghiem_nhieu.png, reports/thi_nghiem_nhieu.csv")
    return result


# ----------------------------------------------------------------------------
# 8. So sanh AdaBoost vs Gradient Boosting vs Random Forest
# ----------------------------------------------------------------------------
def run_ensemble_comparison(Xtr, ytr, Xval, yval, fitted: dict) -> pd.DataFrame:
    """fitted: {ten: (model da fit, thoi gian train)}; Gradient Boosting duoc fit tai day."""
    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                    random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    gb.fit(Xtr, ytr)
    fitted = {**fitted, "Gradient Boosting": (gb, time.perf_counter() - t0)}

    rows = []
    for name, (model, train_time) in fitted.items():
        pred = model.predict(Xval)
        rows.append({"model": name, "accuracy": accuracy_score(yval, pred),
                     "f1": f1_score(yval, pred), "train_time_s": train_time})
        log(f"  {name:20s} Accuracy={rows[-1]['accuracy']:.4f}  F1={rows[-1]['f1']:.4f}  "
            f"[{train_time:.1f}s]")

    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "so_sanh_ensemble.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(result))
    width = 0.35
    ax.bar(x - width / 2, result["accuracy"], width, label="Accuracy", color="#2980b9")
    ax.bar(x + width / 2, result["f1"], width, label="F1-score", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(result["model"])
    ax.set_ylim(0.9, 1.0)
    ax.set_title("So sanh AdaBoost vs Gradient Boosting vs Random Forest (tap validation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "so_sanh_ensemble.png", dpi=130)
    plt.close(fig)
    log("Da luu -> reports/so_sanh_ensemble.png, reports/so_sanh_ensemble.csv")
    return result


# ----------------------------------------------------------------------------
# 9-10. Danh gia tren test NSL-KDD goc + ma tran nham lan + bao dong gia/ngay
# ----------------------------------------------------------------------------
def _test_metrics(y_test: np.ndarray, pred: np.ndarray, name: str, thr: float) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn)
    normal_share = (y_test == 0).mean()
    return {
        "nguong": name, "gia_tri": thr,
        "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
        "f2": fbeta_score(y_test, pred, beta=THRESHOLD_BETA),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": fpr, "fnr": fn / (fn + tp),
        "bao_dong_gia_moi_ngay": ASSUMED_DAILY_CONNECTIONS * normal_share * fpr,
    }


def run_final_test_eval(final_pipe: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray,
                        threshold: float, cv_f1_reference: float):
    """Test goc duoc dung DUNG 1 LAN: nguong da chon tu truoc tren OOF cua train."""
    score_test = final_pipe.decision_function(X_test)
    preds = {
        "Mac dinh (0)": (score_test > 0).astype(int),
        f"Da do F{THRESHOLD_BETA:g} (OOF)": (score_test >= threshold).astype(int),
    }
    table = pd.DataFrame([
        _test_metrics(y_test, preds["Mac dinh (0)"], "Mac dinh (0)", 0.0),
        _test_metrics(y_test, preds[f"Da do F{THRESHOLD_BETA:g} (OOF)"],
                      f"Da do F{THRESHOLD_BETA:g} (OOF)", threshold),
    ])
    table.to_csv(REPORTS_DIR / "danh_gia_test.csv", index=False)
    for _, r in table.iterrows():
        log(f"  TEST {r['nguong']:18s} F1={r['f1']:.4f}  Recall={r['recall']:.4f}  "
            f"FPR={r['fpr']:.4%}  FNR={r['fnr']:.4%}  -> ~{r['bao_dong_gia_moi_ngay']:,.0f} bao dong gia/ngay")
    log(f"  CV F1 (train) = {cv_f1_reference:.4f}  vs  test F1 (nguong 0) = {table['f1'].iloc[0]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (_, r) in zip(axes, table.iterrows()):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "attack"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["normal", "attack"])
        ax.set_xlabel("Du doan"); ax.set_ylabel("Thuc te")
        ax.set_title(f"Nguong {r['nguong']} (t={r['gia_tri']:+.3f})")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.suptitle("Ma tran nham lan - tap test NSL-KDD goc")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix_test.png", dpi=130)
    plt.close(fig)
    log("Da luu -> reports/confusion_matrix_test.png, reports/danh_gia_test.csv")
    return table, score_test, preds


def run_threshold_sensitivity(y_test: np.ndarray, score_test: np.ndarray, threshold: float) -> pd.DataFrame:
    """Phan tich SAU (post-hoc) tren test: KHONG dung de chon nguong, chi de thay
    doi FNR/FPR khi dich nguong -> dich nguong co cuu duoc tan cong bi bo lot khong?"""
    grid = np.round(np.arange(-0.30, 0.101, 0.02), 2)
    rows = []
    for t in grid:
        pred = (score_test >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append({"nguong": float(t), "fpr": fp / (fp + tn), "fnr": fn / (fn + tp),
                     "f1": f1_score(y_test, pred)})
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / "nhay_nguong_test_posthoc.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(table["nguong"], table["fnr"] * 100, marker=".", color="#c0392b", label="FNR (bo lot)")
    ax.plot(table["nguong"], table["fpr"] * 100, marker=".", color="#2980b9", label="FPR (bao dong gia)")
    ax.axvline(0, color="grey", ls="--", label="nguong mac dinh 0")
    ax.axvline(threshold, color="black", ls=":", label=f"nguong OOF {threshold:+.3f}")
    ax.set_xlabel("Nguong tren decision_function")
    ax.set_ylabel("Ty le (%)")
    ax.set_title("Post-hoc tren test: FNR/FPR khi dich nguong (khong dung de chon nguong)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "nhay_nguong_test_posthoc.png", dpi=130)
    plt.close(fig)
    log("Da luu -> reports/nhay_nguong_test_posthoc.png, reports/nhay_nguong_test_posthoc.csv")
    return table


# ----------------------------------------------------------------------------
# 11. FN theo tung loai tan cong: seen vs unseen
# ----------------------------------------------------------------------------
def run_fn_breakdown(test_df: pd.DataFrame, pred: np.ndarray, unseen_labels: list[str],
                     save: bool = True):
    att = test_df.assign(pred=pred)
    att = att[att["binary_label"] == 1].copy()
    att["nhom_thay"] = np.where(att["label"].isin(unseen_labels), "unseen (chi co o test)", "seen (co trong train)")
    att["fn"] = (att["pred"] == 0).astype(int)

    by_label = (att.groupby(["label", "attack_category", "nhom_thay"])
                .agg(so_dong=("fn", "size"), fn=("fn", "sum")).reset_index())
    by_label["recall"] = 1 - by_label["fn"] / by_label["so_dong"]
    by_label["ty_trong_fn"] = by_label["fn"] / att["fn"].sum()
    by_label = by_label.sort_values("fn", ascending=False)

    by_group = (att.groupby("nhom_thay").agg(so_dong=("fn", "size"), fn=("fn", "sum")).reset_index())
    by_group["recall"] = 1 - by_group["fn"] / by_group["so_dong"]
    by_group["ty_trong_fn"] = by_group["fn"] / att["fn"].sum()

    by_cat = (att.groupby(["attack_category", "nhom_thay"])
              .agg(so_dong=("fn", "size"), fn=("fn", "sum")).reset_index())
    by_cat["recall"] = 1 - by_cat["fn"] / by_cat["so_dong"]
    by_cat["ty_trong_fn"] = by_cat["fn"] / att["fn"].sum()

    for _, r in by_group.iterrows():
        log(f"  {r['nhom_thay']:24s} {r['so_dong']:>6,} dong  FN={r['fn']:>5,}  "
            f"recall={r['recall']:.2%}  chiem {r['ty_trong_fn']:.1%} tong FN")
    if not save:
        return by_label, by_group, by_cat

    by_label.to_csv(REPORTS_DIR / "fn_theo_loai_tan_cong.csv", index=False)
    by_group.to_csv(REPORTS_DIR / "fn_seen_vs_unseen.csv", index=False)
    by_cat.to_csv(REPORTS_DIR / "fn_theo_nhom_tan_cong.csv", index=False)
    top = by_label.head(15).iloc[::-1]
    colors = ["#c0392b" if "unseen" in g else "#2980b9" for g in top["nhom_thay"]]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["label"] + " (" + top["attack_category"] + ")", top["fn"], color=colors)
    for i, (fn, n) in enumerate(zip(top["fn"], top["so_dong"])):
        ax.text(fn, i, f" {fn:,}/{n:,}", va="center", fontsize=8)
    ax.set_xlabel("So FN (tan cong bi bo lot) / tong so dong cua loai do")
    ax.set_title("Top 15 loai tan cong bi bo lot nhieu nhat - test NSL-KDD\n"
                 "do = unseen (chi co o test), xanh = seen (co trong train)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "fn_theo_loai_tan_cong.png", dpi=130)
    plt.close(fig)
    log("Da luu -> reports/fn_theo_loai_tan_cong.png, reports/fn_*.csv")
    return by_label, by_group, by_cat


def main() -> None:
    log("Nap du lieu NSL-KDD...")
    train_df = load_raw(TRAIN_PATH)
    test_df = load_raw(TEST_PATH)
    log(f"  Train: {len(train_df):,} dong | attack={train_df['binary_label'].mean():.2%}")
    log(f"  Test : {len(test_df):,} dong | attack={test_df['binary_label'].mean():.2%}")

    u2r_share = (train_df["attack_category"] == "U2R").mean()
    log(f"  Ty le lop U2R (train) = {u2r_share:.4%} -> qua hiem, dung nhi phan normal/attack")

    X_train_full = train_df[FEATURE_COLUMNS]
    y_train_full = train_df["binary_label"].to_numpy()
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["binary_label"].to_numpy()

    log("3. EDA...")
    unseen_labels = run_eda(train_df, test_df)

    log("4. Baseline (Dummy, 1 stump) - 5-fold CV...")
    baseline = run_cv_baseline(X_train_full, y_train_full)

    log(f"5a. Tune AdaBoost: learning_rate {LR_GRID} x n_estimators {N_GRID} - {N_FOLDS}-fold CV...")
    tune_table, oof = run_tuning(X_train_full, y_train_full)
    best = tune_table.loc[tune_table["cv_f1_mean"].idxmax()]
    best_lr, best_n = float(best["learning_rate"]), int(best["n_estimators"])
    old = tune_table[(tune_table["learning_rate"] == OLD_LR) & (tune_table["n_estimators"] == OLD_N)].iloc[0]
    cv_table = pd.concat([baseline, pd.DataFrame([
        {"model": f"AdaBoost cu (lr={OLD_LR}, n={OLD_N})", **old.drop(["learning_rate", "n_estimators"]).to_dict()},
        {"model": f"AdaBoost tuned (lr={best_lr:g}, n={best_n})", **best.drop(["learning_rate", "n_estimators"]).to_dict()},
    ])], ignore_index=True)
    cv_table.to_csv(REPORTS_DIR / "so_sanh_baseline_cv.csv", index=False)

    log(f"5b. Do nguong quyet dinh tren diem out-of-fold (toi da F{THRESHOLD_BETA:g})...")
    thr_info = choose_threshold(y_train_full, oof[(best_lr, best_n)])
    threshold = thr_info["threshold"]

    log("Tach train_sub/validation (80/20, stratify) tu DataFrame THO, fit preprocessor chi tren train_sub...")
    X_sub_df, X_val_df, ytr_sub, yval = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=RANDOM_STATE,
    )
    preprocessor = build_preprocessor()
    Xtr_sub = preprocessor.fit_transform(X_sub_df)
    Xval = preprocessor.transform(X_val_df)
    log(f"  train_sub={Xtr_sub.shape[0]:,}  val={Xval.shape[0]:,}  so chieu sau one-hot={Xtr_sub.shape[1]}")

    log(f"6. Duong F1/Accuracy theo n_estimators = 1..{N_MAX} (lr={best_lr:g})...")
    ada_clean, ada_time, _ = run_staged_curve(Xtr_sub, ytr_sub, Xval, yval, best_lr, best_n)

    log("   Huan luyen Random Forest (300 cay, sach) de tai su dung cho buoc 7-8...")
    rf_clean = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1)
    t0 = time.perf_counter()
    rf_clean.fit(Xtr_sub, ytr_sub)
    rf_time = time.perf_counter() - t0

    log("7. THI NGHIEM NHIEU NHAN (dao 5% nhan train)...")
    noise_result = run_noise_experiment(Xtr_sub, ytr_sub, Xval, yval, ada_clean, rf_clean)

    log("8. So sanh AdaBoost vs Gradient Boosting vs Random Forest...")
    ensemble_result = run_ensemble_comparison(
        Xtr_sub, ytr_sub, Xval, yval,
        {"AdaBoost": (ada_clean, ada_time), "Random Forest": (rf_clean, rf_time)},
    )

    log("9. Huan luyen model AdaBoost CUOI CUNG tren toan bo train...")
    final_pipe = Pipeline([("prep", build_preprocessor()), ("clf", make_adaboost(best_lr, best_n))])
    final_pipe.fit(X_train_full, y_train_full)
    joblib.dump({"pipeline": final_pipe, "threshold": threshold}, MODELS_DIR / "adaboost.joblib")
    log(f"Da luu model + nguong -> models/adaboost.joblib (du doan: decision_function(X) >= {threshold:+.4f})")

    log("10. Danh gia test NSL-KDD goc (1 lan) + ma tran nham lan + bao dong gia/ngay...")
    test_table, score_test, preds = run_final_test_eval(final_pipe, X_test, y_test, threshold,
                                                        cv_f1_reference=float(best["cv_f1_mean"]))
    run_threshold_sensitivity(y_test, score_test, threshold)

    log("11. Phan tich FN theo loai tan cong (seen vs unseen), nguong da do...")
    fn_label, fn_group, fn_cat = run_fn_breakdown(test_df, preds[f"Da do F{THRESHOLD_BETA:g} (OOF)"], unseen_labels)
    log("    (doi chieu nguong mac dinh 0)")
    _, fn_group_default, _ = run_fn_breakdown(test_df, preds["Mac dinh (0)"], unseen_labels, save=False)

    summary = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "attack_rate_train": float(train_df["binary_label"].mean()),
        "attack_rate_test": float(test_df["binary_label"].mean()),
        "u2r_share_train": float(u2r_share),
        "unseen_attack_types_in_test": unseen_labels,
        "n_unseen_rows_in_test": int(test_df["label"].isin(unseen_labels).sum()),
        "cv_baseline": records(cv_table),
        "tuning": {"best_learning_rate": best_lr, "best_n_estimators": best_n,
                   "grid": records(tune_table)},
        "threshold_oof": {"beta": THRESHOLD_BETA, "threshold": threshold,
                          "table": records(thr_info["table"])},
        "noise_experiment": records(noise_result),
        "ensemble_comparison": records(ensemble_result),
        "final_test_eval": {
            "cv_f1_train": float(best["cv_f1_mean"]),
            "assumed_daily_connections": ASSUMED_DAILY_CONNECTIONS,
            "by_threshold": records(test_table),
        },
        "fn_seen_vs_unseen": {"threshold_tuned": records(fn_group), "threshold_default": records(fn_group_default)},
        "fn_by_category": records(fn_cat),
        "fn_top10_labels": records(fn_label.head(10)),
    }
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)

    log("HOAN THANH. Xem ket qua chi tiet trong thu muc reports/.")


if __name__ == "__main__":
    main()
