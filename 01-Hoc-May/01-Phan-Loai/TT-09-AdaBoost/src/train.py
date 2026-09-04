"""
TT-09 - AdaBoost: Phat hien xam nhap mang trong he thong giam sat an ninh (SOC).

Pipeline day du:
 1. Nap du lieu NSL-KDD (train + test goc), gop nhan thanh nhi phan normal(0)/attack(1)
 2. Tien xu ly: one-hot 3 cot phan loai, scale cac cot so (fit CHI tren train)
 3. EDA: phan bo loai tan cong (DoS/Probe/R2L/U2R), ty le normal/attack
 4. Baseline: DummyClassifier + 1 stump don le (depth=1) - 5-fold CV tren train
 5. AdaBoost 300 stump - 5-fold CV tren train, so sanh voi 1 stump
 6. Duong Accuracy/F1 theo n_estimators = 1..300 (staged_predict tren tap validation)
 7. THI NGHIEM NHIEU: dao nguoc 5% nhan train -> AdaBoost vs Random Forest
 8. So sanh AdaBoost vs Gradient Boosting (TT-07) vs Random Forest (TT-03)
 9. Danh gia model AdaBoost cuoi cung (fit tren toan bo train) tren tap test NSL-KDD
    goc (co chua 17 loai tan cong CHUA TUNG THAY - mo phong zero-day)
10. Ma tran nham lan + uoc tinh so bao dong gia/ngay (alert fatigue)

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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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
N_ESTIMATORS = 300
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


# ----------------------------------------------------------------------------
# 1. Nap du lieu + gop nhan nhi phan
# ----------------------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, names=ALL_COLUMNS, header=None)
    df["binary_label"] = (df["label"] != "normal").astype(int)
    df["attack_category"] = np.where(
        df["label"] == "normal", "Normal", df["label"].map(ATTACK_CATEGORY).fillna("Unknown")
    )
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )


# ----------------------------------------------------------------------------
# 3. EDA
# ----------------------------------------------------------------------------
def run_eda(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    cat_train = train_df["attack_category"].value_counts()
    cat_test = test_df["attack_category"].value_counts()
    cat_table = pd.DataFrame({"train": cat_train, "test": cat_test}).fillna(0).astype(int)
    cat_table.to_csv(REPORTS_DIR / "phan_bo_loai_tan_cong.csv")

    unseen_labels = sorted(set(test_df["label"]) - set(train_df["label"]))
    log(f"  So loai tan cong CHI co trong test (zero-day mo phong): {len(unseen_labels)} "
        f"-> {unseen_labels}")

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


# ----------------------------------------------------------------------------
# 4-5. Baseline (Dummy + 1 stump) va AdaBoost 300 stump - 5-fold CV
# ----------------------------------------------------------------------------
def run_cv_baseline(X_train_full: pd.DataFrame, y_train_full: np.ndarray) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "Baseline (Dummy)": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "1 Stump (depth=1)": DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
        f"AdaBoost ({N_ESTIMATORS} stump)": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=N_ESTIMATORS, learning_rate=0.5, random_state=RANDOM_STATE,
        ),
    }

    rows = []
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        t0 = time.perf_counter()
        scores = cross_validate(
            pipe, X_train_full, y_train_full, cv=skf,
            scoring=("accuracy", "f1"), n_jobs=-1,
        )
        elapsed = time.perf_counter() - t0
        row = {
            "model": name,
            "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
            "cv_f1_mean": float(np.mean(scores["test_f1"])),
            "cv_f1_std": float(np.std(scores["test_f1"])),
            "cv_time_s": elapsed,
        }
        rows.append(row)
        log(f"  {name:26s} CV Accuracy={row['cv_accuracy_mean']:.4f} (+-{row['cv_accuracy_std']:.4f})  "
            f"CV F1={row['cv_f1_mean']:.4f} (+-{row['cv_f1_std']:.4f})  [{elapsed:.1f}s]")

    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "so_sanh_baseline_cv.csv", index=False)
    return result


# ----------------------------------------------------------------------------
# 6. Duong F1/Accuracy theo n_estimators (staged_predict)
# ----------------------------------------------------------------------------
def run_staged_curve(Xtr, ytr, Xval, yval) -> AdaBoostClassifier:
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=N_ESTIMATORS, learning_rate=0.5, random_state=RANDOM_STATE,
    )
    ada.fit(Xtr, ytr)

    rows = []
    for i, pred in enumerate(ada.staged_predict(Xval), start=1):
        rows.append({
            "n_estimators": i,
            "accuracy": accuracy_score(yval, pred),
            "f1": f1_score(yval, pred),
        })
    curve = pd.DataFrame(rows)
    curve.to_csv(REPORTS_DIR / "f1_theo_vong_lap.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve["n_estimators"], curve["f1"], label="F1-score", color="#c0392b")
    ax.plot(curve["n_estimators"], curve["accuracy"], label="Accuracy", color="#2980b9", ls="--")
    ax.set_xlabel("So vong lap (n_estimators)")
    ax.set_ylabel("Diem so tren tap validation")
    ax.set_title(f"AdaBoost: F1/Accuracy theo so vong lap (1..{N_ESTIMATORS})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "f1_theo_vong_lap.png", dpi=130)
    plt.close(fig)
    log(f"  F1 tai n=1: {curve['f1'].iloc[0]:.4f}  |  F1 tai n={N_ESTIMATORS}: {curve['f1'].iloc[-1]:.4f}")
    log("Da luu -> reports/f1_theo_vong_lap.png, reports/f1_theo_vong_lap.csv")
    return ada


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

    ada_noisy = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=N_ESTIMATORS, learning_rate=0.5, random_state=RANDOM_STATE,
    )
    ada_noisy.fit(Xtr, ytr_noisy)

    rf_noisy = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_noisy.fit(Xtr, ytr_noisy)

    rows = []
    for model_name, clean_model, noisy_model in [
        ("AdaBoost", ada_clean, ada_noisy),
        ("Random Forest", rf_clean, rf_noisy),
    ]:
        f1_clean = f1_score(yval, clean_model.predict(Xval))
        f1_noisy = f1_score(yval, noisy_model.predict(Xval))
        rows.append({
            "model": model_name,
            "f1_clean": f1_clean,
            "f1_noisy_5pct": f1_noisy,
            "sut_giam_f1": f1_clean - f1_noisy,
        })
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
def run_ensemble_comparison(Xtr, ytr, Xval, yval, ada_clean: AdaBoostClassifier,
                             rf_clean: RandomForestClassifier) -> pd.DataFrame:
    rows = []

    for name, model, already_fitted in [
        ("AdaBoost", ada_clean, True),
        ("Random Forest", rf_clean, True),
        ("Gradient Boosting", GradientBoostingClassifier(
            n_estimators=N_ESTIMATORS, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE,
        ), False),
    ]:
        if already_fitted:
            train_time = None
        else:
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            train_time = time.perf_counter() - t0

        pred = model.predict(Xval)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(yval, pred),
            "f1": f1_score(yval, pred),
            "train_time_s": train_time,
        })
        log(f"  {name:20s} Accuracy={rows[-1]['accuracy']:.4f}  F1={rows[-1]['f1']:.4f}")

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
def run_final_test_eval(final_pipe: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray,
                         cv_f1_reference: float) -> dict:
    pred_test = final_pipe.predict(X_test)
    acc_test = accuracy_score(y_test, pred_test)
    f1_test = f1_score(y_test, pred_test)
    log(f"  F1 tren CV (train) = {cv_f1_reference:.4f}  vs  F1 tren test NSL-KDD goc = {f1_test:.4f}  "
        f"(chenh lech = {cv_f1_reference - f1_test:.4f})")

    cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "attack"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["normal", "attack"])
    ax.set_xlabel("Du doan"); ax.set_ylabel("Thuc te")
    ax.set_title("Ma tran nham lan - tap test NSL-KDD goc")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix_test.png", dpi=130)
    plt.close(fig)
    log("Da luu -> reports/confusion_matrix_test.png")

    normal_share = (y_test == 0).mean()
    assumed_daily_normal = ASSUMED_DAILY_CONNECTIONS * normal_share
    false_alarms_per_day = assumed_daily_normal * fpr
    log(f"  FPR (tap test) = {fpr:.4%}  ->  uoc tinh {false_alarms_per_day:,.0f} bao dong gia/ngay "
        f"(gia dinh {ASSUMED_DAILY_CONNECTIONS:,} ket noi/ngay)")

    return {
        "accuracy_test": float(acc_test),
        "f1_test": float(f1_test),
        "cv_f1_train": float(cv_f1_reference),
        "gap_cv_vs_test_f1": float(cv_f1_reference - f1_test),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "assumed_daily_connections": ASSUMED_DAILY_CONNECTIONS,
        "estimated_false_alarms_per_day": float(false_alarms_per_day),
    }


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

    log("EDA...")
    run_eda(train_df, test_df)

    log("4-5. Baseline (Dummy, 1 stump) vs AdaBoost 300 stump - 5-fold CV...")
    cv_result = run_cv_baseline(X_train_full, y_train_full)
    ada_cv_f1 = cv_result.loc[cv_result["model"].str.startswith("AdaBoost"), "cv_f1_mean"].iloc[0]

    log("Chuan bi tap train_sub/validation (80/20, stratify) cho cac thi nghiem con lai...")
    preprocessor = build_preprocessor()
    Xtr_full_mat = preprocessor.fit_transform(X_train_full)
    Xtest_mat = preprocessor.transform(X_test)
    Xtr_sub, Xval, ytr_sub, yval = train_test_split(
        Xtr_full_mat, y_train_full, test_size=0.2, stratify=y_train_full, random_state=RANDOM_STATE,
    )
    log(f"  train_sub={Xtr_sub.shape[0]:,}  val={Xval.shape[0]:,}  so chieu sau one-hot={Xtr_sub.shape[1]}")

    log(f"6. Duong F1/Accuracy theo n_estimators = 1..{N_ESTIMATORS}...")
    ada_clean = run_staged_curve(Xtr_sub, ytr_sub, Xval, yval)

    log(f"   Huan luyen Random Forest ({N_ESTIMATORS} cay, sach) de tai su dung cho buoc 7-8...")
    rf_clean = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_clean.fit(Xtr_sub, ytr_sub)

    log("7. THI NGHIEM NHIEU NHAN (dao 5% nhan train)...")
    noise_result = run_noise_experiment(Xtr_sub, ytr_sub, Xval, yval, ada_clean, rf_clean)

    log("8. So sanh AdaBoost vs Gradient Boosting vs Random Forest...")
    ensemble_result = run_ensemble_comparison(Xtr_sub, ytr_sub, Xval, yval, ada_clean, rf_clean)

    log("9. Huan luyen model AdaBoost CUOI CUNG tren toan bo train, danh gia tren test NSL-KDD goc...")
    final_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=N_ESTIMATORS, learning_rate=0.5, random_state=RANDOM_STATE,
        )),
    ])
    final_pipe.fit(X_train_full, y_train_full)
    joblib.dump(final_pipe, MODELS_DIR / "adaboost.joblib")
    log("Da luu model -> models/adaboost.joblib")

    log("10. Ma tran nham lan + uoc tinh bao dong gia/ngay...")
    test_summary = run_final_test_eval(final_pipe, X_test, y_test, cv_f1_reference=ada_cv_f1)

    summary = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "attack_rate_train": float(train_df["binary_label"].mean()),
        "attack_rate_test": float(test_df["binary_label"].mean()),
        "u2r_share_train": float(u2r_share),
        "n_unseen_attack_types_in_test": int(len(set(test_df["label"]) - set(train_df["label"]))),
        "cv_baseline": cv_result.to_dict(orient="records"),
        "noise_experiment": noise_result.to_dict(orient="records"),
        "ensemble_comparison": ensemble_result.to_dict(orient="records"),
        "final_test_eval": test_summary,
    }
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("HOAN THANH. Xem ket qua chi tiet trong thu muc reports/.")


if __name__ == "__main__":
    main()
