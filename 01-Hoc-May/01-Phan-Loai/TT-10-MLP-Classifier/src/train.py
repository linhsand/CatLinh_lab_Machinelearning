"""
TT-10 - MLP Classifier: Doc so viet tay tren sec / phieu chuyen khoan (MNIST).

Pipeline day du (theo 13 buoc trong README.md cap de):
 1. Khoi dong voi load_digits() (1.797 anh 8x8) - hieu quy trinh nhanh
 2. Nap MNIST (mnist_784, 70.000 anh 28x28) qua fetch_openml, lay mau stratify
    de gioi han thoi gian chay (xem data/DATA_SOURCE.md)
 3. Baseline: Logistic Regression tren MNIST
 4. MLP KHONG chuan hoa (pixel 0-255 tho) - ghi lai hien tuong khong hoi tu
 5. MLP CO chuan hoa (/255) - so sanh voi buoc 4
 6. So sanh 4 kien truc: (64), (128), (128,64), (256,128,64)
    -> bang accuracy, so tham so, thoi gian train + do thi loss chong nhau
 7. (gop vao buoc 6) Ve duong loss theo epoch (mlp.loss_curve_)
 8. So sanh 3 activation: relu / tanh / logistic
 9. So sanh 3 learning_rate_init: 1e-2 / 1e-3 / 1e-4 - ve 3 duong loss chong nhau
10. Ma tran nham lan 10x10 + cap so hay bi nham nhat
11. Hien thi 20 anh du doan SAI
12. Co che human-in-the-loop: nguong tin cay 99% -> % tu dong / % chuyen nguoi
13. So sanh dinh tinh voi CNN (TT-26) - ghi trong bao cao, khong huan luyen lai

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
from sklearn.datasets import fetch_openml, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42
# Lay mau stratify tu MNIST goc de gioi han thoi gian chay toan bo script
# (4 kien truc + 3 activation + 3 learning rate + model cuoi = 11 lan fit MLP).
# Dat = None de chay tren toan bo 60.000/10.000 mau chuan cua MNIST.
TRAIN_SAMPLE = 12_000
TEST_SAMPLE = 3_000
CONFIDENCE_THRESHOLD = 0.99

ARCHITECTURES = {
    "(64)": (64,),
    "(128)": (128,),
    "(128,64)": (128, 64),
    "(256,128,64)": (256, 128, 64),
}
BEST_ARCHITECTURE = (128, 64)  # chon sau buoc 6, dung lam co so cho buoc 8-9


def log(msg: str) -> None:
    print(f"[TT-10] {msg}")


def divide_by_255(X: np.ndarray) -> np.ndarray:
    return X / 255.0


def count_params(input_dim: int, hidden_layers: tuple[int, ...], n_classes: int) -> int:
    layer_sizes = [input_dim, *hidden_layers, n_classes]
    return sum(a * b + b for a, b in zip(layer_sizes[:-1], layer_sizes[1:]))


# ----------------------------------------------------------------------------
# 1. Khoi dong voi load_digits()
# ----------------------------------------------------------------------------
def warmup_digits() -> None:
    digits = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(
        digits.data, digits.target, test_size=0.2, stratify=digits.target, random_state=RANDOM_STATE
    )
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=RANDOM_STATE)
    mlp.fit(Xtr / 16.0, ytr)  # pixel load_digits() trong [0,16]
    acc = accuracy_score(yte, mlp.predict(Xte / 16.0))
    log(f"  Khoi dong load_digits(): {len(digits.data)} anh 8x8, MLP(64) accuracy={acc:.4f}")


# ----------------------------------------------------------------------------
# 2. Nap MNIST, lay mau stratify
# ----------------------------------------------------------------------------
def load_mnist_sample() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    log("  Dang tai/doc cache MNIST (mnist_784) qua fetch_openml...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)

    # MNIST goc: 60.000 mau dau la train, 10.000 mau cuoi la test (quy uoc chuan).
    X_train_full, y_train_full = X[:60_000], y[:60_000]
    X_test_full, y_test_full = X[60_000:], y[60_000:]

    if TRAIN_SAMPLE is not None:
        X_train, _, y_train, _ = train_test_split(
            X_train_full, y_train_full, train_size=TRAIN_SAMPLE,
            stratify=y_train_full, random_state=RANDOM_STATE,
        )
    else:
        X_train, y_train = X_train_full, y_train_full

    if TEST_SAMPLE is not None:
        X_test, _, y_test, _ = train_test_split(
            X_test_full, y_test_full, train_size=TEST_SAMPLE,
            stratify=y_test_full, random_state=RANDOM_STATE,
        )
    else:
        X_test, y_test = X_test_full, y_test_full

    log(f"  Train mau: {X_train.shape[0]:,} | Test mau: {X_test.shape[0]:,} | "
        f"pixel range=[{X_train.min():.0f},{X_train.max():.0f}]")
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------------
# 3. Baseline Logistic Regression
# ----------------------------------------------------------------------------
def run_baseline_logreg(Xtr, ytr, Xte, yte) -> float:
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(Xtr / 255.0, ytr)
    acc = accuracy_score(yte, clf.predict(Xte / 255.0))
    log(f"  Baseline Logistic Regression: accuracy={acc:.4f}")
    return acc


# ----------------------------------------------------------------------------
# 4-5. MLP KHONG chuan hoa vs CO chuan hoa
# ----------------------------------------------------------------------------
def run_normalization_comparison(Xtr, ytr, Xte, yte) -> pd.DataFrame:
    rows = []
    for name, Xtr_use, Xte_use in [
        ("KHONG chuan hoa (0-255)", Xtr, Xte),
        ("CO chuan hoa (/255)", Xtr / 255.0, Xte / 255.0),
    ]:
        mlp = MLPClassifier(
            hidden_layer_sizes=BEST_ARCHITECTURE, activation="relu", solver="adam",
            alpha=1e-4, batch_size=128, learning_rate_init=1e-3, max_iter=40,
            early_stopping=True, n_iter_no_change=8, random_state=RANDOM_STATE,
        )
        t0 = time.perf_counter()
        mlp.fit(Xtr_use, ytr)
        elapsed = time.perf_counter() - t0
        acc = accuracy_score(yte, mlp.predict(Xte_use))
        rows.append({
            "cau_hinh": name, "accuracy": acc, "so_epoch_thuc_te": mlp.n_iter_,
            "loss_cuoi": mlp.loss_, "thoi_gian_s": elapsed,
        })
        log(f"  {name:26s} accuracy={acc:.4f}  epoch={mlp.n_iter_:3d}  loss_cuoi={mlp.loss_:.4f}  "
            f"[{elapsed:.1f}s]")

    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "so_sanh_chuan_hoa.csv", index=False)
    log("Da luu -> reports/so_sanh_chuan_hoa.csv")
    return result


# ----------------------------------------------------------------------------
# 6-7. So sanh 4 kien truc + duong loss chong nhau
# ----------------------------------------------------------------------------
def run_architecture_comparison(Xtr, ytr, Xte, yte) -> pd.DataFrame:
    rows = []
    loss_curves = {}
    for name, hidden in ARCHITECTURES.items():
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden, activation="relu", solver="adam",
            alpha=1e-4, batch_size=128, learning_rate_init=1e-3, max_iter=100,
            early_stopping=True, n_iter_no_change=10, random_state=RANDOM_STATE,
        )
        t0 = time.perf_counter()
        mlp.fit(Xtr, ytr)
        elapsed = time.perf_counter() - t0
        acc = accuracy_score(yte, mlp.predict(Xte))
        n_params = count_params(Xtr.shape[1], hidden, 10)
        rows.append({
            "kien_truc": name, "so_tham_so": n_params, "accuracy": acc,
            "so_epoch_thuc_te": mlp.n_iter_, "thoi_gian_s": elapsed,
        })
        loss_curves[name] = mlp.loss_curve_
        log(f"  MLP{name:16s} params={n_params:,}  accuracy={acc:.4f}  "
            f"epoch={mlp.n_iter_:3d}  [{elapsed:.1f}s]")

    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "kien_truc_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(result["kien_truc"], result["accuracy"], color="#2980b9")
    axes[0].set_title("Accuracy theo kien truc")
    axes[0].set_ylabel("Accuracy (test)")
    for i, v in enumerate(result["accuracy"]):
        axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    ax2 = axes[0].twinx()
    ax2.plot(result["kien_truc"], result["so_tham_so"], color="#c0392b", marker="o", label="So tham so")
    ax2.set_ylabel("So tham so", color="#c0392b")

    for name, curve in loss_curves.items():
        axes[1].plot(curve, label=name)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Training loss")
    axes[1].set_title("Duong loss theo epoch - 4 kien truc")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "kien_truc_comparison.png", dpi=130)
    plt.close(fig)

    # Bieu do rieng loss_curve_ (yeu cau muc 6 cua tieu chi hoan thanh)
    fig2, ax = plt.subplots(figsize=(9, 5.5))
    for name, curve in loss_curves.items():
        ax.plot(curve, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("MLP: duong loss theo epoch cho 4 kien truc")
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(REPORTS_DIR / "loss_curves.png", dpi=130)
    plt.close(fig2)

    log("Da luu -> reports/kien_truc_comparison.png, reports/loss_curves.png, "
        "reports/kien_truc_comparison.csv")
    return result


# ----------------------------------------------------------------------------
# 8. So sanh 3 activation
# ----------------------------------------------------------------------------
def run_activation_comparison(Xtr, ytr, Xte, yte) -> pd.DataFrame:
    rows = []
    curves = {}
    for act in ["relu", "tanh", "logistic"]:
        mlp = MLPClassifier(
            hidden_layer_sizes=BEST_ARCHITECTURE, activation=act, solver="adam",
            alpha=1e-4, batch_size=128, learning_rate_init=1e-3, max_iter=100,
            early_stopping=True, n_iter_no_change=10, random_state=RANDOM_STATE,
        )
        mlp.fit(Xtr, ytr)
        acc = accuracy_score(yte, mlp.predict(Xte))
        rows.append({
            "activation": act, "accuracy": acc,
            "so_epoch_thuc_te": mlp.n_iter_, "loss_cuoi": mlp.loss_,
        })
        curves[act] = mlp.loss_curve_
        log(f"  activation={act:10s} accuracy={acc:.4f}  epoch={mlp.n_iter_:3d}  loss_cuoi={mlp.loss_:.4f}")

    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "activation_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(result["activation"], result["accuracy"], color=["#2980b9", "#27ae60", "#c0392b"])
    axes[0].set_title("Accuracy theo activation")
    for i, v in enumerate(result["accuracy"]):
        axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    for act, curve in curves.items():
        axes[1].plot(curve, label=act)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Training loss")
    axes[1].set_title("Duong loss - 3 activation (vanishing gradient o logistic)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "activation_comparison.png", dpi=130)
    plt.close(fig)
    log("Da luu -> reports/activation_comparison.png, reports/activation_comparison.csv")
    return result


# ----------------------------------------------------------------------------
# 9. So sanh 3 learning_rate_init
# ----------------------------------------------------------------------------
def run_learning_rate_comparison(Xtr, ytr, Xte, yte) -> pd.DataFrame:
    rows = []
    curves = {}
    for lr in [1e-2, 1e-3, 1e-4]:
        mlp = MLPClassifier(
            hidden_layer_sizes=BEST_ARCHITECTURE, activation="relu", solver="adam",
            alpha=1e-4, batch_size=128, learning_rate_init=lr, max_iter=100,
            early_stopping=True, n_iter_no_change=10, random_state=RANDOM_STATE,
        )
        mlp.fit(Xtr, ytr)
        acc = accuracy_score(yte, mlp.predict(Xte))
        rows.append({"learning_rate_init": lr, "accuracy": acc, "so_epoch_thuc_te": mlp.n_iter_})
        curves[lr] = mlp.loss_curve_
        log(f"  learning_rate_init={lr:<7g} accuracy={acc:.4f}  epoch={mlp.n_iter_:3d}")

    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "learning_rate_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for lr, curve in curves.items():
        ax.plot(curve, label=f"lr={lr:g}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Duong loss theo 3 learning_rate_init")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "learning_rate_curves.png", dpi=130)
    plt.close(fig)
    log("Da luu -> reports/learning_rate_curves.png, reports/learning_rate_comparison.csv")
    return result


# ----------------------------------------------------------------------------
# 10-11. Ma tran nham lan 10x10 + 20 anh du doan sai
# ----------------------------------------------------------------------------
def run_confusion_and_errors(final_pipe: Pipeline, Xte: np.ndarray, yte: np.ndarray) -> dict:
    pred = final_pipe.predict(Xte)
    cm = confusion_matrix(yte, pred, labels=list(range(10)))

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Du doan"); ax.set_ylabel("Thuc te")
    ax.set_title("Ma tran nham lan 10x10 - MLP tren MNIST")
    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_10x10.png", dpi=130)
    plt.close(fig)

    # Cap so hay bi nham nhat (bo qua duong cheo)
    pairs = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                pairs.append({"thuc_te": i, "du_doan": j, "so_lan": int(cm[i, j])})
    pairs_df = pd.DataFrame(pairs).sort_values("so_lan", ascending=False)
    pairs_df.to_csv(REPORTS_DIR / "cap_nham_lan.csv", index=False)
    top5 = pairs_df.head(5).to_dict(orient="records")
    log(f"  5 cap hay nham nhat: {top5}")
    log("Da luu -> reports/confusion_10x10.png, reports/cap_nham_lan.csv")

    # 11. 20 anh du doan sai
    wrong_idx = np.where(pred != yte)[0]
    rng = np.random.RandomState(RANDOM_STATE)
    show_idx = rng.choice(wrong_idx, size=min(20, len(wrong_idx)), replace=False)

    fig, axes = plt.subplots(4, 5, figsize=(11, 9))
    for ax, idx in zip(axes.ravel(), show_idx):
        ax.imshow(Xte[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"that={yte[idx]} doan={pred[idx]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("20 anh bi du doan SAI (chon ngau nhien)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "anh_sai.png", dpi=130)
    plt.close(fig)
    log(f"  So mau sai tren tap test: {len(wrong_idx)} / {len(yte)} "
        f"({len(wrong_idx) / len(yte):.2%})")
    log("Da luu -> reports/anh_sai.png")

    return {"top5_cap_nham_lan": top5, "so_mau_sai": int(len(wrong_idx)), "tong_test": int(len(yte))}


# ----------------------------------------------------------------------------
# 12. Human-in-the-loop: nguong tin cay 99%
# ----------------------------------------------------------------------------
def run_human_in_the_loop(final_pipe: Pipeline, Xte: np.ndarray, yte: np.ndarray) -> pd.DataFrame:
    proba = final_pipe.predict_proba(Xte)
    pred = proba.argmax(axis=1)
    confidence = proba.max(axis=1)

    rows = []
    for threshold in [0.90, 0.95, 0.99, 0.999]:
        auto_mask = confidence >= threshold
        n_auto = int(auto_mask.sum())
        pct_auto = n_auto / len(yte)
        acc_auto = accuracy_score(yte[auto_mask], pred[auto_mask]) if n_auto else float("nan")
        rows.append({
            "nguong_tin_cay": threshold,
            "pct_tu_dong": pct_auto,
            "pct_chuyen_nguoi": 1 - pct_auto,
            "accuracy_tren_phan_tu_dong": acc_auto,
        })
        log(f"  Nguong={threshold:.3f}  tu dong={pct_auto:.2%}  chuyen nguoi={1 - pct_auto:.2%}  "
            f"accuracy(tu dong)={acc_auto:.4%}")

    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "human_in_the_loop.csv", index=False)
    log("Da luu -> reports/human_in_the_loop.csv")
    return result


def main() -> None:
    log("1. Khoi dong voi load_digits()...")
    warmup_digits()

    log("2. Nap MNIST (lay mau stratify)...")
    Xtr, Xte, ytr, yte = load_mnist_sample()

    log("3. Baseline Logistic Regression...")
    baseline_acc = run_baseline_logreg(Xtr, ytr, Xte, yte)

    log("4-5. So sanh MLP KHONG chuan hoa vs CO chuan hoa...")
    norm_result = run_normalization_comparison(Xtr, ytr, Xte, yte)

    log("Chuan hoa du lieu (/255) cho cac buoc con lai...")
    Xtr_n, Xte_n = Xtr / 255.0, Xte / 255.0

    log("6-7. So sanh 4 kien truc + duong loss...")
    arch_result = run_architecture_comparison(Xtr_n, ytr, Xte_n, yte)

    log("8. So sanh 3 activation...")
    act_result = run_activation_comparison(Xtr_n, ytr, Xte_n, yte)

    log("9. So sanh 3 learning_rate_init...")
    lr_result = run_learning_rate_comparison(Xtr_n, ytr, Xte_n, yte)

    log("Huan luyen model CUOI CUNG (kien truc tot nhat, /255 trong Pipeline)...")
    final_pipe = Pipeline([
        ("scale", FunctionTransformer(divide_by_255, validate=False)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=BEST_ARCHITECTURE, activation="relu", solver="adam",
            alpha=1e-4, batch_size=128, learning_rate_init=1e-3, max_iter=150,
            early_stopping=True, n_iter_no_change=10, random_state=RANDOM_STATE, verbose=False,
        )),
    ])
    final_pipe.fit(Xtr, ytr)
    final_acc = accuracy_score(yte, final_pipe.predict(Xte))
    log(f"  Accuracy model cuoi tren test: {final_acc:.4f}")
    joblib.dump(final_pipe, MODELS_DIR / "mlp_pipeline.joblib")
    log("Da luu model -> models/mlp_pipeline.joblib")

    log("10-11. Ma tran nham lan + 20 anh du doan sai...")
    error_summary = run_confusion_and_errors(final_pipe, Xte, yte)

    log("12. Human-in-the-loop (nguong tin cay)...")
    hitl_result = run_human_in_the_loop(final_pipe, Xte, yte)

    summary = {
        "train_sample": int(Xtr.shape[0]),
        "test_sample": int(Xte.shape[0]),
        "baseline_logreg_accuracy": float(baseline_acc),
        "normalization_comparison": norm_result.to_dict(orient="records"),
        "architecture_comparison": arch_result.to_dict(orient="records"),
        "activation_comparison": act_result.to_dict(orient="records"),
        "learning_rate_comparison": lr_result.to_dict(orient="records"),
        "final_model": {
            "hidden_layer_sizes": list(BEST_ARCHITECTURE),
            "accuracy_test": float(final_acc),
        },
        "error_analysis": error_summary,
        "human_in_the_loop_threshold_99pct": hitl_result[
            hitl_result["nguong_tin_cay"] == CONFIDENCE_THRESHOLD
        ].to_dict(orient="records")[0],
    }
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("HOAN THANH. Xem ket qua chi tiet trong thu muc reports/.")


if __name__ == "__main__":
    main()
