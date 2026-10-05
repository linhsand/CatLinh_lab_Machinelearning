"""
TT-10 - MLP Classifier: Doc so viet tay tren sec / phieu chuyen khoan (MNIST).

Nguyen tac danh gia (chong "chon tren test"):
  * MNIST chuan: 60.000 train / 10.000 test.
  * 60.000 train -> tach stratify 50.000 TRAIN + 10.000 VALIDATION.
  * MOI lua chon (kien truc, activation, learning rate, nguong human-in-the-loop,
    so epoch CNN) deu cham bang VALIDATION.
  * TEST chi duoc dung 1 LAN o buoc cuoi, sau khi moi thu da co dinh.

Cac buoc:
 1. Khoi dong voi load_digits() (1.797 anh 8x8, chia /16)
 2. Nap MNIST day du, tach train / val / test
 3. Baseline Logistic Regression
 4-5. MLP KHONG chuan hoa vs CO chuan hoa (/255)
 6-7. So sanh 4 kien truc (accuracy val, so tham so, thoi gian, loss_curve_)
      -> CHON kien truc tot nhat theo val (khong hard-code)
 8. So sanh 3 activation tren kien truc da chon -> chon theo val
 9. So sanh 3 learning_rate_init -> chon theo val
    -> Model cuoi = Pipeline(/255 -> MLP) voi bo sieu tham so da chon
12a. Chon nguong human-in-the-loop tren VAL (muc tieu: sec sai khong nhieu hon nguoi nhap)
13. CNN nho (PyTorch) tren cung split, chon epoch theo val
--- TEST (1 lan) ---
    Bang test cuoi: LogReg / MLP / CNN
10-11. Ma tran nham lan 10x10 + 20 anh sai
12b. Human-in-the-loop tren test + phan tich chi phi (nhan su, loi/ngay)
    Do ben: dich 2-3px, xoay 10 do -> MLP vs CNN

Chay: python src/train.py
"""
from __future__ import annotations

import json
import math
import time
from functools import partial
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
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
VAL_SIZE = 10_000

ARCHITECTURES = {
    "(64)": (64,),
    "(128)": (128,),
    "(128,64)": (128, 64),
    "(256,128,64)": (256, 128, 64),
}
ACTIVATIONS = ["relu", "tanh", "logistic"]
LEARNING_RATES = [1e-2, 1e-3, 1e-4]

# Human-in-the-loop & nghiep vu (so lieu trong de bai)
HUMAN_ERROR_RATE = 0.005           # nguoi nhap sai ~0,5% so SEC
CHEQUES_PER_DAY = 8_000
CHEQUES_PER_STAFF = 300
BASE_STAFF = math.ceil(CHEQUES_PER_DAY / CHEQUES_PER_STAFF)   # 27 nguoi
DIGITS_PER_CHEQUE = 7              # vd so tien 1.250.000 -> 7 chu so
# Sec tu dong chi dung khi CA 7 chu so dung -> de sec sai <= 0,5% thi moi chu so
# tu dong phai dung >= 0,995^(1/7) ~ 99,93% (khong phai 99,5%)
TARGET_AUTO_ACCURACY = (1 - HUMAN_ERROR_RATE) ** (1 / DIGITS_PER_CHEQUE)
README_THRESHOLD = 0.99            # nguong de bai yeu cau bao cao
THRESHOLD_GRID = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999, 0.99999]

# Chia 255 dang partial(np.multiply) -> pickle/joblib duoc ma khong can import file nay
scale_255 = partial(np.multiply, 1.0 / 255.0)

_mlp_cache: dict = {}


def log(msg: str) -> None:
    print(f"[TT-10] {msg}", flush=True)


def count_params(input_dim: int, hidden_layers: tuple[int, ...], n_classes: int) -> int:
    layer_sizes = [input_dim, *hidden_layers, n_classes]
    return sum(a * b + b for a, b in zip(layer_sizes[:-1], layer_sizes[1:]))


def make_mlp(hidden, activation="relu", lr=1e-3) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden, activation=activation, solver="adam",
        alpha=1e-4, batch_size=128, learning_rate_init=lr, max_iter=100,
        early_stopping=True, n_iter_no_change=10, random_state=RANDOM_STATE,
    )


def fit_mlp(hidden, activation, lr, Xtr, ytr, Xval, yval, normalized=True) -> dict:
    """Huan luyen 1 cau hinh MLP, cham tren VALIDATION. Co cache de khong train lai."""
    key = (hidden, activation, lr, normalized)
    if key not in _mlp_cache:
        mlp = make_mlp(hidden, activation, lr)
        t0 = time.perf_counter()
        mlp.fit(Xtr, ytr)
        elapsed = time.perf_counter() - t0
        _mlp_cache[key] = {
            "model": mlp,
            "acc_val": accuracy_score(yval, mlp.predict(Xval)),
            "so_epoch": mlp.n_iter_,
            "loss_cuoi": mlp.loss_,
            "thoi_gian_s": elapsed,
        }
    return _mlp_cache[key]


# ----------------------------------------------------------------------------
# 1. Khoi dong voi load_digits()
# ----------------------------------------------------------------------------
def warmup_digits() -> float:
    digits = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(
        digits.data, digits.target, test_size=0.2, stratify=digits.target, random_state=RANDOM_STATE
    )
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=RANDOM_STATE)
    mlp.fit(Xtr / 16.0, ytr)  # pixel load_digits() trong [0,16]
    acc = accuracy_score(yte, mlp.predict(Xte / 16.0))
    log(f"  load_digits(): {len(digits.data)} anh 8x8, MLP(64) accuracy={acc:.4f}")
    return acc


# ----------------------------------------------------------------------------
# 2. Nap MNIST day du, tach train / val / test
# ----------------------------------------------------------------------------
def load_mnist():
    log("  Dang tai/doc cache MNIST (mnist_784) qua fetch_openml...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float32)
    y = y.astype(int)

    # Quy uoc chuan MNIST: 60.000 dau la train, 10.000 cuoi la test
    X_train_full, y_train_full = X[:60_000], y[:60_000]
    Xte, yte = X[60_000:], y[60_000:]
    Xtr, Xval, ytr, yval = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SIZE,
        stratify=y_train_full, random_state=RANDOM_STATE,
    )
    log(f"  Train: {len(ytr):,} | Val: {len(yval):,} | Test: {len(yte):,} | "
        f"pixel range=[{Xtr.min():.0f},{Xtr.max():.0f}]")
    return Xtr, Xval, Xte, ytr, yval, yte


# ----------------------------------------------------------------------------
# 3. Baseline Logistic Regression
# ----------------------------------------------------------------------------
def run_baseline_logreg(Xtr, ytr, Xval, yval):
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    clf.fit(scale_255(Xtr), ytr)
    elapsed = time.perf_counter() - t0
    acc = accuracy_score(yval, clf.predict(scale_255(Xval)))
    log(f"  Baseline Logistic Regression: accuracy val={acc:.4f}  [{elapsed:.1f}s]")
    return clf, acc


# ----------------------------------------------------------------------------
# 4-5. MLP KHONG chuan hoa vs CO chuan hoa
# ----------------------------------------------------------------------------
def run_normalization_comparison(Xtr, ytr, Xval, yval) -> pd.DataFrame:
    # Cau hinh tham chieu cua de bai (128,64) relu lr=1e-3, chi khac buoc chia 255
    hidden = (128, 64)
    rows = []
    for name, normalized in [("KHONG chuan hoa (0-255)", False), ("CO chuan hoa (/255)", True)]:
        Xa, Xb = (scale_255(Xtr), scale_255(Xval)) if normalized else (Xtr, Xval)
        r = fit_mlp(hidden, "relu", 1e-3, Xa, ytr, Xb, yval, normalized=normalized)
        rows.append({"cau_hinh": name, "accuracy_val": r["acc_val"], "so_epoch": r["so_epoch"],
                     "loss_cuoi": r["loss_cuoi"], "thoi_gian_s": r["thoi_gian_s"]})
        log(f"  {name:26s} acc_val={r['acc_val']:.4f}  epoch={r['so_epoch']:3d}  "
            f"loss_cuoi={r['loss_cuoi']:.4f}  [{r['thoi_gian_s']:.1f}s]")
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "so_sanh_chuan_hoa.csv", index=False)
    return result


# ----------------------------------------------------------------------------
# 6-7. So sanh 4 kien truc + duong loss / validation score
# ----------------------------------------------------------------------------
def run_architecture_comparison(Xtr_n, ytr, Xval_n, yval):
    rows, models = [], {}
    for name, hidden in ARCHITECTURES.items():
        r = fit_mlp(hidden, "relu", 1e-3, Xtr_n, ytr, Xval_n, yval)
        n_params = count_params(Xtr_n.shape[1], hidden, 10)
        rows.append({"kien_truc": name, "so_tham_so": n_params, "accuracy_val": r["acc_val"],
                     "so_epoch": r["so_epoch"], "thoi_gian_s": r["thoi_gian_s"]})
        models[name] = r["model"]
        log(f"  MLP{name:14s} params={n_params:>7,}  acc_val={r['acc_val']:.4f}  "
            f"epoch={r['so_epoch']:3d}  [{r['thoi_gian_s']:.1f}s]")
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "kien_truc_comparison.csv", index=False)

    best_name = result.loc[result["accuracy_val"].idxmax(), "kien_truc"]
    best_hidden = ARCHITECTURES[best_name]
    log(f"  => Kien truc chon theo VAL: {best_name}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(result["kien_truc"], result["accuracy_val"], color="#2980b9")
    axes[0].set_ylim(result["accuracy_val"].min() - 0.01, 1.0)
    axes[0].set_title("Accuracy VALIDATION theo kien truc")
    for i, (v, p) in enumerate(zip(result["accuracy_val"], result["so_tham_so"])):
        axes[0].text(i, v + 0.001, f"{v:.4f}\n{p:,} tham so", ha="center", fontsize=8)
    for name, m in models.items():
        axes[1].plot(m.loss_curve_, label=f"{name} - training loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Training loss")
    ax2 = axes[1].twinx()
    for name, m in models.items():
        ax2.plot(m.validation_scores_, linestyle="--", alpha=0.7)
    ax2.set_ylabel("Accuracy early-stopping (net dut)")
    axes[1].set_title("Loss (net lien) va accuracy val noi bo (net dut)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "kien_truc_comparison.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, m in models.items():
        axes[0].plot(m.loss_curve_, label=name)
        axes[1].plot(m.validation_scores_, label=name)
    axes[0].set_yscale("log"); axes[0].set_title("Training loss theo epoch (thang log)")
    axes[1].set_title("Accuracy tren 10% val noi bo cua early_stopping")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "loss_curves.png", dpi=130)
    plt.close(fig)
    return result, best_hidden


# ----------------------------------------------------------------------------
# 8. So sanh 3 activation (tren kien truc da chon)
# ----------------------------------------------------------------------------
def run_activation_comparison(hidden, Xtr_n, ytr, Xval_n, yval):
    rows, curves = [], {}
    for act in ACTIVATIONS:
        r = fit_mlp(hidden, act, 1e-3, Xtr_n, ytr, Xval_n, yval)
        m = r["model"]
        rows.append({"activation": act, "accuracy_val": r["acc_val"], "so_epoch": r["so_epoch"],
                     "loss_epoch_1": m.loss_curve_[0], "loss_epoch_5": m.loss_curve_[min(4, len(m.loss_curve_) - 1)],
                     "loss_cuoi": r["loss_cuoi"], "thoi_gian_s": r["thoi_gian_s"]})
        curves[act] = m.loss_curve_
        log(f"  activation={act:9s} acc_val={r['acc_val']:.4f}  epoch={r['so_epoch']:3d}  "
            f"loss ep1={m.loss_curve_[0]:.4f}  loss_cuoi={r['loss_cuoi']:.4f}")
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "activation_comparison.csv", index=False)
    best_act = result.loc[result["accuracy_val"].idxmax(), "activation"]
    log(f"  => Activation chon theo VAL: {best_act}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(result["activation"], result["accuracy_val"], color=["#2980b9", "#27ae60", "#c0392b"])
    axes[0].set_ylim(result["accuracy_val"].min() - 0.01, 1.0)
    axes[0].set_title("Accuracy VALIDATION theo activation")
    for i, v in enumerate(result["accuracy_val"]):
        axes[0].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
    for act, curve in curves.items():
        axes[1].plot(curve, label=act)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Training loss (log)")
    axes[1].set_title("Duong loss - 3 activation")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "activation_comparison.png", dpi=130)
    plt.close(fig)
    return result, best_act


# ----------------------------------------------------------------------------
# 9. So sanh 3 learning_rate_init
# ----------------------------------------------------------------------------
def run_learning_rate_comparison(hidden, act, Xtr_n, ytr, Xval_n, yval):
    rows, curves = [], {}
    for lr in LEARNING_RATES:
        r = fit_mlp(hidden, act, lr, Xtr_n, ytr, Xval_n, yval)
        m = r["model"]
        rows.append({"learning_rate_init": lr, "accuracy_val": r["acc_val"], "so_epoch": r["so_epoch"],
                     "loss_cuoi": r["loss_cuoi"], "thoi_gian_s": r["thoi_gian_s"]})
        curves[lr] = m.loss_curve_
        log(f"  learning_rate_init={lr:<7g} acc_val={r['acc_val']:.4f}  epoch={r['so_epoch']:3d}  "
            f"loss_cuoi={r['loss_cuoi']:.4f}")
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "learning_rate_comparison.csv", index=False)
    best_lr = float(result.loc[result["accuracy_val"].idxmax(), "learning_rate_init"])
    log(f"  => learning_rate_init chon theo VAL: {best_lr:g}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for lr, curve in curves.items():
        ax.plot(curve, label=f"lr={lr:g}")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss (log)")
    ax.set_title("Duong loss theo 3 learning_rate_init")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "learning_rate_curves.png", dpi=130)
    plt.close(fig)
    return result, best_lr


def build_final_pipeline(hidden, act, lr, Xtr, ytr, Xval, yval) -> Pipeline:
    """Model cuoi = MLP da train voi bo sieu tham so chon tren val, boc chung buoc /255.

    Khong train lai tren train+val: nguong HITL duoc hieu chinh tren val voi CHINH model nay,
    train lai se lam thay doi phan bo do tin cay va nguong khong con dung.
    """
    mlp = fit_mlp(hidden, act, lr, scale_255(Xtr), ytr, scale_255(Xval), yval)["model"]
    scaler = FunctionTransformer(scale_255, validate=False).fit(Xtr[:1])
    pipe = Pipeline([("scale", scaler), ("mlp", mlp)])
    joblib.dump(pipe, MODELS_DIR / "mlp_pipeline.joblib")
    log(f"  Model cuoi MLP{hidden} act={act} lr={lr:g}: "
        f"acc_val={accuracy_score(yval, pipe.predict(Xval)):.4f} (nhan anh tho 0-255)")
    log("  Da luu -> models/mlp_pipeline.joblib")
    return pipe


# ----------------------------------------------------------------------------
# 12. Human-in-the-loop
# ----------------------------------------------------------------------------
def hitl_table(proba: np.ndarray, y: np.ndarray, thresholds) -> pd.DataFrame:
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    rows = []
    for t in thresholds:
        auto = conf >= t
        n_auto = int(auto.sum())
        rows.append({
            "nguong": t,
            "pct_tu_dong": n_auto / len(y),
            "pct_chuyen_nguoi": 1 - n_auto / len(y),
            "accuracy_phan_tu_dong": accuracy_score(y[auto], pred[auto]) if n_auto else float("nan"),
            "so_loi_lot_qua": int((auto & (pred != y)).sum()),
        })
    return pd.DataFrame(rows)


def choose_threshold(proba_val, yval, name: str):
    """Nguong NHO nhat (tu dong nhieu nhat) ma accuracy chu so tu dong tren VAL >= TARGET_AUTO_ACCURACY."""
    table = hitl_table(proba_val, yval, THRESHOLD_GRID)
    table.insert(0, "model", name)
    ok = table[table["accuracy_phan_tu_dong"] >= TARGET_AUTO_ACCURACY]
    chosen = float(ok["nguong"].min()) if len(ok) else float(max(THRESHOLD_GRID))
    log(f"  [{name}] nguong chon tren VAL (acc tu dong >= {TARGET_AUTO_ACCURACY:.3%}): {chosen:g}")
    for _, r in table.iterrows():
        log(f"    nguong={r['nguong']:<7g} tu dong={r['pct_tu_dong']:.2%}  "
            f"acc(tu dong)={r['accuracy_phan_tu_dong']:.4%}")
    return table, chosen


def cheque_cost(proba, y, threshold, n_digits=DIGITS_PER_CHEQUE, n_sim=200_000) -> dict:
    """Mo phong sec gom n_digits chu so lay ngau nhien tu tap danh gia.

    Che do A (ca sec): sec duoc tu dong chi khi MOI chu so >= nguong, nguoc lai ca sec sang nguoi.
    Che do B (tung chu so): nguoi chi go lai cac chu so bi gan co (gia dinh thoi gian ~ so chu so).
    """
    conf, correct = proba.max(axis=1), proba.argmax(axis=1) == y
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.integers(0, len(y), size=(n_sim, n_digits))
    digit_auto = conf[idx] >= threshold
    digit_wrong = ~correct[idx]

    sec_auto = digit_auto.all(axis=1)
    pct_sec_auto = sec_auto.mean()
    err_sec_auto = (sec_auto & digit_wrong.any(axis=1)).sum() / max(sec_auto.sum(), 1)
    sec_auto_day = CHEQUES_PER_DAY * pct_sec_auto
    sec_human_day = CHEQUES_PER_DAY - sec_auto_day
    loi_A = sec_auto_day * err_sec_auto + sec_human_day * HUMAN_ERROR_RATE

    pct_digit_flag = 1 - digit_auto.mean()
    sec_has_flag = (~digit_auto).any(axis=1).mean()
    sec_auto_wrong_B = (digit_auto & digit_wrong).any(axis=1).mean()
    loi_B = CHEQUES_PER_DAY * (sec_auto_wrong_B + sec_has_flag * HUMAN_ERROR_RATE)

    return {
        "nguong": threshold,
        "so_chu_so_moi_sec": n_digits,
        "pct_chu_so_tu_dong": 1 - pct_digit_flag,
        "A_pct_sec_tu_dong": pct_sec_auto,
        "A_sec_chuyen_nguoi_moi_ngay": round(sec_human_day),
        "A_nhan_vien_can": math.ceil(sec_human_day / CHEQUES_PER_STAFF),
        "A_ty_le_sai_sec_tu_dong": err_sec_auto,
        "A_sec_sai_moi_ngay": loi_A,
        "B_nhan_vien_can": math.ceil(BASE_STAFF * pct_digit_flag),
        "B_sec_sai_moi_ngay": loi_B,
    }


# ----------------------------------------------------------------------------
# 13. CNN nho (PyTorch)
# ----------------------------------------------------------------------------
class SmallCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3), torch.nn.ReLU(), torch.nn.MaxPool2d(2),   # 28 -> 26 -> 13
            torch.nn.Conv2d(32, 64, 3), torch.nn.ReLU(), torch.nn.MaxPool2d(2),  # 13 -> 11 -> 5
        )
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(64 * 5 * 5, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.head(self.features(x))


def _to_tensor(X):
    return torch.from_numpy(scale_255(X).astype(np.float32).reshape(-1, 1, 28, 28))


def cnn_predict_proba(model: SmallCNN, X: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for xb in torch.split(_to_tensor(X), 2048):
            out.append(torch.softmax(model(xb), dim=1).numpy())
    return np.concatenate(out)


def train_cnn(Xtr, ytr, Xval, yval, max_epochs=10, patience=3):
    torch.manual_seed(RANDOM_STATE)
    model = SmallCNN()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    Xt, yt = _to_tensor(Xtr), torch.from_numpy(ytr.astype(np.int64))
    gen = torch.Generator().manual_seed(RANDOM_STATE)

    best_acc, best_state, best_epoch, wait, rows = -1.0, None, 0, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(yt), generator=gen)
        total = 0.0
        for i in range(0, len(yt), 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        acc_val = accuracy_score(yval, cnn_predict_proba(model, Xval).argmax(axis=1))
        rows.append({"epoch": epoch, "train_loss": total / len(yt), "accuracy_val": acc_val})
        log(f"  CNN epoch {epoch:2d}: train_loss={total / len(yt):.4f}  acc_val={acc_val:.4f}")
        if acc_val > best_acc:
            best_acc, best_epoch, wait = acc_val, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    elapsed = time.perf_counter() - t0
    model.load_state_dict(best_state)
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "cnn_epochs.csv", index=False)
    log(f"  => CNN: {n_params:,} tham so, epoch tot nhat theo VAL={best_epoch}, "
        f"acc_val={best_acc:.4f}  [{elapsed:.1f}s]")
    return model, {"so_tham_so": n_params, "epoch_chon": best_epoch, "accuracy_val": best_acc,
                   "thoi_gian_s": elapsed}


# ----------------------------------------------------------------------------
# TEST - chi chay 1 lan, sau khi moi lua chon da co dinh
# ----------------------------------------------------------------------------
def final_test_table(logreg, mlp_pipe, cnn, mlp_info, cnn_info, Xte, yte):
    rows = [
        {"model": "Logistic Regression (baseline)", "so_tham_so": 784 * 10 + 10,
         "accuracy_test": accuracy_score(yte, logreg.predict(scale_255(Xte)))},
        {"model": f"MLP{mlp_info['hidden']} {mlp_info['activation']} lr={mlp_info['lr']:g}",
         "so_tham_so": count_params(784, mlp_info["hidden"], 10),
         "accuracy_test": accuracy_score(yte, mlp_pipe.predict(Xte))},
        {"model": "CNN 2 conv + 1 dense", "so_tham_so": cnn_info["so_tham_so"],
         "accuracy_test": accuracy_score(yte, cnn_predict_proba(cnn, Xte).argmax(axis=1))},
    ]
    result = pd.DataFrame(rows)
    result["so_loi_tren_10k"] = ((1 - result["accuracy_test"]) * len(yte)).round().astype(int)
    result.to_csv(REPORTS_DIR / "ket_qua_test.csv", index=False)
    for _, r in result.iterrows():
        log(f"  {r['model']:34s} params={r['so_tham_so']:>7,}  acc_test={r['accuracy_test']:.4f}  "
            f"sai={r['so_loi_tren_10k']}")
    return result


def run_confusion_and_errors(pipe: Pipeline, Xte: np.ndarray, yte: np.ndarray) -> pd.DataFrame:
    pred = pipe.predict(Xte)
    cm = confusion_matrix(yte, pred, labels=list(range(10)))

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=1, vmax=cm.max()))
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Du doan"); ax.set_ylabel("Thuc te")
    ax.set_title("Ma tran nham lan 10x10 - MLP tren MNIST test (mau thang log)")
    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > 100 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_10x10.png", dpi=130)
    plt.close(fig)

    pairs = pd.DataFrame([
        {"thuc_te": i, "du_doan": j, "so_lan": int(cm[i, j]), "pct_cua_lop_thuc_te": cm[i, j] / cm[i].sum()}
        for i in range(10) for j in range(10) if i != j and cm[i, j] > 0
    ]).sort_values("so_lan", ascending=False)
    pairs.to_csv(REPORTS_DIR / "cap_nham_lan.csv", index=False)

    # Gop 2 chieu (a<->b) de thay cap nao kho phan biet nhat
    sym = cm + cm.T
    two_way = pd.DataFrame([{"cap": f"{i}<->{j}", "tong_so_lan": int(sym[i, j])}
                            for i in range(10) for j in range(i + 1, 10) if sym[i, j] > 0])
    two_way = two_way.sort_values("tong_so_lan", ascending=False)
    two_way.to_csv(REPORTS_DIR / "cap_nham_lan_hai_chieu.csv", index=False)

    per_class = pd.DataFrame({"chu_so": range(10), "so_mau": cm.sum(axis=1),
                              "recall": np.diag(cm) / cm.sum(axis=1)})
    per_class.to_csv(REPORTS_DIR / "recall_theo_chu_so.csv", index=False)
    log(f"  Top 5 cap (1 chieu): {pairs.head(5)[['thuc_te', 'du_doan', 'so_lan']].values.tolist()}")
    log(f"  Top 5 cap (2 chieu): {two_way.head(5).values.tolist()}")

    wrong_idx = np.where(pred != yte)[0]
    rng = np.random.RandomState(RANDOM_STATE)
    show_idx = rng.choice(wrong_idx, size=min(20, len(wrong_idx)), replace=False)
    conf = pipe.predict_proba(Xte).max(axis=1)
    fig, axes = plt.subplots(4, 5, figsize=(11, 9.5))
    for ax, idx in zip(axes.ravel(), show_idx):
        ax.imshow(Xte[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"that={yte[idx]} doan={pred[idx]}\ntin cay={conf[idx]:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle("20 anh bi du doan SAI (chon ngau nhien) - kem do tin cay")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "anh_sai.png", dpi=130)
    plt.close(fig)
    log(f"  So mau sai: {len(wrong_idx)} / {len(yte)}; trong 20 anh sai hien thi, "
        f"{int((conf[show_idx] < README_THRESHOLD).sum())} anh co tin cay < {README_THRESHOLD}")
    return pairs


def shift_images(X: np.ndarray, dx: int, dy: int) -> np.ndarray:
    imgs = X.reshape(-1, 28, 28)
    out = np.zeros_like(imgs)
    src_x, dst_x = (slice(0, 28 - dx), slice(dx, 28)) if dx >= 0 else (slice(-dx, 28), slice(0, 28 + dx))
    src_y, dst_y = (slice(0, 28 - dy), slice(dy, 28)) if dy >= 0 else (slice(-dy, 28), slice(0, 28 + dy))
    out[:, dst_y, dst_x] = imgs[:, src_y, src_x]
    return out.reshape(len(X), -1)


def rotate_images(X: np.ndarray, angle: float) -> np.ndarray:
    imgs = ndimage.rotate(X.reshape(-1, 28, 28), angle, axes=(2, 1), reshape=False, order=1)
    return np.clip(imgs, 0, 255).reshape(len(X), -1)


def run_robustness(mlp_pipe, cnn, Xte, yte) -> pd.DataFrame:
    variants = {
        "goc": Xte,
        "dich phai 2px": shift_images(Xte, 2, 0),
        "dich phai 3px": shift_images(Xte, 3, 0),
        "dich cheo 3px (phai+xuong)": shift_images(Xte, 3, 3),
        "xoay 10 do": rotate_images(Xte, 10),
        "xoay 20 do": rotate_images(Xte, 20),
    }
    rows = []
    for name, Xv in variants.items():
        rows.append({"bien_doi": name,
                     "MLP": accuracy_score(yte, mlp_pipe.predict(Xv)),
                     "CNN": accuracy_score(yte, cnn_predict_proba(cnn, Xv).argmax(axis=1))})
        log(f"  {name:28s} MLP={rows[-1]['MLP']:.4f}  CNN={rows[-1]['CNN']:.4f}")
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "do_ben_dich_xoay.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(result))
    ax.bar(x - 0.2, result["MLP"], 0.4, label="MLP", color="#2980b9")
    ax.bar(x + 0.2, result["CNN"], 0.4, label="CNN", color="#27ae60")
    ax.set_xticks(x); ax.set_xticklabels(result["bien_doi"], rotation=15, fontsize=8)
    ax.set_ylabel("Accuracy test"); ax.set_title("Do ben voi dich chuyen / xoay: MLP vs CNN")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "do_ben_dich_xoay.png", dpi=130)
    plt.close(fig)
    return result


def run_hitl_and_cost(proba_by_model: dict, chosen_by_model: dict, yte) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hitl_rows, cost_rows = [], []
    for name, proba in proba_by_model.items():
        thresholds = sorted({chosen_by_model[name], README_THRESHOLD})
        t = hitl_table(proba, yte, thresholds)
        t.insert(0, "model", name)
        t.insert(2, "nguong_chon_tren_val", t["nguong"] == chosen_by_model[name])
        hitl_rows.append(t)
        for thr in thresholds:
            c = cheque_cost(proba, yte, thr)
            cost_rows.append({"model": name, "nguong_chon_tren_val": thr == chosen_by_model[name], **c})
    hitl = pd.concat(hitl_rows, ignore_index=True)
    hitl.to_csv(REPORTS_DIR / "human_in_the_loop.csv", index=False)

    baseline = {"model": "100% nguoi nhap (hien tai)", "nguong_chon_tren_val": False, "nguong": float("nan"),
                "so_chu_so_moi_sec": DIGITS_PER_CHEQUE, "pct_chu_so_tu_dong": 0.0, "A_pct_sec_tu_dong": 0.0,
                "A_sec_chuyen_nguoi_moi_ngay": CHEQUES_PER_DAY, "A_nhan_vien_can": BASE_STAFF,
                "A_ty_le_sai_sec_tu_dong": float("nan"), "A_sec_sai_moi_ngay": CHEQUES_PER_DAY * HUMAN_ERROR_RATE,
                "B_nhan_vien_can": BASE_STAFF, "B_sec_sai_moi_ngay": CHEQUES_PER_DAY * HUMAN_ERROR_RATE}
    cost = pd.DataFrame([baseline, *cost_rows])
    cost.to_csv(REPORTS_DIR / "chi_phi_hitl.csv", index=False)

    # Do nhay theo so chu so moi sec (MLP, nguong chon tren val)
    mlp_name = next(iter(proba_by_model))
    sens = pd.DataFrame([cheque_cost(proba_by_model[mlp_name], yte, chosen_by_model[mlp_name], n_digits=k)
                         for k in [1, 4, 6, 7, 9]])
    sens.insert(0, "model", mlp_name)
    sens.to_csv(REPORTS_DIR / "chi_phi_theo_so_chu_so.csv", index=False)

    for _, r in cost.iterrows():
        log(f"  {r['model']:28s} nguong={r['nguong']:<7g} chu so tu dong={r['pct_chu_so_tu_dong']:.2%}  "
            f"[A] sec tu dong={r['A_pct_sec_tu_dong']:.2%} NV={r['A_nhan_vien_can']} "
            f"sai/ngay={r['A_sec_sai_moi_ngay']:.1f}  [B] NV={r['B_nhan_vien_can']} "
            f"sai/ngay={r['B_sec_sai_moi_ngay']:.1f}")
    return hitl, cost, sens


def save_summary(n_split, warmup_acc, logreg_val, norm_result, arch_result, act_result, lr_result,
                 mlp_info, thr_mlp, thr_cnn, cnn_info, test_result, pairs, hitl, cost, robust) -> None:
    summary = {
        "du_lieu": dict(zip(["train", "val", "test"], map(int, n_split))),
        "warmup_load_digits_accuracy": float(warmup_acc),
        "baseline_logreg_accuracy_val": float(logreg_val),
        "normalization_comparison_val": norm_result.to_dict(orient="records"),
        "architecture_comparison_val": arch_result.to_dict(orient="records"),
        "activation_comparison_val": act_result.to_dict(orient="records"),
        "learning_rate_comparison_val": lr_result.to_dict(orient="records"),
        "muc_tieu_accuracy_chu_so_tu_dong": TARGET_AUTO_ACCURACY,
        "lua_chon_theo_val": {"hidden_layer_sizes": list(mlp_info["hidden"]), "activation": mlp_info["activation"],
                              "learning_rate_init": mlp_info["lr"], "nguong_hitl_mlp": thr_mlp,
                              "nguong_hitl_cnn": thr_cnn, "cnn_epoch": cnn_info["epoch_chon"]},
        "cnn": cnn_info,
        "ket_qua_test": test_result.to_dict(orient="records"),
        "top5_cap_nham_lan_test": pairs.head(5).to_dict(orient="records"),
        "human_in_the_loop_test": hitl.to_dict(orient="records"),
        "chi_phi_nghiep_vu_test": cost.to_dict(orient="records"),
        "do_ben_test": robust.to_dict(orient="records"),
    }
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    log("Da luu -> reports/tom_tat.json")


def main() -> None:
    log("1. Khoi dong voi load_digits()...")
    warmup_acc = warmup_digits()

    log("2. Nap MNIST day du, tach train/val/test...")
    Xtr, Xval, Xte, ytr, yval, yte = load_mnist()
    Xtr_n, Xval_n = scale_255(Xtr), scale_255(Xval)

    log("3. Baseline Logistic Regression...")
    logreg, logreg_val = run_baseline_logreg(Xtr, ytr, Xval, yval)

    log("4-5. MLP KHONG chuan hoa vs CO chuan hoa (cham tren VAL)...")
    norm_result = run_normalization_comparison(Xtr, ytr, Xval, yval)

    log("6-7. So sanh 4 kien truc (cham tren VAL)...")
    arch_result, best_hidden = run_architecture_comparison(Xtr_n, ytr, Xval_n, yval)

    log("8. So sanh 3 activation (cham tren VAL)...")
    act_result, best_act = run_activation_comparison(best_hidden, Xtr_n, ytr, Xval_n, yval)

    log("9. So sanh 3 learning_rate_init (cham tren VAL)...")
    lr_result, best_lr = run_learning_rate_comparison(best_hidden, best_act, Xtr_n, ytr, Xval_n, yval)

    log("Dong goi model cuoi (Pipeline /255 -> MLP)...")
    mlp_pipe = build_final_pipeline(best_hidden, best_act, best_lr, Xtr, ytr, Xval, yval)
    mlp_info = {"hidden": best_hidden, "activation": best_act, "lr": best_lr}

    log("13. Huan luyen CNN nho (chon epoch theo VAL)...")
    cnn, cnn_info = train_cnn(Xtr, ytr, Xval, yval)

    log("12a. Chon nguong human-in-the-loop tren VAL...")
    mlp_name, cnn_name = "MLP", "CNN"
    val_mlp, thr_mlp = choose_threshold(mlp_pipe.predict_proba(Xval), yval, mlp_name)
    val_cnn, thr_cnn = choose_threshold(cnn_predict_proba(cnn, Xval), yval, cnn_name)
    pd.concat([val_mlp, val_cnn], ignore_index=True).to_csv(REPORTS_DIR / "chon_nguong_val.csv", index=False)

    log("===== TEST (dung 1 lan, moi lua chon da co dinh) =====")
    test_result = final_test_table(logreg, mlp_pipe, cnn, mlp_info, cnn_info, Xte, yte)

    log("10-11. Ma tran nham lan + 20 anh sai (MLP, test)...")
    pairs = run_confusion_and_errors(mlp_pipe, Xte, yte)

    log("12b. Human-in-the-loop + chi phi nghiep vu (test)...")
    proba_te = {mlp_name: mlp_pipe.predict_proba(Xte), cnn_name: cnn_predict_proba(cnn, Xte)}
    hitl, cost, _ = run_hitl_and_cost(proba_te, {mlp_name: thr_mlp, cnn_name: thr_cnn}, yte)

    log("Do ben dich chuyen / xoay (test)...")
    robust = run_robustness(mlp_pipe, cnn, Xte, yte)

    save_summary(n_split=(len(ytr), len(yval), len(yte)), warmup_acc=warmup_acc, logreg_val=logreg_val,
                 norm_result=norm_result, arch_result=arch_result, act_result=act_result, lr_result=lr_result,
                 mlp_info=mlp_info, thr_mlp=thr_mlp, thr_cnn=thr_cnn, cnn_info=cnn_info, test_result=test_result,
                 pairs=pairs, hitl=hitl, cost=cost, robust=robust)

    log("HOAN THANH. Xem ket qua chi tiet trong thu muc reports/.")


if __name__ == "__main__":
    main()
