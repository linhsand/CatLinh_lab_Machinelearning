"""TT-26 - CNN: sang loc viem phoi tren anh X-quang nguc (Chest X-Ray Images (Pneumonia), Kaggle).

Nguyen tac danh gia:
  * Val goc (16 anh) vo dung -> gop vao train, TU TACH 15% val theo benh nhan + stratify. Test goc giu nguyen.
  * EarlyStopping / chon mo hinh / chon nguong: chi nhin VAL. TEST chi dung 1 lan de bao cao.
  * Metric chinh: RECALL lop PNEUMONIA (yeu cau >= 0,97 tren val, >= 0,96 tren test) + so ca BO SOT.
    ROC-AUC de so sanh chat luong xep hang giua cac mo hinh. Accuracy chi de tham khao.

Cac buoc (theo README de bai):
  1.    Dem anh moi lop moi tap -> 3 van de                    -> run_counts()
  2.    Tu tach lai val 15%                                     -> load()
  3.    8 anh moi lop                                           -> show_samples()
  4.    tf.data + augmentation (khong lat)                      -> show_augmentation()
  5.    Baseline CNN nho train tu dau                           -> run_scratch()
  6.    Transfer Learning giai doan 1 (dong bang base)          -> run_transfer()
  7.    Fine-tuning giai doan 2 (lr 1e-5) + doi chung lr 1e-3   -> run_finetune(), run_finetune_lr_lon()
  8.    Learning curves                                         -> plot_learning_curves()
  9.    Nguong dat recall >= 0,97 tren val                      -> choose_threshold(), compare_models()
  10.   Test 1 lan: ma tran nham lan, so ca bo sot              -> evaluate_test()
  11.   Grad-CAM 6 ca + do dinh luong vung chu y                -> run_gradcam()
  12.   10 ca du doan sai                                       -> show_errors()

Moi lan train luu checkpoint (models/runs/) + history + du doan. Chay lai se DOC LAI thay vi train lai
(train lai: dat bien moi truong TT26_RETRAIN=1). Chay: python src/train.py (~1,5-2 gio tren CPU lan dau).
"""
from __future__ import annotations

import json
import os
import shutil
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight

import data_pipeline as D
import model as M

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
RUNS_DIR = MODELS_DIR / "runs"           # checkpoint + du doan cua tung lan train (khong commit)
for d in (REPORTS_DIR, MODELS_DIR, RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH = 32
RETRAIN = os.environ.get("TT26_RETRAIN") == "1"
EPOCHS = {"cnn_tu_dau": 20, "tl_dong_bang": 12, "tl_finetune": 12, "tl_finetune_lr_lon": 4}
TARGET_RECALL = 0.97
N_UNFREEZE = 30
NAMES = {"cnn_tu_dau": "CNN tu dau (3 khoi Conv)", "tl_dong_bang": "EfficientNetB0 dong bang",
         "tl_finetune": "EfficientNetB0 fine-tune (lr 1e-5)",
         "tl_finetune_lr_lon": "EfficientNetB0 fine-tune lr 1e-3 (doi chung)"}


def log(msg: str) -> None:
    print(msg, flush=True)


def save_fig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 1-2. Du lieu
def load() -> dict:
    """Doc cache anh, tu tach val 15% (theo benh nhan, stratify). Tra ve dict anh/nhan theo tap."""
    df, images = D.load_index()
    df = D.resplit(df)
    split = D.split_table(df)
    split.to_csv(REPORTS_DIR / "chia_du_lieu.csv")
    # khong benh nhan nao nam o ca train va val. (Test goc danh so personN rieng: cung so nhung khac tac nhan,
    # anh khac han nhau -> khong phai cung benh nhan, nen khong kiem tra train/test theo ma.)
    groups = df.groupby("tap")["benh_nhan"].apply(set)
    assert not (groups["train"] & groups["val"])
    data = {"df": df, "images": images}
    for t in ["train", "val", "test"]:
        idx = np.flatnonzero(df["tap"].to_numpy() == t)
        data[t] = {"idx": idx, "X": images[idx], "y": df["y"].to_numpy()[idx]}
    log(f"[load] {len(df):,} anh | train/val/test = "
        + "/".join(f"{len(data[t]['y']):,}" for t in ["train", "val", "test"]))
    return data


def run_counts(data: dict) -> pd.DataFrame:
    """Buoc 1: dem anh theo tap goc -> 3 van de (val 16 anh, mat can bang, train != test)."""
    t = D.count_table(data["df"])
    t.to_csv(REPORTS_DIR / "so_luong_anh.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    t[D.CLASSES].plot.bar(ax=axes[0], color=["#4C72B0", "#C44E52"], rot=0)
    for i, (n, p) in enumerate(zip(t["NORMAL"], t["PNEUMONIA"])):
        axes[0].text(i - 0.12, n, f"{n:,}", ha="center", va="bottom", fontsize=8)
        axes[0].text(i + 0.12, p, f"{p:,}", ha="center", va="bottom", fontsize=8)
    axes[0].set_title("So anh theo tap goc (Kaggle)"); axes[0].set_xlabel("")
    s = D.split_table(data["df"])
    axes[1].bar(s.index, s["ti_le_PNEUMONIA"], color="#C44E52")
    for i, (r, n) in enumerate(zip(s["ti_le_PNEUMONIA"], s["so_anh"])):
        axes[1].text(i, r, f"{r:.1%}\n(n={n:,})", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, 1); axes[1].set_title("Ti le PNEUMONIA sau khi tach lai val")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    save_fig(fig, "so_luong_anh.png")
    return t


def show_samples(data: dict, n: int = 8) -> None:
    """Buoc 3: 8 anh moi lop (tu tap train), kem tac nhan voi PNEUMONIA."""
    df = data["df"]; rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.8))
    for r, lop in enumerate(D.CLASSES):
        pick = rng.choice(np.flatnonzero((df["tap"] == "train") & (df["lop"] == lop)), n, replace=False)
        for ax, i in zip(axes[r], pick):
            ax.imshow(data["images"][i], cmap="gray"); ax.axis("off")
            ax.set_title(f"{lop}" + (f"\n{df['tac_nhan'].iat[i]}" if lop == "PNEUMONIA" else "\n"), fontsize=8)
    save_fig(fig, "mau_anh.png")


def show_augmentation(data: dict, n: int = 7) -> None:
    """Buoc 4: 1 anh goc + n ban augmentation. Tim van o ben PHAI anh (= ben trai benh nhan) o moi ban."""
    aug = D.augmenter(SEED)
    x = D.to_batch(data["train"]["X"][:1])
    fig, axes = plt.subplots(1, n + 1, figsize=(2.1 * (n + 1), 2.5))
    axes[0].imshow(x[0, ..., 0], cmap="gray", vmin=0, vmax=255); axes[0].set_title("goc", fontsize=9)
    for k, ax in enumerate(axes[1:]):
        a = np.clip(aug(x, training=True).numpy(), 0, 255)
        ax.imshow(a[0, ..., 0], cmap="gray", vmin=0, vmax=255); ax.set_title(f"augment {k + 1}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    save_fig(fig, "augmentation.png")


# ------------------------------------------------------------------ Huan luyen (co cache)
def class_weights(y: np.ndarray) -> dict:
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    return {0: float(w[0]), 1: float(w[1])}


def fit_run(name: str, build, data: dict, lr: float, patience: int = 4) -> dict:
    """Train 1 cau hinh (hoac doc lai neu da co). build() tra ve model CHUA compile.

    Luu: models/runs/<name>.keras (ban tot nhat theo val_auc), reports/history_<name>.csv,
    models/runs/<name>_meta.json (thoi gian, so tham so), models/runs/<name>_pred.npz (xac suat val/test).
    """
    ckpt, hist_csv = RUNS_DIR / f"{name}.keras", REPORTS_DIR / f"history_{name}.csv"
    meta_p, pred_p = RUNS_DIR / f"{name}_meta.json", RUNS_DIR / f"{name}_pred.npz"
    if not RETRAIN and all(p.exists() for p in (ckpt, hist_csv, meta_p, pred_p)):
        log(f"[{name}] doc lai ket qua da train ({ckpt.name})")
        model = tf.keras.models.load_model(ckpt)
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        pred = np.load(pred_p)
        return {"name": name, "model": model, "history": pd.read_csv(hist_csv), **meta,
                "s_val": pred["s_val"], "s_test": pred["s_test"], "checkpoint": ckpt}

    tf.keras.utils.set_random_seed(SEED)
    model = M.compile_model(build(), lr)
    tr = D.make_dataset(data["train"]["X"], data["train"]["y"], training=True, batch=BATCH, seed=SEED)
    va = D.make_dataset(data["val"]["X"], data["val"]["y"], training=False, batch=BATCH)
    cw = class_weights(data["train"]["y"])
    n_train = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    log(f"[{name}] train toi da {EPOCHS[name]} epoch | lr={lr:g} | tham so train duoc = {n_train:,} "
        f"/ {model.count_params():,} | class_weight={ {k: round(v, 3) for k, v in cw.items()} }")
    t0 = time.time()
    h = model.fit(tr, validation_data=va, epochs=EPOCHS[name], class_weight=cw, verbose=2,
                  callbacks=M.make_callbacks(ckpt, patience=patience))
    secs = time.time() - t0
    hist = pd.DataFrame(h.history); hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    hist.to_csv(hist_csv, index=False)
    best = int(hist["val_auc"].idxmax()) + 1
    s_val = model.predict(D.make_dataset(data["val"]["X"], data["val"]["y"], False), verbose=0)[:, 0]
    s_test = model.predict(D.make_dataset(data["test"]["X"], data["test"]["y"], False), verbose=0)[:, 0]
    np.savez(pred_p, s_val=s_val, s_test=s_test)
    meta = {"lr": lr, "so_epoch": len(hist), "epoch_tot_nhat": best, "thoi_gian_train_s": round(secs, 1),
            "tham_so": int(model.count_params()), "tham_so_train_duoc": n_train, "class_weight": cw}
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"[{name}] xong {len(hist)} epoch trong {secs / 60:.1f} phut | epoch tot nhat {best} | "
        f"val_auc={hist['val_auc'].max():.4f}")
    return {"name": name, "model": model, "history": hist, **meta, "s_val": s_val, "s_test": s_test,
            "checkpoint": ckpt}


def run_scratch(data: dict) -> dict:
    """Buoc 5: CNN nho train tu dau (co augmentation + class_weight nhu cac mo hinh khac)."""
    return fit_run("cnn_tu_dau", M.build_small_cnn, data, lr=1e-3, patience=5)


def run_transfer(data: dict) -> dict:
    """Buoc 6: EfficientNetB0 ImageNet dong bang, chi train dau phan loai (1.281 tham so)."""
    return fit_run("tl_dong_bang", M.build_transfer, data, lr=1e-3)


def _from_frozen(tl: dict):
    """build() cho giai doan 2: nap lai model giai doan 1 tot nhat roi mo khoa N_UNFREEZE lop cuoi."""
    def build():
        model = tf.keras.models.load_model(tl["checkpoint"], compile=False)
        M.unfreeze_top(model, N_UNFREEZE)
        return model
    return build


def run_finetune(data: dict, tl: dict) -> dict:
    """Buoc 7: fine-tune 30 lop cuoi (tru BatchNorm) voi lr 1e-5 = nho hon 100 lan giai doan 1."""
    return fit_run("tl_finetune", _from_frozen(tl), data, lr=1e-5)


def run_finetune_lr_lon(data: dict, tl: dict) -> dict:
    """Doi chung cho cam bay "fine-tune voi lr 1e-3": cung diem xuat phat, cung so lop mo khoa, 4 epoch."""
    return fit_run("tl_finetune_lr_lon", _from_frozen(tl), data, lr=1e-3, patience=10)


# ------------------------------------------------------------------ 8. Learning curves
def plot_learning_curves(runs: dict) -> None:
    """Hang 1: CNN tu dau. Hang 2: TL giai doan 1 noi tiep giai doan 2 (va doi chung lr 1e-3)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5))
    metrics = [("loss", "Loss"), ("auc", "ROC-AUC"), ("recall", "Recall @0,5")]
    h = runs["cnn_tu_dau"]["history"]
    for ax, (m, title) in zip(axes[0], metrics):
        ax.plot(h["epoch"], h[m], "o-", ms=3, c="#4C72B0", label="train")
        ax.plot(h["epoch"], h[f"val_{m}"], "o-", ms=3, c="#C44E52", label="val")
        ax.axvline(runs["cnn_tu_dau"]["epoch_tot_nhat"], ls=":", c="gray")
        ax.set_title(f"CNN tu dau - {title}"); ax.set_xlabel("epoch")
    axes[0][0].legend()
    h1 = runs["tl_dong_bang"]["history"]; n1 = len(h1)
    for ax, (m, title) in zip(axes[1], metrics):
        ax.plot(h1["epoch"], h1[m], "o-", ms=3, c="#4C72B0", label="GD1 dong bang - train")
        ax.plot(h1["epoch"], h1[f"val_{m}"], "o-", ms=3, c="#C44E52", label="GD1 dong bang - val")
        for key, ls, lab in [("tl_finetune", "-", "GD2 lr 1e-5"), ("tl_finetune_lr_lon", "--", "GD2 lr 1e-3")]:
            if key not in runs:
                continue
            h2 = runs[key]["history"]; ep = n1 + h2["epoch"]
            # GD2 bat dau tu trong so tot nhat cua GD1 -> noi tu diem do
            b = runs["tl_dong_bang"]["epoch_tot_nhat"] - 1
            ax.plot([h1["epoch"].iat[b], *ep], [h1[m].iat[b], *h2[m]], ls, c="#55A868" if key == "tl_finetune" else "#8172B2",
                    lw=1.2, label=f"{lab} - train")
            ax.plot([h1["epoch"].iat[b], *ep], [h1[f"val_{m}"].iat[b], *h2[f"val_{m}"]], ls, marker="s", ms=3,
                    c="#DD8452" if key == "tl_finetune" else "#937860", label=f"{lab} - val")
        ax.axvline(n1 + 0.5, ls=":", c="gray"); ax.set_title(f"Transfer Learning - {title}"); ax.set_xlabel("epoch")
    axes[1][0].legend(fontsize=7)
    for ax in axes[:, 2]:
        ax.set_ylim(0.5, 1.01)
    axes[0, 1].set_ylim(0.45, 1.01)    # CNN tu dau: val AUC roi xuong 0,5 nhieu lan -> de thay
    axes[1, 1].set_ylim(0.8, 1.005)
    save_fig(fig, "learning_curves.png")


# ------------------------------------------------------------------ 9-10. Nguong & danh gia
def choose_threshold(y: np.ndarray, s: np.ndarray, target: float = TARGET_RECALL) -> float:
    """Nguong CAO NHAT van giu recall >= target tren val (cao nhat = it bao dong gia nhat)."""
    fpr, tpr, thr = roc_curve(y, s)
    ok = np.flatnonzero(tpr >= target)
    return float(min(thr[ok[0]], 1.0))


def metrics_at(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, (s >= thr).astype(int), labels=[0, 1]).ravel()
    return {"recall": tp / (tp + fn), "specificity": tn / (tn + fp), "precision": tp / max(tp + fp, 1),
            "accuracy": (tp + tn) / len(y), "bo_sot_FN": int(fn), "bao_dong_gia_FP": int(fp), "TP": int(tp),
            "TN": int(tn)}


def compare_models(runs: dict, data: dict) -> tuple[pd.DataFrame, str]:
    """Bang so sanh. Moi mo hinh tu chon nguong recall >= 0,97 tren VAL cua no. Chon mo hinh theo val ROC-AUC
    (doi chung lr 1e-3 khong tham gia chon)."""
    y_va, y_te = data["val"]["y"], data["test"]["y"]
    rows = []
    for k, r in runs.items():
        thr = choose_threshold(y_va, r["s_val"])
        r["nguong"] = thr
        mv, m5, mt = metrics_at(y_va, r["s_val"], thr), metrics_at(y_te, r["s_test"], 0.5), metrics_at(y_te, r["s_test"], thr)
        rows.append({"mo_hinh": NAMES[k], "tham_so": r["tham_so"], "tham_so_train_duoc": r["tham_so_train_duoc"],
                     "so_epoch": r["so_epoch"], "thoi_gian_phut": r["thoi_gian_train_s"] / 60,
                     "auc_val": roc_auc_score(y_va, r["s_val"]), "nguong_val": thr,
                     "specificity_val": mv["specificity"], "auc_test": roc_auc_score(y_te, r["s_test"]),
                     "recall_test_0.5": m5["recall"], "accuracy_test_0.5": m5["accuracy"],
                     "recall_test": mt["recall"], "specificity_test": mt["specificity"],
                     "accuracy_test": mt["accuracy"], "bo_sot_test": mt["bo_sot_FN"],
                     "bao_dong_gia_test": mt["bao_dong_gia_FP"], "_key": k})
    table = pd.DataFrame(rows)
    cand = table[table["_key"] != "tl_finetune_lr_lon"]
    best = cand.loc[cand["auc_val"].idxmax(), "_key"]
    table.drop(columns="_key").to_csv(REPORTS_DIR / "so_sanh_mo_hinh.csv", index=False)
    return table.drop(columns="_key"), best


def evaluate_test(final: dict, data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Buoc 10: ma tran nham lan tren TEST o nguong 0,5 va o nguong chon tren val; ti le bo sot theo tac nhan."""
    y, s, thr = data["test"]["y"], final["s_test"], final["nguong"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, t, title in [(axes[0], 0.5, "Nguong 0,5"), (axes[1], thr, f"Nguong {thr:.3f} (recall val >= 0,97)")]:
        cm = confusion_matrix(y, (s >= t).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            lab = {(1, 0): "\nBO SOT", (0, 1): "\nbao dong gia"}.get((i, j), "")
            ax.text(j, i, f"{v}{lab}", ha="center", va="center", fontsize=12,
                    color="white" if v > cm.max() / 2 else ("#C44E52" if (i, j) == (1, 0) else "black"),
                    fontweight="bold" if (i, j) == (1, 0) else None)
        m = metrics_at(y, s, t)
        ax.set_xticks([0, 1], ["du doan NORMAL", "du doan PNEUMONIA"]); ax.set_yticks([0, 1], ["that NORMAL", "that PNEUMONIA"])
        ax.set_title(f"{title}\nrecall {m['recall']:.3f} | specificity {m['specificity']:.3f} | acc {m['accuracy']:.3f}",
                     fontsize=10)
    fig.suptitle(f"TEST (624 anh) - {NAMES[final['name']]}", fontsize=11)
    save_fig(fig, "confusion_matrix.png")
    df = data["df"].iloc[data["test"]["idx"]].assign(p=s, du_doan=(s >= thr).astype(int))
    by = df.groupby("tac_nhan").apply(lambda g: pd.Series({
        "so_anh": len(g), "du_doan_PNEUMONIA": int(g["du_doan"].sum()),
        "ti_le_du_doan_PNEUMONIA": g["du_doan"].mean(), "xac_suat_TB": g["p"].mean()}), include_groups=False)
    by.to_csv(REPORTS_DIR / "test_theo_tac_nhan.csv")
    out = pd.DataFrame([{"nguong": 0.5, **metrics_at(y, s, 0.5)}, {"nguong": thr, **metrics_at(y, s, thr)}])
    out.to_csv(REPORTS_DIR / "ket_qua_test.csv", index=False)
    return out, by


# ------------------------------------------------------------------ 11. Grad-CAM
def cam_stats(cams: np.ndarray, border: float = 0.12) -> dict:
    """Ti le "nang luong" CAM nam o vien anh (dai 12% moi canh = noi thuong co chu/ky hieu R, L, ten benh vien)
    va o 4 goc. Neu CAM phan bo deu: vien chiem 1-(1-2*0,12)^2 = 42% dien tich, 4 goc 5,8%."""
    H, W = cams.shape[1:]
    b = int(round(border * H))
    mask_in = np.zeros((H, W), bool); mask_in[b:H - b, b:W - b] = True
    corner = np.zeros((H, W), bool)
    for rs in (slice(0, b), slice(H - b, H)):
        for cs in (slice(0, b), slice(W - b, W)):
            corner[rs, cs] = True
    tot = cams.sum(axis=(1, 2)) + 1e-8
    return {"ti_le_o_vien": float(np.mean(cams[:, ~mask_in].sum(1) / tot)),
            "ti_le_4_goc": float(np.mean(cams[:, corner].sum(1) / tot)),
            "dien_tich_vien": float((~mask_in).mean()), "dien_tich_4_goc": float(corner.mean()),
            "so_anh_dinh_CAM_o_vien": float(np.mean(~mask_in.ravel()[cams.reshape(len(cams), -1).argmax(1)]))}


def run_gradcam(final: dict, other: dict, data: dict) -> pd.DataFrame:
    """6 ca test (2 viem phoi do vi khuan, 1 do virus, 2 binh thuong, 1 ca sai) + ban do CAM trung binh.
    So sanh mo hinh cuoi voi 1 mo hinh khac (other): CNN tu dau <-> EfficientNet fine-tune."""
    df = data["df"].iloc[data["test"]["idx"]].reset_index(drop=True)
    X = D.to_batch(data["test"]["X"])
    s, thr = final["s_test"], final["nguong"]
    pred = (s >= thr).astype(int)
    rng = np.random.default_rng(SEED)
    pick = []
    for cond, k in [((df["tac_nhan"] == "bacteria") & (pred == 1), 2), ((df["tac_nhan"] == "virus") & (pred == 1), 1),
                    ((df["y"] == 0) & (pred == 0), 2)]:
        pick += list(rng.choice(np.flatnonzero(cond), k, replace=False))
    # ca thu 6: ca bo sot (FN) co xac suat thap nhat; neu khong co FN thi bao dong gia tu tin nhat
    fn = np.flatnonzero((df["y"] == 1) & (pred == 0)); fp = np.flatnonzero((df["y"] == 0) & (pred == 1))
    pick.append(fn[np.argmin(s[fn])] if len(fn) else fp[np.argmax(s[fp])])
    cams, _ = M.grad_cam(final["model"], X[pick])
    cams_s, _ = M.grad_cam(other["model"], X[pick])
    fig, axes = plt.subplots(3, 6, figsize=(16, 8.4))
    for c, (i, cam, cam_s) in enumerate(zip(pick, cams, cams_s)):
        img = data["test"]["X"][i]
        axes[0, c].imshow(img, cmap="gray")
        axes[0, c].set_title(f"that: {df['lop'][i]} ({df['tac_nhan'][i]})\np={s[i]:.3f} -> "
                             f"{'PNEUMONIA' if pred[i] else 'NORMAL'}", fontsize=8,
                             color="#C44E52" if pred[i] != df["y"][i] else "black")
        for r, cm_, lab in [(1, cam, NAMES[final["name"]]), (2, cam_s, NAMES[other["name"]])]:
            axes[r, c].imshow(img, cmap="gray"); axes[r, c].imshow(cm_, cmap="jet", alpha=0.4, vmin=0, vmax=1)
            axes[r, c].set_title(lab, fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    save_fig(fig, "gradcam_examples.png")

    # Dinh luong tren TOAN BO test: CAM co nam o vung phoi (giua anh) hay o vien/goc (noi co chu)?
    rows, mean_cams = [], {}
    for r in (final, other):
        all_cams, _ = M.grad_cam(r["model"], X)
        st = cam_stats(all_cams)
        rows.append({"mo_hinh": NAMES[r["name"]], **st})
        mean_cams[NAMES[r["name"]]] = all_cams.mean(0)
    fig, axes = plt.subplots(1, len(mean_cams) + 1, figsize=(4.2 * (len(mean_cams) + 1), 4))
    axes[0].imshow(data["test"]["X"].mean(0), cmap="gray"); axes[0].set_title("Anh test trung binh", fontsize=9)
    for ax, (k, mc) in zip(axes[1:], mean_cams.items()):
        ax.imshow(data["test"]["X"].mean(0), cmap="gray"); ax.imshow(mc / mc.max(), cmap="jet", alpha=0.45)
        ax.set_title(f"CAM trung binh 624 anh test\n{k}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    save_fig(fig, "gradcam_trung_binh.png")
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / "gradcam_vung_chu_y.csv", index=False)
    pd.DataFrame({"file": df["file"][pick].to_numpy(), "lop": df["lop"][pick].to_numpy(),
                  "tac_nhan": df["tac_nhan"][pick].to_numpy(), "p": s[pick]}).to_csv(
        REPORTS_DIR / "gradcam_ca_chon.csv", index=False)
    return table


# ------------------------------------------------------------------ 12. Ca du doan sai
def show_errors(final: dict, data: dict, n: int = 10) -> pd.DataFrame:
    """10 ca sai tren test: TAT CA ca bo sot (FN) truoc, roi cac bao dong gia (FP) tu tin nhat. Kem Grad-CAM."""
    df = data["df"].iloc[data["test"]["idx"]].reset_index(drop=True)
    s, thr = final["s_test"], final["nguong"]
    y = df["y"].to_numpy(); pred = (s >= thr).astype(int)
    fn = np.flatnonzero((y == 1) & (pred == 0)); fn = fn[np.argsort(s[fn])]
    fp = np.flatnonzero((y == 0) & (pred == 1)); fp = fp[np.argsort(-s[fp])]
    pick = np.concatenate([fn, fp])[:n]
    cams, _ = M.grad_cam(final["model"], D.to_batch(data["test"]["X"][pick]))
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.4))
    for ax, i, cam in zip(axes.ravel(), pick, cams):
        ax.imshow(data["test"]["X"][i], cmap="gray"); ax.imshow(cam, cmap="jet", alpha=0.3, vmin=0, vmax=1)
        kind = "BO SOT (FN)" if y[i] == 1 else "bao dong gia (FP)"
        ax.set_title(f"{kind}\nthat {df['lop'][i]} ({df['tac_nhan'][i]}) | p={s[i]:.3f}\n{df['file'][i]}", fontsize=8,
                     color="#C44E52" if y[i] == 1 else "black")
        ax.axis("off")
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    save_fig(fig, "ca_du_doan_sai.png")
    out = pd.DataFrame({"file": df["file"][pick].to_numpy(), "lop_that": df["lop"][pick].to_numpy(),
                        "tac_nhan": df["tac_nhan"][pick].to_numpy(), "xac_suat_PNEUMONIA": s[pick],
                        "loai_sai": np.where(y[pick] == 1, "FN bo sot", "FP bao dong gia")})
    out.to_csv(REPORTS_DIR / "ca_du_doan_sai.csv", index=False)
    return out


def save_final_model(final: dict) -> None:
    M.export_clean(final["model"], MODELS_DIR / "best_model.keras")
    (MODELS_DIR / "nguong.json").write_text(json.dumps(
        {"mo_hinh": NAMES[final["name"]], "nguong": final["nguong"],
         "y_nghia": "xac suat PNEUMONIA >= nguong -> gan co uu tien doc phim; nguong chon de recall val >= 0,97"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[save] {final['name']} -> models/best_model.keras (nguong {final['nguong']:.4f})")


def save_summary(**kw) -> None:
    def conv(o):
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(kw, f, ensure_ascii=False, indent=2, default=conv)
    log("[save] reports/tom_tat.json")


def main() -> None:
    data = load()
    counts = run_counts(data)
    show_samples(data)
    show_augmentation(data)
    runs = {"cnn_tu_dau": run_scratch(data)}
    runs["tl_dong_bang"] = run_transfer(data)
    runs["tl_finetune"] = run_finetune(data, runs["tl_dong_bang"])
    runs["tl_finetune_lr_lon"] = run_finetune_lr_lon(data, runs["tl_dong_bang"])
    plot_learning_curves(runs)
    table, best = compare_models(runs, data)
    log(table.to_string())
    final = runs[best]
    save_final_model(final)
    test, by = evaluate_test(final, data)
    log(test.to_string()); log(by.to_string())
    other = runs["tl_finetune"] if best == "cnn_tu_dau" else runs["cnn_tu_dau"]
    cam = run_gradcam(final, other, data)
    log(cam.to_string())
    errs = show_errors(final, data)
    log(errs.to_string())
    save_summary(so_luong_anh=counts.reset_index(), chia_du_lieu=D.split_table(data["df"]).reset_index(),
                 so_sanh=table, mo_hinh_chon=best, nguong=final["nguong"], test=test,
                 test_theo_tac_nhan=by.reset_index(), gradcam=cam, ca_sai=errs)


if __name__ == "__main__":
    main()
