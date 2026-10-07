"""TT-25 - MLP voi Keras: cham diem khach hang tiem nang mua bao hiem o to (ban cheo).

Nguyen tac danh gia:
  * Chia 70/15/15 co stratify. Tien xu ly chi fit tren TRAIN.
  * EarlyStopping / chon kien truc / chon class_weight / chon nguong: chi nhin tap VAL.
    Tap TEST chi dung de BAO CAO (va so sanh cuoi MLP vs LightGBM).
  * Metric: PR-AUC (= average precision cua sklearn) va Precision@K. KHONG dung accuracy
    (doan "khong quan tam" cho tat ca da duoc 87,7%).

Cac buoc (theo README de bai):
  1.     EDA ti le quan tam theo nhom                       -> run_eda()
  2-3.   Tien xu ly + chia train/val/test                    -> load()
  4.     Baseline cay LightGBM (mac dinh + co tinh chinh)     -> run_tree_baselines()
  5-6.   MLP co ban vs + Dropout/BatchNorm                    -> run_dropout_bn()
  7.     Duong hoc loss + PR-AUC theo epoch                   -> plot_learning_curves()
  8.     3 kien truc (64) / (128,64) / (256,128,64)           -> run_architectures()
  9.     class_weight vs khong                                -> run_class_weight()
  10.    Embedding vs one-hot cho 2 bien nhieu muc            -> run_embedding()
  Them:  Do dao dong theo seed (MLP va LightGBM)              -> run_seed_variance()
  11.    Precision@3000, chon nguong tren val                 -> run_precision_at_k()
  12.    Bang ket luan MLP vs LightGBM + duong PR             -> run_final_comparison()

Chay: python src/train.py
"""
from __future__ import annotations

import json
import os
import shutil
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import data as D
import model as M

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
RUNS_DIR = MODELS_DIR / "runs"           # checkpoint cua tung thi nghiem (khong commit)
for d in (REPORTS_DIR, MODELS_DIR, RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = int(os.environ.get("TT25_EPOCHS", 100))   # tran tren; EarlyStopping thuong dung som hon nhieu
BATCH = 2048
N_CUSTOMERS = 381_109                    # tong so khach dang mua bao hiem suc khoe
CALLS_PER_DAY = 3_000
DAILY_FRACTION = CALLS_PER_DAY / N_CUSTOMERS   # 0,79% -> top 0,79% diem cao nhat moi ngay

ARCHS = {"(64)": (64,), "(128,64)": (128, 64), "(256,128,64)": (256, 128, 64)}
DROPOUTS = {"(64)": (0.3,), "(128,64)": (0.3, 0.2), "(256,128,64)": (0.3, 0.3, 0.2)}


def log(msg: str) -> None:
    print(msg, flush=True)


def save_fig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def pr_auc(y, s) -> float:
    return float(average_precision_score(y, s))


def precision_at_k(y, s, k: int) -> float:
    top = np.argsort(-s, kind="stable")[:k]
    return float(np.mean(y[top]))


# ------------------------------------------------------------------ 1-3. Du lieu
def load() -> dict:
    """Nap, chia 70/15/15, tien xu ly 2 cach (one-hot / embedding) + ma tran cho cay."""
    df = D.load_raw()
    tr, va, te = D.split(df)
    pre_oh = D.Preprocessor("onehot").fit(tr)
    # Embedding: ma xuat hien < 10 lan o train gop vao chi so 0 "khac" (vector cua ma hiem hoc kem)
    pre_emb = D.Preprocessor("embedding", min_count=10).fit(tr)
    data = {"df": df, "frames": {"train": tr, "val": va, "test": te}, "pre_oh": pre_oh, "pre_emb": pre_emb,
            "oh": {k: pre_oh.transform(v) for k, v in [("train", tr), ("val", va), ("test", te)]},
            "emb": {k: pre_emb.transform(v) for k, v in [("train", tr), ("val", va), ("test", te)]},
            "tree": {k: D.tree_matrix(v) for k, v in [("train", tr), ("val", va), ("test", te)]}}
    data["y"] = {k: v.y for k, v in data["oh"].items()}
    sizes = pd.DataFrame({k: {"so_dong": len(v), "ti_le_quan_tam": v[D.TARGET].mean()}
                          for k, v in data["frames"].items()}).T
    sizes.to_csv(REPORTS_DIR / "chia_du_lieu.csv")
    log(f"[load] train/val/test = {len(tr):,}/{len(va):,}/{len(te):,} | "
        f"so cot one-hot = {data['oh']['train'].X_dense.shape[1]} | "
        f"so cot dense (embedding) = {data['emb']['train'].X_dense.shape[1]}")
    return data


def run_eda(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Ti le quan tam theo Previously_Insured, Vehicle_Damage, nhom tuoi, tuoi xe + kiem tra rui ro ro ri."""
    d = df.copy()
    d["Nhom_tuoi"] = pd.cut(d["Age"], [19, 25, 30, 40, 50, 60, 86], right=False,
                            labels=["20-24", "25-29", "30-39", "40-49", "50-59", "60+"])
    out = {}
    for col in ["Previously_Insured", "Vehicle_Damage", "Nhom_tuoi", "Vehicle_Age", "Gender", "Driving_License"]:
        g = d.groupby(col, observed=True)[D.TARGET].agg(so_khach="size", ti_le_quan_tam="mean")
        g["ti_trong"] = g["so_khach"] / len(d)
        out[col] = g
    cross = d.groupby(["Previously_Insured", "Vehicle_Damage"])[D.TARGET].agg(so_khach="size",
                                                                             ti_le_quan_tam="mean")
    out["Previously_Insured_x_Vehicle_Damage"] = cross
    pd.concat({k: v.reset_index().rename(columns={k: "muc"}) for k, v in out.items()
               if k != "Previously_Insured_x_Vehicle_Damage"}).to_csv(REPORTS_DIR / "eda_ti_le_quan_tam.csv")
    cross.to_csv(REPORTS_DIR / "eda_previously_x_damage.csv")

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    base = d[D.TARGET].mean()
    for ax, col in zip(axes, ["Previously_Insured", "Vehicle_Damage", "Nhom_tuoi", "Vehicle_Age"]):
        g = out[col]
        ax.bar(g.index.astype(str), g["ti_le_quan_tam"], color="#4C72B0")
        ax.axhline(base, ls="--", c="gray", lw=1, label=f"chung {base:.1%}")
        for i, (r, n) in enumerate(zip(g["ti_le_quan_tam"], g["so_khach"])):
            ax.text(i, r, f"{r:.1%}\n(n={n:,})", ha="center", va="bottom", fontsize=8)
        ax.set_title(col); ax.set_ylim(0, max(g["ti_le_quan_tam"]) * 1.35)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Ti le quan tam"); axes[0].legend(fontsize=8)
    save_fig(fig, "eda_ti_le_quan_tam.png")

    # Annual_Premium lech phai -> log1p
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(d["Annual_Premium"], bins=100, color="#DD8452"); axes[0].set_title(
        f"Annual_Premium (skew = {d['Annual_Premium'].skew():.2f})")
    axes[1].hist(np.log1p(d["Annual_Premium"]), bins=100, color="#55A868"); axes[1].set_title(
        f"log1p(Annual_Premium) (skew = {np.log1p(d['Annual_Premium']).skew():.2f})")
    save_fig(fig, "eda_annual_premium.png")
    log(f"[eda] Previously_Insured=1: {out['Previously_Insured'].loc[1, 'ti_le_quan_tam']:.3%} quan tam | "
        f"=0: {out['Previously_Insured'].loc[0, 'ti_le_quan_tam']:.1%}")
    return out


# ------------------------------------------------------------------ 4. Baseline cay
def fit_lgbm(data: dict, tuned: bool = True, seed: int = SEED) -> tuple[lgb.LGBMClassifier, dict]:
    """tuned=False: tham so mac dinh (100 cay) - 'cong suc tinh chinh' = 0.
    tuned=True : learning_rate nho + EarlyStopping theo average_precision tren VAL."""
    X, y = data["tree"], data["y"]
    if tuned:
        clf = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, cat_smooth=20,
                                 random_state=seed, verbose=-1)
        fit_kw = dict(eval_set=[(X["val"], y["val"])], eval_metric="average_precision",
                      callbacks=[lgb.early_stopping(200, verbose=False)])
    else:
        clf = lgb.LGBMClassifier(random_state=seed, verbose=-1)
        fit_kw = {}
    t0 = time.perf_counter()
    clf.fit(X["train"], y["train"], categorical_feature=D.HIGH_CARD, **fit_kw)
    secs = time.perf_counter() - t0
    s_val, s_te = clf.predict_proba(X["val"])[:, 1], clf.predict_proba(X["test"])[:, 1]
    rec = {"mo_hinh": "LightGBM (tinh chinh + early stopping)" if tuned else "LightGBM (mac dinh)",
           "so_cay": int(clf.best_iteration_ or clf.n_estimators) if tuned else clf.n_estimators,
           "thoi_gian_train_s": secs, "pr_auc_val": pr_auc(y["val"], s_val), "pr_auc_test": pr_auc(y["test"], s_te),
           "roc_auc_test": float(roc_auc_score(y["test"], s_te)), "s_val": s_val, "s_test": s_te}
    return clf, rec


def run_tree_baselines(data: dict) -> tuple[pd.DataFrame, dict]:
    recs = {}
    for tuned in (False, True):
        clf, rec = fit_lgbm(data, tuned)
        recs["lgbm_tuned" if tuned else "lgbm_default"] = rec
        log(f"[lgbm] {rec['mo_hinh']}: {rec['so_cay']} cay, {rec['thoi_gian_train_s']:.1f}s, "
            f"PR-AUC val {rec['pr_auc_val']:.4f} test {rec['pr_auc_test']:.4f}")
    imp = pd.Series(clf.booster_.feature_importance("gain"), index=clf.feature_name_).sort_values(ascending=False)
    (imp / imp.sum()).rename("ti_trong_gain").to_csv(REPORTS_DIR / "lgbm_feature_importance.csv")
    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("s_")} for r in recs.values()])
    table.to_csv(REPORTS_DIR / "baseline_lightgbm.csv", index=False)
    return table, recs


# ------------------------------------------------------------------ Huan luyen Keras
def class_weights(y) -> dict:
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y.astype(int))
    return {0: float(w[0]), 1: float(w[1])}


def inputs(prep: D.Prepared, embedding: bool):
    if embedding:
        return {"dense": prep.X_dense, "region": prep.region, "channel": prep.channel}
    return prep.X_dense


def fit_keras(name: str, data: dict, hidden=(128, 64), dropout=(0.3, 0.2), batchnorm: bool = True,
              use_class_weight: bool = True, embedding: bool = False, seed: int = SEED,
              emb_dims=(8, 12), verbose: bool = True) -> dict:
    """Xay + compile + fit 1 cau hinh. Tra ve dict: thong so, thoi gian, PR-AUC (sklearn) val/test,
    history (de ve duong hoc), diem du doan val/test va duong dan checkpoint."""
    tf.keras.utils.set_random_seed(seed)       # cung seed -> cung trong so khoi tao, cung thu tu batch
    sets = data["emb"] if embedding else data["oh"]
    if embedding:
        net = M.build_embedding_mlp(sets["train"].X_dense.shape[1], data["pre_emb"].vocab_size("Region_Code"),
                                    data["pre_emb"].vocab_size("Policy_Sales_Channel"), emb_dims[0], emb_dims[1],
                                    hidden, dropout, batchnorm)
    else:
        net = M.build_mlp(sets["train"].X_dense.shape[1], hidden, dropout, batchnorm)
    M.compile_model(net)
    ckpt = RUNS_DIR / f"{name}.keras"
    cw = class_weights(sets["train"].y) if use_class_weight else None
    t0 = time.perf_counter()
    hist = net.fit(inputs(sets["train"], embedding), sets["train"].y, batch_size=BATCH, epochs=EPOCHS,
                   validation_data=(inputs(sets["val"], embedding), sets["val"].y),
                   class_weight=cw, callbacks=M.make_callbacks(ckpt), verbose=0)
    secs = time.perf_counter() - t0
    h = pd.DataFrame(hist.history)
    h.index = np.arange(1, len(h) + 1)
    s_val = net.predict(inputs(sets["val"], embedding), batch_size=8192, verbose=0).ravel()
    s_te = net.predict(inputs(sets["test"], embedding), batch_size=8192, verbose=0).ravel()
    rec = {"ten": name, "kien_truc": str(tuple(hidden)), "dropout_bn": bool(batchnorm or any(np.ravel(dropout))),
           "class_weight": use_class_weight, "ma_hoa": "embedding" if embedding else "one-hot",
           "so_dau_vao": sets["train"].X_dense.shape[1], "so_tham_so": int(net.count_params()),
           "so_epoch": len(h), "epoch_tot_nhat": int(h["val_pr_auc"].idxmax()),
           "thoi_gian_train_s": secs, "s_moi_epoch": secs / len(h),
           "pr_auc_val": pr_auc(sets["val"].y, s_val), "pr_auc_test": pr_auc(sets["test"].y, s_te),
           "roc_auc_test": float(roc_auc_score(sets["test"].y, s_te)),
           "history": h, "s_val": s_val, "s_test": s_te, "checkpoint": ckpt, "model": net}
    if verbose:
        log(f"[mlp] {name:<28} params={rec['so_tham_so']:>7,} epochs={rec['so_epoch']:>3} "
            f"(best {rec['epoch_tot_nhat']:>3}) {secs:6.1f}s | PR-AUC val {rec['pr_auc_val']:.4f} "
            f"test {rec['pr_auc_test']:.4f}")
    return rec


def to_table(recs: list[dict]) -> pd.DataFrame:
    drop = {"history", "s_val", "s_test", "checkpoint", "model"}
    return pd.DataFrame([{k: v for k, v in r.items() if k not in drop} for r in recs])


# ------------------------------------------------------------------ 5-6. Dropout + BatchNorm
def run_dropout_bn(data: dict) -> tuple[pd.DataFrame, dict]:
    base = fit_keras("co_ban_128_64", data, (128, 64), dropout=0, batchnorm=False)
    reg = fit_keras("dropout_bn_128_64", data, (128, 64), dropout=(0.3, 0.2), batchnorm=True)
    table = to_table([base, reg])
    table.to_csv(REPORTS_DIR / "dropout_bn_comparison.csv", index=False)
    return table, {"co_ban": base, "dropout_bn": reg}


# ------------------------------------------------------------------ 7. Duong hoc
def plot_learning_curves(runs: dict) -> None:
    """Hang = cau hinh, cot = loss / PR-AUC. Danh dau epoch tot nhat (theo val_pr_auc)."""
    fig, axes = plt.subplots(len(runs), 2, figsize=(12, 3.6 * len(runs)), squeeze=False)
    for row, (label, rec) in zip(axes, runs.items()):
        h, best = rec["history"], rec["epoch_tot_nhat"]
        for ax, m, title in [(row[0], "loss", "Binary cross-entropy"), (row[1], "pr_auc", "PR-AUC")]:
            ax.plot(h.index, h[m], label="train", c="#4C72B0")
            ax.plot(h.index, h[f"val_{m}"], label="val", c="#C44E52")
            ax.axvline(best, ls="--", c="gray", lw=1, label=f"epoch tot nhat = {best}")
            ax.set_title(f"{label} — {title}"); ax.set_xlabel("epoch"); ax.legend(fontsize=8)
    save_fig(fig, "learning_curves.png")
    pd.concat({k: v["history"] for k, v in runs.items()}).to_csv(REPORTS_DIR / "learning_curves.csv")


# ------------------------------------------------------------------ 8. Kien truc
def run_architectures(data: dict, use_class_weight: bool = True) -> tuple[pd.DataFrame, dict]:
    recs = {k: fit_keras(f"kien_truc_{k.strip('()').replace(',', '_')}", data, ARCHS[k], DROPOUTS[k], True,
                         use_class_weight) for k in ARCHS}
    table = to_table(list(recs.values()))
    table.insert(0, "ten_kien_truc", list(recs))
    table.to_csv(REPORTS_DIR / "kien_truc_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    x = np.arange(len(table))
    axes[0].bar(x - 0.2, table["pr_auc_val"], 0.4, label="val"); axes[0].bar(x + 0.2, table["pr_auc_test"], 0.4,
                                                                              label="test")
    lo = table[["pr_auc_val", "pr_auc_test"]].min().min()
    axes[0].set_ylim(lo - 0.01, table[["pr_auc_val", "pr_auc_test"]].max().max() + 0.005)
    axes[0].set_title("PR-AUC"); axes[0].legend()
    axes[1].bar(x, table["so_tham_so"], color="#8172B2"); axes[1].set_title("So tham so")
    axes[2].bar(x, table["thoi_gian_train_s"], color="#937860"); axes[2].set_title("Thoi gian train (s)")
    for ax in axes:
        ax.set_xticks(x, table["ten_kien_truc"])
    for i, (p, e) in enumerate(zip(table["so_tham_so"], table["so_epoch"])):
        axes[1].text(i, p, f"{p:,}", ha="center", va="bottom", fontsize=8)
        axes[2].text(i, table["thoi_gian_train_s"][i], f"{e} epoch", ha="center", va="bottom", fontsize=8)
    save_fig(fig, "kien_truc_comparison.png")
    return table, recs


def best_by_val(recs: dict) -> str:
    return max(recs, key=lambda k: recs[k]["pr_auc_val"])


# ------------------------------------------------------------------ 9. class_weight
def run_class_weight(data: dict, arch: str, with_cw: dict) -> tuple[pd.DataFrame, dict]:
    """Tai dung run da co class_weight (tu buoc 8), chi train them ban KHONG class_weight.
    Kiem tra them do hieu chinh: trung binh xac suat du doan tren test so voi ti le thuc 12,3%."""
    no_cw = fit_keras(f"khong_class_weight_{arch.strip('()').replace(',', '_')}", data, ARCHS[arch],
                      DROPOUTS[arch], True, use_class_weight=False)
    recs = {"co class_weight": with_cw, "khong class_weight": no_cw}
    table = to_table(list(recs.values()))
    table.insert(0, "cau_hinh", list(recs))
    y_te = data["y"]["test"]
    table["xac_suat_tb_test"] = [r["s_test"].mean() for r in recs.values()]
    table["ti_le_thuc_test"] = y_te.mean()
    table["recall_nguong_0.5"] = [float(((r["s_test"] >= 0.5) & (y_te == 1)).sum() / y_te.sum())
                                  for r in recs.values()]
    table["ti_le_du_doan_1_nguong_0.5"] = [float((r["s_test"] >= 0.5).mean()) for r in recs.values()]
    table.to_csv(REPORTS_DIR / "class_weight_comparison.csv", index=False)
    log(f"[cw] cw={class_weights(data['y']['train'])}")
    return table, recs


# ------------------------------------------------------------------ 10. Embedding
def run_embedding(data: dict, arch: str, use_class_weight: bool, onehot_rec: dict) -> tuple[pd.DataFrame, dict]:
    emb = fit_keras(f"embedding_{arch.strip('()').replace(',', '_')}", data, ARCHS[arch], DROPOUTS[arch], True,
                    use_class_weight, embedding=True)
    recs = {"one-hot": onehot_rec, "embedding": emb}
    table = to_table(list(recs.values()))
    table.insert(0, "cach_ma_hoa", list(recs))
    # tham so cua rieng phan ma hoa 2 bien nhieu muc
    n_reg = data["pre_oh"].ohe.categories_[1].size
    n_ch = data["pre_oh"].ohe.categories_[2].size
    first = ARCHS[arch][0]
    vr, vc = data["pre_emb"].vocab_size("Region_Code"), data["pre_emb"].vocab_size("Policy_Sales_Channel")
    table["tham_so_cho_2_bien_nhieu_muc"] = [(n_reg + n_ch) * first, vr * 8 + vc * 12 + (8 + 12) * first]
    table.to_csv(REPORTS_DIR / "embedding_vs_onehot.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    x = np.arange(2)
    axes[0].bar(x - 0.2, table["pr_auc_val"], 0.4, label="val"); axes[0].bar(x + 0.2, table["pr_auc_test"], 0.4,
                                                                              label="test")
    lo = table[["pr_auc_val", "pr_auc_test"]].min().min()
    axes[0].set_ylim(lo - 0.01, table[["pr_auc_val", "pr_auc_test"]].max().max() + 0.005)
    axes[0].set_title("PR-AUC"); axes[0].legend()
    axes[1].bar(x - 0.2, table["so_dau_vao"], 0.4, label="so cot dau vao", color="#64B5CD")
    axes[1].bar(x + 0.2, table["tham_so_cho_2_bien_nhieu_muc"] / 100, 0.4, label="tham so 2 bien (x100)",
                color="#8172B2")
    axes[1].set_title("Kich thuoc dau vao"); axes[1].legend(fontsize=8)
    axes[2].bar(x, table["so_tham_so"], color="#937860"); axes[2].set_title("Tong so tham so")
    for i, p in enumerate(table["so_tham_so"]):
        axes[2].text(i, p, f"{p:,}", ha="center", va="bottom", fontsize=9)
    for ax in axes:
        ax.set_xticks(x, table["cach_ma_hoa"])
    save_fig(fig, "embedding_vs_onehot.png")
    return table, recs


def channel_embedding_neighbors(data: dict, emb_rec: dict, top: int = 8) -> pd.DataFrame:
    """Embedding hoc duoc gi? Voi cac kenh ban lon nhat: ti le quan tam thuc te + kenh 'gan nhat' (cosine)."""
    W = emb_rec["model"].get_layer("emb_channel").get_weights()[0]
    vocab = data["pre_emb"].vocab["Policy_Sales_Channel"]
    inv = {i: c for c, i in vocab.items()}
    tr = data["frames"]["train"]
    stats = tr.groupby("Policy_Sales_Channel")[D.TARGET].agg(["size", "mean"])
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    rows = []
    for code in stats.sort_values("size", ascending=False).index[:top]:
        i = vocab[code]
        sim = Wn @ Wn[i]
        sim[[0, i]] = -np.inf
        j = int(np.argmax(sim))
        rows.append({"kenh": code, "so_khach_train": int(stats.loc[code, "size"]),
                     "ti_le_quan_tam": stats.loc[code, "mean"], "kenh_gan_nhat": inv[j],
                     "cosine": float(sim[j]), "ti_le_quan_tam_kenh_gan_nhat": stats.loc[inv[j], "mean"]})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "embedding_kenh_gan_nhau.csv", index=False)
    return out


# ------------------------------------------------------------------ Do dao dong theo seed
def run_seed_variance(data: dict, arch: str, use_class_weight: bool, embedding: bool,
                      seeds=(1, 2, 3)) -> pd.DataFrame:
    """Chenh lech giua cac cau hinh co lon hon nhieu ngau nhien khong? Train lai MLP cuoi va LightGBM voi 3 seed."""
    rows = []
    for s in seeds:
        r = fit_keras(f"seed_{s}", data, ARCHS[arch], DROPOUTS[arch], True, use_class_weight, embedding, seed=s,
                      verbose=False)
        rows.append({"mo_hinh": "MLP cuoi", "seed": s, "pr_auc_val": r["pr_auc_val"], "pr_auc_test": r["pr_auc_test"],
                     "thoi_gian_train_s": r["thoi_gian_train_s"]})
        _, g = fit_lgbm(data, tuned=True, seed=s)
        rows.append({"mo_hinh": "LightGBM tinh chinh", "seed": s, "pr_auc_val": g["pr_auc_val"],
                     "pr_auc_test": g["pr_auc_test"], "thoi_gian_train_s": g["thoi_gian_train_s"]})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "dao_dong_theo_seed.csv", index=False)
    log(out.groupby("mo_hinh")[["pr_auc_val", "pr_auc_test"]].agg(["mean", "std"]).to_string())
    return out


# ------------------------------------------------------------------ 11. Precision@K
def run_precision_at_k(y_val, y_te, scores: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """scores = {ten: (s_val, s_test)}.

    * Ngan sach 3.000 cuoc/ngay tren 381.109 khach = goi top 0,79%. Test (57.167 khach) la mau ngau nhien
      15% -> top 0,79% cua test (~450 khach) mo phong dung danh sach goi 1 ngay.
    * Nguong chon tren VAL: diem cua khach thu ceil(0,79% x |val|). Ap nguyen nguong do sang TEST.
    * Bao cao them Precision@3000 theo nghia den tren test va Precision o nhieu muc ngan sach.
    """
    k_val = int(np.ceil(DAILY_FRACTION * len(y_val)))
    k_te = int(np.ceil(DAILY_FRACTION * len(y_te)))
    base = float(y_te.mean())
    rows, curve = [], []
    for name, (s_val, s_te) in scores.items():
        thr = float(np.sort(s_val)[::-1][k_val - 1])
        sel = s_te >= thr
        p_day = precision_at_k(y_te, s_te, k_te)
        rows.append({"mo_hinh": name, "nguong_chon_tren_val": thr, "so_khach_chon_o_test": int(sel.sum()),
                     "precision_tai_nguong_test": float(y_te[sel].mean()),
                     f"precision_top{k_te}_test (=3000/ngay)": p_day,
                     "precision@3000_test (nghia den)": precision_at_k(y_te, s_te, 3000),
                     "lift_3000/ngay": p_day / base,
                     "khach_quan_tam_moi_ngay_uoc_tinh": p_day * CALLS_PER_DAY,
                     "neu_goi_ngau_nhien": base * CALLS_PER_DAY})
        for k in [100, 250, k_te, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000]:
            curve.append({"mo_hinh": name, "k": k, "ti_le_goi": k / len(y_te), "precision": precision_at_k(y_te, s_te, k),
                          "recall": float(y_te[np.argsort(-s_te)[:k]].sum() / y_te.sum())})
    table, curve = pd.DataFrame(rows), pd.DataFrame(curve)
    table.to_csv(REPORTS_DIR / "precision_at_3000.csv", index=False)
    curve.to_csv(REPORTS_DIR / "precision_at_k.csv", index=False)
    return table, curve


# ------------------------------------------------------------------ 12. Ket luan
def run_final_comparison(y_te, scores: dict, rows: list[dict]) -> pd.DataFrame:
    """Ve duong PR + precision@K tren TEST cho cac model chinh, luu bang ket luan."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    base = y_te.mean()
    for (name, s), c in zip(scores.items(), ["#C44E52", "#4C72B0", "#55A868", "#8172B2", "#DD8452"]):
        p, r, _ = precision_recall_curve(y_te, s)
        axes[0].plot(r, p, c=c, lw=1.4, label=f"{name} (AP={pr_auc(y_te, s):.4f})")
        ks = np.unique(np.geomspace(50, len(s), 120).astype(int))
        order = np.argsort(-s)
        cum = np.cumsum(y_te[order])
        axes[1].plot(ks / len(s), cum[ks - 1] / ks, c=c, lw=1.4, label=name)
    axes[0].axhline(base, ls="--", c="gray", lw=1, label=f"ngau nhien ({base:.3f})")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision"); axes[0].set_title("Duong Precision-Recall (test)")
    axes[0].legend(fontsize=8)
    axes[1].axvline(DAILY_FRACTION, ls="--", c="k", lw=1, label="3.000 cuoc/ngay (0,79%)")
    axes[1].axhline(base, ls="--", c="gray", lw=1)
    axes[1].set_xscale("log"); axes[1].set_xlabel("Ti le khach duoc goi (top theo diem)")
    axes[1].set_ylabel("Precision"); axes[1].set_title("Precision theo ngan sach goi (test)")
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); axes[1].legend(fontsize=8)
    save_fig(fig, "pr_curve.png")
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / "ket_luan_mlp_vs_lightgbm.csv", index=False)
    return table


def conclusion_rows(trees: dict, final: dict, seeds: pd.DataFrame, p_table: pd.DataFrame, n_mlp_runs: int) -> list:
    """Bang ket luan: hieu qua (PR-AUC, Precision 3000/ngay) + chi phi (thoi gian, so lan train, tien xu ly)."""
    sd = seeds.groupby("mo_hinh")["pr_auc_test"].agg(["mean", "std"])
    p_col = [c for c in p_table.columns if c.startswith("precision_top")][0]
    p = p_table.set_index("mo_hinh")[p_col]
    return [
        {"mo_hinh": "LightGBM mac dinh", "pr_auc_val": trees["lgbm_default"]["pr_auc_val"],
         "pr_auc_test": trees["lgbm_default"]["pr_auc_test"], "pr_auc_test_tb_3_seed": np.nan,
         "pr_auc_test_std_3_seed": np.nan, "precision_3000_ngay": p["LightGBM mac dinh"],
         "thoi_gian_train_s": trees["lgbm_default"]["thoi_gian_train_s"], "so_lan_train_de_chon": 1,
         "tien_xu_ly": "khong (giu nguyen ma so, khai bao 2 cot categorical)"},
        {"mo_hinh": "LightGBM tinh chinh", "pr_auc_val": trees["lgbm_tuned"]["pr_auc_val"],
         "pr_auc_test": trees["lgbm_tuned"]["pr_auc_test"],
         "pr_auc_test_tb_3_seed": sd.loc["LightGBM tinh chinh", "mean"],
         "pr_auc_test_std_3_seed": sd.loc["LightGBM tinh chinh", "std"],
         "precision_3000_ngay": p["LightGBM tinh chinh"],
         "thoi_gian_train_s": trees["lgbm_tuned"]["thoi_gian_train_s"], "so_lan_train_de_chon": 1,
         "tien_xu_ly": "khong (+ early stopping tren val)"},
        {"mo_hinh": f"MLP cuoi {final['kien_truc']} {final['ma_hoa']}", "pr_auc_val": final["pr_auc_val"],
         "pr_auc_test": final["pr_auc_test"], "pr_auc_test_tb_3_seed": sd.loc["MLP cuoi", "mean"],
         "pr_auc_test_std_3_seed": sd.loc["MLP cuoi", "std"], "precision_3000_ngay": p["MLP cuoi"],
         "thoi_gian_train_s": final["thoi_gian_train_s"], "so_lan_train_de_chon": n_mlp_runs,
         "tien_xu_ly": "log1p + StandardScaler + one-hot/Embedding, class_weight, 3 callback"},
    ]


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


def save_final_model(rec: dict) -> None:
    shutil.copy(rec["checkpoint"], MODELS_DIR / "best.keras")
    log(f"[save] {rec['checkpoint'].name} -> models/best.keras")


def main() -> None:
    tf.config.experimental.enable_op_determinism()
    data = load()
    run_eda(data["df"])
    tree_table, trees = run_tree_baselines(data)
    dbn_table, dbn = run_dropout_bn(data)
    plot_learning_curves({"MLP co ban (128,64)": dbn["co_ban"], "MLP + Dropout + BatchNorm (128,64)": dbn["dropout_bn"]})
    arch_table, archs = run_architectures(data)
    arch = best_by_val(archs)
    cw_table, cws = run_class_weight(data, arch, archs[arch])
    use_cw = cws["co class_weight"]["pr_auc_val"] >= cws["khong class_weight"]["pr_auc_val"]
    best_oh = cws["co class_weight" if use_cw else "khong class_weight"]
    emb_table, embs = run_embedding(data, arch, use_cw, best_oh)
    channel_embedding_neighbors(data, embs["embedding"])
    use_emb = embs["embedding"]["pr_auc_val"] > best_oh["pr_auc_val"]
    final = embs["embedding"] if use_emb else best_oh
    save_final_model(final)
    seeds = run_seed_variance(data, arch, use_cw, use_emb)
    scores = {"MLP cuoi": (final["s_val"], final["s_test"]),
              "LightGBM tinh chinh": (trees["lgbm_tuned"]["s_val"], trees["lgbm_tuned"]["s_test"]),
              "LightGBM mac dinh": (trees["lgbm_default"]["s_val"], trees["lgbm_default"]["s_test"])}
    p_table, _ = run_precision_at_k(data["y"]["val"], data["y"]["test"], scores)
    concl = run_final_comparison(data["y"]["test"], {k: v[1] for k, v in scores.items()},
                                 conclusion_rows(trees, final, seeds, p_table, n_mlp_runs=7))
    log(concl.to_string())
    save_summary(kien_truc_chon=arch, dung_class_weight=bool(use_cw), dung_embedding=bool(use_emb),
                 baseline=tree_table, dropout_bn=dbn_table, kien_truc=arch_table, class_weight=cw_table,
                 embedding=emb_table, precision_3000=p_table, ket_luan=concl)


if __name__ == "__main__":
    main()
