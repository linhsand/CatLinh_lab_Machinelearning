"""TT-28 - Transformer: phan loai chu de tin tuc AG News (World / Sports / Business / Sci-Tech).

Nguyen tac danh gia:
  * Test goc (7.600 tin) chi dung de BAO CAO. Tach 10% train lam VAL (12.000 tin) cho EarlyStopping / chon tham so.
  * Moi mo hinh: accuracy, thoi gian train, so tham so, do tre du doan 1 tin (tinh ca buoc tien xu ly).

Cac buoc (theo README de bai):
  1.    Can bang lop, do dai, 5 mau moi lop, rac HTML       -> run_eda()
  2.    Baseline TF-IDF + LinearSVC (full + 20k mau)         -> run_tfidf()
  3.    Tu dien + padding                                    -> make_vectorizer()
  4.    Transformer tu xay (Embedding + PE + 2 khoi)         -> run_transformer()
  5.    Bo positional encoding                               -> run_transformer(use_pos=False)
  6.    So head 1 / 2 / 4 / 8                                -> run_heads()
  7.    DistilBERT (finetune_bert.py)                        -> finetune_bert.run()
  8.    Bang so sanh 4 cach (+ BiLSTM)                       -> compare()
  9.    Ma tran nham lan 4x4                                 -> run_confusion()
  10.   Heatmap attention 3 tin                              -> run_attention()
  11.   Human-in-the-loop nguong tin cay 95%                 -> run_hitl()

Moi mo hinh luu ket qua vao models/runs/; chay lai se DOC LAI (TT28_RETRAIN=1 de train lai).
Chay: python src/train.py   (DistilBERT ~2 gio tren CPU, phan con lai ~1 gio)
"""
from __future__ import annotations

import json
import os
import pickle
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import data as D
import transformer_block as B

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
RUNS_DIR = MODELS_DIR / "runs"
for d in (REPORTS_DIR, MODELS_DIR, RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
RETRAIN = os.environ.get("TT28_RETRAIN") == "1"
VOCAB = 20000
MAX_LEN = 80            # tu; p99 do dai tin = 70 tu
BATCH = 128
EPOCHS = 8
HEADS = (1, 2, 4, 8)
N_SUBSET = 20000        # cung so mau voi DistilBERT
THRESHOLDS = (0.80, 0.90, 0.95, 0.99)
NEWS_PER_DAY, PER_EDITOR = 5000, 400


def log(msg: str) -> None:
    print(msg, flush=True)


def save_fig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _cache(name: str):
    return RUNS_DIR / f"{name}_meta.json", RUNS_DIR / f"{name}_pred.npz"


def _load_cached(name: str):
    meta_p, pred_p = _cache(name)
    if RETRAIN or not (meta_p.exists() and pred_p.exists()):
        return None
    meta = json.loads(meta_p.read_text(encoding="utf-8")); pred = np.load(pred_p)
    log(f"[{name}] doc lai ket qua da train")
    return {**meta, "p_val": pred["p_val"], "p_test": pred["p_test"]}


def _save(name: str, meta: dict, p_val, p_test) -> dict:
    meta_p, pred_p = _cache(name)
    np.savez(pred_p, p_val=p_val, p_test=p_test)
    meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**meta, "p_val": p_val, "p_test": p_test}


# ------------------------------------------------------------------ 1. Du lieu
def load() -> dict:
    data = D.load()
    data["y"] = {k: data[k]["label"].to_numpy() for k in ["train", "val", "test"]}
    log("[load] train/val/test = " + "/".join(f"{len(data[k]):,}" for k in ["train", "val", "test"]))
    return data


def run_eda(data: dict) -> dict:
    full = pd.concat([data["train"].assign(tap="train"), data["val"].assign(tap="val"),
                      data["test"].assign(tap="test")])
    full["ten_lop"] = full["label"].map(dict(enumerate(D.LABELS)))
    can_bang = full.pivot_table(index="tap", columns="ten_lop", values="label", aggfunc="size").reindex(
        ["train", "val", "test"])[D.LABELS]
    full["so_tu"] = full["text"].str.split().str.len()
    do_dai = full.groupby("ten_lop")["so_tu"].describe(percentiles=[0.5, 0.95, 0.99])[["mean", "50%", "95%", "99%", "max"]]
    mau = data["train"].groupby("label", group_keys=False).sample(5, random_state=SEED)
    mau = mau.assign(lop=mau["label"].map(dict(enumerate(D.LABELS))))[["lop", "text"]]
    rac = D.artifact_table(full)
    nguon = full.assign(nguon=full["text"].str.extract(r"\((Reuters|AP|AFP)\)", expand=False).fillna("khac")) \
        .pivot_table(index="nguon", columns="ten_lop", values="label", aggfunc="size")[D.LABELS]
    can_bang.to_csv(REPORTS_DIR / "can_bang_lop.csv"); do_dai.to_csv(REPORTS_DIR / "do_dai_van_ban.csv")
    mau.to_csv(REPORTS_DIR / "mau_moi_lop.csv", index=False); rac.to_csv(REPORTS_DIR / "rac_html.csv")
    nguon.to_csv(REPORTS_DIR / "nguon_tin_theo_lop.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for lab in D.LABELS:
        axes[0].hist(full.loc[full["ten_lop"] == lab, "so_tu"], bins=np.arange(0, 120, 3), histtype="step", lw=1.4,
                     label=lab)
    axes[0].axvline(MAX_LEN, ls="--", c="k", lw=1, label=f"max_len = {MAX_LEN}")
    axes[0].set_xlabel("so tu / tin"); axes[0].set_title("Do dai tin (da lam sach)"); axes[0].legend(fontsize=8)
    can_bang.T.plot.bar(ax=axes[1], rot=0); axes[1].set_title("So tin moi lop"); axes[1].set_xlabel("")
    save_fig(fig, "eda.png")
    return {"can_bang": can_bang, "do_dai": do_dai, "mau": mau, "rac": rac, "nguon": nguon}


# ------------------------------------------------------------------ 2. TF-IDF + LinearSVC
def _latency(fn, text: str, n: int = 50) -> float:
    ts = []
    for _ in range(n + 3):
        t0 = time.perf_counter(); fn([text]); ts.append(time.perf_counter() - t0)
    return float(np.median(ts[3:]) * 1000)


def run_tfidf(data: dict, n: int | None = None) -> dict:
    """TF-IDF (tu + cap tu, sublinear tf) + LinearSVC. C chon tren val trong {0,1; 0,3; 1}.
    Xac suat (cho human-in-the-loop) lay tu CalibratedClassifierCV(LinearSVC, cv=3) fit them 1 lan."""
    name = "tfidf_svc" + (f"_{n // 1000}k" if n else "")
    cached = _load_cached(name)
    if cached:
        return cached
    tr = D.subset(data["train"], n) if n else data["train"]
    vec = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    scores = {}
    for C in (0.1, 0.3, 1.0):
        pipe = make_pipeline(vec(), LinearSVC(C=C)).fit(tr["text"], tr["label"])
        scores[C] = accuracy_score(data["y"]["val"], pipe.predict(data["val"]["text"]))
    C = max(scores, key=scores.get)
    t0 = time.time()
    pipe = make_pipeline(vec(), LinearSVC(C=C)).fit(tr["text"], tr["label"])
    secs = time.time() - t0
    cal = make_pipeline(vec(), CalibratedClassifierCV(LinearSVC(C=C), cv=3)).fit(tr["text"], tr["label"])
    p = {k: cal.predict_proba(data[k]["text"]) for k in ["val", "test"]}
    hard = {k: pipe.predict(data[k]["text"]) for k in ["val", "test"]}
    n_feat = len(pipe[0].vocabulary_)
    meta = {"mo_hinh": "TF-IDF + LinearSVC" + (f" ({n // 1000}k mau)" if n else ""), "n_train": len(tr), "C": C,
            "acc_val_theo_C": {str(k): v for k, v in scores.items()},
            "acc_val": accuracy_score(data["y"]["val"], hard["val"]),
            "acc_test": accuracy_score(data["y"]["test"], hard["test"]),
            "tham_so": int(n_feat * len(D.LABELS) + len(D.LABELS)), "so_dac_trung": n_feat,
            "thoi_gian_train_s": round(secs, 1), "do_tre_1_tin_ms": _latency(pipe.predict, data["test"]["text"].iloc[0])}
    log(f"[{name}] C={C} | {n_feat:,} dac trung | {secs:.0f}s | acc val {meta['acc_val']:.4f} test {meta['acc_test']:.4f}")
    if not n:
        with open(MODELS_DIR / "tfidf_svc.pkl", "wb") as f:
            pickle.dump(pipe, f)
    # p_* = xac suat da hieu chinh; acc_* = du doan cua LinearSVC goc
    return _save(name, meta, p["val"].astype(np.float32), p["test"].astype(np.float32))


# ------------------------------------------------------------------ 3. Tu dien
def make_vectorizer(data: dict) -> tf.keras.layers.TextVectorization:
    """Tu dien 20.000 tu pho bien nhat CUA TAP TRAIN, cat/pad ve 80 tu. 0 = padding, 1 = tu ngoai tu dien."""
    vocab_p = MODELS_DIR / "vocab.txt"
    vec = tf.keras.layers.TextVectorization(max_tokens=VOCAB, output_sequence_length=MAX_LEN,
                                            standardize="lower_and_strip_punctuation")
    if vocab_p.exists() and not RETRAIN:
        vec.set_vocabulary(vocab_p.read_text(encoding="utf-8").split("\n")[2:])   # bo "" va [UNK]
    else:
        vec.adapt(tf.data.Dataset.from_tensor_slices(data["train"]["text"].to_numpy()).batch(4096))
        vocab_p.write_text("\n".join(vec.get_vocabulary()), encoding="utf-8")
    data["vec"] = vec
    data["ids"] = {k: vec(data[k]["text"].to_numpy()).numpy().astype(np.int32) for k in ["train", "val", "test"]}
    return vec


def vocab_stats(data: dict) -> dict:
    ids = data["ids"]["test"]
    real = ids > 0
    return {"kich_thuoc_tu_dien": len(data["vec"].get_vocabulary()),
            "ti_le_tu_ngoai_tu_dien_test": float((ids == 1).sum() / real.sum()),
            "ti_le_tin_bi_cat_test": float((data["test"]["text"].str.split().str.len() > MAX_LEN).mean())}


# ------------------------------------------------------------------ Keras chung
def fit_keras(name: str, build, data: dict, label: str, lr: float = 1e-3) -> dict:
    cached = _load_cached(name)
    if cached:
        cached["history"] = pd.read_csv(RUNS_DIR / f"{name}_history.csv")
        cached["checkpoint"] = RUNS_DIR / f"{name}.keras"
        return cached
    tf.keras.utils.set_random_seed(SEED)
    model = build()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=2, restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5)]
    t0 = time.time()
    h = model.fit(data["ids"]["train"], data["y"]["train"], validation_data=(data["ids"]["val"], data["y"]["val"]),
                  epochs=EPOCHS, batch_size=BATCH, callbacks=cb, verbose=0)
    secs = time.time() - t0
    hist = pd.DataFrame(h.history); hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    hist.to_csv(RUNS_DIR / f"{name}_history.csv", index=False)
    model.save(RUNS_DIR / f"{name}.keras")
    p = {k: model.predict(data["ids"][k], batch_size=1024, verbose=0) for k in ["val", "test"]}
    vec = data["vec"]
    one = tf.function(lambda t: model(vec(t), training=False), input_signature=[tf.TensorSpec([None], tf.string)])
    lat = _latency(lambda t: one(tf.constant(t)), data["test"]["text"].iloc[0])
    meta = {"mo_hinh": label, "n_train": len(data["train"]), "tham_so": int(model.count_params()),
            "so_epoch": len(hist), "epoch_tot_nhat": int(hist["val_accuracy"].idxmax()) + 1,
            "thoi_gian_train_s": round(secs, 1), "s_moi_epoch": round(secs / len(hist), 1), "do_tre_1_tin_ms": lat,
            "acc_val": accuracy_score(data["y"]["val"], p["val"].argmax(1)),
            "acc_test": accuracy_score(data["y"]["test"], p["test"].argmax(1))}
    log(f"[{name}] {len(hist)} epoch, {secs / 60:.1f} ph, {meta['tham_so']:,} tham so | acc val {meta['acc_val']:.4f} "
        f"test {meta['acc_test']:.4f}")
    out = _save(name, meta, p["val"], p["test"])
    out["history"], out["checkpoint"] = hist, RUNS_DIR / f"{name}.keras"
    return out


def run_lstm(data: dict) -> dict:
    return fit_keras("bilstm", lambda: B.build_lstm(VOCAB, MAX_LEN), data, "BiLSTM (64)")


def run_transformer(data: dict, n_heads: int = 4, use_pos: bool = True) -> dict:
    name = f"transformer_h{n_heads}" + ("" if use_pos else "_khong_pe")
    label = f"Transformer tu xay ({n_heads} head" + (")" if use_pos else ", KHONG positional encoding)")
    return fit_keras(name, lambda: B.build_transformer(VOCAB, MAX_LEN, n_heads=n_heads, use_pos=use_pos), data, label)


def run_heads(data: dict, base: dict) -> tuple[pd.DataFrame, dict]:
    runs = {h: (base if h == 4 else run_transformer(data, h)) for h in HEADS}
    table = pd.DataFrame([{"so_head": h, "key_dim": 64 // h, **{k: r[k] for k in
                          ["tham_so", "so_epoch", "thoi_gian_train_s", "s_moi_epoch", "do_tre_1_tin_ms", "acc_val", "acc_test"]}}
                          for h, r in runs.items()])
    table.to_csv(REPORTS_DIR / "so_head.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(table["so_head"], table["acc_val"], "o-", label="val"); ax[0].plot(table["so_head"], table["acc_test"], "s-", label="test")
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(HEADS, [str(h) for h in HEADS]); ax[0].set_xlabel("so head")
    ax[0].set_ylabel("accuracy"); ax[0].legend(); ax[0].set_title("Accuracy theo so head (tham so khong doi)")
    ax[1].bar([str(h) for h in HEADS], table["s_moi_epoch"], color="#4C72B0"); ax[1].set_xlabel("so head")
    ax[1].set_ylabel("giay / epoch"); ax[1].set_title("Thoi gian train moi epoch")
    save_fig(fig, "so_head.png")
    return table, runs


# ------------------------------------------------------------------ 8. So sanh
def compare(data: dict, runs: dict) -> pd.DataFrame:
    rows = []
    for key, r in runs.items():
        rows.append({"cach": key, "mo_hinh": r["mo_hinh"], "n_train": r["n_train"], "acc_val": r["acc_val"],
                     "acc_test": r["acc_test"], "thoi_gian_train_s": r["thoi_gian_train_s"], "tham_so": r["tham_so"],
                     "do_tre_1_tin_ms": r["do_tre_1_tin_ms"]})
    t = pd.DataFrame(rows)
    t.to_csv(REPORTS_DIR / "so_sanh_4_cach.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    order = t.sort_values("acc_test")
    col = ["#55A868" if "TF-IDF" in m else "#C44E52" if "BERT" in m else "#4C72B0" for m in order["mo_hinh"]]
    axes[0].barh(order["mo_hinh"], order["acc_test"], color=col)
    for i, v in enumerate(order["acc_test"]):
        axes[0].text(v, i, f" {v:.2%}", va="center", fontsize=8)
    axes[0].set_xlim(0.85, 0.97); axes[0].set_title("Accuracy test (7.600 tin)")
    for ax, c, title in [(axes[1], "thoi_gian_train_s", "Thoi gian train (giay, log)"),
                         (axes[2], "do_tre_1_tin_ms", "Do tre du doan 1 tin (ms, log)")]:
        ax.barh(order["mo_hinh"], order[c], color=col); ax.set_xscale("log"); ax.set_title(title)
        ax.set_yticklabels([])
        for i, v in enumerate(order[c]):
            ax.text(v, i, f" {v:,.0f}" if v >= 10 else f" {v:.1f}", va="center", fontsize=8)
    save_fig(fig, "so_sanh_4_cach.png")
    return t


# ------------------------------------------------------------------ 9. Ma tran nham lan
def run_confusion(data: dict, runs: dict) -> pd.DataFrame:
    y = data["y"]["test"]
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.6))
    pairs = []
    for ax, (key, r) in zip(np.atleast_1d(axes), runs.items()):
        pred = r["p_test"].argmax(1) if "TF-IDF" not in r["mo_hinh"] else _tfidf_hard(r, data)
        cm = confusion_matrix(y, pred, labels=range(4))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=10)
        ax.set_xticks(range(4), D.LABELS, rotation=20); ax.set_yticks(range(4), D.LABELS)
        ax.set_xlabel("du doan"); ax.set_ylabel("that")
        ax.set_title(f"{r['mo_hinh']}\naccuracy {np.mean(pred == y):.2%}", fontsize=10)
        for i in range(4):
            for j in range(4):
                if i != j:
                    pairs.append({"mo_hinh": r["mo_hinh"], "that": D.LABELS[i], "du_doan": D.LABELS[j], "so_tin": int(cm[i, j])})
    save_fig(fig, "confusion_4x4.png")
    t = pd.DataFrame(pairs)
    t.to_csv(REPORTS_DIR / "cap_lop_nham.csv", index=False)
    return t


def _tfidf_hard(r: dict, data: dict) -> np.ndarray:
    """Du doan cua LinearSVC goc (khong phai argmax xac suat da hieu chinh) cho dung voi acc_test da bao cao."""
    with open(MODELS_DIR / "tfidf_svc.pkl", "rb") as f:
        return pickle.load(f).predict(data["test"]["text"])


def error_examples(data: dict, r: dict, n: int = 12) -> pd.DataFrame:
    p = r["p_test"]; y = data["y"]["test"]; pred = p.argmax(1)
    wrong = np.flatnonzero(pred != y)
    wrong = wrong[np.argsort(-p[wrong].max(1))][:n]
    t = pd.DataFrame({"that": [D.LABELS[i] for i in y[wrong]], "du_doan": [D.LABELS[i] for i in pred[wrong]],
                      "tin_cay": p[wrong].max(1), "text": data["test"]["text"].to_numpy()[wrong]})
    t.to_csv(REPORTS_DIR / "tin_sai_tu_tin_nhat.csv", index=False)
    return t


# ------------------------------------------------------------------ 10. Attention
def run_attention(data: dict, r: dict, idx=None) -> list[int]:
    """Heatmap attention (trung binh cac head) cua khoi 1 va khoi 2 cho 3 tin test ngan, 1 tin / lop khac nhau."""
    model = tf.keras.models.load_model(r["checkpoint"])
    vocab = np.array(data["vec"].get_vocabulary())
    ids = data["ids"]["test"]; y = data["y"]["test"]
    n_tok = (ids > 0).sum(1)
    if idx is None:
        pred = r["p_test"].argmax(1)
        idx = []
        for lab in (0, 2, 3):          # World, Business, Sci/Tech
            ok = np.flatnonzero((y == lab) & (pred == lab) & (n_tok >= 12) & (n_tok <= 16))
            if not len(ok):                      # du phong: bo dieu kien du doan dung
                ok = np.flatnonzero((y == lab) & (n_tok >= 12) & (n_tok <= 16))
            idx.append(int(ok[0]))
    maps = B.attention_maps(model, ids[idx])
    fig, axes = plt.subplots(len(idx), 2, figsize=(15, 6.2 * len(idx)))
    rows = []
    for r_i, i in enumerate(idx):
        L = int(n_tok[i]); words = vocab[ids[i, :L]]
        for b in range(2):
            A = maps[b][r_i].mean(0)[:L, :L]
            ax = axes[r_i, b]
            ax.imshow(A, cmap="viridis")
            ax.set_xticks(range(L), words, rotation=90, fontsize=8); ax.set_yticks(range(L), words, fontsize=8)
            ax.set_title(f"Tin {r_i + 1} ({D.LABELS[y[i]]}) - khoi {b + 1}, TB {maps[b].shape[1]} head\n"
                         f"hang = tu hoi (query), cot = tu duoc chu y (key)", fontsize=9)
            recv = A.mean(0)
            for w, v in sorted(zip(words, recv), key=lambda t: -t[1])[:5]:
                rows.append({"tin": r_i + 1, "lop": D.LABELS[y[i]], "khoi": b + 1, "tu": w, "chu_y_nhan_TB": float(v)})
    save_fig(fig, "attention_heatmap.png")
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "attention_tu_noi_bat.csv", index=False)
    pd.DataFrame({"tin": range(1, len(idx) + 1), "chi_so_test": idx,
                  "text": data["test"]["text"].to_numpy()[idx]}).to_csv(REPORTS_DIR / "attention_tin_chon.csv", index=False)
    return idx


# ------------------------------------------------------------------ 11. Human-in-the-loop
def run_hitl(data: dict, runs: dict) -> pd.DataFrame:
    """Tin co xac suat lon nhat >= nguong -> tu dong; con lai -> bien tap vien. 5.000 tin/ngay, 400 tin/nguoi."""
    y = data["y"]["test"]; rows = []
    for key, r in runs.items():
        p = r["p_test"]; conf = p.max(1); pred = p.argmax(1)
        for thr in THRESHOLDS:
            auto = conf >= thr
            manual = 1 - auto.mean()
            rows.append({"mo_hinh": r["mo_hinh"], "nguong": thr, "ti_le_tu_dong": auto.mean(),
                         "acc_phan_tu_dong": float((pred[auto] == y[auto]).mean()) if auto.any() else np.nan,
                         "so_tin_sai_lot_qua_moi_ngay": NEWS_PER_DAY * float(((pred != y) & auto).mean()),
                         "ti_le_can_nguoi": manual, "so_bien_tap_vien": int(np.ceil(NEWS_PER_DAY * manual / PER_EDITOR))})
    t = pd.DataFrame(rows)
    t.to_csv(REPORTS_DIR / "human_in_the_loop.csv", index=False)
    return t


def save_summary(**kw) -> None:
    def conv(o):
        if isinstance(o, pd.DataFrame):
            return o.reset_index().to_dict(orient="records")
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(kw, f, ensure_ascii=False, indent=2, default=conv)
    log("[save] reports/tom_tat.json")


def main() -> None:
    tf.config.experimental.enable_op_determinism()
    data = load()
    eda = run_eda(data)
    tfidf = run_tfidf(data); tfidf20 = run_tfidf(data, N_SUBSET)
    make_vectorizer(data); log(str(vocab_stats(data)))
    lstm = run_lstm(data)
    tr4 = run_transformer(data)
    no_pe = run_transformer(data, use_pos=False)
    heads, head_runs = run_heads(data, tr4)
    best_h = int(heads.loc[heads["acc_val"].idxmax(), "so_head"])
    tr = head_runs[best_h]
    shutil.copy(tr["checkpoint"], MODELS_DIR / "transformer_tu_xay.keras")
    import finetune_bert as F
    bert = F.run(data, retrain=RETRAIN)
    table = compare(data, {"tfidf": tfidf, "tfidf_20k": tfidf20, "lstm": lstm, "transformer_khong_pe": no_pe,
                           "transformer": tr, "distilbert": bert})
    log(table.to_string())
    cm = run_confusion(data, {"tfidf": tfidf, "transformer": tr, "distilbert": bert})
    error_examples(data, bert)
    run_attention(data, tr)
    hitl = run_hitl(data, {"tfidf": tfidf, "transformer": tr, "distilbert": bert}); log(hitl.to_string())
    save_summary(can_bang=eda["can_bang"], rac=eda["rac"], so_sanh=table, so_head=heads, head_chon=best_h,
                 hitl=hitl, nham=cm.sort_values("so_tin", ascending=False).head(12))


if __name__ == "__main__":
    main()
