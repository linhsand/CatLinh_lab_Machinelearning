"""Nhanh B: fine-tune DistilBERT (Hugging Face, PyTorch) cho TT-28.

Gioi han phan cung: may chi co CPU (i9-13900H laptop). Do toc do thuc te: ~6 mau/giay khi train voi max_length 96
va batch gom cac tin dai gan bang nhau (khong gom: ~3,6 mau/giay; bf16 tren CPU nay con cham hon).
2 epoch x 108.000 mau se mat ~10 gio -> fine-tune tren TAP CON 20.000 mau (stratify, 5.000 / lop), 2 epoch.
Danh gia tren TOAN BO 7.600 tin test. Bao cao so sanh them TF-IDF train tren CUNG 20.000 mau de tach bach
"BERT tot hon" voi "BERT it du lieu hon".

* lr 3e-5 (khoang 2e-5 - 5e-5 cua de), AdamW weight_decay 0,01, warmup 6% roi giam tuyen tinh ve 0.
* max_length 96 token: tin dai nhat ~1% bi cat (trung binh 53 token, p99 ~130 tren mau 2.000 tin).
* Chon epoch theo accuracy tren 2.000 tin val (stratify). Ket qua luu vao models/runs/distilbert/ va doc lai neu co.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data as D

MODEL_NAME = "distilbert-base-uncased"
N_TRAIN = int(os.environ.get("TT28_BERT_N", 20000))
N_VAL = 2000
EPOCHS = 2
LR = 3e-5
BATCH = 32
MAX_LEN = 96
SEED = 42
OUT = D.BASE_DIR / "models" / "runs" / "distilbert"


def _batches(texts: list[str], batch: int, rng: np.random.Generator | None):
    """Chia batch, moi batch gom cac tin co do dai gan nhau (it padding hon -> nhanh hon).
    Train: xao tron, chia thanh khoi 50 batch, sap xep trong khoi theo do dai, roi xao thu tu batch."""
    idx = np.arange(len(texts))
    lens = np.array([len(t) for t in texts])
    if rng is None:
        order = np.argsort(lens)
        return [order[i:i + batch] for i in range(0, len(order), batch)]
    rng.shuffle(idx)
    chunk = batch * 50
    out = []
    for s in range(0, len(idx), chunk):
        c = idx[s:s + chunk]
        c = c[np.argsort(lens[c])]
        out += [c[i:i + batch] for i in range(0, len(c), batch)]
    rng.shuffle(out)
    return out


def predict_proba(model, tok, texts: list[str], batch: int = 64) -> np.ndarray:
    import torch
    model.eval()
    probs = np.zeros((len(texts), len(D.LABELS)), dtype=np.float32)
    with torch.no_grad():
        for b in _batches(texts, batch, None):
            enc = tok([texts[i] for i in b], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            probs[b] = torch.softmax(model(**enc).logits, -1).numpy()
    return probs


def latency_ms(model, tok, text: str, n: int = 30) -> float:
    """Thoi gian du doan 1 tin (tokenize + forward), trung vi n lan."""
    import torch
    model.eval(); ts = []
    with torch.no_grad():
        for _ in range(n + 3):
            t0 = time.perf_counter()
            model(**tok([text], truncation=True, max_length=MAX_LEN, return_tensors="pt"))
            ts.append(time.perf_counter() - t0)
    return float(np.median(ts[3:]) * 1000)


def run(data: dict, retrain: bool = False) -> dict:
    meta_p, pred_p, hist_p = OUT / "meta.json", OUT / "pred.npz", OUT / "history.csv"
    if not retrain and all(p.exists() for p in (meta_p, pred_p, hist_p)):
        meta = json.loads(meta_p.read_text(encoding="utf-8")); pred = np.load(pred_p)
        print(f"[bert] doc lai ket qua da train ({OUT})", flush=True)
        return {**meta, "p_val": pred["p_val"], "p_test": pred["p_test"], "y_val": pred["y_val"],
                "history": pd.read_csv(hist_p)}

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tr = D.subset(data["train"], N_TRAIN)
    va = D.subset(data["val"], N_VAL)
    te = data["test"]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(D.LABELS))
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(tr) / BATCH))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    texts, y = tr["text"].tolist(), torch.tensor(tr["label"].to_numpy())
    print(f"[bert] {MODEL_NAME}: {n_params:,} tham so | train {len(tr):,} mau x {EPOCHS} epoch = {steps} buoc | "
          f"lr {LR:g} | max_length {MAX_LEN}", flush=True)
    hist, best_acc, t_train = [], -1.0, 0.0
    for ep in range(1, EPOCHS + 1):
        model.train(); t0 = time.time(); losses = []
        batches = _batches(texts, BATCH, rng)
        for k, b in enumerate(batches):
            enc = tok([texts[i] for i in b], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            loss = model(**enc, labels=y[b]).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(loss.item())
            if (k + 1) % 50 == 0:
                el = time.time() - t0
                print(f"[bert] epoch {ep} buoc {k + 1}/{len(batches)} | loss TB 50 buoc {np.mean(losses[-50:]):.4f} | "
                      f"{(k + 1) * BATCH / el:.1f} mau/s | con ~{el / (k + 1) * (len(batches) - k - 1) / 60:.0f} ph",
                      flush=True)
        t_train += time.time() - t0
        p_val = predict_proba(model, tok, va["text"].tolist())
        acc = float((p_val.argmax(1) == va["label"].to_numpy()).mean())
        hist.append({"epoch": ep, "loss_train": float(np.mean(losses)), "acc_val": acc,
                     "thoi_gian_epoch_s": time.time() - t0})
        print(f"[bert] epoch {ep}: loss train {np.mean(losses):.4f} | acc val {acc:.4f} | {(time.time() - t0) / 60:.0f} ph",
              flush=True)
        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(OUT / "model"); tok.save_pretrained(OUT / "model")
    model = AutoModelForSequenceClassification.from_pretrained(OUT / "model")
    p_val = predict_proba(model, tok, va["text"].tolist())
    t0 = time.time()
    p_test = predict_proba(model, tok, te["text"].tolist())
    t_pred = time.time() - t0
    lat = latency_ms(model, tok, te["text"].iloc[0])
    pd.DataFrame(hist).to_csv(hist_p, index=False)
    np.savez(pred_p, p_val=p_val, p_test=p_test, y_val=va["label"].to_numpy())
    meta = {"mo_hinh": f"DistilBERT fine-tune ({len(tr) // 1000}k mau)", "n_train": len(tr), "epochs": EPOCHS,
            "epoch_tot_nhat": int(pd.DataFrame(hist)["acc_val"].idxmax()) + 1, "lr": LR, "max_length": MAX_LEN,
            "tham_so": int(n_params), "thoi_gian_train_s": round(t_train, 1),
            "thoi_gian_du_doan_test_s": round(t_pred, 1), "do_tre_1_tin_ms": round(lat, 1),
            "acc_val": float((p_val.argmax(1) == va["label"].to_numpy()).mean()),
            "acc_test": float((p_test.argmax(1) == te["label"].to_numpy()).mean())}
    meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[bert] xong: acc val {meta['acc_val']:.4f} | acc test {meta['acc_test']:.4f} | train {t_train / 60:.0f} ph",
          flush=True)
    return {**meta, "p_val": p_val, "p_test": p_test, "y_val": va["label"].to_numpy(), "history": pd.DataFrame(hist)}


if __name__ == "__main__":
    run(D.load(), retrain=os.environ.get("TT28_RETRAIN") == "1")
