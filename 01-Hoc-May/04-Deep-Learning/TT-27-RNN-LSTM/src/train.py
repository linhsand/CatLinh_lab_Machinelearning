"""TT-27 - RNN / LSTM / GRU: du bao luu luong giao thong theo gio (Metro Interstate Traffic Volume, UCI).

Nguyen tac danh gia:
  * Chia THEO THOI GIAN 70/15/15, khong shuffle khi chia (duoc shuffle cac CUA SO khi train).
  * Scaler fit CHI tren train. Cua so chi chua qua khu, khong bac qua lo hong (sequences.py co assert).
  * Moi mo hinh (baseline, XGBoost, RNN/LSTM/GRU, moi do dai cua so) danh gia tren CUNG MOT tap gio.
  * Chon kien truc / do dai cua so: chi nhin VAL. TEST chi de bao cao. Metric chinh: MAE (xe/gio).

Cac buoc (theo README de bai):
  1-2.  Lam sach 4 van de, reindex, xu ly lo hong          -> load()
  3.    EDA theo gio / thu / thang                         -> run_eda()
  4-5.  Dac trung + chia theo thoi gian                    -> load()
  6.    Baseline naive (hom qua) + seasonal naive (tuan truoc) -> baselines()
  7.    Baseline cay XGBoost + lag                         -> run_xgb()
  8.    SimpleRNN vs LSTM vs GRU (+ LSTM khong thong tin phu) -> run_rnn_lstm_gru()
  9.    Do dai cua so 6 / 12 / 24 / 48                     -> run_windows()
  Them: Dao dong theo seed                                 -> run_seeds()
  10.   Du bao vs thuc te 1 tuan cuoi                      -> plot_week()
  11.   Du bao 1h / 3h / 6h toi                            -> run_multistep()
  12.   Phan tich loi theo gio / thu / ngay le             -> run_error_analysis()

Moi mang no-ron luu checkpoint + history + du doan vao models/runs/; chay lai se DOC LAI (TT27_RETRAIN=1 de train lai).
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from tensorflow.keras import layers, models

import data as D
import sequences as S

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
RUNS_DIR = MODELS_DIR / "runs"
for d in (REPORTS_DIR, MODELS_DIR, RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH = 256
EPOCHS = int(os.environ.get("TT27_EPOCHS", 60))
RETRAIN = os.environ.get("TT27_RETRAIN") == "1"
HORIZONS = (1, 3, 6)
WINDOWS = (6, 12, 24, 48)
MAIN_WINDOW = 24
KINDS = {"rnn": ("SimpleRNN", layers.SimpleRNN), "lstm": ("LSTM", layers.LSTM), "gru": ("GRU", layers.GRU)}


def log(msg: str) -> None:
    print(msg, flush=True)


def save_fig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


# ------------------------------------------------------------------ 1-5. Du lieu
def load() -> dict:
    raw = D.load_raw()
    clean, rep = D.clean(raw)
    rep.to_csv(REPORTS_DIR / "van_de_du_lieu.csv")
    feat = D.add_features(clean)
    rngs = D.split_times(feat)
    scaler = S.Scaler().fit(feat, rngs["train"])
    data = {"raw": raw, "clean": clean, "feat": feat, "feat_s": scaler.transform(feat), "rngs": rngs,
            "scaler": scaler, "anchors": {}, "_win": {}}
    for h in HORIZONS:
        for k, r in rngs.items():
            data["anchors"][(k, h)] = S.valid_anchors(feat, r, h)
    rows = []
    for k, (a, b) in rngs.items():
        seg = feat.loc[a:b]
        rows.append({"tap": k, "tu": a, "den": b, "so_gio": len(seg), "gio_co_gia_tri": int(seg[D.TARGET].notna().sum()),
                     "so_mau_danh_gia_h1": len(data["anchors"][(k, 1)]),
                     "luu_luong_tb": seg[D.TARGET].mean()})
    split = pd.DataFrame(rows).set_index("tap")
    split.to_csv(REPORTS_DIR / "chia_du_lieu.csv")
    data["split"] = split
    log(f"[load] {len(raw):,} dong goc -> {len(feat):,} gio | mau h=1 train/val/test = "
        + "/".join(f"{len(data['anchors'][(k, 1)]):,}" for k in rngs))
    return data


def windows(data: dict, split: str, window: int, h: int) -> S.Windows:
    key = (split, window, h)
    if key not in data["_win"]:
        data["_win"][key] = S.make_windows(data["feat_s"], data["feat"], data["anchors"][(split, h)], window, h,
                                           data["scaler"])
    return data["_win"][key]


# ------------------------------------------------------------------ 3. EDA
def run_eda(data: dict) -> dict:
    f = data["feat"].copy()
    f["gio"], f["thu"], f["thang"] = f.index.hour, f.index.dayofweek, f.index.month
    f["loai_ngay"] = np.where(f["holiday"] == 1, "Ngay le", np.where(f["weekend"] == 1, "Cuoi tuan", "Ngay thuong"))
    by_hour = f.pivot_table(index="gio", columns="loai_ngay", values=D.TARGET, aggfunc="mean")
    by_dow = f.groupby("thu")[D.TARGET].mean()
    by_month = f.groupby("thang")[D.TARGET].mean()
    pd.concat({"theo_gio": by_hour}).to_csv(REPORTS_DIR / "eda_theo_gio.csv")
    pd.DataFrame({"luu_luong_tb": by_dow}).to_csv(REPORTS_DIR / "eda_theo_thu.csv")
    pd.DataFrame({"luu_luong_tb": by_month}).to_csv(REPORTS_DIR / "eda_theo_thang.csv")

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(2, 3, 1)
    for c, col in zip(["#4C72B0", "#DD8452", "#C44E52"], ["Ngay thuong", "Cuoi tuan", "Ngay le"]):
        ax.plot(by_hour.index, by_hour[col], "o-", ms=3, c=c, label=col)
    ax.set_title("Luu luong TB theo gio trong ngay"); ax.set_xlabel("gio"); ax.set_ylabel("xe/gio"); ax.legend()
    ax.set_xticks(range(0, 24, 3))
    ax = fig.add_subplot(2, 3, 2)
    ax.bar(["T2", "T3", "T4", "T5", "T6", "T7", "CN"], by_dow, color="#4C72B0"); ax.set_title("Theo thu")
    ax = fig.add_subplot(2, 3, 3)
    ax.bar(by_month.index, by_month, color="#4C72B0"); ax.set_title("Theo thang"); ax.set_xticks(range(1, 13))
    ax.set_ylim(by_month.min() * 0.9, by_month.max() * 1.03)
    ax = fig.add_subplot(2, 1, 2)
    daily = f[D.TARGET].resample("D").mean()
    ax.plot(daily.index, daily, lw=0.6, c="#4C72B0", label="TB ngay")
    miss = f[D.TARGET].isna().resample("D").mean()
    ax.fill_between(miss.index, 0, miss * daily.max(), color="#C44E52", alpha=0.35, step="mid",
                    label="ti le gio thieu trong ngay")
    for k, c in zip(["val", "test"], ["#55A868", "#8172B2"]):
        ax.axvline(data["rngs"][k][0], ls="--", c=c, lw=1.2, label=f"bat dau {k}")
    ax.set_title("Luu luong TB theo ngay (2012-2018) va lo hong du lieu"); ax.legend(fontsize=8, loc="lower left")
    save_fig(fig, "eda_theo_gio.png")
    return {"theo_gio": by_hour, "theo_thu": by_dow, "theo_thang": by_month}


# ------------------------------------------------------------------ 6. Baseline naive
def baselines(data: dict, h: int) -> list[dict]:
    rows = []
    for name, attr in [("Naive (cung gio hom qua)", "naive24"), ("Seasonal naive (cung gio tuan truoc)", "naive168")]:
        r = {"mo_hinh": name, "h": h}
        for k in ["val", "test"]:
            w = windows(data, k, MAIN_WINDOW, h)
            r[f"mae_{k}"] = mae(w.y_raw, getattr(w, attr)); r[f"rmse_{k}"] = rmse(w.y_raw, getattr(w, attr))
        rows.append(r)
    return rows


# ------------------------------------------------------------------ 7. XGBoost
def run_xgb(data: dict, h: int) -> dict:
    """XGBoost + lag features. Early stopping theo MAE tren val."""
    X = {k: S.tree_matrix(data["feat"], data["anchors"][(k, h)], h) for k in ["train", "val", "test"]}
    y = {k: windows(data, k, MAIN_WINDOW, h).y_raw for k in X}
    model = xgb.XGBRegressor(n_estimators=3000, learning_rate=0.03, max_depth=7, min_child_weight=5, subsample=0.8,
                             colsample_bytree=0.8, objective="reg:squarederror", eval_metric="mae",
                             early_stopping_rounds=100, random_state=SEED, n_jobs=-1)
    t0 = time.time()
    model.fit(X["train"], y["train"], eval_set=[(X["val"], y["val"])], verbose=False)
    secs = time.time() - t0
    p = {k: model.predict(X[k]) for k in ["val", "test"]}
    imp = pd.Series(model.feature_importances_, index=X["train"].columns).sort_values(ascending=False)
    log(f"[xgb h={h}] {model.best_iteration + 1} cay, {secs:.1f}s | MAE val {mae(y['val'], p['val']):.1f} "
        f"test {mae(y['test'], p['test']):.1f}")
    return {"mo_hinh": "XGBoost + lag", "h": h, "mae_val": mae(y["val"], p["val"]), "mae_test": mae(y["test"], p["test"]),
            "rmse_val": rmse(y["val"], p["val"]), "rmse_test": rmse(y["test"], p["test"]),
            "thoi_gian_train_s": secs, "so_cay": model.best_iteration + 1, "p_val": p["val"], "p_test": p["test"],
            "importance": imp, "model": model}


# ------------------------------------------------------------------ 8. Mang hoi quy
def build_rnn(kind: str, window: int, n_feat: int, n_side: int, use_side: bool = True,
              units=(64, 32), dropout: float = 0.2) -> tf.keras.Model:
    """2 lop hoi quy (64 -> 32) + Dropout nhu de bai. Lop dau return_sequences=True (tra ca chuoi cho lop sau),
    lop cuoi chi tra trang thai cuoi. use_side: noi them thong tin biet truoc ve gio dich (lich + y hom qua /
    tuan truoc) vao truoc lop Dense. Dense(1) cuoi KHONG activation (hoi quy)."""
    Layer = KINDS[kind][1]
    seq = layers.Input((window, n_feat), name="cua_so")
    x = Layer(units[0], return_sequences=True, name=f"{kind}_1")(seq)
    x = layers.Dropout(dropout)(x)
    x = Layer(units[1], name=f"{kind}_2")(x)
    x = layers.Dropout(dropout)(x)
    inputs = [seq]
    if use_side:
        side = layers.Input((n_side,), name="biet_truoc")
        x = layers.Concatenate()([x, side])
        x = layers.Dense(32, activation="relu")(x)
        inputs.append(side)
    out = layers.Dense(1, name="du_bao")(x)
    return models.Model(inputs, out, name=f"{kind}_w{window}")


def fit_seq(data: dict, kind: str = "lstm", window: int = MAIN_WINDOW, h: int = 1, use_side: bool = True,
            seed: int = SEED) -> dict:
    """Train 1 cau hinh (hoac doc lai). loss MSE tren y da chuan hoa, theo doi MAE. EarlyStopping theo val_loss."""
    name = f"{kind}_w{window}_h{h}" + ("" if use_side else "_thuan") + ("" if seed == SEED else f"_s{seed}")
    ckpt, meta_p, pred_p = RUNS_DIR / f"{name}.keras", RUNS_DIR / f"{name}_meta.json", RUNS_DIR / f"{name}_pred.npz"
    hist_p = RUNS_DIR / f"{name}_history.csv"
    label = f"{KINDS[kind][0]} w={window} h={h}" + ("" if use_side else " (chi cua so)")
    if not RETRAIN and all(p.exists() for p in (ckpt, meta_p, pred_p, hist_p)):
        meta = json.loads(meta_p.read_text(encoding="utf-8")); pred = np.load(pred_p)
        return {**meta, "p_val": pred["p_val"], "p_test": pred["p_test"], "history": pd.read_csv(hist_p),
                "checkpoint": ckpt}
    W = {k: windows(data, k, window, h) for k in ["train", "val", "test"]}
    inp = lambda w: [w.X, w.side] if use_side else w.X
    tf.keras.utils.set_random_seed(seed)
    model = build_rnn(kind, window, W["train"].X.shape[2], W["train"].side.shape[1], use_side)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)]
    t0 = time.time()
    hist = model.fit(inp(W["train"]), W["train"].y, validation_data=(inp(W["val"]), W["val"].y), epochs=EPOCHS,
                     batch_size=BATCH, shuffle=True, callbacks=cb, verbose=0)
    secs = time.time() - t0
    sc = data["scaler"]
    p = {k: sc.y_from_scaled(model.predict(inp(W[k]), batch_size=2048, verbose=0)[:, 0]) for k in ["val", "test"]}
    h_df = pd.DataFrame(hist.history); h_df.insert(0, "epoch", np.arange(1, len(h_df) + 1))
    # MAE theo epoch quy ve xe/gio (Keras tinh tren y chuan hoa)
    for c in ["mae", "val_mae"]:
        h_df[c + "_xe"] = h_df[c] * sc.sd
    h_df.to_csv(hist_p, index=False)
    model.save(ckpt)
    np.savez(pred_p, p_val=p["val"], p_test=p["test"])
    meta = {"ten": name, "mo_hinh": label, "kind": kind, "window": window, "h": h, "use_side": use_side, "seed": seed,
            "tham_so": int(model.count_params()), "so_epoch": len(h_df),
            "epoch_tot_nhat": int(h_df["val_loss"].idxmin()) + 1, "thoi_gian_train_s": round(secs, 1),
            "mae_val": mae(W["val"].y_raw, p["val"]), "mae_test": mae(W["test"].y_raw, p["test"]),
            "rmse_val": rmse(W["val"].y_raw, p["val"]), "rmse_test": rmse(W["test"].y_raw, p["test"])}
    meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[{name}] {meta['so_epoch']} epoch ({secs:.0f}s), {meta['tham_so']:,} tham so | MAE val {meta['mae_val']:.1f} "
        f"test {meta['mae_test']:.1f}")
    return {**meta, "p_val": p["val"], "p_test": p["test"], "history": h_df, "checkpoint": ckpt}


def _row(r: dict) -> dict:
    keep = ["mo_hinh", "h", "tham_so", "so_epoch", "thoi_gian_train_s", "mae_val", "mae_test", "rmse_val", "rmse_test"]
    return {k: r.get(k, np.nan) for k in keep}


def run_rnn_lstm_gru(data: dict, xgb1: dict) -> tuple[pd.DataFrame, dict]:
    """Buoc 8: 3 kieu lop hoi quy, cung cua so 24h, h=1, cung thong tin phu. + LSTM chi dung cua so (nhu de bai)."""
    runs = {k: fit_seq(data, k) for k in KINDS}
    runs["lstm_thuan"] = fit_seq(data, "lstm", use_side=False)
    table = pd.DataFrame([_row(r) for r in runs.values()] + [_row(r) for r in baselines(data, 1)] + [_row(xgb1)])
    table.to_csv(REPORTS_DIR / "rnn_lstm_gru.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    t = table.sort_values("mae_test", ascending=False)
    colors = ["#C44E52" if "naive" in m.lower() else "#55A868" if "XGB" in m else "#4C72B0" for m in t["mo_hinh"]]
    axes[0].barh(t["mo_hinh"], t["mae_test"], color=colors)
    for i, (v, vv) in enumerate(zip(t["mae_test"], t["mae_val"])):
        axes[0].text(v, i, f" test {v:.0f} | val {vv:.0f}", va="center", fontsize=8)
    axes[0].set_xlabel("MAE (xe/gio)"); axes[0].set_title("Du bao 1 gio toi: MAE test (do = baseline naive)")
    axes[0].set_xlim(0, t["mae_test"].max() * 1.35)
    for k, r in runs.items():
        hh = r["history"]
        axes[1].plot(hh["epoch"], hh["val_mae_xe"], "-", lw=1.4, label=f"{r['mo_hinh']} - val")
        axes[1].plot(hh["epoch"], hh["mae_xe"], ":", lw=1, c=axes[1].lines[-1].get_color())
    sn = [r for r in baselines(data, 1) if "Seasonal" in r["mo_hinh"]][0]
    axes[1].axhline(sn["mae_val"], ls="--", c="#C44E52", lw=1, label="seasonal naive - val")
    axes[1].axhline(xgb1["mae_val"], ls="--", c="#55A868", lw=1, label="XGBoost - val")
    axes[1].set_ylim(0, max(sn["mae_val"] * 1.3, 500)); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("MAE (xe/gio)")
    axes[1].set_title("Duong hoc (lien = val, cham = train co Dropout)"); axes[1].legend(fontsize=7)
    save_fig(fig, "rnn_lstm_gru.png")
    return table, runs


def run_windows(data: dict, kind: str, base: dict) -> pd.DataFrame:
    """Buoc 9: do dai cua so 6 / 12 / 24 / 48 gio (kieu lop tot nhat, h=1). Cung tap danh gia."""
    runs = {w: (base if w == MAIN_WINDOW else fit_seq(data, kind, w)) for w in WINDOWS}
    table = pd.DataFrame([{"cua_so_gio": w, **_row(r)} for w, r in runs.items()])
    table.to_csv(REPORTS_DIR / "do_dai_cua_so.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(table["cua_so_gio"], table["mae_val"], "o-", label="val")
    ax.plot(table["cua_so_gio"], table["mae_test"], "s-", label="test")
    for _, r in table.iterrows():
        ax.annotate(f"{r['thoi_gian_train_s']:.0f}s", (r["cua_so_gio"], r["mae_val"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="gray")
    ax.set_xscale("log", base=2); ax.set_xticks(WINDOWS, [str(w) for w in WINDOWS])
    ax.set_xlabel("do dai cua so (gio)"); ax.set_ylabel("MAE (xe/gio)")
    ax.set_title(f"{KINDS[kind][0]}, du bao 1 gio toi (so xam = thoi gian train)"); ax.legend()
    save_fig(fig, "do_dai_cua_so.png")
    return table, runs


def run_seeds(data: dict, kind: str, window: int, base: dict, seeds=(1, 2)) -> pd.DataFrame:
    """Dao dong do seed cua cau hinh tot nhat: chenh lech giua cac cau hinh co lon hon nhieu khong?"""
    runs = [base] + [fit_seq(data, kind, window, seed=s) for s in seeds]
    table = pd.DataFrame([{"seed": r["seed"], **_row(r)} for r in runs])
    table.to_csv(REPORTS_DIR / "dao_dong_theo_seed.csv", index=False)
    return table


# ------------------------------------------------------------------ 10. 1 tuan cuoi
def plot_week(data: dict, nn: dict, xg: dict) -> None:
    w = windows(data, "test", MAIN_WINDOW, 1)
    end = w.t.max(); start = end - pd.Timedelta(days=7)
    m = w.t > start
    fig, ax = plt.subplots(figsize=(16, 4.8))
    ax.plot(w.t[m], w.y_raw[m], c="k", lw=2, label="thuc te")
    ax.plot(w.t[m], nn["p_test"][m], c="#4C72B0", lw=1.3, label=f"{nn['mo_hinh']} (MAE tuan {mae(w.y_raw[m], nn['p_test'][m]):.0f})")
    ax.plot(w.t[m], xg["p_test"][m], c="#55A868", lw=1.3, ls="--", label=f"XGBoost (MAE tuan {mae(w.y_raw[m], xg['p_test'][m]):.0f})")
    ax.plot(w.t[m], w.naive168[m], c="#C44E52", lw=1, ls=":", label=f"seasonal naive (MAE tuan {mae(w.y_raw[m], w.naive168[m]):.0f})")
    ax.set_ylabel("xe/gio"); ax.set_title(f"Du bao 1 gio toi vs thuc te: tuan cuoi tap test ({start:%d/%m} - {end:%d/%m/%Y})")
    ax.legend(fontsize=8, loc="upper left"); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d/%m"))
    save_fig(fig, "du_bao_vs_thuc_te.png")


# ------------------------------------------------------------------ 11. Nhieu buoc
def run_multistep(data: dict, kind: str, window: int, nn1: dict, xgbs: dict) -> pd.DataFrame:
    rows, runs = [], {1: nn1}
    for h in HORIZONS:
        if h not in runs:
            runs[h] = fit_seq(data, kind, window, h)
        rows += [_row(r) for r in baselines(data, h)] + [_row(xgbs[h]), _row(runs[h])]
    table = pd.DataFrame(rows)
    table["mo_hinh_goc"] = table["mo_hinh"].str.replace(r" h=\d", "", regex=True)
    table.to_csv(REPORTS_DIR / "du_bao_nhieu_buoc.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, g in table.groupby("mo_hinh_goc", sort=False):
        ax.plot(g["h"], g["mae_test"], "o-", label=name)
        for _, r in g.iterrows():
            ax.annotate(f"{r['mae_test']:.0f}", (r["h"], r["mae_test"]), textcoords="offset points", xytext=(6, -3),
                        fontsize=8)
    ax.set_xticks(HORIZONS, [f"{h}h" for h in HORIZONS]); ax.set_xlabel("du bao truoc"); ax.set_ylabel("MAE test (xe/gio)")
    ax.set_title("MAE theo tam du bao"); ax.legend(fontsize=8)
    save_fig(fig, "du_bao_nhieu_buoc.png")
    return table, runs


# ------------------------------------------------------------------ 12. Phan tich loi
def run_error_analysis(data: dict, nn: dict, xg: dict) -> dict:
    w = windows(data, "test", MAIN_WINDOW, 1)
    df = pd.DataFrame({"t": w.t, "y": w.y_raw, "nn": nn["p_test"], "xgb": xg["p_test"], "sn": w.naive168})
    for c in ["nn", "xgb", "sn"]:
        df[f"err_{c}"] = (df[c] - df["y"]).abs()
    df["gio"], df["thu"] = df["t"].dt.hour, df["t"].dt.dayofweek
    df["le"] = data["feat"]["holiday"].reindex(df["t"]).to_numpy() == 1
    by_hour = df.groupby("gio")[["err_nn", "err_xgb", "err_sn", "y"]].mean()
    by_dow = df.groupby("thu")[["err_nn", "err_xgb", "err_sn"]].mean()
    by_hol = df.groupby("le")[["err_nn", "err_xgb", "err_sn"]].agg(["mean", "size"])
    daily = df.groupby(df["t"].dt.date).agg(mae_nn=("err_nn", "mean"), mae_sn=("err_sn", "mean"),
                                            luu_luong_tb=("y", "mean"), so_gio=("y", "size"))
    worst = daily[daily["so_gio"] >= 12].sort_values("mae_nn", ascending=False).head(10)
    worst["thu"] = pd.to_datetime(worst.index).day_name()
    hol_name = data["clean"]["holiday"].groupby(data["clean"].index.date).first()
    worst["ngay_le"] = hol_name.reindex(worst.index).to_numpy()
    by_hour.to_csv(REPORTS_DIR / "loi_theo_gio.csv"); by_dow.to_csv(REPORTS_DIR / "loi_theo_thu.csv")
    worst.to_csv(REPORTS_DIR / "ngay_sai_nhieu_nhat.csv")
    big = df.nlargest(15, "err_nn")[["t", "y", "nn", "err_nn", "sn"]]
    big.to_csv(REPORTS_DIR / "gio_sai_nhieu_nhat.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4), gridspec_kw={"width_ratios": [2, 1.1, 1.6]})
    for c, lab, col in [("err_nn", nn["mo_hinh"], "#4C72B0"), ("err_xgb", "XGBoost", "#55A868"),
                        ("err_sn", "seasonal naive", "#C44E52")]:
        axes[0].plot(by_hour.index, by_hour[c], "o-", ms=3, c=col, label=lab)
    ax2 = axes[0].twinx(); ax2.fill_between(by_hour.index, by_hour["y"], color="gray", alpha=0.15)
    ax2.set_ylabel("luu luong TB (nen xam)", color="gray")
    axes[0].set_title("MAE test theo gio trong ngay (h=1)"); axes[0].set_xlabel("gio"); axes[0].legend(fontsize=8)
    axes[0].set_xticks(range(0, 24, 2)); axes[0].set_zorder(ax2.get_zorder() + 1); axes[0].patch.set_visible(False)
    x = np.arange(7)
    for i, (c, col) in enumerate([("err_nn", "#4C72B0"), ("err_xgb", "#55A868"), ("err_sn", "#C44E52")]):
        axes[1].bar(x + (i - 1) * 0.27, by_dow[c], 0.27, color=col)
    axes[1].set_xticks(x, ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]); axes[1].set_title("MAE test theo thu")
    lab = [f"{d:%d/%m} {('(' + str(n) + ')') if n not in (None, 'None') else ''}" for d, n in zip(worst.index, worst["ngay_le"])]
    axes[2].barh(lab[::-1], worst["mae_nn"][::-1], color="#4C72B0", label=nn["mo_hinh"])
    axes[2].barh(lab[::-1], worst["mae_sn"][::-1], color="#C44E52", alpha=0.35, label="seasonal naive")
    axes[2].set_title("10 ngay test sai nhieu nhat"); axes[2].set_xlabel("MAE ngay (xe/gio)"); axes[2].legend(fontsize=8)
    save_fig(fig, "phan_tich_loi.png")
    return {"theo_gio": by_hour, "theo_thu": by_dow, "theo_ngay_le": by_hol, "ngay_sai": worst, "gio_sai": big}


def save_final_model(best: dict, data: dict) -> None:
    shutil.copy(best["checkpoint"], MODELS_DIR / "lstm_traffic.keras")
    sc = data["scaler"].sc
    (MODELS_DIR / "scaler.json").write_text(json.dumps(
        {"mo_hinh": best["mo_hinh"], "cot_chuan_hoa": S.SCALE_COLS, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
         "dau_vao": {"cua_so": f"({best['window']}, so dac trung)", "biet_truoc": D.CALENDAR_COLS + ["y-24 (chuan hoa)", "y-168 (chuan hoa)"]},
         "dau_ra": "y chuan hoa -> xe/gio = y * scale[0] + mean[0]"}, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[save] {best['ten']} -> models/lstm_traffic.keras")


def save_summary(**kw) -> None:
    def conv(o):
        if isinstance(o, pd.DataFrame):
            return o.reset_index().to_dict(orient="records")
        if isinstance(o, pd.Series):
            return o.to_dict()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(kw, f, ensure_ascii=False, indent=2, default=conv)
    log("[save] reports/tom_tat.json")


def best_kind(runs: dict) -> str:
    return min(KINDS, key=lambda k: runs[k]["mae_val"])


def main() -> None:
    tf.config.experimental.enable_op_determinism()
    data = load()
    run_eda(data)
    xgbs = {h: run_xgb(data, h) for h in HORIZONS}
    table, runs = run_rnn_lstm_gru(data, xgbs[1]); log(table.to_string())
    kind = best_kind(runs)
    win_table, wins = run_windows(data, kind, runs[kind]); log(win_table.to_string())
    window = int(win_table.loc[win_table["mae_val"].idxmin(), "cua_so_gio"])
    best = wins[window]
    seeds = run_seeds(data, kind, window, best); log(seeds.to_string())
    plot_week(data, best, xgbs[1])
    multi, _ = run_multistep(data, kind, window, best, xgbs); log(multi.to_string())
    err = run_error_analysis(data, best, xgbs[1]); log(err["ngay_sai"].to_string())
    save_final_model(best, data)
    save_summary(van_de=pd.read_csv(REPORTS_DIR / "van_de_du_lieu.csv"), chia=data["split"], rnn_lstm_gru=table,
                 kieu_chon=kind, cua_so_chon=window, cua_so=win_table, seed=seeds, nhieu_buoc=multi,
                 xgb_importance=xgbs[1]["importance"].head(15))


if __name__ == "__main__":
    main()
