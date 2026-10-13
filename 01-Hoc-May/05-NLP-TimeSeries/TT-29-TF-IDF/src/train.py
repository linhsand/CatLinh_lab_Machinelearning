"""TT-29 - TF-IDF: phan loai van ban 20 Newsgroups (luyen cho bai chuyen ticket CSKH ve dung bo phan).

Nguyen tac danh gia:
  * Train/test cua sklearn chia SAN theo thoi gian -> giu nguyen. Tach 20% train lam VAL de chon tham so.
  * Vectorizer luon nam TRONG pipeline -> chi fit tren du lieu train cua lan fit do (khong ro ri tu vung / IDF).
  * Tu buoc 2: chi dung ban da bo headers + footers + quotes.

Cac buoc (theo README de bai):
  1.   Ro ri: co / khong bo headers, footers, quotes (+ giu Subject)   -> run_leakage()
  2.   EDA: so van ban / lop, do dai, tu pho bien                       -> run_eda()
  3.   Baseline Dummy + Naive Bayes                                     -> run_baselines()
  4-5. TF-IDF + LinearSVC, khao sat ngram / min_df / sublinear_tf      -> run_sweep()
  6.   NB vs LogReg vs LinearSVC tren cung TF-IDF                       -> run_classifiers()
  7.   Top 15 tu moi lop                                                -> run_top_words()
  8.   Ma tran nham lan                                                 -> run_confusion()
  9.   Nguong tin cay 0,6                                               -> run_threshold()
  9b.  Chon ban trien khai: can xac suat + < 5 ms / ticket              -> run_deploy_options()
  10.  Thoi gian train + du doan 1 van ban                              -> run_timing()
  11.  10 ca sai                                                        -> run_errors()
  Mo rong: LSA (TruncatedSVD)                                           -> run_lsa()

Chay: python src/train.py   (~5 phut tren CPU)
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
from sklearn.base import clone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import preprocess as P

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
for d in (REPORTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
THRESHOLD = 0.6
# tham so mac dinh cua de (README muc 4)
DEFAULT_TFIDF = dict(lowercase=True, ngram_range=(1, 2), min_df=3, max_df=0.8, sublinear_tf=True, max_features=50000)


def log(msg: str) -> None:
    print(msg, flush=True)


def save_fig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def tfidf(**kw) -> TfidfVectorizer:
    return TfidfVectorizer(**{**DEFAULT_TFIDF, **kw})


def evaluate(pipe, Xtr, ytr, Xev, yev) -> dict:
    t0 = time.time(); pipe.fit(Xtr, ytr); secs = time.time() - t0
    pred = pipe.predict(Xev)
    return {"accuracy": accuracy_score(yev, pred), "macro_f1": f1_score(yev, pred, average="macro"),
            "thoi_gian_fit_s": secs, "pred": pred}


# ------------------------------------------------------------------ 1. Ro ri
LEAK_VARIANTS = [
    ("Giu nguyen (khong bo gi)", (), False),
    ("Bo headers", ("headers",), False),
    ("Bo headers + footers", ("headers", "footers"), False),
    ("Bo headers + footers + quotes", P.REMOVE_ALL, False),
    ("Bo ca 3 nhung GIU Subject", P.REMOVE_ALL, True),
]


def run_leakage() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cung pipeline (TF-IDF mac dinh + LinearSVC), chi khac phan van ban duoc giu lai. Danh gia tren TEST."""
    rows, top = [], {}
    for name, remove, subj in LEAK_VARIANTS:
        d = P.load(remove=remove, keep_subject=subj, anonymize=False)
        pipe = make_pipeline(tfidf(), LinearSVC(C=1.0))
        r = evaluate(pipe, d["train"]["text"], d["train"]["label"], d["test"]["text"], d["test"]["label"])
        rows.append({"bien_the": name, "accuracy_test": r["accuracy"], "macro_f1_test": r["macro_f1"],
                     "so_tu_TB": d["train"]["text"].str.split().str.len().mean(),
                     "van_ban_rong_train": int((d["train"]["text"].str.len() == 0).sum())})
        if not remove:
            vec, clf = pipe[0], pipe[1]
            feats = np.array(vec.get_feature_names_out())
            top = {d["names"][c]: list(feats[np.argsort(-clf.coef_[c])[:10]]) for c in range(len(d["names"]))}
        log(f"[leak] {name}: acc test {r['accuracy']:.4f}")
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / "leakage_comparison.csv", index=False)
    top_df = pd.DataFrame(top).T
    top_df.to_csv(REPORTS_DIR / "leakage_top_tu_ban_goc.csv")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    c = ["#C44E52", "#DD8452", "#DD8452", "#4C72B0", "#55A868"]
    ax.barh(table["bien_the"][::-1], table["accuracy_test"][::-1], color=c[::-1])
    for i, v in enumerate(table["accuracy_test"][::-1]):
        ax.text(v, i, f" {v:.1%}", va="center")
    ax.set_xlim(0, 1); ax.set_xlabel("accuracy test (20 lop)")
    ax.set_title("Cung TF-IDF + LinearSVC, chi khac phan van ban giu lai")
    save_fig(fig, "leakage_comparison.png")
    return table, top_df


# ------------------------------------------------------------------ Du lieu sach
def load() -> dict:
    d = P.load()                                  # bo ca 3 + an danh
    d["tr"], d["va"] = P.split_val(d["train"])
    log(f"[load] train {len(d['tr']):,} | val {len(d['va']):,} | test {len(d['test']):,} | train day du {len(d['train']):,}")
    return d


def run_eda(d: dict) -> dict:
    full = pd.concat([d["train"].assign(tap="train"), d["test"].assign(tap="test")])
    full["nhom"] = [d["names"][i] for i in full["label"]]
    full["so_tu"] = full["text"].str.split().str.len()
    counts = full.pivot_table(index="nhom", columns="tap", values="label", aggfunc="size")[["train", "test"]]
    length = full.groupby("nhom")["so_tu"].agg(["mean", "median", lambda s: int((s == 0).sum())])
    length.columns = ["so_tu_TB", "so_tu_trung_vi", "so_van_ban_rong"]
    cv = CountVectorizer(stop_words="english", min_df=3, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = cv.fit_transform(d["train"]["text"])
    freq = pd.Series(np.asarray(X.sum(0)).ravel(), index=cv.get_feature_names_out()).sort_values(ascending=False)
    counts.join(length).to_csv(REPORTS_DIR / "eda_theo_lop.csv")
    freq.head(40).to_csv(REPORTS_DIR / "eda_tu_pho_bien.csv", header=["so_lan"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    names = [P.short_name(n) for n in counts.index]
    axes[0].barh(names, counts["train"], color="#4C72B0", label="train")
    axes[0].barh(names, counts["test"], left=counts["train"], color="#DD8452", label="test")
    axes[0].set_title("So van ban moi nhom"); axes[0].legend(); axes[0].invert_yaxis()
    axes[1].barh(names, length["so_tu_trung_vi"], color="#4C72B0"); axes[1].invert_yaxis()
    axes[1].set_title("So tu trung vi / van ban (sau khi bo header, footer, quote)")
    top = freq.head(25)[::-1]
    axes[2].barh(top.index, top.values, color="#55A868"); axes[2].set_title("25 tu pho bien nhat (bo stopword)")
    save_fig(fig, "eda.png")
    return {"counts": counts, "length": length, "freq": freq}


# ------------------------------------------------------------------ 3. Baseline
def run_baselines(d: dict) -> pd.DataFrame:
    rows = []
    for name, pipe in [("Dummy (doan lop dong nhat)", make_pipeline(tfidf(), DummyClassifier(strategy="most_frequent"))),
                       ("MultinomialNB (alpha=0,01)", make_pipeline(tfidf(), MultinomialNB(alpha=0.01))),
                       ("ComplementNB (alpha=0,3)", make_pipeline(tfidf(), ComplementNB(alpha=0.3)))]:
        rv = evaluate(pipe, d["tr"]["text"], d["tr"]["label"], d["va"]["text"], d["va"]["label"])
        rt = evaluate(pipe, d["train"]["text"], d["train"]["label"], d["test"]["text"], d["test"]["label"])
        rows.append({"mo_hinh": name, "acc_val": rv["accuracy"], "acc_test": rt["accuracy"],
                     "macro_f1_test": rt["macro_f1"], "thoi_gian_fit_s": rt["thoi_gian_fit_s"]})
    t = pd.DataFrame(rows)
    t.to_csv(REPORTS_DIR / "baseline.csv", index=False)
    return t


# ------------------------------------------------------------------ 4-5. Khao sat vectorizer
def run_sweep(d: dict) -> tuple[pd.DataFrame, dict]:
    """ngram (1,1)/(1,2) x min_df 1/3/5 x sublinear_tf T/F. KHONG gioi han max_features de thay so dac trung that."""
    rows = []
    for ng in [(1, 1), (1, 2)]:
        for mdf in [1, 3, 5]:
            for sub in [True, False]:
                pipe = make_pipeline(tfidf(ngram_range=ng, min_df=mdf, sublinear_tf=sub, max_features=None),
                                     LinearSVC(C=1.0))
                r = evaluate(pipe, d["tr"]["text"], d["tr"]["label"], d["va"]["text"], d["va"]["label"])
                rows.append({"ngram_range": str(ng), "min_df": mdf, "sublinear_tf": sub,
                             "so_dac_trung": len(pipe[0].vocabulary_), "acc_val": r["accuracy"],
                             "macro_f1_val": r["macro_f1"], "thoi_gian_fit_s": r["thoi_gian_fit_s"]})
    t = pd.DataFrame(rows)
    # max_features cua de (50.000) tren cau hinh tot nhat
    b = t.loc[t["acc_val"].idxmax()]
    best = {"ngram_range": eval(b["ngram_range"]), "min_df": int(b["min_df"]), "sublinear_tf": bool(b["sublinear_tf"])}
    for mf in [20000, 50000, None]:
        pipe = make_pipeline(tfidf(**best, max_features=mf), LinearSVC(C=1.0))
        r = evaluate(pipe, d["tr"]["text"], d["tr"]["label"], d["va"]["text"], d["va"]["label"])
        rows.append({**{k: str(v) if k == "ngram_range" else v for k, v in best.items()}, "max_features": mf or "khong gioi han",
                     "so_dac_trung": len(pipe[0].vocabulary_), "acc_val": r["accuracy"], "macro_f1_val": r["macro_f1"],
                     "thoi_gian_fit_s": r["thoi_gian_fit_s"]})
    t = pd.DataFrame(rows)
    t.to_csv(REPORTS_DIR / "khao_sat_vectorizer.csv", index=False)
    mf_rows = t[t["max_features"].notna()] if "max_features" in t else t.iloc[0:0]
    mf_best = mf_rows.loc[mf_rows["acc_val"].idxmax(), "max_features"]
    best["max_features"] = None if mf_best == "khong gioi han" else int(mf_best)

    g = t[t.get("max_features").isna()] if "max_features" in t else t
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for (ng, sub), gg in g.groupby(["ngram_range", "sublinear_tf"]):
        lab = f"ngram {ng}, sublinear={sub}"
        axes[0].plot(gg["min_df"], gg["acc_val"], "o-", label=lab)
        axes[1].plot(gg["min_df"], gg["so_dac_trung"], "o-", label=lab)
    axes[0].set_xlabel("min_df"); axes[0].set_ylabel("accuracy val"); axes[0].legend(fontsize=8)
    axes[0].set_title("Accuracy val theo tham so vectorizer"); axes[0].set_xticks([1, 3, 5])
    axes[1].set_xlabel("min_df"); axes[1].set_yscale("log"); axes[1].set_ylabel("so dac trung (log)")
    axes[1].set_title("So dac trung sinh ra"); axes[1].set_xticks([1, 3, 5])
    save_fig(fig, "khao_sat_vectorizer.png")
    return t, best


# ------------------------------------------------------------------ 6. 3 bo phan loai
def run_classifiers(d: dict, best: dict) -> tuple[pd.DataFrame, dict]:
    """Moi bo phan loai chon 1 sieu tham so tren val, roi fit lai tren toan bo train -> test."""
    grids = {"ComplementNB": [(a, lambda a=a: ComplementNB(alpha=a)) for a in (0.1, 0.3, 1.0)],
             "LogisticRegression": [(C, lambda C=C: LogisticRegression(C=C, max_iter=3000)) for C in (3, 10, 30)],
             "LinearSVC": [(C, lambda C=C: LinearSVC(C=C)) for C in (0.1, 0.3, 1.0)]}
    rows, chosen = [], {}
    for name, grid in grids.items():
        scores = {}
        for p, make in grid:
            r = evaluate(make_pipeline(tfidf(**best), make()), d["tr"]["text"], d["tr"]["label"], d["va"]["text"],
                         d["va"]["label"])
            scores[p] = r["accuracy"]
        p_best = max(scores, key=scores.get)
        make = dict(grid)[p_best]
        rt = evaluate(make_pipeline(tfidf(**best), make()), d["train"]["text"], d["train"]["label"], d["test"]["text"],
                      d["test"]["label"])
        rows.append({"mo_hinh": name, "tham_so_chon": p_best, "acc_val_theo_tham_so": json.dumps(
            {str(k): round(v, 4) for k, v in scores.items()}), "acc_val": scores[p_best], "acc_test": rt["accuracy"],
            "macro_f1_test": rt["macro_f1"], "thoi_gian_fit_s": rt["thoi_gian_fit_s"]})
        chosen[name] = p_best
        log(f"[clf] {name} ({p_best}): val {scores[p_best]:.4f} test {rt['accuracy']:.4f} ({rt['thoi_gian_fit_s']:.1f}s)")
    t = pd.DataFrame(rows)
    t.to_csv(REPORTS_DIR / "so_sanh_bo_phan_loai.csv", index=False)
    C = chosen["LinearSVC"]
    svc = make_pipeline(tfidf(**best), LinearSVC(C=C)).fit(d["train"]["text"], d["train"]["label"])
    return t, {"svc": svc, "C": C, "alpha": chosen["ComplementNB"]}


def _latency_ms(pipe, texts: list[str]) -> float:
    ts = []
    for t in texts:
        t0 = time.perf_counter(); pipe.predict_proba([t]); ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000)


def run_deploy_options(d: dict, best: dict, C: float, alpha: float) -> tuple[pd.DataFrame, object]:
    """Ban trien khai can XAC SUAT (co che nguong) va du doan 1 ticket < 5 ms. So 5 cach lay xac suat.
    ensemble=False: 1 mo hinh fit tren toan bo train + bo hieu chinh hoc tu du doan cross-val (thay vi trung binh
    3 mo hinh) -> du doan nhanh gap ~3 lan."""
    opts = {"ComplementNB, xac suat goc": ComplementNB(alpha=alpha),
            "ComplementNB + Calibrated (cv=3)": CalibratedClassifierCV(ComplementNB(alpha=alpha), cv=3),
            "ComplementNB + Calibrated (cv=3, ensemble=False)": CalibratedClassifierCV(ComplementNB(alpha=alpha), cv=3,
                                                                                        ensemble=False),
            "LinearSVC + Calibrated (cv=3)": CalibratedClassifierCV(LinearSVC(C=C), cv=3),
            "LinearSVC + Calibrated (cv=3, ensemble=False)": CalibratedClassifierCV(LinearSVC(C=C), cv=3, ensemble=False)}
    y = d["test"]["label"].to_numpy()
    sample = d["test"]["text"].sample(300, random_state=SEED).tolist()
    rows, pipes = [], {}
    for name, clf in opts.items():
        pipe = make_pipeline(tfidf(**best), clf)
        t0 = time.time(); pipe.fit(d["train"]["text"], d["train"]["label"]); fit_s = time.time() - t0
        pv = make_pipeline(tfidf(**best), clone(clf)).fit(d["tr"]["text"], d["tr"]["label"])
        acc_val = accuracy_score(d["va"]["label"], pv.predict(d["va"]["text"]))
        p = pipe.predict_proba(d["test"]["text"]); conf, pred = p.max(1), p.argmax(1); auto = conf >= THRESHOLD
        lat = _latency_ms(pipe, sample)
        rows.append({"cach": name, "acc_val": acc_val, "acc_test": float(np.mean(pred == y)),
                     "tin_cay_TB": float(conf.mean()), f"ti_le_tu_dong_{THRESHOLD}": float(auto.mean()),
                     f"acc_phan_tu_dong_{THRESHOLD}": float(np.mean(pred[auto] == y[auto])) if auto.any() else np.nan,
                     "du_doan_1_ticket_ms": lat, "dat_duoi_5ms": lat < 5, "thoi_gian_fit_s": fit_s})
        pipes[name] = pipe
        log(f"[deploy] {name}: acc test {rows[-1]['acc_test']:.4f} | {lat:.1f} ms | tu dong {auto.mean():.1%}")
    t = pd.DataFrame(rows)
    t.to_csv(REPORTS_DIR / "lua_chon_trien_khai.csv", index=False)
    ok = t[t["dat_duoi_5ms"]]
    choice = ok.loc[ok["acc_val"].idxmax(), "cach"]          # trong cac cach dat < 5 ms, chon theo VAL
    final = pipes[choice]
    joblib.dump(final, MODELS_DIR / "tfidf_pipeline.joblib")
    log(f"[deploy] chon: {choice}")
    return t, final


# ------------------------------------------------------------------ 7. Top tu
def run_top_words(d: dict, svc, k: int = 15) -> pd.DataFrame:
    vec, clf = svc[0], svc[1]
    feats = np.array(vec.get_feature_names_out())
    top = {d["names"][c]: list(feats[np.argsort(-clf.coef_[c])[:k]]) for c in range(len(d["names"]))}
    t = pd.DataFrame(top).T
    t.columns = [f"top_{i + 1}" for i in range(k)]
    t.to_csv(REPORTS_DIR / "top_tu_moi_lop.csv")
    fig, axes = plt.subplots(4, 5, figsize=(22, 17))
    for c, ax in enumerate(axes.ravel()):
        idx = np.argsort(-clf.coef_[c])[:k][::-1]
        ax.barh(feats[idx], clf.coef_[c][idx], color="#4C72B0")
        ax.set_title(d["names"][c], fontsize=11); ax.tick_params(axis="y", labelsize=9)
    save_fig(fig, "top_tu_moi_lop.png")
    return t


# ------------------------------------------------------------------ 8. Ma tran nham lan
def run_confusion(d: dict, final) -> pd.DataFrame:
    y = d["test"]["label"].to_numpy(); pred = final.predict(d["test"]["text"])
    cm = confusion_matrix(y, pred)
    cmn = cm / cm.sum(1, keepdims=True)
    names = [P.short_name(n) for n in d["names"]]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cmn):
        if v >= 0.05:
            ax.text(j, i, f"{v:.0%}"[:-1], ha="center", va="center", fontsize=7, color="white" if v > 0.5 else "black")
    ax.set_xticks(range(20), names, rotation=70, fontsize=8); ax.set_yticks(range(20), names, fontsize=8)
    ax.set_xlabel("du doan"); ax.set_ylabel("that"); fig.colorbar(im, fraction=0.04)
    ax.set_title(f"Ma tran nham lan TEST (ti le theo hang, %) - accuracy {np.mean(pred == y):.1%}")
    save_fig(fig, "confusion_matrix.png")
    pairs = [{"that": d["names"][i], "du_doan": d["names"][j], "so_van_ban": int(cm[i, j]),
              "ti_le_cua_lop_that": cmn[i, j]} for i in range(20) for j in range(20) if i != j]
    t = pd.DataFrame(pairs).sort_values("so_van_ban", ascending=False)
    t.to_csv(REPORTS_DIR / "cap_nhom_hay_nham.csv", index=False)
    per = pd.DataFrame({"nhom": d["names"], "recall": np.diag(cmn), "so_van_ban_test": cm.sum(1)})
    per.sort_values("recall").to_csv(REPORTS_DIR / "recall_theo_nhom.csv", index=False)
    return t, per


# ------------------------------------------------------------------ 9. Nguong tin cay
def run_threshold(d: dict, final) -> pd.DataFrame:
    y = d["test"]["label"].to_numpy(); p = final.predict_proba(d["test"]["text"])
    conf, pred = p.max(1), p.argmax(1)
    rows = []
    for thr in np.round(np.arange(0.1, 0.96, 0.05), 2):
        auto = conf >= thr
        rows.append({"nguong": thr, "ti_le_tu_dong": auto.mean(),
                     "acc_phan_tu_dong": (pred[auto] == y[auto]).mean() if auto.any() else np.nan,
                     "ti_le_can_nguoi": 1 - auto.mean(), "acc_phan_can_nguoi_neu_de_model": (pred[~auto] == y[~auto]).mean()
                     if (~auto).any() else np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(REPORTS_DIR / "nguong_tin_cay.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(t["nguong"], t["ti_le_tu_dong"], "o-", label="% ticket tu dong")
    ax.plot(t["nguong"], t["acc_phan_tu_dong"], "s-", label="accuracy phan tu dong")
    ax.axvline(THRESHOLD, ls="--", c="k", lw=1, label=f"nguong {THRESHOLD}")
    ax.axhline(np.mean(pred == y), ls=":", c="gray", label=f"accuracy neu tu dong 100% ({np.mean(pred == y):.1%})")
    ax.set_xlabel("nguong max(proba)"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Co che tin cay: duoi nguong -> chuyen nguoi xu ly")
    save_fig(fig, "nguong_tin_cay.png")
    return t


# ------------------------------------------------------------------ 10. Thoi gian
def run_timing(d: dict, final, n: int = 300) -> dict:
    texts = d["test"]["text"].sample(n, random_state=SEED).tolist()
    ts = []
    for t in texts:
        t0 = time.perf_counter(); final.predict_proba([t]); ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000
    t0 = time.time(); final.predict_proba(d["test"]["text"]); batch = time.time() - t0
    t0 = time.time(); joblib.load(MODELS_DIR / "tfidf_pipeline.joblib"); load_s = time.time() - t0
    out = {"du_doan_1_van_ban_ms_trung_vi": float(np.median(ts)), "du_doan_1_van_ban_ms_p95": float(np.percentile(ts, 95)),
           "du_doan_1_van_ban_ms_max": float(ts.max()), "du_doan_ca_test_s": batch,
           "van_ban_moi_giay_batch": len(d["test"]) / batch, "nap_model_s": load_s,
           "kich_thuoc_model_MB": (MODELS_DIR / "tfidf_pipeline.joblib").stat().st_size / 1e6}
    pd.Series(out).to_csv(REPORTS_DIR / "thoi_gian.csv", header=["gia_tri"])
    return out


# ------------------------------------------------------------------ 11. Ca sai
def run_errors(d: dict, final, n: int = 10) -> pd.DataFrame:
    y = d["test"]["label"].to_numpy(); p = final.predict_proba(d["test"]["text"])
    pred, conf = p.argmax(1), p.max(1)
    wrong = np.flatnonzero(pred != y)
    rng = np.random.default_rng(SEED)
    # 5 ca sai TU TIN nhat + 5 ca sai ngau nhien (dai dien cho ca sai thong thuong)
    sure = wrong[np.argsort(-conf[wrong])[:5]]
    rand = rng.choice(np.setdiff1d(wrong, sure), 5, replace=False)
    pick = np.concatenate([sure, rand])
    texts = d["test"]["text"].to_numpy()
    t = pd.DataFrame({"loai": ["sai tu tin"] * 5 + ["sai ngau nhien"] * 5, "that": [d["names"][i] for i in y[pick]],
                      "du_doan": [d["names"][i] for i in pred[pick]], "tin_cay": conf[pick],
                      "so_tu": [len(texts[i].split()) for i in pick], "text_200_ky_tu": [texts[i][:200] for i in pick]})
    t.to_csv(REPORTS_DIR / "ca_sai.csv", index=False)
    lens = np.array([len(s.split()) for s in texts])
    stat = pd.DataFrame({"nhom_do_dai": pd.cut(lens, [-1, 0, 10, 50, 200, 10**6], labels=["rong", "1-10", "11-50", "51-200", ">200"]),
                         "dung": pred == y}).groupby("nhom_do_dai", observed=True)["dung"].agg(["size", "mean"])
    stat.columns = ["so_van_ban", "accuracy"]
    stat.to_csv(REPORTS_DIR / "accuracy_theo_do_dai.csv")
    return t, stat


# ------------------------------------------------------------------ Mo rong: LSA
def run_lsa(d: dict, best: dict, C: float) -> pd.DataFrame:
    rows = []
    for k in [100, 300, 1000]:
        pipe = make_pipeline(tfidf(**best), TruncatedSVD(k, random_state=SEED), LinearSVC(C=C))
        r = evaluate(pipe, d["train"]["text"], d["train"]["label"], d["test"]["text"], d["test"]["label"])
        rows.append({"so_chieu_LSA": k, "acc_test": r["accuracy"], "thoi_gian_fit_s": r["thoi_gian_fit_s"],
                     "phuong_sai_giu_lai": float(pipe[1].explained_variance_ratio_.sum())})
        log(f"[lsa] {k} chieu: acc test {r['accuracy']:.4f} ({r['thoi_gian_fit_s']:.0f}s)")
    t = pd.DataFrame(rows)
    t.to_csv(REPORTS_DIR / "lsa.csv", index=False)
    return t


def save_summary(**kw) -> None:
    def conv(o):
        if isinstance(o, pd.DataFrame):
            return o.reset_index().to_dict(orient="records")
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        return str(o)
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(kw, f, ensure_ascii=False, indent=2, default=conv)
    log("[save] reports/tom_tat.json")


def main() -> None:
    leak, leak_top = run_leakage(); log(leak.to_string())
    d = load()
    eda = run_eda(d)
    base = run_baselines(d); log(base.to_string())
    sweep, best = run_sweep(d); log(sweep.to_string()); log(f"[sweep] chon {best}")
    clf, m = run_classifiers(d, best); log(clf.to_string())
    deploy, final = run_deploy_options(d, best, m["C"], m["alpha"]); log(deploy.to_string())
    top = run_top_words(d, m["svc"])
    pairs, per = run_confusion(d, final); log(pairs.head(10).to_string())
    thr = run_threshold(d, final); log(thr.to_string())
    timing = run_timing(d, final); log(str(timing))
    errs, by_len = run_errors(d, final); log(by_len.to_string())
    lsa = run_lsa(d, best, m["C"])
    save_summary(leakage=leak, baseline=base, vectorizer_chon=best, bo_phan_loai=clf, trien_khai=deploy, nguong=thr,
                 thoi_gian=timing, lsa=lsa, nham=pairs.head(10), theo_do_dai=by_len)


if __name__ == "__main__":
    main()
