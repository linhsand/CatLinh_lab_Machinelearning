"""
TT-06 — NAIVE BAYES: LỌC TIN NHẮN RÁC CHO TỔNG ĐÀI VIỄN THÔNG
==============================================================
Chạy:  python src/train.py            (từ bất kỳ thư mục nào)
       python src/train.py --help     (xem tuỳ chọn)

Đầu vào : data/spam.csv — bộ SMS Spam Collection (UCI #228), 5.572 dòng.
          Nếu chưa có: python data/download_data.py
Đầu ra  : reports/*.png, reports/*.csv, reports/ket_qua.md,
          models/nb_pipeline.joblib

NGUYÊN TẮC ĐÁNH GIÁ (điểm sửa chính so với bản trước)
------------------------------------------------------
Tập TEST được khoá lại ngay sau khi tách và CHỈ được chạm đúng MỘT LẦN ở
bước cuối. Mọi quyết định lựa chọn — tổ hợp vector hoá/mô hình, alpha,
ngram_range, và NGƯỠNG quyết định — đều chấm bằng K-fold cross-validation
trên tập TRAIN (xác suất out-of-fold). Bản trước chọn ngưỡng bằng
precision_recall_curve(y_test, ...) rồi báo cáo trên chính test đó, nên
con số Precision/Recall công bố là ước lượng lạc quan giả tạo.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import make_pipeline

# ── Đường dẫn neo theo VỊ TRÍ FILE, không theo cwd ────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "spam.csv"
DEFAULT_REPORTS = ROOT / "reports"
DEFAULT_MODELS = ROOT / "models"

POS = "spam"  # nhãn dương
EXPECTED_ROWS = 5572
EXPECTED_LABELS = {"ham": 4825, "spam": 747}


# ══════════════════════════════════════════════════════════════════════════
# Tiện ích
# ══════════════════════════════════════════════════════════════════════════
def banner(text: str) -> str:
    line = "=" * 74
    out = f"\n{line}\n{text}\n{line}"
    print(out)
    return out


def spam_column(estimator) -> int:
    """Chỉ số cột ứng với lớp 'spam' trong predict_proba."""
    return int(list(estimator.classes_).index(POS))


def prf(y_true, y_pred) -> dict:
    return {
        "Precision": precision_score(y_true, y_pred, pos_label=POS, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=POS, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=POS, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
    }


# ══════════════════════════════════════════════════════════════════════════
# BƯỚC 1 — Nạp và XÁC THỰC dữ liệu
# ══════════════════════════════════════════════════════════════════════════
def load_data(path: Path, skip_check: bool) -> pd.DataFrame:
    banner("BƯỚC 1 — Nạp & xác thực dữ liệu")

    if not path.exists():
        print(
            f"[LỖI] Không tìm thấy dữ liệu tại: {path}\n"
            f"      Bộ dữ liệu KHÔNG được commit vào repo (xem data/.gitignore).\n"
            f"      Chạy lệnh sau rồi thử lại:\n\n"
            f"          python data/download_data.py\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # File gốc UCI mã hoá latin-1 (chứa £, €); đọc bằng utf-8 sẽ UnicodeDecodeError.
    df = pd.read_csv(path, encoding="latin-1")
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    else:
        df = df.iloc[:, :2]
        df.columns = ["label", "text"]

    counts = df["label"].value_counts().to_dict()
    print(f"Số dòng thô     : {len(df):,}")
    print(f"Phân phối nhãn  : {counts}")
    print(f"Tỉ lệ spam      : {(df['label'] == POS).mean():.2%}")

    if not skip_check:
        if len(df) != EXPECTED_ROWS or counts != EXPECTED_LABELS:
            print(
                f"\n[LỖI TOÀN VẸN DỮ LIỆU]\n"
                f"  Kỳ vọng {EXPECTED_ROWS:,} dòng {EXPECTED_LABELS}\n"
                f"  Nhận được {len(df):,} dòng {counts}\n"
                f"  Đây KHÔNG phải bộ SMS Spam Collection gốc. Mọi số liệu sinh ra\n"
                f"  sẽ không so sánh được với README. Chạy: python data/download_data.py --force\n"
                f"  (hoặc --skip-data-check nếu bạn CỐ Ý dùng bộ dữ liệu khác)",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print("[OK] Khớp bộ SMS Spam Collection gốc của UCI.")
    else:
        print("[CẢNH BÁO] Bỏ qua kiểm tra toàn vẹn (--skip-data-check).")

    # BƯỚC 2 — loại trùng lặp TRƯỚC khi chia tập (chống rò rỉ train↔test)
    n_before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"\nLoại trùng lặp  : {n_before - len(df)} dòng → còn {len(df):,} dòng duy nhất")
    print(f"Phân phối sau lọc: {df['label'].value_counts().to_dict()}")
    return df


# ══════════════════════════════════════════════════════════════════════════
# BƯỚC 3 — EDA (chỉ trên TRAIN)
# ══════════════════════════════════════════════════════════════════════════
def eda_length(X_train: pd.Series, y_train: pd.Series, reports: Path) -> pd.DataFrame:
    banner("BƯỚC 3 — EDA độ dài tin nhắn (CHỈ trên tập TRAIN)")
    tmp = pd.DataFrame({"text": X_train, "label": y_train})
    tmp["length"] = tmp["text"].str.len()
    stats = tmp.groupby("label")["length"].describe()[["count", "mean", "std", "50%", "min", "max"]]
    print(stats.to_string())
    print("\n(Thống kê tính trên TRAIN để không hé lộ phân bố của tập TEST.)")

    plt.figure(figsize=(9, 5))
    for lbl, color, name in [("ham", "#2b5c8f", "HAM (tin thường)"), (POS, "#d9534f", "SPAM (tin rác)")]:
        plt.hist(tmp.loc[tmp.label == lbl, "length"], bins=50, alpha=0.65,
                 label=name, color=color, density=True)
    plt.xlabel("Độ dài tin nhắn (số ký tự)")
    plt.ylabel("Mật độ phân bố")
    plt.title("Phân bố độ dài tin nhắn: HAM vs SPAM (tập train)", fontweight="bold")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(reports / "do_dai_tin.png", dpi=150)
    plt.close()
    print(f"→ đã lưu {reports / 'do_dai_tin.png'}")
    return stats


# ══════════════════════════════════════════════════════════════════════════
# Định nghĩa các pipeline ứng viên
# ══════════════════════════════════════════════════════════════════════════
def build_combos(alpha: float, ngram: tuple[int, int]) -> dict:
    return {
        "CountVectorizer + MultinomialNB": make_pipeline(
            CountVectorizer(lowercase=True, ngram_range=ngram, min_df=2, max_df=0.9),
            MultinomialNB(alpha=alpha),
        ),
        "TfidfVectorizer + MultinomialNB": make_pipeline(
            TfidfVectorizer(lowercase=True, ngram_range=ngram, min_df=2, max_df=0.9, sublinear_tf=True),
            MultinomialNB(alpha=alpha),
        ),
        "TfidfVectorizer + BernoulliNB": make_pipeline(
            TfidfVectorizer(lowercase=True, ngram_range=ngram, min_df=2, max_df=0.9, sublinear_tf=True),
            BernoulliNB(alpha=alpha),
        ),
    }


def build_logreg(ngram: tuple[int, int], seed: int):
    return make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=ngram, min_df=2, max_df=0.9, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )


# ══════════════════════════════════════════════════════════════════════════
# BƯỚC 5-7 — So sánh tổ hợp bằng CROSS-VALIDATION trên TRAIN
# ══════════════════════════════════════════════════════════════════════════
def compare_combos(X_train, y_train, cv, reports: Path, seed: int) -> tuple[pd.DataFrame, str]:
    banner("BƯỚC 5-7 — So sánh tổ hợp bằng Cross-Validation trên TRAIN (test chưa đụng)")

    candidates = build_combos(alpha=0.1, ngram=(1, 2))
    candidates["Logistic Regression + TFIDF"] = build_logreg((1, 2), seed)
    candidates["Baseline: DummyClassifier"] = make_pipeline(
        CountVectorizer(), DummyClassifier(strategy="most_frequent")
    )

    rows = []
    for name, pipe in candidates.items():
        t0 = time.perf_counter()
        y_oof = cross_val_predict(pipe, X_train, y_train, cv=cv, n_jobs=1)
        cv_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0

        m = prf(y_train, y_oof)
        rows.append({
            "Tổ hợp": name,
            "Precision (CV)": round(m["Precision"], 4),
            "Recall (CV)": round(m["Recall"], 4),
            "F1 (CV)": round(m["F1"], 4),
            "Accuracy (CV)": round(m["Accuracy"], 4),
            "Train 1 lần (s)": round(fit_seconds, 4),
            f"{cv.get_n_splits()}-fold CV (s)": round(cv_seconds, 3),
        })

    df = pd.DataFrame(rows).sort_values("F1 (CV)", ascending=False).reset_index(drop=True)
    print(df.to_string(index=False))
    df.to_csv(reports / "bang_so_sanh_3_to_hop.csv", index=False)
    print(f"\n→ đã lưu {reports / 'bang_so_sanh_3_to_hop.csv'}")

    nb_only = df[~df["Tổ hợp"].str.contains("Logistic|Baseline")]
    best_name = str(nb_only.iloc[0]["Tổ hợp"])
    print(f"\n>> Tổ hợp Naive Bayes tốt nhất theo F1 out-of-fold: {best_name}")
    return df, best_name


# ══════════════════════════════════════════════════════════════════════════
# BƯỚC 8 — Lưới alpha × ngram, chấm bằng CV (KHÔNG dùng y_test)
# ══════════════════════════════════════════════════════════════════════════
def grid_search(X_train, y_train, cv, best_name: str, reports: Path) -> tuple[pd.DataFrame, float, tuple]:
    banner("BƯỚC 8 — Dò alpha × ngram_range bằng Cross-Validation trên TRAIN")

    rows = []
    for alpha in [0.01, 0.1, 0.5, 1.0]:
        for ngram in [(1, 1), (1, 2)]:
            pipe = build_combos(alpha, ngram)[best_name]
            y_oof = cross_val_predict(pipe, X_train, y_train, cv=cv, n_jobs=1)
            m = prf(y_train, y_oof)
            rows.append({
                "alpha": alpha,
                "ngram_range": str(ngram),
                "Precision (CV)": round(m["Precision"], 4),
                "Recall (CV)": round(m["Recall"], 4),
                "F1 (CV)": round(m["F1"], 4),
                "Accuracy (CV)": round(m["Accuracy"], 4),
            })

    df = pd.DataFrame(rows).sort_values("F1 (CV)", ascending=False).reset_index(drop=True)
    print(df.to_string(index=False))
    df.to_csv(reports / "grid_alpha_ngram.csv", index=False)
    print(f"\n→ đã lưu {reports / 'grid_alpha_ngram.csv'}")

    best_alpha = float(df.iloc[0]["alpha"])
    best_ngram = eval(df.iloc[0]["ngram_range"])  # noqa: S307 — chuỗi do chính ta sinh
    print(f"\n>> Cấu hình tốt nhất (out-of-fold): alpha={best_alpha}, ngram_range={best_ngram}")
    return df, best_alpha, best_ngram


def demo_alpha_zero(X_train, y_train) -> str:
    """Minh hoạ zero-probability một cách CÓ KIỂM SOÁT.

    Điểm tinh tế thường bị hiểu sai: một từ hoàn toàn lạ (chưa từng có trong
    train) KHÔNG gây lỗi — CountVectorizer lặng lẽ bỏ nó vì nó không nằm trong
    từ vựng. Thủ phạm thật là từ CÓ trong từ vựng nhưng đếm được 0 lần ở MỘT
    lớp. Khi alpha = 0, P(w|lớp đó) = 0 và cả tích xác suất của lớp đó sụp về 0,
    bất kể mọi bằng chứng còn lại nói gì.
    """
    banner("BƯỚC 8b — Vì sao alpha = 0 phá huỷ mô hình (zero-probability)")
    lines: list[str] = []

    vec = CountVectorizer()
    Xc = vec.fit_transform(X_train)
    names = np.array(vec.get_feature_names_out())
    is_spam = (y_train.values == POS)
    cnt_spam = np.asarray(Xc[is_spam].sum(axis=0)).ravel()
    cnt_ham = np.asarray(Xc[~is_spam].sum(axis=0)).ravel()

    # Từ chỉ xuất hiện ở spam (đếm ở ham = 0) và ngược lại, chọn từ phổ biến nhất
    only_spam = names[(cnt_ham == 0) & (cnt_spam > 0)][
        np.argsort(cnt_spam[(cnt_ham == 0) & (cnt_spam > 0)])[::-1]
    ][:1]
    only_ham = names[(cnt_spam == 0) & (cnt_ham > 0)][
        np.argsort(cnt_ham[(cnt_spam == 0) & (cnt_ham > 0)])[::-1]
    ][:1]
    w_spam_only = str(only_spam[0])
    w_ham_only = str(only_ham[0])

    base_ham = "ok lar joking wif u oni see you later at home"
    base_spam = "URGENT you have won a free prize call now to claim your reward"
    unseen = "xyzzyqwerty7788"

    probes = [
        ("Tin HAM bình thường", base_ham),
        (f"Tin HAM + 1 từ chỉ có ở spam ({w_spam_only!r})", f"{base_ham} {w_spam_only}"),
        ("Tin SPAM bình thường", base_spam),
        (f"Tin SPAM + 1 từ chỉ có ở ham ({w_ham_only!r})", f"{base_spam} {w_ham_only}"),
        (f"Tin SPAM + 1 từ HOÀN TOÀN lạ ({unseen!r})", f"{base_spam} {unseen}"),
    ]

    proba, label = {}, {}
    for alpha, tag in [(0.0, "a0"), (0.1, "a01")]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipe = make_pipeline(CountVectorizer(), MultinomialNB(alpha=alpha))
            pipe.fit(X_train, y_train)
            j = spam_column(pipe)
            texts = [t for _, t in probes]
            proba[tag] = pipe.predict_proba(texts)[:, j]
            label[tag] = pipe.predict(texts)
            if alpha == 0.0:
                for w in caught:
                    lines.append(f"[cảnh báo sklearn] {w.category.__name__}: {w.message}")

    lines.append(f"Từ chỉ xuất hiện trong SPAM ở tập train: {w_spam_only!r} "
                 f"({int(cnt_spam[names == w_spam_only][0])} lần spam / 0 lần ham)")
    lines.append(f"Từ chỉ xuất hiện trong HAM ở tập train : {w_ham_only!r} "
                 f"({int(cnt_ham[names == w_ham_only][0])} lần ham / 0 lần spam)")
    lines.append("")
    lines.append(f"{'Tình huống':<50} {'P(spam) α=0':>12} {'nhãn':>6} "
                 f"{'P(spam) α=0.1':>14} {'nhãn':>6}")
    lines.append("-" * 92)
    for i, (name, _) in enumerate(probes):
        p0 = proba["a0"][i]
        s0 = "  NaN" if np.isnan(p0) else f"{p0:.2e}"
        lines.append(f"{name:<50} {s0:>12} {label['a0'][i]:>6} "
                     f"{proba['a01'][i]:>14.2e} {label['a01'][i]:>6}")
    lines.append("")
    lines.append(
        "Đọc bảng — hai dòng có từ 'độc' cho ra NaN, không phải một xác suất sai. "
        "Với alpha = 0 thì P(w|c) = 0 nên log P(w|c) = −∞. Tin nhắn đó chứa cả từ "
        "vắng mặt ở ham lẫn từ vắng mặt ở spam, nên log-likelihood của CẢ HAI lớp "
        "đều bằng −∞; khi chuẩn hoá, −∞ − (−∞) = NaN. Bộ phân loại không trả về "
        "phán quyết sai — nó không trả về gì cả, và nhãn dự đoán trở thành tuỳ ý."
    )
    lines.append(
        "Dòng cuối cho thấy từ HOÀN TOÀN lạ lại vô hại: 'xyzzyqwerty7788' không nằm "
        "trong từ vựng nên bị CountVectorizer loại thẳng từ khâu vector hoá. Thủ "
        "phạm thật là từ CÓ trong từ vựng nhưng đếm được 0 lần ở một lớp — đúng "
        "tình huống mà Laplace smoothing sinh ra để xử lý."
    )
    lines.append(
        "Với alpha = 0.1, cùng những tin đó vẫn cho xác suất hữu hạn và nhãn đúng "
        "hướng. Lưu ý Naive Bayes vốn quá tự tin (xác suất bão hoà về ~1e-16 hoặc "
        "~1.0) do giả định độc lập nhân dồn hàng chục thừa số — đó cũng chính là lý "
        "do phải chọn ngưỡng bằng thực nghiệm ở bước 10 thay vì mặc định 0.5."
    )
    text = "\n".join(lines)
    print(text)
    return text


# ══════════════════════════════════════════════════════════════════════════
# BƯỚC 9 — Từ đặc trưng của SPAM
# ══════════════════════════════════════════════════════════════════════════
def top_spam_words(pipe, reports: Path, k: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    banner(f"BƯỚC 9 — Top {k} từ đặc trưng của SPAM")

    steps = list(pipe.named_steps.values())
    vec, clf = steps[0], steps[1]
    names = np.array(vec.get_feature_names_out())
    j = spam_column(clf)
    log_spam = clf.feature_log_prob_[j]
    log_ham = clf.feature_log_prob_[1 - j]

    # (a) Xếp theo log P(w | spam) — đúng yêu cầu đề bài
    idx = np.argsort(log_spam)[::-1][:k]
    raw_df = pd.DataFrame({
        "Hạng": range(1, k + 1),
        "Từ/Cụm từ": names[idx],
        "log P(từ | spam)": np.round(log_spam[idx], 4),
    })
    print("(a) Xếp theo log P(từ | spam):")
    print(raw_df.to_string(index=False))

    # (b) Xếp theo log-odds — loại bỏ hư từ, thấy đúng "chữ ký" của spam
    log_odds = log_spam - log_ham
    idx2 = np.argsort(log_odds)[::-1][:k]
    odds_df = pd.DataFrame({
        "Hạng": range(1, k + 1),
        "Từ/Cụm từ": names[idx2],
        "log P(w|spam) - log P(w|ham)": np.round(log_odds[idx2], 4),
    })
    print("\n(b) Xếp theo log-odds spam/ham (phân biệt tốt hơn):")
    print(odds_df.to_string(index=False))

    raw_df.to_csv(reports / "top_tu_spam.csv", index=False)
    odds_df.to_csv(reports / "top_tu_spam_logodds.csv", index=False)

    for data, col, fname, title, xlabel in [
        (raw_df, "log P(từ | spam)", "top_tu_spam.png",
         f"Top {k} từ có log P(từ | SPAM) cao nhất", "log P(từ | SPAM)"),
        (odds_df, "log P(w|spam) - log P(w|ham)", "top_tu_spam_logodds.png",
         f"Top {k} từ theo log-odds SPAM/HAM", "log P(w|spam) − log P(w|ham)"),
    ]:
        plt.figure(figsize=(9, 6.5))
        plt.barh(range(k), data[col].values[::-1], color="#c9302c", edgecolor="#7a1c1a")
        plt.yticks(range(k), data["Từ/Cụm từ"].values[::-1], fontsize=10)
        plt.xlabel(xlabel)
        plt.title(title, fontweight="bold")
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(reports / fname, dpi=150)
        plt.close()
        print(f"→ đã lưu {reports / fname}")

    return raw_df, odds_df


# ══════════════════════════════════════════════════════════════════════════
# BƯỚC 10 — Chọn ngưỡng trên xác suất OUT-OF-FOLD của TRAIN
# ══════════════════════════════════════════════════════════════════════════
def choose_threshold(final_pipe, X_train, y_train, cv, target_precision: float,
                     reports: Path) -> dict:
    banner(f"BƯỚC 10 — Chọn ngưỡng đạt Precision ≥ {target_precision} (trên OOF của TRAIN)")

    proba_oof = cross_val_predict(final_pipe, X_train, y_train, cv=cv,
                                  method="predict_proba", n_jobs=1)
    # cross_val_predict trả cột theo thứ tự np.unique(y) = ['ham', 'spam']
    j = int(np.where(np.unique(y_train) == POS)[0][0])
    p_oof = proba_oof[:, j]
    y_bin = (y_train.values == POS).astype(int)

    precisions, recalls, thresholds = precision_recall_curve(y_bin, p_oof)
    ok = precisions[:-1] >= target_precision
    if ok.any():
        # ngưỡng NHỎ NHẤT đạt đủ precision ⇒ recall lớn nhất có thể
        i = int(np.argmax(ok))
        thr = float(thresholds[i])
        p_val, r_val = float(precisions[i]), float(recalls[i])
        met = True
    else:
        thr, met = 0.5, False
        pred = np.where(p_oof >= thr, POS, "ham")
        p_val = precision_score(y_train, pred, pos_label=POS, zero_division=0)
        r_val = recall_score(y_train, pred, pos_label=POS, zero_division=0)
        print(f"[cảnh báo] Không cấu hình nào đạt Precision ≥ {target_precision} trên OOF; dùng T=0.5.")

    # Precision tại ngưỡng đó trên TỪNG fold — để thấy sai số ước lượng
    fold_prec, fold_rec = [], []
    for _, va in cv.split(X_train, y_train):
        pred = np.where(p_oof[va] >= thr, POS, "ham")
        truth = y_train.values[va]
        fold_prec.append(precision_score(truth, pred, pos_label=POS, zero_division=0))
        fold_rec.append(recall_score(truth, pred, pos_label=POS, zero_division=0))
    fold_prec, fold_rec = np.array(fold_prec), np.array(fold_rec)

    print(f"Ngưỡng chọn (từ dữ liệu validation) : T = {thr:.6g}")
    print(f"→ Precision ước lượng trên OOF      : {p_val:.4f}")
    print(f"→ Recall ước lượng trên OOF         : {r_val:.4f}")
    print(f"→ Precision theo từng fold          : {np.round(fold_prec, 4).tolist()}")
    print(f"   trung bình {fold_prec.mean():.4f} ± {fold_prec.std(ddof=1):.4f} (độ lệch chuẩn)")
    print(f"   sai số chuẩn ≈ {fold_prec.std(ddof=1) / np.sqrt(len(fold_prec)):.4f}"
          f"  → khoảng ±1 s.e. là [{fold_prec.mean() - fold_prec.std(ddof=1) / np.sqrt(len(fold_prec)):.4f}, "
          f"{fold_prec.mean() + fold_prec.std(ddof=1) / np.sqrt(len(fold_prec)):.4f}]")
    print("Ngưỡng này được ĐÓNG BĂNG trước khi chạm tập test.")
    print("Ràng buộc Precision ≥ mục tiêu chỉ đúng THEO KỲ VỌNG; trên một tập test\n"
          "hữu hạn, kết quả thực tế có thể rơi thấp hơn — xem bước 12.")

    plt.figure(figsize=(7.5, 5))
    plt.plot(recalls, precisions, color="#2b5c8f", lw=2, label="Đường Precision–Recall (OOF trên train)")
    plt.axhline(target_precision, ls="--", color="#d9534f", lw=1.5,
                label=f"Ràng buộc nghiệp vụ: Precision = {target_precision}")
    plt.scatter([r_val], [p_val], color="#d9534f", zorder=5, s=70,
                label=f"Điểm vận hành đã chọn (T = {thr:.2g})")
    plt.xlabel("Recall (lớp SPAM)")
    plt.ylabel("Precision (lớp SPAM)")
    plt.title("Chọn ngưỡng trên dữ liệu VALIDATION, không phải test", fontweight="bold")
    plt.grid(linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(reports / "chon_nguong_pr_curve.png", dpi=150)
    plt.close()
    print(f"→ đã lưu {reports / 'chon_nguong_pr_curve.png'}")

    return {
        "threshold": thr,
        "precision_oof": p_val,
        "recall_oof": r_val,
        "reached_target": met,
        "precision_fold_mean": float(fold_prec.mean()),
        "precision_fold_std": float(fold_prec.std(ddof=1)),
        "recall_fold_mean": float(fold_rec.mean()),
        "precision_folds": np.round(fold_prec, 4).tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════
# BƯỚC 11 — Đo độ trễ suy luận
# ══════════════════════════════════════════════════════════════════════════
def measure_latency(pipe, X_test, reports: Path, n_single: int = 300) -> pd.DataFrame:
    banner("BƯỚC 11 — Đo độ trễ suy luận (SLA gateway < 5 ms/tin)")

    sample = list(X_test[:n_single])
    times = []
    for msg in sample:
        t0 = time.perf_counter()
        pipe.predict_proba([msg])
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)

    t0 = time.perf_counter()
    pipe.predict_proba(list(X_test))
    batch_ms = (time.perf_counter() - t0) / len(X_test) * 1000

    df = pd.DataFrame([{
        "Chế độ": "Từng tin một (giống gateway thật)",
        "Trung vị p50 (ms)": round(float(np.percentile(times, 50)), 4),
        "p95 (ms)": round(float(np.percentile(times, 95)), 4),
        "p99 (ms)": round(float(np.percentile(times, 99)), 4),
        "Trung bình (ms)": round(float(times.mean()), 4),
    }, {
        "Chế độ": f"Theo lô {len(X_test)} tin (vector hoá 1 lần)",
        "Trung vị p50 (ms)": None,
        "p95 (ms)": None,
        "p99 (ms)": None,
        "Trung bình (ms)": round(batch_ms, 4),
    }])
    print(df.to_string(index=False))
    print("\nLưu ý: bản trước chỉ đo chế độ THEO LÔ rồi gọi đó là 'ms/tin' — con số này\n"
          "lạc quan hơn thực tế vì chi phí vector hoá được chia đều. Gateway xử lý\n"
          "từng tin, nên p95 của chế độ đầu mới là số cần đối chiếu với SLA.")
    df.to_csv(reports / "do_tre_suy_luan.csv", index=False)
    print(f"→ đã lưu {reports / 'do_tre_suy_luan.csv'}")
    return df


# ══════════════════════════════════════════════════════════════════════════
# BƯỚC 12 — Đánh giá cuối cùng trên TEST (chạm đúng một lần)
# ══════════════════════════════════════════════════════════════════════════
def final_evaluation(final_pipe, logreg_pipe, dummy_pipe, X_train, y_train, X_test, y_test,
                     threshold: float, best_name: str, reports: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    banner("BƯỚC 12 — ĐÁNH GIÁ CUỐI CÙNG TRÊN TEST (lần chạm duy nhất)")

    j = spam_column(final_pipe)
    p_test = final_pipe.predict_proba(X_test)[:, j]
    y_pred_thr = np.where(p_test >= threshold, POS, "ham")

    rows = []
    m = prf(y_test, y_pred_thr)
    rows.append({"Mô hình": f"{best_name} @ T={threshold:.4g}", **{k: round(v, 4) for k, v in m.items()}})

    m = prf(y_test, final_pipe.predict(X_test))
    rows.append({"Mô hình": f"{best_name} @ T=0.5 (mặc định)", **{k: round(v, 4) for k, v in m.items()}})

    m = prf(y_test, logreg_pipe.predict(X_test))
    rows.append({"Mô hình": "Logistic Regression (balanced) + TFIDF", **{k: round(v, 4) for k, v in m.items()}})

    m = prf(y_test, dummy_pipe.predict(X_test))
    rows.append({"Mô hình": "Baseline: đoán toàn 'ham'", **{k: round(v, 4) for k, v in m.items()}})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(reports / "ket_qua_cuoi_cung.csv", index=False)
    print(f"\n→ đã lưu {reports / 'ket_qua_cuoi_cung.csv'}")

    cm = confusion_matrix(y_test, y_pred_thr, labels=["ham", POS])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title(f"Ma trận nhầm lẫn trên TEST (ngưỡng T = {threshold:.4g})", fontweight="bold", fontsize=11)
    plt.xticks([0, 1], ["Dự đoán: HAM", "Dự đoán: SPAM"])
    plt.yticks([0, 1], ["Thực tế: HAM", "Thực tế: SPAM"])
    plt.xlabel("Nhãn dự đoán")
    plt.ylabel("Nhãn thực tế")
    for a in range(2):
        for b in range(2):
            plt.text(b, a, f"{cm[a, b]:,}", ha="center", va="center", fontsize=15,
                     fontweight="bold", color="white" if cm[a, b] > cm.max() / 2 else "black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(reports / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"\nTN (ham đúng)              : {cm[0, 0]}")
    print(f"FP (chặn nhầm ham/OTP)     : {cm[0, 1]}")
    print(f"FN (lọt tin rác)           : {cm[1, 0]}")
    print(f"TP (bắt đúng spam)         : {cm[1, 1]}")
    print(f"→ đã lưu {reports / 'confusion_matrix.png'}")
    return df, cm, y_pred_thr


# ══════════════════════════════════════════════════════════════════════════
# BƯỚC 13 — Phân tích ca sai
# ══════════════════════════════════════════════════════════════════════════
def error_analysis(final_pipe, X_test, y_test, y_pred_thr, threshold: float,
                   reports: Path, k: int = 10) -> pd.DataFrame:
    banner(f"BƯỚC 13 — Phân tích {k} ca dự đoán SAI trên test")

    j = spam_column(final_pipe)
    p_test = final_pipe.predict_proba(X_test)[:, j]
    Xr = X_test.reset_index(drop=True)
    yr = y_test.reset_index(drop=True)
    mask = y_pred_thr != yr.values

    err = pd.DataFrame({
        "loai_loi": np.where(yr[mask] == "ham", "False Positive", "False Negative"),
        "thuc_te": yr[mask].values,
        "du_doan": y_pred_thr[mask],
        "xac_suat_spam": np.round(p_test[mask], 8),
        "do_dai": Xr[mask].str.len().values,
        "noi_dung": Xr[mask].values,
    })
    # FP xếp trước vì tốn kém hơn (chặn nhầm OTP), trong mỗi nhóm sắp theo
    # mức độ "sát ngưỡng": FP có p(spam) thấp nhất và FN có p(spam) cao nhất
    # là những ca đáng soi nhất vì chỉ suýt vượt/suýt trượt ngưỡng.
    err["_uu_tien"] = np.where(err["loai_loi"] == "False Positive", 0, 1)
    err["_khoang_cach"] = (err["xac_suat_spam"] - threshold).abs()
    err = (err.sort_values(["_uu_tien", "_khoang_cach"])
              .drop(columns=["_uu_tien", "_khoang_cach"])
              .reset_index(drop=True))

    n_fp = int((err["loai_loi"] == "False Positive").sum())
    n_fn = int((err["loai_loi"] == "False Negative").sum())
    print(f"Tổng số ca sai: {len(err)}  (FP = {n_fp}, FN = {n_fn})\n")

    shown = err.head(k)
    for i, row in shown.iterrows():
        print(f"Ca #{i + 1} [{row['loai_loi']}] {row['thuc_te'].upper()} → {row['du_doan'].upper()}"
              f"  p(spam) = {row['xac_suat_spam']:.3e}  (ngưỡng {threshold:.3e})")
        print(f"   {row['noi_dung'][:150]!r}\n")

    err.to_csv(reports / "ca_du_doan_sai.csv", index=False)
    print(f"→ đã lưu TOÀN BỘ {len(err)} ca sai vào {reports / 'ca_du_doan_sai.csv'}")
    if len(err) == 0:
        print("[CẢNH BÁO] Không có ca sai nào — dấu hiệu điển hình của dữ liệu giả lập/rò rỉ.")
    return err


# ══════════════════════════════════════════════════════════════════════════
# Sinh báo cáo Markdown để README không thể lệch khỏi số liệu thật
# ══════════════════════════════════════════════════════════════════════════
def write_markdown_report(path: Path, ctx: dict) -> None:
    def md(df: pd.DataFrame) -> str:
        return df.to_markdown(index=False)

    cm = ctx["cm"]
    err = ctx["errors"]
    text = f"""# KẾT QUẢ THỰC NGHIỆM — SINH TỰ ĐỘNG

> File này do `src/train.py` sinh ra, **không sửa tay**. Mọi bảng số trong
> `README.md` đều trích từ đây, nên README và `reports/` không thể mâu thuẫn nhau.

* Thời điểm chạy: `{ctx['timestamp']}`
* Lệnh: `python src/train.py --seed {ctx['seed']} --folds {ctx['folds']} --target-precision {ctx['target_precision']}`
* Dữ liệu: `{ctx['data_path']}` — {ctx['n_raw']:,} dòng thô → {ctx['n_dedup']:,} dòng sau khi lọc trùng
* Chia tập: train {ctx['n_train']:,} / test {ctx['n_test']:,} (80/20, stratify, seed {ctx['seed']})
* Phiên bản: scikit-learn {ctx['sklearn_version']}, pandas {ctx['pandas_version']}

## 1. Thống kê độ dài tin nhắn (tập train)

{md(ctx['length_stats'].reset_index())}

## 2. So sánh tổ hợp — chấm bằng {ctx['folds']}-fold CV trên TRAIN

Đây là bảng dùng để **chọn** mô hình, nên bắt buộc tính trên out-of-fold của train.

{md(ctx['combo_df'])}

Tổ hợp Naive Bayes được chọn: **{ctx['best_name']}**

## 3. Lưới alpha × ngram_range — chấm bằng {ctx['folds']}-fold CV trên TRAIN

{md(ctx['grid_df'])}

Cấu hình được chọn: **alpha = {ctx['best_alpha']}, ngram_range = {ctx['best_ngram']}**

## 4. Ngưỡng quyết định

| Hạng mục | Giá trị |
| :--- | :--- |
| Ràng buộc nghiệp vụ | Precision (spam) ≥ {ctx['target_precision']} |
| Ngưỡng chọn trên OOF của train | **T = {ctx['thr']['threshold']:.6g}** |
| Precision ước lượng (OOF gộp) | {ctx['thr']['precision_oof']:.4f} |
| Recall ước lượng (OOF gộp) | {ctx['thr']['recall_oof']:.4f} |
| Precision theo từng fold | {ctx['thr']['precision_folds']} |
| Precision trung bình ± độ lệch chuẩn | {ctx['thr']['precision_fold_mean']:.4f} ± {ctx['thr']['precision_fold_std']:.4f} |
| Đạt ràng buộc trên validation | {"Có" if ctx['thr']['reached_target'] else "Không"} |

Độ lệch chuẩn giữa các fold cho thấy ràng buộc Precision ≥ {ctx['target_precision']} chỉ đúng
**theo kỳ vọng**. Trên một tập test hữu hạn (chỉ {ctx['n_test_spam']} tin spam), sai lệch một
vài False Positive đã đủ kéo Precision xuống dưới mục tiêu — đây là hạn chế
thống kê, không phải lỗi lập trình.

## 5. Top 20 từ đặc trưng của SPAM

### 5a. Theo log P(từ | spam)

{md(ctx['top_raw'])}

### 5b. Theo log-odds log P(w|spam) − log P(w|ham)

{md(ctx['top_odds'])}

## 6. Độ trễ suy luận

{md(ctx['latency_df'])}

## 7. KẾT QUẢ CUỐI CÙNG TRÊN TEST (lần chạm duy nhất)

{md(ctx['final_df'])}

Ma trận nhầm lẫn tại T = {ctx['thr']['threshold']:.6g}:

| | Dự đoán HAM | Dự đoán SPAM |
| :--- | ---: | ---: |
| **Thực tế HAM** | {cm[0, 0]} (TN) | {cm[0, 1]} (FP) |
| **Thực tế SPAM** | {cm[1, 0]} (FN) | {cm[1, 1]} (TP) |

## 8. Phân tích ca sai

Tổng {len(err)} ca sai trên test: {int((err['loai_loi'] == 'False Positive').sum())} False Positive,
{int((err['loai_loi'] == 'False Negative').sum())} False Negative.
Danh sách đầy đủ: `reports/ca_du_doan_sai.csv`.

{md(err.head(10)[['loai_loi', 'thuc_te', 'du_doan', 'xac_suat_spam', 'noi_dung']].assign(noi_dung=lambda d: d['noi_dung'].str.slice(0, 110)))}

## 9. Minh hoạ alpha = 0

```
{ctx['alpha0_text']}
```
"""
    path.write_text(text, encoding="utf-8")
    print(f"\n→ đã lưu báo cáo tổng hợp {path}")


# ══════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Huấn luyện bộ lọc SMS rác bằng Naive Bayes và sinh toàn bộ báo cáo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Đường dẫn spam.csv")
    ap.add_argument("--reports", type=Path, default=DEFAULT_REPORTS, help="Thư mục báo cáo")
    ap.add_argument("--models", type=Path, default=DEFAULT_MODELS, help="Thư mục mô hình")
    ap.add_argument("--seed", type=int, default=42, help="random_state")
    ap.add_argument("--test-size", type=float, default=0.2, help="Tỉ lệ tập test")
    ap.add_argument("--folds", type=int, default=5, help="Số fold cross-validation trên train")
    ap.add_argument("--target-precision", type=float, default=0.98,
                    help="Ràng buộc Precision tối thiểu cho lớp spam")
    ap.add_argument("--skip-data-check", action="store_true",
                    help="Bỏ qua kiểm tra toàn vẹn dữ liệu (KHÔNG khuyến nghị)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)

    # Tạo thư mục đầu ra — thiếu bước này thì clone mới sẽ crash ở lần savefig đầu tiên
    args.reports.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)

    print(f"Thư mục gốc dự án : {ROOT}")
    print(f"Dữ liệu           : {args.data}")
    print(f"Báo cáo           : {args.reports}")
    print(f"Mô hình           : {args.models}")

    df = load_data(args.data, args.skip_data_check)
    n_dedup = len(df)

    # ── Tách TEST ra ngay và khoá lại ───────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=args.test_size,
        stratify=df["label"], random_state=args.seed,
    )
    banner("BƯỚC 2b — Chia tập (test bị khoá cho đến bước 12)")
    print(f"Train : {len(X_train):,} tin (spam {int((y_train == POS).sum()):,} / "
          f"ham {int((y_train == 'ham').sum()):,})")
    print(f"Test  : {len(X_test):,} tin (spam {int((y_test == POS).sum()):,} / "
          f"ham {int((y_test == 'ham').sum()):,})")

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    length_stats = eda_length(X_train, y_train, args.reports)
    combo_df, best_name = compare_combos(X_train, y_train, cv, args.reports, args.seed)
    grid_df, best_alpha, best_ngram = grid_search(X_train, y_train, cv, best_name, args.reports)
    alpha0_text = demo_alpha_zero(X_train, y_train)

    # Huấn luyện bản cuối trên TOÀN BỘ train (vẫn chưa chạm test)
    final_pipe = build_combos(best_alpha, best_ngram)[best_name]
    final_pipe.fit(X_train, y_train)

    top_raw, top_odds = top_spam_words(final_pipe, args.reports)
    thr = choose_threshold(final_pipe, X_train, y_train, cv, args.target_precision, args.reports)

    logreg_pipe = build_logreg(best_ngram, args.seed).fit(X_train, y_train)
    dummy_pipe = make_pipeline(CountVectorizer(), DummyClassifier(strategy="most_frequent"))
    dummy_pipe.fit(X_train, y_train)

    latency_df = measure_latency(final_pipe, X_test, args.reports)
    final_df, cm, y_pred_thr = final_evaluation(
        final_pipe, logreg_pipe, dummy_pipe, X_train, y_train, X_test, y_test,
        thr["threshold"], best_name, args.reports,
    )
    errors = error_analysis(final_pipe, X_test, y_test, y_pred_thr, thr["threshold"], args.reports)

    # ── Lưu mô hình ────────────────────────────────────────────────────
    artifact = {
        "pipeline": final_pipe,
        "threshold": thr["threshold"],
        "positive_label": POS,
        "target_precision": args.target_precision,
        "selection": {
            "combo": best_name, "alpha": best_alpha, "ngram_range": best_ngram,
            "chosen_on": f"{args.folds}-fold CV out-of-fold trên tập train",
        },
        "validation_metrics": {"precision": thr["precision_oof"], "recall": thr["recall_oof"]},
        "test_metrics": final_df.iloc[0].to_dict(),
        "data_fingerprint": {"n_raw": EXPECTED_ROWS, "n_dedup": n_dedup,
                             "n_train": len(X_train), "n_test": len(X_test)},
        "seed": args.seed,
        "sklearn_version": sklearn.__version__,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    model_path = args.models / "nb_pipeline.joblib"
    joblib.dump(artifact, model_path)
    print(f"\n→ đã lưu mô hình {model_path}")

    write_markdown_report(args.reports / "ket_qua.md", {
        "timestamp": artifact["created_at"], "seed": args.seed, "folds": args.folds,
        "target_precision": args.target_precision,
        "data_path": args.data.relative_to(ROOT) if args.data.is_relative_to(ROOT) else args.data,
        "n_raw": EXPECTED_ROWS, "n_dedup": n_dedup,
        "n_train": len(X_train), "n_test": len(X_test),
        "n_test_spam": int((y_test == POS).sum()),
        "sklearn_version": sklearn.__version__, "pandas_version": pd.__version__,
        "length_stats": length_stats, "combo_df": combo_df, "best_name": best_name,
        "grid_df": grid_df, "best_alpha": best_alpha, "best_ngram": best_ngram,
        "thr": thr, "top_raw": top_raw, "top_odds": top_odds,
        "latency_df": latency_df, "final_df": final_df, "cm": cm,
        "errors": errors, "alpha0_text": alpha0_text,
    })

    (args.reports / "run_metadata.json").write_text(
        json.dumps({k: str(v) for k, v in artifact.items()
                    if k not in ("pipeline",)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    banner("HOÀN TẤT")
    print("Mọi bảng số trong README được trích từ reports/ket_qua.md của lần chạy này.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
