"""
TT-06 — Naive Bayes lọc SMS rác
================================
Chạy: python src/train.py
Đầu vào: data/spam.csv (cột v1=label, v2=text, encoding latin-1)
Đầu ra: reports/*.png, models/nb_pipeline.joblib, in bảng so sánh ra màn hình
"""
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, precision_recall_curve
)

DATA_PATH = "data/spam.csv"
REPORT_DIR = "reports"
MODEL_DIR = "models"
RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────────
# BƯỚC 1: Nạp dữ liệu, kiểm tra encoding
# ──────────────────────────────────────────────────────────────────
print("=" * 70)
print("BƯỚC 1: Nạp dữ liệu")
print("=" * 70)
# File gốc UCI là latin-1, KHÔNG phải utf-8 -> đọc bằng utf-8 sẽ crash
df = pd.read_csv(DATA_PATH, encoding="latin-1")
df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
print(f"Số dòng ban đầu : {len(df)}")
print(df["label"].value_counts())
print(f"Tỉ lệ spam      : {(df['label']=='spam').mean():.1%}")

# ──────────────────────────────────────────────────────────────────
# BƯỚC 2: Loại tin nhắn trùng lặp
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BƯỚC 2: Loại trùng lặp")
print("=" * 70)
n_before = len(df)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
print(f"Trùng lặp bị loại: {n_before - len(df)} dòng -> còn lại {len(df)} dòng")

# ──────────────────────────────────────────────────────────────────
# BƯỚC 3: EDA - độ dài tin ham vs spam
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BƯỚC 3: EDA độ dài tin nhắn")
print("=" * 70)
df["length"] = df["text"].str.len()
print(df.groupby("label")["length"].describe()[["mean", "50%", "max"]])

plt.figure(figsize=(7, 4.5))
for lbl, color in [("ham", "#4C72B0"), ("spam", "#C44E52")]:
    plt.hist(df.loc[df.label == lbl, "length"], bins=30, alpha=0.6, label=lbl, color=color)
plt.xlabel("Độ dài tin nhắn (số ký tự)")
plt.ylabel("Số lượng")
plt.title("Phân bố độ dài: HAM vs SPAM")
plt.legend()
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/do_dai_tin.png", dpi=130)
plt.close()
print(f"-> Đã lưu {REPORT_DIR}/do_dai_tin.png")

# Chia train/test TRƯỚC khi vector hoá để tránh rò rỉ
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE
)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

# ──────────────────────────────────────────────────────────────────
# BƯỚC 4: Baseline ngây thơ
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BƯỚC 4: Baseline (DummyClassifier)")
print("=" * 70)
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
acc_dummy = accuracy_score(y_test, dummy.predict(X_test))
print(f"Baseline accuracy (đoán toàn 'ham'): {acc_dummy:.4f}  <-- cao nhưng VÔ DỤNG")

# ──────────────────────────────────────────────────────────────────
# BƯỚC 5-7: 3 tổ hợp Count/TFIDF x Multinomial/Bernoulli
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BƯỚC 5-7: So sánh 3 tổ hợp vector hoá x mô hình")
print("=" * 70)

combos = {
    "CountVectorizer + MultinomialNB": make_pipeline(
        CountVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9),
        MultinomialNB(alpha=0.1),
    ),
    "TfidfVectorizer + MultinomialNB": make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True),
        MultinomialNB(alpha=0.1),
    ),
    "TfidfVectorizer + BernoulliNB": make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True),
        BernoulliNB(alpha=0.1),
    ),
}

results = []
fitted = {}
for name, pipe in combos.items():
    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = pipe.predict(X_test)
    predict_ms_per_msg = (time.time() - t0) / len(X_test) * 1000

    results.append({
        "Tổ hợp": name,
        "Precision": precision_score(y_test, y_pred, pos_label="spam"),
        "Recall": recall_score(y_test, y_pred, pos_label="spam"),
        "F1": f1_score(y_test, y_pred, pos_label="spam"),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Train (s)": round(train_time, 3),
        "Predict (ms/tin)": round(predict_ms_per_msg, 4),
    })
    fitted[name] = pipe

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print(results_df.to_string(index=False))
results_df.to_csv(f"{REPORT_DIR}/bang_so_sanh_3_to_hop.csv", index=False)

best_name = results_df.iloc[0]["Tổ hợp"]
best_pipe = fitted[best_name]
print(f"\n>> Tổ hợp tốt nhất theo F1: {best_name}")

# ──────────────────────────────────────────────────────────────────
# BƯỚC 8: Dò alpha và ngram_range
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BƯỚC 8: Dò alpha và ngram_range (trên TfidfVectorizer + MultinomialNB)")
print("=" * 70)
grid_results = []
for alpha in [0.01, 0.1, 0.5, 1.0]:
    for ngram in [(1, 1), (1, 2)]:
        pipe = make_pipeline(
            TfidfVectorizer(lowercase=True, ngram_range=ngram, min_df=2, max_df=0.9, sublinear_tf=True),
            MultinomialNB(alpha=alpha),
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        grid_results.append({
            "alpha": alpha,
            "ngram_range": str(ngram),
            "Precision": round(precision_score(y_test, y_pred, pos_label="spam"), 4),
            "Recall": round(recall_score(y_test, y_pred, pos_label="spam"), 4),
            "F1": round(f1_score(y_test, y_pred, pos_label="spam"), 4),
        })
grid_df = pd.DataFrame(grid_results).sort_values("F1", ascending=False)
print(grid_df.to_string(index=False))
grid_df.to_csv(f"{REPORT_DIR}/grid_alpha_ngram.csv", index=False)

# alpha = 0 minh hoạ lỗi (chỉ minh hoạ, KHÔNG dùng thật)
print("\n[Minh hoạ vì sao alpha=0 nguy hiểm]")
try:
    demo_pipe = make_pipeline(CountVectorizer(), MultinomialNB(alpha=0.0))
    demo_pipe.fit(X_train, y_train)
    probs = demo_pipe.predict_proba(["a completely unseen made up word xyzzyqwerty"])
    print(f"P(ham), P(spam) cho tin chứa từ CHƯA TỪNG THẤY: {probs}")
except Exception as e:
    print(f"Lỗi/ cảnh báo khi alpha=0: {e}")
print("-> alpha=0 khiến xác suất của từ lạ = 0, làm hỏng toàn bộ tích xác suất Naive Bayes.")

# ──────────────────────────────────────────────────────────────────
# BƯỚC 9: Top 20 từ có xác suất cao nhất cho lớp spam
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BƯỚC 9: Top 20 từ đặc trưng của SPAM")
print("=" * 70)
vec = best_pipe.named_steps[list(best_pipe.named_steps.keys())[0]]
clf = best_pipe.named_steps[list(best_pipe.named_steps.keys())[1]]
feature_names = np.array(vec.get_feature_names_out())
spam_idx = list(clf.classes_).index("spam")
log_prob_spam = clf.feature_log_prob_[spam_idx]
top20_idx = np.argsort(log_prob_spam)[::-1][:20]
top20_words = feature_names[top20_idx]
print(", ".join(top20_words))

plt.figure(figsize=(7, 6))
plt.barh(range(20), log_prob_spam[top20_idx][::-1], color="#C44E52")
plt.yticks(range(20), top20_words[::-1])
plt.xlabel("log P(từ | spam)")
plt.title("Top 20 từ đặc trưng của SPAM")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/top_tu_spam.png", dpi=130)
plt.close()
print(f"-> Đã lưu {REPORT_DIR}/top_tu_spam.png")

# ──────────────────────────────────────────────────────────────────
# BƯỚC 10: Chọn ngưỡng đạt precision >= 0.98
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BƯỚC 10: Chọn ngưỡng theo Precision >= 0.98")
print("=" * 70)
y_test_bin = (y_test == "spam").astype(int)
proba_spam = best_pipe.predict_proba(X_test)[:, spam_idx]
precisions, recalls, thresholds = precision_recall_curve(y_test_bin, proba_spam)

target_precision = 0.98
valid = precisions[:-1] >= target_precision
if valid.any():
    chosen_idx = np.argmax(valid)  # ngưỡng nhỏ nhất đạt đủ precision -> recall cao nhất có thể
    chosen_threshold = thresholds[chosen_idx]
    chosen_recall = recalls[chosen_idx]
    chosen_precision = precisions[chosen_idx]
else:
    chosen_threshold = 0.5
    chosen_precision, chosen_recall = precisions[len(precisions)//2], recalls[len(precisions)//2]

print(f"Ngưỡng chọn: {chosen_threshold:.3f}")
print(f"-> Precision đạt được: {chosen_precision:.4f}")
print(f"-> Recall tương ứng : {chosen_recall:.4f}")
print("(Precision cao ưu tiên vì chặn NHẦM tin thật, ví dụ mã OTP, gây thiệt hại nặng hơn lọt 1 tin rác)")

y_pred_thresholded = np.where(proba_spam >= chosen_threshold, "spam", "ham")
cm = confusion_matrix(y_test, y_pred_thresholded, labels=["ham", "spam"])

plt.figure(figsize=(5, 4.5))
plt.imshow(cm, cmap="Blues")
plt.title(f"Ma trận nhầm lẫn (ngưỡng={chosen_threshold:.2f})")
plt.xticks([0, 1], ["ham", "spam"])
plt.yticks([0, 1], ["ham", "spam"])
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
plt.colorbar()
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/confusion_matrix.png", dpi=130)
plt.close()
print(f"-> Đã lưu {REPORT_DIR}/confusion_matrix.png")
print(f"Ma trận nhầm lẫn:\n{cm}")

# ──────────────────────────────────────────────────────────────────
# BƯỚC 11: Phân tích 10 ca dự đoán sai
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BƯỚC 11: 10 tin bị phân loại SAI")
print("=" * 70)
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)
wrong_mask = y_pred_thresholded != y_test_reset.values
wrong_df = pd.DataFrame({
    "text": X_test_reset[wrong_mask],
    "thuc_te": y_test_reset[wrong_mask],
    "du_doan": y_pred_thresholded[wrong_mask],
    "xac_suat_spam": proba_spam[wrong_mask],
}).head(10)
if len(wrong_df) == 0:
    print("(Không có ca sai nào ở ngưỡng này trong tập test demo)")
else:
    for _, row in wrong_df.iterrows():
        print(f"[{row['thuc_te']} -> dự đoán {row['du_doan']}, p(spam)={row['xac_suat_spam']:.3f}] {row['text'][:80]}")
wrong_df.to_csv(f"{REPORT_DIR}/ca_du_doan_sai.csv", index=False)

# ──────────────────────────────────────────────────────────────────
# BƯỚC 12: So sánh với Logistic Regression + TF-IDF
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BƯỚC 12: So sánh với Logistic Regression")
print("=" * 70)
logreg_pipe = make_pipeline(
    TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
t0 = time.time()
logreg_pipe.fit(X_train, y_train)
lr_train_time = time.time() - t0
y_pred_lr = logreg_pipe.predict(X_test)

print(f"{'Mô hình':35s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'Train(s)':>10s}")
print(f"{best_name:35s} "
      f"{precision_score(y_test, best_pipe.predict(X_test), pos_label='spam'):10.4f} "
      f"{recall_score(y_test, best_pipe.predict(X_test), pos_label='spam'):8.4f} "
      f"{f1_score(y_test, best_pipe.predict(X_test), pos_label='spam'):8.4f} "
      f"{results_df.iloc[0]['Train (s)']:10.4f}")
print(f"{'Logistic Regression + TFIDF':35s} "
      f"{precision_score(y_test, y_pred_lr, pos_label='spam'):10.4f} "
      f"{recall_score(y_test, y_pred_lr, pos_label='spam'):8.4f} "
      f"{f1_score(y_test, y_pred_lr, pos_label='spam'):8.4f} "
      f"{lr_train_time:10.4f}")

# ──────────────────────────────────────────────────────────────────
# Lưu mô hình cuối cùng
# ──────────────────────────────────────────────────────────────────
joblib.dump({"pipeline": best_pipe, "threshold": chosen_threshold}, f"{MODEL_DIR}/nb_pipeline.joblib")
print(f"\nĐã lưu mô hình vào {MODEL_DIR}/nb_pipeline.joblib")
print("\nHOÀN TẤT.")
