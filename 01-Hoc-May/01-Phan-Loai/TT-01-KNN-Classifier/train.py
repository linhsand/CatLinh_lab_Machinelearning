"""
TT-01 — KNN CLASSIFIER: Sàng lọc nguy cơ tiểu đường
Thực hiện đầy đủ 11 bước theo README.md
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (recall_score, precision_score, f1_score,
                              accuracy_score, confusion_matrix,
                              classification_report, average_precision_score)

RANDOM_STATE = 42
COLS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
COT_KHONG_THE_BANG_0 = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

results_log = {}

# ============================================================
# BƯỚC 1 — Nạp dữ liệu, in describe() để phát hiện cột có min=0 phi lý
# ============================================================
print("=" * 70)
print("BƯỚC 1: NẠP DỮ LIỆU")
print("=" * 70)
df = pd.read_csv("data/pima-indians-diabetes.csv", header=None, names=COLS)
print(f"Kích thước: {df.shape}")
print(df.describe().T[['min', 'max', 'mean']])

min_zero_cols = [c for c in COT_KHONG_THE_BANG_0 if df[c].min() == 0]
print(f"\n⚠️  Các cột có min=0 (phi lý về mặt y sinh): {min_zero_cols}")

# --> các cột Glucose, BloodPressure, SkinThickness, Insulin, BMI không thể bằng 0 về mặt y sinh học
# ============================================================
# BƯỚC 2 — Thay 0 -> NaN ở 5 cột y sinh, đếm % thiếu
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 2: THAY 0 -> NaN, ĐẾM % THIẾU")
print("=" * 70)
df_clean = df.copy()
df_clean[COT_KHONG_THE_BANG_0] = df_clean[COT_KHONG_THE_BANG_0].replace(0, np.nan)
missing_pct = df_clean[COT_KHONG_THE_BANG_0].isna().mean().mul(100).round(1) #isna là True/False
                                                                            #tính mean ( true=1 false=0)
                                                                            #mul(100) để ra % và round(1) làm tròn 1 chữ số thập phân
print(missing_pct.to_string()) #missing_pct là series, in ra to_string() để hiển thị đẹp hơn
results_log["missing_pct"] = missing_pct.to_dict()

# ============================================================
# BƯỚC 3 — EDA: histogram Glucose theo nhóm Outcome
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 3: EDA — Glucose theo Outcome")
print("=" * 70)
fig, ax = plt.subplots(figsize=(7, 5))
for outcome, color, label in [(0, "#4C72B0", "Không tiểu đường (0)"),
                               (1, "#C44E52", "Tiểu đường (1)")]:
    vals = df_clean.loc[df_clean["Outcome"] == outcome, "Glucose"].dropna()
    ax.hist(vals, bins=25, alpha=0.6, color=color, label=label)
ax.set_xlabel("Glucose")
ax.set_ylabel("Số bệnh nhân")
ax.set_title("Phân bố Glucose theo nhóm Outcome")
ax.legend()
fig.tight_layout()
fig.savefig("reports/glucose_histogram.png", dpi=130)
plt.close(fig)
print("Đã lưu reports/glucose_histogram.png — Glucose tách nhóm khá rõ, đúng như kỳ vọng.")

# ============================================================
# BƯỚC 4 — Chia train/test stratify, random_state=42
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 4: CHIA TRAIN/TEST (stratify, random_state=42)")
print("=" * 70)
X = df_clean.drop(columns=["Outcome"])
y = df_clean["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Tỷ lệ dương tính - train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

# ============================================================
# BƯỚC 5 — Baseline: DummyClassifier(most_frequent)
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 5: BASELINE (DummyClassifier)")
print("=" * 70)
dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
dummy.fit(X_train, y_train) #đếm xem lớp nào nhiều nhất
y_pred_dummy = dummy.predict(X_test) 
baseline_recall = recall_score(y_test, y_pred_dummy) #recall tính theo công thức TP/(TP+FN)
baseline_acc = accuracy_score(y_test, y_pred_dummy) #acc=số dự đoán đúng/ tổng số dự đoán 

print(f"Baseline recall = {baseline_recall:.3f}, accuracy = {baseline_acc:.3f}")
results_log["baseline"] = {"recall": baseline_recall, "accuracy": baseline_acc}

#recall=0 -> model dự đoán tất cả là âm tính, không phát hiện ra dương tính nào.
#acc=0.649-> 64.9% bệnh nhân được dự đoán đúng -> đây là kết quả đánh lừa vì dummy classifier luôn dự đoán outcome=0
# ============================================================
# BƯỚC 6 — KNN với K=5 mặc định (có pipeline chuẩn: impute + scale)
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 6: KNN K=5 MẶC ĐỊNH (CÓ chuẩn hoá)")
print("=" * 70)
pipe_k5 = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
pipe_k5.fit(X_train, y_train) #pipeline tự động làm train-> imputer.fit()-> sca;e.fit()-> knn.fit()
y_pred_k5 = pipe_k5.predict(X_test)
recall_k5 = recall_score(y_test, y_pred_k5) 
acc_k5 = accuracy_score(y_test, y_pred_k5)
print(f"K=5 (chuẩn hoá): recall = {recall_k5:.3f}, accuracy = {acc_k5:.3f}")
results_log["knn_k5_scaled"] = {"recall": recall_k5, "accuracy": acc_k5}

#dùng KNN -> acc=0.753-> mô hình dự đoán đúng 75.3% bệnh nhân-> so với baseline thì acc của KNN 
                                                                #tăng 10.4 điểm %
#recall=0.611-> trong tất cả bệnh nhân mắc tiểu đường, KNN dự đoán đúng 61.1% -> so với baseline thì recall của KNN tăng 61.1 điểm %
#vd: có 100 ng tiểu đường, KNN dự đoán được 61 người, bỏ sót 39 người

#KNN đoán đúng nhiều hơn baseline

# ============================================================
# BƯỚC 7 — Chạy KNN KHÔNG chuẩn hoá, so sánh
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 7: KNN K=5 KHÔNG CHUẨN HOÁ (chứng minh vì sao cần scale)")
print("=" * 70)
pipe_noscale = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
pipe_noscale.fit(X_train, y_train)
y_pred_noscale = pipe_noscale.predict(X_test)
recall_noscale = recall_score(y_test, y_pred_noscale)
acc_noscale = accuracy_score(y_test, y_pred_noscale)
print(f"K=5 (KHÔNG chuẩn hoá): recall = {recall_noscale:.3f}, accuracy = {acc_noscale:.3f}")
results_log["knn_k5_unscaled"] = {"recall": recall_noscale, "accuracy": acc_noscale}

comparison_table = pd.DataFrame({
    "Baseline (Dummy)": [baseline_recall, baseline_acc],
    "KNN K=5 — KHÔNG chuẩn hoá": [recall_noscale, acc_noscale],
    "KNN K=5 — CÓ chuẩn hoá": [recall_k5, acc_k5],
}, index=["Recall", "Accuracy"]).round(3)
print("\nBẢNG SO SÁNH:")
print(comparison_table.to_string())
comparison_table.to_csv("reports/comparison_scale_vs_noscale.csv")

#KNN có chuẩn hóa cho kết quả tốt nhất
# ============================================================
# BƯỚC 8 — GridSearchCV dò K, weights, metric (tối ưu recall)
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 8: GRIDSEARCHCV (tối ưu RECALL)")
print("=" * 70)
grid_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("knn", KNeighborsClassifier()),
]) #tạo pipeline để gridsearchcv dò các tham số của knn -> khi gridsearchcv chạy, mọi mô hình đều được eda giống nhau
grid = {
    "knn__n_neighbors": list(range(1, 32, 2)), #sinh ra chạy từ 1-32 step=2-> 16 giá trị
    "knn__weights": ["uniform", "distance"], 
    "knn__metric": ["euclidean", "manhattan"], #manhattan: khoảng cách theo trục, euclidean: khoảng cách theo đường ché0
    #tổng số mô hình được thử= 16*2*2=64 mô hình
} #tạo lưới tham số-> danh sách các giá trị mà gridsearchcv sẽ thử
gs = GridSearchCV(
    grid_pipe, grid, #fridpipe: pipeline chứa knn, grid: lưới tham số
    cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), #dữ liệu train được chia thành 5 fold,
                                                                    #sau 5 lần lấy recall trung bình
    scoring="recall", n_jobs=-1, #chọn mô hình có recall cao nhất, n_jobs=-1: chạy hết tất cả core của cpu 
)#khởi tạo gridsearchcv
gs.fit(X_train, y_train) #huấn luyện 64 mô hình, mỗi mô hình CV 5 lần, tính recall trung bình, so sánh -> chọn mô hình tốt nhất
print(f"Best params: {gs.best_params_}")
print(f"Best CV recall: {gs.best_score_:.3f}")
results_log["gridsearch_best_params"] = gs.best_params_
results_log["gridsearch_best_cv_recall"] = gs.best_score_

# ============================================================
# BƯỚC 9 — Vẽ Recall & Accuracy theo K (weights='uniform', metric tốt nhất)
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 9: ĐƯỜNG RECALL/ACCURACY THEO K")
print("=" * 70)
best_weights = gs.best_params_["knn__weights"] #giữ nguyên  lấy weights và metrics tốt nhất
best_metric = gs.best_params_["knn__metric"]
k_values = list(range(1, 32, 2)) #tạo list k
recalls_by_k, accs_by_k = [], []
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

for k in k_values: #thử với mỗi k
    p = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights=best_weights, metric=best_metric)),
    ])
    fold_recalls, fold_accs = [], []
    for tr_idx, val_idx in cv.split(X_train, y_train): #chia train thành 5 fold, tương tự các bước ở trên 
        p.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pred = p.predict(X_train.iloc[val_idx])
        fold_recalls.append(recall_score(y_train.iloc[val_idx], pred))
        fold_accs.append(accuracy_score(y_train.iloc[val_idx], pred))
    recalls_by_k.append(np.mean(fold_recalls))
    accs_by_k.append(np.mean(fold_accs)) #lấy được mean của recall và fold sau tất cả các lần huấn luyện với k đó

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_values, recalls_by_k, marker="o", label="Recall (CV)", color="#C44E52")
ax.plot(k_values, accs_by_k, marker="s", label="Accuracy (CV)", color="#4C72B0")
ax.axvline(gs.best_params_["knn__n_neighbors"], color="gray", linestyle="--",
           label=f"K chọn = {gs.best_params_['knn__n_neighbors']}")
ax.set_xlabel("K (số láng giềng)")
ax.set_ylabel("Điểm (CV, 5-fold)")
ax.set_title(f"Recall & Accuracy theo K (weights={best_weights}, metric={best_metric})")
ax.legend()
fig.tight_layout()
fig.savefig("reports/recall_theo_K.png", dpi=130)
plt.close(fig)
print("Đã lưu reports/recall_theo_K.png")
print("Giải thích hình dạng: K nhỏ -> variance cao, dễ overfit theo nhiễu cục bộ "
      "(recall dao động mạnh). K lớn -> bias cao, model 'mượt' quá mức, "
      "nghiêng về lớp đa số (âm tính) nên recall lớp dương tính giảm dần.")

#Biểu đồ cho thấy khi K tăng từ 1 lên 13, cả Recall và Accuracy đều cải thiện. 
# Điều này cho thấy việc xem xét nhiều hàng xóm hơn giúp giảm ảnh hưởng của nhiễu và tăng khả năng tổng quát hóa của mô hình. 
# Tuy nhiên, khi K tiếp tục tăng sau 13, Recall giảm dần trong khi Accuracy gần như giữ nguyên. 
# Điều này phản ánh mô hình bắt đầu trở nên quá "mượt" (underfitting), có xu hướng dự đoán theo lớp đa số (không tiểu đường),
# dẫn đến bỏ sót nhiều bệnh nhân mắc tiểu đường hơn. Vì vậy, K = 13 là điểm cân bằng tốt giữa khả năng phát hiện bệnh (Recall) 
# và độ chính xác tổng thể (Accuracy), phù hợp với kết quả mà GridSearchCV đã lựa chọn.

# ============================================================
# BƯỚC 10 — Chạm tập TEST 1 lần với model tốt nhất, ma trận nhầm lẫn
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 10: ĐÁNH GIÁ TRÊN TẬP TEST (1 LẦN DUY NHẤT)")
print("=" * 70)
best_model = gs.best_estimator_ #là pipeline tốt nhất từ gridsearchcv
y_pred_best = best_model.predict(X_test) #dự đoán trên tập test, mô hình lần đầu thấy X_test-> làm đầy đủ các bước, KNN với k=13

test_recall = recall_score(y_test, y_pred_best) #recall = 0.519 -> trong tất cả bệnh nhân dương tính, mô hình dự đoán đúng khoảng 52%-> bỏ sót 45% bệnh nhân dương tính
test_precision = precision_score(y_test, y_pred_best) #precision = 0.609 -> trong tất cả các bệnh nhân mô hình dự đoán dương tính, có khoảng 60.9% thật sự dương tính
test_f1 = f1_score(y_test, y_pred_best) #f1= 0.560 -> là trung bình điều hòa của recall và precision-> vừa phát  hiện được nhiều bệnh nhân( recall ) vừa hạn chế báo động giả ( precision)
test_acc = accuracy_score(y_test, y_pred_best) #acc= 0.714 -> mô hình dự đoán đúng khoảng 71.4% bệnh nhân trên tập test 
cm = confusion_matrix(y_test, y_pred_best) # ma trận nhầm lẫn: [[84 16] [23 24]] ->
#trong 100 bệnh nhân dương tính, mô hình dự đoán đúng 24 người, bỏ sót 23 người;
# trong 100 bệnh nhân âm tính, mô hình dự đoán đúng 84 người, nhầm 16 người

print(f"Recall    = {test_recall:.3f}")
print(f"Precision = {test_precision:.3f}")
print(f"F1        = {test_f1:.3f}")
print(f"Accuracy  = {test_acc:.3f}")
print("\nMa trận nhầm lẫn:")
print(cm)
print("\n" + classification_report(y_test, y_pred_best, target_names=["Âm tính", "Dương tính"]))

results_log["test_best_model"] = {
    "recall": test_recall, "precision": test_precision,
    "f1": test_f1, "accuracy": test_acc,
    "confusion_matrix": cm.tolist(),
    "params": gs.best_params_,
}

fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Dự đoán 0", "Dự đoán 1"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Thực 0", "Thực 1"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16)
ax.set_title(f"Ma trận nhầm lẫn — KNN tối ưu (K={gs.best_params_['knn__n_neighbors']})")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig("reports/confusion_matrix.png", dpi=130)
plt.close(fig)
print("Đã lưu reports/confusion_matrix.png")

# K=1 sanity check trên TRAIN (giải thích tiêu chí hoàn thành)
p_k1 = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=1)),
])
p_k1.fit(X_train, y_train)
train_acc_k1 = accuracy_score(y_train, p_k1.predict(X_train))
print(f"\nK=1 accuracy trên TRAIN = {train_acc_k1:.3f} "
      f"(≈1.0 vì mỗi điểm train là láng giềng gần nhất của chính nó -> overfitting, "
      f"không phản ánh khả năng tổng quát hoá).")
results_log["k1_train_accuracy_sanity_check"] = train_acc_k1

# ============================================================
# BƯỚC 11 — So sánh KNN với Logistic Regression
# ============================================================
print("\n" + "=" * 70)
print("BƯỚC 11: SO SÁNH KNN vs LOGISTIC REGRESSION")
print("=" * 70)
logreg_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("logreg", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
])
logreg_pipe.fit(X_train, y_train)
y_pred_lr = logreg_pipe.predict(X_test)
lr_recall = recall_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_acc = accuracy_score(y_test, y_pred_lr)

final_compare = pd.DataFrame({
    "KNN (tối ưu)": [test_recall, test_precision, test_f1, test_acc],
    "Logistic Regression": [lr_recall, lr_precision, lr_f1, lr_acc],
}, index=["Recall", "Precision", "F1", "Accuracy"]).round(3)
print(final_compare.to_string())
final_compare.to_csv("reports/comparison_knn_vs_logreg.csv")
results_log["knn_vs_logreg"] = final_compare.to_dict()

print("\nNhận xét: Logistic Regression với class_weight='balanced' thường cho recall "
      "cao hơn hoặc tương đương KNN vì tối ưu trực tiếp theo ranh giới xác suất và "
      "xử lý mất cân bằng lớp tốt hơn, đồng thời cho hệ số diễn giải được (feature "
      "importance) — điều KNN không có. KNN có thể nhỉnh hơn nếu ranh giới quyết định "
      "phi tuyến mạnh, nhưng chậm hơn khi dự đoán trên dữ liệu lớn.")

# ============================================================
# LƯU MODEL & LOG
# ============================================================
joblib.dump(best_model, "models/knn_pipeline.joblib")
with open("reports/results_log.json", "w", encoding="utf-8") as f:
    json.dump(results_log, f, ensure_ascii=False, indent=2, default=str)

print("\n" + "=" * 70)
print("HOÀN TẤT. Model đã lưu: models/knn_pipeline.joblib")
print("=" * 70)
