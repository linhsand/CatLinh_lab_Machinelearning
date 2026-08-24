"""
TT-01 — KNN CLASSIFIER: Sàng lọc nguy cơ tiểu đường
Thực hiện đầy đủ 11 bước theo README.md
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "pima-indians-diabetes.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")

RANDOM_STATE = 42
COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
# Các cột không thể bằng 0 về mặt y sinh học -> 0 thực chất là dữ liệu thiếu
COT_KHONG_THE_BANG_0 = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


# ============================================================
# BƯỚC 1-2 — Nạp dữ liệu, thay 0 -> NaN, đếm % thiếu
# ============================================================
def load_and_clean_data(results_log):
    print("=" * 70)
    print("BƯỚC 1: NẠP DỮ LIỆU")
    print("=" * 70)
    df = pd.read_csv(DATA_PATH, header=None, names=COLS)
    print(f"Kích thước: {df.shape}")
    print(df.describe().T[["min", "max", "mean"]])

    min_zero_cols = [c for c in COT_KHONG_THE_BANG_0 if df[c].min() == 0]
    print(f"\n⚠️  Các cột có min=0 (phi lý về mặt y sinh): {min_zero_cols}")

    print("\n" + "=" * 70)
    print("BƯỚC 2: THAY 0 -> NaN, ĐẾM % THIẾU")
    print("=" * 70)
    df_clean = df.copy()
    df_clean[COT_KHONG_THE_BANG_0] = df_clean[COT_KHONG_THE_BANG_0].replace(0, np.nan)
    # isna() trả True/False, mean() tính tỷ lệ True, mul(100) đổi ra %
    missing_pct = df_clean[COT_KHONG_THE_BANG_0].isna().mean().mul(100).round(1)
    print(missing_pct.to_string())
    results_log["missing_pct"] = missing_pct.to_dict()
    return df_clean


# ============================================================
# BƯỚC 3 — EDA: histogram Glucose theo nhóm Outcome
# ============================================================
def eda_glucose_histogram(df_clean):
    print("\n" + "=" * 70)
    print("BƯỚC 3: EDA — Glucose theo Outcome")
    print("=" * 70)
    fig, ax = plt.subplots(figsize=(7, 5))
    for outcome, color, label in [
        (0, "#4C72B0", "Không tiểu đường (0)"),
        (1, "#C44E52", "Tiểu đường (1)"),
    ]:
        vals = df_clean.loc[df_clean["Outcome"] == outcome, "Glucose"].dropna()
        ax.hist(vals, bins=25, alpha=0.6, color=color, label=label)
    ax.set_xlabel("Glucose")
    ax.set_ylabel("Số bệnh nhân")
    ax.set_title("Phân bố Glucose theo nhóm Outcome")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "glucose_histogram.png"), dpi=130)
    plt.close(fig)
    print("Đã lưu reports/glucose_histogram.png — Glucose tách nhóm khá rõ theo Outcome.")


# ============================================================
# BƯỚC 4 — Chia train/test stratify, random_state=42
# ============================================================
def split_data(df_clean):
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
    return X_train, X_test, y_train, y_test


# ============================================================
# BƯỚC 5 — Baseline: DummyClassifier(most_frequent)
# ============================================================
def run_baseline(X_train, y_train, X_test, y_test, results_log):
    print("\n" + "=" * 70)
    print("BƯỚC 5: BASELINE (DummyClassifier)")
    print("=" * 70)
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    baseline_recall = recall_score(y_test, y_pred_dummy)
    baseline_acc = accuracy_score(y_test, y_pred_dummy)

    print(f"Baseline recall = {baseline_recall:.3f}, accuracy = {baseline_acc:.3f}")
    results_log["baseline"] = {"recall": baseline_recall, "accuracy": baseline_acc}
    return baseline_recall, baseline_acc


# ============================================================
# BƯỚC 6-7 — KNN K=5 có/không chuẩn hoá, bảng so sánh
# ============================================================
def run_knn_scaled_vs_unscaled(
    X_train, y_train, X_test, y_test, baseline_recall, baseline_acc, results_log
):
    print("\n" + "=" * 70)
    print("BƯỚC 6: KNN K=5 MẶC ĐỊNH (CÓ chuẩn hoá)")
    print("=" * 70)
    pipe_k5 = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ])
    pipe_k5.fit(X_train, y_train)
    y_pred_k5 = pipe_k5.predict(X_test)
    recall_k5 = recall_score(y_test, y_pred_k5)
    acc_k5 = accuracy_score(y_test, y_pred_k5)
    print(f"K=5 (chuẩn hoá): recall = {recall_k5:.3f}, accuracy = {acc_k5:.3f}")
    results_log["knn_k5_scaled"] = {"recall": recall_k5, "accuracy": acc_k5}

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
    comparison_table.to_csv(os.path.join(REPORTS_DIR, "comparison_scale_vs_noscale.csv"))


# ============================================================
# BƯỚC 8 — GridSearchCV dò K, weights, metric (tối ưu recall)
# Pipeline (impute+scale+knn) đặt TRONG GridSearchCV -> chống rò rỉ dữ liệu
# ============================================================
def run_grid_search(X_train, y_train, results_log):
    print("\n" + "=" * 70)
    print("BƯỚC 8: GRIDSEARCHCV (tối ưu RECALL)")
    print("=" * 70)
    grid_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])
    grid = {
        "knn__n_neighbors": list(range(1, 32, 2)),
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }
    gs = GridSearchCV(
        grid_pipe, grid,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="recall", n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    print(f"Best params: {gs.best_params_}")
    print(f"Best CV recall: {gs.best_score_:.3f}")
    results_log["gridsearch_best_params"] = gs.best_params_
    results_log["gridsearch_best_cv_recall"] = gs.best_score_
    return gs


# ============================================================
# BƯỚC 9 — Vẽ Recall & Accuracy theo K (weights='uniform', metric tốt nhất)
# ============================================================
def plot_recall_accuracy_vs_k(X_train, y_train, gs):
    print("\n" + "=" * 70)
    print("BƯỚC 9: ĐƯỜNG RECALL/ACCURACY THEO K")
    print("=" * 70)
    best_weights = gs.best_params_["knn__weights"]
    best_metric = gs.best_params_["knn__metric"]
    k_values = list(range(1, 32, 2))
    recalls_by_k, accs_by_k = [], []
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    for k in k_values:
        p = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights=best_weights, metric=best_metric)),
        ])
        fold_recalls, fold_accs = [], []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            p.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            pred = p.predict(X_train.iloc[val_idx])
            fold_recalls.append(recall_score(y_train.iloc[val_idx], pred))
            fold_accs.append(accuracy_score(y_train.iloc[val_idx], pred))
        recalls_by_k.append(np.mean(fold_recalls))
        accs_by_k.append(np.mean(fold_accs))

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
    fig.savefig(os.path.join(REPORTS_DIR, "recall_theo_K.png"), dpi=130)
    plt.close(fig)
    print("Đã lưu reports/recall_theo_K.png")
    print("Giải thích hình dạng: K nhỏ -> variance cao, dễ overfit theo nhiễu cục bộ "
          "(recall dao động mạnh). K lớn -> bias cao, model 'mượt' quá mức, "
          "nghiêng về lớp đa số (âm tính) nên recall lớp dương tính giảm dần.")


# ============================================================
# Dò ngưỡng phân loại (threshold) bằng CV trên TRAIN, ưu tiên Recall
# với ràng buộc Precision >= MIN_PRECISION (nếu không đạt thì chọn F1 cao nhất)
#
# GIẢ ĐỊNH NGHIỆP VỤ (cần phòng khám xác nhận lại): bỏ sót một ca dương tính
# thật (FN) tốn kém hơn nhiều so với một báo động giả (FP, chỉ tốn thêm một
# lượt xét nghiệm khẳng định) -> ưu tiên recall. Nhưng nếu precision quá thấp,
# số ca phải gọi lại xét nghiệm sẽ vượt quá năng lực vận hành của phòng khám
# -> đặt sàn precision tối thiểu để giữ số báo động giả trong tầm kiểm soát.
# MIN_PRECISION = 0.50 là giả định làm việc (mỗi 2 ca cảnh báo có tối đa 1 ca
# âm tính thật), KHÔNG dựa trên số liệu thực tế của phòng khám nào.
#
# Về lưới threshold: model là KNN với weights="uniform" nên predict_proba chỉ
# nhận các giá trị rời rạc dạng k/n_neighbors (k = 0..n_neighbors). Quét một
# lưới liên tục bước 0.01 sẽ tạo ảo giác về độ mịn: rất nhiều threshold liền
# kề cho ra cùng một tập dự đoán. Thay vào đó, các threshold ứng viên được
# suy ra trực tiếp từ các giá trị xác suất thực sự xuất hiện trên các fold
# validation (điểm giữa hai giá trị xác suất liền kề khác nhau) -> mỗi ứng
# viên đại diện cho một hành vi phân loại thực sự khác nhau. Cách này cũng
# tổng quát cho mọi mô hình (kể cả Logistic Regression với proba liên tục).
# ============================================================
def find_best_threshold(model, X_train, y_train, min_precision=0.50):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fold_data = []
    all_probs = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        fold_model = clone(model)
        fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        y_prob = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
        y_true = y_train.iloc[val_idx].to_numpy()
        fold_data.append((y_prob, y_true))
        all_probs.append(y_prob)

    unique_probs = np.unique(np.concatenate(all_probs))
    if len(unique_probs) > 1:
        midpoints = (unique_probs[:-1] + unique_probs[1:]) / 2
    else:
        midpoints = unique_probs
    # Thêm 0.0 và 1.0 để bao trọn hai đầu (toàn bộ dương tính / toàn bộ âm tính)
    thresholds = np.unique(np.concatenate([[0.0], midpoints, [1.0]]))

    threshold_results = []
    for threshold in thresholds:
        fold_recalls, fold_precisions = [], []
        for y_prob, y_true in fold_data:
            y_pred = (y_prob >= threshold).astype(int)
            fold_recalls.append(recall_score(y_true, y_pred))
            fold_precisions.append(precision_score(y_true, y_pred, zero_division=0))
        threshold_results.append({
            "threshold": float(threshold),
            "recall": float(np.mean(fold_recalls)),
            "precision": float(np.mean(fold_precisions)),
        })

    threshold_df = pd.DataFrame(threshold_results)
    valid = threshold_df[threshold_df["precision"] >= min_precision]

    if valid.empty:
        threshold_df["f1"] = (
            2 * threshold_df["precision"] * threshold_df["recall"]
            / (threshold_df["precision"] + threshold_df["recall"]).replace(0, np.nan)
        ).fillna(0.0)
        best_row = threshold_df.loc[threshold_df["f1"].idxmax()]
    else:
        best_row = valid.loc[valid["recall"].idxmax()]

    return (
        float(best_row["threshold"]),
        float(best_row["recall"]),
        float(best_row["precision"]),
        threshold_df,
    )


# ============================================================
# BƯỚC 10 — Chạm tập TEST 1 lần với model tốt nhất, ma trận nhầm lẫn
# ============================================================
def evaluate_on_test(gs, X_train, y_train, X_test, y_test, results_log):
    print("\n" + "=" * 70)
    print("BƯỚC 10: ĐÁNH GIÁ TRÊN TẬP TEST (1 LẦN DUY NHẤT)")
    print("=" * 70)
    best_model = gs.best_estimator_

    best_threshold, threshold_cv_recall, threshold_cv_precision, threshold_df = (
        find_best_threshold(best_model, X_train, y_train)
    )
    print(f"Best threshold = {best_threshold:.2f}")
    print(f"CV Recall tại threshold = {threshold_cv_recall:.3f}")
    print(f"CV Precision tại threshold = {threshold_cv_precision:.3f}")
    print(f"Số điểm vận hành (threshold) thực sự khác nhau đã xét: {len(threshold_df)}")

    y_prob_best = best_model.predict_proba(X_test)[:, 1]
    y_pred_best = (y_prob_best >= best_threshold).astype(int)

    test_recall = recall_score(y_test, y_pred_best)
    test_precision = precision_score(y_test, y_pred_best)
    test_f1 = f1_score(y_test, y_pred_best)
    test_acc = accuracy_score(y_test, y_pred_best)
    test_pr_auc = average_precision_score(y_test, y_prob_best)

    cm = confusion_matrix(y_test, y_pred_best)
    # cm[0][0]=TN, cm[0][1]=FP, cm[1][0]=FN, cm[1][1]=TP

    print(f"Recall    = {test_recall:.3f}")
    print(f"Precision = {test_precision:.3f}")
    print(f"F1        = {test_f1:.3f}")
    print(f"Accuracy  = {test_acc:.3f}")
    print(f"PR-AUC    = {test_pr_auc:.3f}")
    print(f"Threshold = {best_threshold:.2f}")
    print("\nMa trận nhầm lẫn:")
    print(cm)
    print("\n" + classification_report(y_test, y_pred_best, target_names=["Âm tính", "Dương tính"]))

    results_log["test_best_model"] = {
        "recall": float(test_recall),
        "precision": float(test_precision),
        "f1": float(test_f1),
        "accuracy": float(test_acc),
        "pr_auc": float(test_pr_auc),
        "threshold": float(best_threshold),
        "confusion_matrix": cm.tolist(),
        "params": gs.best_params_,
    }
    results_log["threshold_tuning"] = {
        "best_threshold": float(best_threshold),
        "cv_recall": float(threshold_cv_recall),
        "cv_precision": float(threshold_cv_precision),
        "min_precision_constraint": 0.50,
        "n_candidate_thresholds": int(len(threshold_df)),
        "candidates": threshold_df.round(4).to_dict(orient="records"),
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
    fig.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=130)
    plt.close(fig)
    print("Đã lưu reports/confusion_matrix.png")

    return best_model


def k1_sanity_check(X_train, y_train, results_log):
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
# BƯỚC 11 — So sánh KNN với Logistic Regression (chỉ TRAIN + 5-fold CV)
# ============================================================
def compare_knn_vs_logreg(gs, X_train, y_train, results_log):
    print("\n" + "=" * 70)
    print("BƯỚC 11: SO SÁNH KNN vs LOGISTIC REGRESSION")
    print("=" * 70)

    logreg_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        )),
    ])
    cv_compare = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"recall": "recall", "precision": "precision", "f1": "f1", "accuracy": "accuracy"}

    knn_cv = cross_validate(gs.best_estimator_, X_train, y_train, cv=cv_compare,
                             scoring=scoring, n_jobs=-1)
    lr_cv = cross_validate(logreg_pipe, X_train, y_train, cv=cv_compare,
                            scoring=scoring, n_jobs=-1)

    final_compare = pd.DataFrame({
        "KNN (tối ưu)": [
            knn_cv["test_recall"].mean(), knn_cv["test_precision"].mean(),
            knn_cv["test_f1"].mean(), knn_cv["test_accuracy"].mean(),
        ],
        "Logistic Regression": [
            lr_cv["test_recall"].mean(), lr_cv["test_precision"].mean(),
            lr_cv["test_f1"].mean(), lr_cv["test_accuracy"].mean(),
        ],
    }, index=["Recall", "Precision", "F1", "Accuracy"]).round(3)

    print(final_compare.to_string())
    final_compare.to_csv(os.path.join(REPORTS_DIR, "comparison_knn_vs_logreg.csv"))
    results_log["knn_vs_logreg_cv"] = final_compare.to_dict()


def save_outputs(best_model, results_log):
    joblib.dump(best_model, os.path.join(MODELS_DIR, "knn_pipeline.joblib"))
    with open(os.path.join(REPORTS_DIR, "results_log.json"), "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2, default=str)
    print("\n" + "=" * 70)
    print("HOÀN TẤT. Model đã lưu: models/knn_pipeline.joblib")
    print("=" * 70)


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    results_log = {}

    df_clean = load_and_clean_data(results_log)
    eda_glucose_histogram(df_clean)
    X_train, X_test, y_train, y_test = split_data(df_clean)

    baseline_recall, baseline_acc = run_baseline(X_train, y_train, X_test, y_test, results_log)
    run_knn_scaled_vs_unscaled(
        X_train, y_train, X_test, y_test, baseline_recall, baseline_acc, results_log
    )

    gs = run_grid_search(X_train, y_train, results_log)
    plot_recall_accuracy_vs_k(X_train, y_train, gs)

    best_model = evaluate_on_test(gs, X_train, y_train, X_test, y_test, results_log)
    k1_sanity_check(X_train, y_train, results_log)
    compare_knn_vs_logreg(gs, X_train, y_train, results_log)

    save_outputs(best_model, results_log)


if __name__ == "__main__":
    main()