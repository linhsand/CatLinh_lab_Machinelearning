"""
TT-03 — Random Forest — Ai sẽ mở sổ tiết kiệm?
Chạy: python src/train.py  (từ thư mục gốc project, hoặc từ bất kỳ đâu — path tự resolve theo vị trí file)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend không tương tác — bắt buộc để chạy headless (server/CI) không bị treo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# Đường dẫn — resolve theo vị trí file này (src/train.py -> project root),
# KHÔNG phụ thuộc cwd lúc gọi lệnh
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
REPORTS_DIR = ROOT / "reports"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"
for d in (REPORTS_DIR, OUTPUTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

CAT_COLS = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "day_of_week", "poutcome"]
NUM_COLS = ["age", "campaign", "pdays", "previous",
            "emp.var.rate", "cons.price.idx", "cons.conf.idx",
            "euribor3m", "nr.employed"]

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Chứng minh rò rỉ dữ liệu từ cột `duration`
#    SỬA: split TRƯỚC rồi mới one-hot (fit trên train, transform trên test)
#    thay vì pd.get_dummies trên toàn bộ df trước khi split.
# ---------------------------------------------------------------------------
def quick_auc(df: pd.DataFrame, use_duration: bool) -> float:
    num_cols = NUM_COLS + (["duration"] if use_duration else [])
    X = df[CAT_COLS + num_cols]
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pre = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", "passthrough", num_cols),
    ])
    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("model", RandomForestClassifier(
            n_estimators=300, max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ])
    pipe.fit(X_train, y_train)  # encoder chỉ nhìn thấy X_train, không rò rỉ sang test
    proba = pipe.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, proba)


# ---------------------------------------------------------------------------
# Precision@k / Lift@k
# ---------------------------------------------------------------------------
def precision_at_k(y_true, y_proba, k: int) -> float:
    y_true = pd.Series(y_true).reset_index(drop=True)
    idx = np.argsort(y_proba)[-k:]
    return y_true.iloc[idx].mean()


def lift_at_k(y_true, y_proba, k: int) -> float:
    base_rate = pd.Series(y_true).mean()
    return precision_at_k(y_true, y_proba, k) / base_rate


def cumulative_gain(y_true, y_proba, n_points: int = 100):
    y_true = pd.Series(y_true).reset_index(drop=True)
    n = len(y_true)
    order = np.argsort(y_proba)[::-1]
    y_sorted = y_true.iloc[order].values
    cum_positives = np.cumsum(y_sorted)
    total_positives = y_sorted.sum()

    ks = np.unique(np.linspace(1, n, n_points).astype(int))
    frac_called = ks / n
    frac_captured = cum_positives[ks - 1] / total_positives
    return frac_called, frac_captured


def main():
    metrics = {}

    df = pd.read_csv(DATA_PATH, sep=";")
    print(df.shape)
    print(df["y"].value_counts(normalize=True))

    df["y"] = (df["y"].str.strip().str.lower() == "yes").astype(int)

    # -----------------------------------------------------------------
    # Bước 2: chứng minh rò rỉ duration
    # -----------------------------------------------------------------
    auc_no_duration = quick_auc(df, use_duration=False)
    auc_with_duration = quick_auc(df, use_duration=True)
    print(f"KHÔNG duration: {auc_no_duration:.4f}")
    print(f"CÓ duration   : {auc_with_duration:.4f}")
    metrics["auc_no_duration"] = auc_no_duration
    metrics["auc_with_duration"] = auc_with_duration

    # -----------------------------------------------------------------
    # Bước 3-4: pdays=999 -> cờ never_contacted, xem qua 'unknown'
    # -----------------------------------------------------------------
    df["never_contacted"] = (df["pdays"] == 999).astype(int)
    print(df["never_contacted"].mean())

    unknown_counts = {}
    for col in ["job", "education", "housing", "loan", "default"]:
        n = int((df[col] == "unknown").sum())
        unknown_counts[col] = n
        print(f"{col}: {n} dòng 'unknown'")
    metrics["unknown_counts"] = unknown_counts

    # -----------------------------------------------------------------
    # Bước 5: pipeline chính thức — KHÔNG có duration
    # -----------------------------------------------------------------
    numeric_cols = NUM_COLS + ["never_contacted"]
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", "passthrough", numeric_cols),
    ])

    def make_pipeline(model):
        return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    feature_cols = CAT_COLS + numeric_cols
    X = df[feature_cols]
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # -----------------------------------------------------------------
    # Bước 6: baseline
    # -----------------------------------------------------------------
    dummy = make_pipeline(DummyClassifier(strategy="stratified", random_state=RANDOM_STATE))
    dummy.fit(X_train, y_train)
    dummy_auc = roc_auc_score(y_test, dummy.predict_proba(X_test)[:, 1])

    tree = make_pipeline(DecisionTreeClassifier(max_depth=8, class_weight="balanced",
                                                 random_state=RANDOM_STATE))
    tree.fit(X_train, y_train)
    tree_auc = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])

    print(f"Dummy: {dummy_auc:.4f}")
    print(f"Tree : {tree_auc:.4f}")
    metrics["dummy_auc"] = dummy_auc
    metrics["tree_auc"] = tree_auc

    # -----------------------------------------------------------------
    # Bước 7: Random Forest chính thức
    # -----------------------------------------------------------------
    rf_pipe = make_pipeline(
        RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            oob_score=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    )
    rf_pipe.fit(X_train, y_train)

    oob_score = rf_pipe.named_steps["model"].oob_score_
    rf_proba = rf_pipe.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba)

    print(f"OOB score : {oob_score:.4f}")
    print(f"Test AUC  : {rf_auc:.4f}")
    metrics["rf_oob_score"] = oob_score
    metrics["rf_test_auc"] = rf_auc

    # -----------------------------------------------------------------
    # Bước 8: đường AUC/OOB theo n_estimators
    # SỬA: chọn n_estimators bằng OOB score, KHÔNG dùng test AUC
    #      (dùng test để tune là rò rỉ thông tin từ test vào lựa chọn siêu tham số).
    #      Đồng thời SỬA: dùng lại preprocessor ĐÃ FIT bên trong rf_pipe,
    #      không tạo/fit lại preprocessor riêng (tránh hai bản encode lệch nhau).
    # -----------------------------------------------------------------
    fitted_pre = rf_pipe.named_steps["preprocess"]
    X_train_enc = fitted_pre.transform(X_train)

    n_values = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500]
    oob_scores = []
    for n in n_values:
        rf_n = RandomForestClassifier(n_estimators=n, max_features="sqrt",
                                       min_samples_leaf=5,
                                       class_weight="balanced_subsample",
                                       oob_score=True,
                                       n_jobs=-1, random_state=RANDOM_STATE)
        rf_n.fit(X_train_enc, y_train)
        oob_scores.append(rf_n.oob_score_)

    metrics["oob_by_n_estimators"] = dict(zip(n_values, oob_scores))

    plt.figure()
    plt.plot(n_values, oob_scores, marker="o")
    plt.xlabel("n_estimators")
    plt.ylabel("OOB score")
    plt.title("OOB score theo số cây")
    plt.savefig(REPORTS_DIR / "auc_theo_so_cay.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------
    # Bước 9: Gini importance vs permutation importance
    # SỬA: dùng lại fitted_pre.transform (không fit lại) + lưu biểu đồ
    #      reports/permutation_importance.png (trước đây chỉ print).
    # -----------------------------------------------------------------
    X_test_enc = fitted_pre.transform(X_test)

    feature_names = fitted_pre.get_feature_names_out()
    default_imp = pd.Series(
        rf_pipe.named_steps["model"].feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    result = permutation_importance(
        rf_pipe, X_test, y_test, n_repeats=10,
        scoring="average_precision", random_state=RANDOM_STATE, n_jobs=-1
    )
    perm_imp = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)

    print("Top 10 theo Gini (mặc định):")
    print(default_imp.head(10))
    print("\nTop 10 theo permutation:")
    print(perm_imp.head(10))

    metrics["top10_gini_importance"] = default_imp.head(10).round(4).to_dict()
    metrics["top10_permutation_importance"] = perm_imp.head(10).round(4).to_dict()

    top_perm = perm_imp.head(15).sort_values()
    plt.figure(figsize=(8, 6))
    top_perm.plot(kind="barh")
    plt.xlabel("Permutation importance (average_precision)")
    plt.title("Top 15 đặc trưng theo permutation importance")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "permutation_importance.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------
    # Bước 10: Precision@5000 / Lift@5000
    # SỬA: quy mô nghiệp vụ đúng — "gọi 5000/45188 khách" (~11%), KHÔNG PHẢI
    #      5000/8238 khách test (~61%). Đo trên test set nhưng scale k theo
    #      cùng tỉ lệ với quy mô nghiệp vụ thật, để precision/lift phản ánh
    #      đúng việc "chỉ chọn top ~11% khách có xác suất cao nhất".
    # -----------------------------------------------------------------
    k_business = 5000
    k_test = round(k_business * len(y_test) / len(df))

    prec = precision_at_k(y_test, rf_proba, k_test)
    lift = lift_at_k(y_test, rf_proba, k_test)
    print(f"PRECISION@{k_test} (tương đương top {k_business}/{len(df)} thật) = {prec:.4f}")
    print(f"LIFT@{k_test} = {lift:.2f}x")
    metrics["k_business"] = k_business
    metrics["k_test_equivalent"] = k_test
    metrics["precision_at_k_test"] = prec
    metrics["lift_at_k_test"] = lift

    # -----------------------------------------------------------------
    # Bước 11: Cumulative gain / lift curve
    # -----------------------------------------------------------------
    frac_called, frac_captured = cumulative_gain(y_test, rf_proba)

    plt.figure()
    plt.plot(frac_called * 100, frac_captured * 100, label="Random Forest")
    plt.plot([0, 100], [0, 100], "--", color="gray", label="Gọi ngẫu nhiên")
    plt.xlabel("% khách được gọi")
    plt.ylabel("% khách 'yes' bắt được")
    plt.legend()
    plt.savefig(REPORTS_DIR / "lift_curve.png", dpi=150)
    plt.close()

    print(f"Decision Tree đơn : {tree_auc:.4f}")
    print(f"Random Forest     : {rf_auc:.4f}")

    # -----------------------------------------------------------------
    # Sản phẩm bàn giao: danh_sach_goi_top5000.csv
    # SỬA: chấm điểm trên TOÀN BỘ 45.188 khách (df[feature_cols]), không
    #      chỉ trên 8.238 dòng test — vì đây là danh sách vận hành thật cho
    #      telesales (top 5000/45188 ≈ 11%), không phải để báo cáo metric
    #      (metric sạch đã tách riêng ở bước 10, chỉ dùng test set).
    # -----------------------------------------------------------------
    full_proba = rf_pipe.predict_proba(df[feature_cols])[:, 1]
    call_list = df[["age", "job", "marital", "education"]].copy()
    call_list["xac_suat_dong_y"] = full_proba
    call_list = call_list.sort_values("xac_suat_dong_y", ascending=False).head(k_business)
    call_list.to_csv(OUTPUTS_DIR / "danh_sach_goi_top5000.csv", index_label="customer_index")

    joblib.dump(rf_pipe, MODELS_DIR / "rf_pipeline.joblib", compress=3)

    # -----------------------------------------------------------------
    # Lưu toàn bộ metric ra file — trước đây chỉ print(), không có bằng
    # chứng nào tồn tại sau khi script chạy xong.
    # -----------------------------------------------------------------
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=float)

    print(f"\nĐã lưu metrics.json, các biểu đồ, danh sách gọi và model vào {ROOT}")


if __name__ == "__main__":
    main()