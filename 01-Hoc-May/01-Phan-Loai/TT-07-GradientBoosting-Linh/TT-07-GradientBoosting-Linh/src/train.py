"""
TT-07 — Gradient Boosting: Dự đoán thu nhập để chấm điểm hồ sơ vay tiêu dùng
============================================================================
Script huấn luyện độc lập (chạy: python src/train.py). Toàn bộ logic ở đây
cũng được trình bày lại theo từng bước, có giải thích, trong
notebooks/gradient_boosting_income.ipynb.
"""

import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "adult.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42

CAT_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
NUM_COLS = [
    "age", "education-num", "capital-gain", "capital-loss", "hours-per-week",
]


# ---------------------------------------------------------------------------
# 1. Đọc & làm sạch dữ liệu
# ---------------------------------------------------------------------------
def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Bẫy 1+2: giá trị thiếu ghi bằng ' ?' (có thể có/không có khoảng trắng
    # tuỳ nguồn tải) và chuỗi có khoảng trắng thừa ở đầu/cuối.
    # -> strip() TRƯỚC, rồi mới so sánh với "?" để chắc chắn bắt được cả 2 trường hợp.
    str_cols = df.select_dtypes(include="object").columns
    for c in str_cols:
        df[c] = df[c].str.strip()
    df[str_cols] = df[str_cols].replace("?", np.nan)

    # Bẫy 3: education và education-num là cùng một thông tin -> bỏ bản chữ,
    # giữ bản số (đã encode thứ tự sẵn, tiện cho model).
    df = df.drop(columns=["education"])
    CAT_COLS_LOCAL = [c for c in CAT_COLS if c != "education"]

    # Bẫy 4: fnlwgt là trọng số điều tra dân số, không phải đặc trưng cá nhân.
    df = df.drop(columns=["fnlwgt"])

    # Nhãn nhị phân
    df["income"] = df["income"].str.strip()
    df["target"] = (df["income"] == ">50K").astype(int)
    df = df.drop(columns=["income"])

    return df, CAT_COLS_LOCAL


def build_pipeline(cat_cols, num_cols, model):
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def main():
    df, cat_cols = load_and_clean(DATA_PATH)
    # hàng thiếu occupation/workclass/native-country -> để nguyên, OneHotEncoder
    # sẽ không xử lý NaN trực tiếp nên ta điền "Missing" cho các cột phân loại
    for c in cat_cols:
        df[c] = df[c].fillna("Missing")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = {}

    # ---- Baselines ----
    dummy = build_pipeline(cat_cols, NUM_COLS, DummyClassifier(strategy="most_frequent"))
    dummy.fit(X_train, y_train)
    results["Dummy (most_frequent)"] = average_precision_score(
        y_test, dummy.predict_proba(X_test)[:, 1]
    )

    dtree = build_pipeline(cat_cols, NUM_COLS, DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE))
    dtree.fit(X_train, y_train)
    results["Decision Tree (depth=6)"] = average_precision_score(
        y_test, dtree.predict_proba(X_test)[:, 1]
    )

    # ---- Gradient Boosting (tham số mặc định theo README) ----
    gb = build_pipeline(
        cat_cols, NUM_COLS,
        GradientBoostingClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=3,
            subsample=0.8, validation_fraction=0.1, n_iter_no_change=20,
            random_state=RANDOM_STATE,
        ),
    )
    t0 = time.time()
    gb.fit(X_train, y_train)
    gb_time = time.time() - t0
    gb_proba = gb.predict_proba(X_test)[:, 1]
    results["Gradient Boosting"] = average_precision_score(y_test, gb_proba)

    print("PR-AUC (Average Precision):")
    for k, v in results.items():
        print(f"  {k:28s}: {v:.4f}")
    print(f"ROC-AUC (GB): {roc_auc_score(y_test, gb_proba):.4f}")
    print(f"Accuracy (GB): {accuracy_score(y_test, gb.predict(X_test)):.4f}")

    # ---- Train/validation loss theo số cây ----
    gb_model = gb.named_steps["model"]
    Xtr_enc = gb.named_steps["pre"].transform(X_train)
    Xte_enc = gb.named_steps["pre"].transform(X_test)
    train_loss = [log_loss(y_train, p) for p in gb_model.staged_predict_proba(Xtr_enc)]
    test_loss = [log_loss(y_test, p) for p in gb_model.staged_predict_proba(Xte_enc)]
    best_iter = int(np.argmin(test_loss))

    plt.figure(figsize=(7, 5))
    plt.plot(train_loss, label="Train log loss")
    plt.plot(test_loss, label="Validation (test) log loss")
    plt.axvline(best_iter, color="red", linestyle="--", label=f"Điểm tốt nhất (cây #{best_iter})")
    plt.xlabel("Số cây (boosting stage)")
    plt.ylabel("Log loss")
    plt.title("Train vs Validation loss theo số cây — điểm bắt đầu overfit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "loss_theo_so_cay.png", dpi=120)
    plt.close()

    # ---- Lưới learning_rate x n_estimators (3x3) ----
    grid = {
        "learning_rate": [0.3, 0.1, 0.05],
        "n_estimators": [50, 200, 500],
    }
    grid_results = []
    for params in ParameterGrid(grid):
        m = build_pipeline(
            cat_cols, NUM_COLS,
            GradientBoostingClassifier(max_depth=3, random_state=RANDOM_STATE, **params),
        )
        m.fit(X_train, y_train)
        auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
        grid_results.append({**params, "roc_auc": auc})

    grid_df = pd.DataFrame(grid_results)
    pivot = grid_df.pivot(index="learning_rate", columns="n_estimators", values="roc_auc")
    pivot.to_csv(REPORTS_DIR / "lr_vs_nestimators.csv")

    plt.figure(figsize=(6, 5))
    im = plt.imshow(pivot.values, cmap="viridis", aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("n_estimators")
    plt.ylabel("learning_rate")
    plt.title("ROC-AUC theo learning_rate × n_estimators")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            plt.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", color="white")
    plt.colorbar(im, label="ROC-AUC")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "lr_vs_nestimators.png", dpi=120)
    plt.close()

    # ---- So sánh Bagging vs Boosting vs AdaBoost ----
    models_to_compare = {
        "Random Forest (Bagging)": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=RANDOM_STATE),
    }
    compare_rows = []
    for name, model in models_to_compare.items():
        pipe = build_pipeline(cat_cols, NUM_COLS, model)
        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0
        proba = pipe.predict_proba(X_test)[:, 1]
        n_params = sum(getattr(est, "tree_", None).node_count if hasattr(est, "tree_") else 0
                        for est in (model.estimators_ if hasattr(model, "estimators_") else []))
        compare_rows.append({
            "model": name,
            "pr_auc": average_precision_score(y_test, proba),
            "roc_auc": roc_auc_score(y_test, proba),
            "train_time_s": train_time,
            "n_estimators": getattr(model, "n_estimators", None),
        })
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(REPORTS_DIR / "bagging_vs_boosting_vs_adaboost.csv", index=False)
    print("\nSo sánh 3 thuật toán:\n", compare_df)

    # ---- GradientBoosting vs HistGradientBoosting (thời gian) ----
    hgb = build_pipeline(
        cat_cols, NUM_COLS,
        HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True, random_state=RANDOM_STATE),
    )
    t0 = time.time()
    hgb.fit(X_train, y_train)
    hgb_time = time.time() - t0
    hgb_auc = roc_auc_score(y_test, hgb.predict_proba(X_test)[:, 1])
    speed_row = {
        "GradientBoostingClassifier_s": gb_time,
        "HistGradientBoostingClassifier_s": hgb_time,
        "speedup_x": gb_time / hgb_time if hgb_time > 0 else None,
        "GB_roc_auc": roc_auc_score(y_test, gb_proba),
        "HGB_roc_auc": hgb_auc,
    }
    with open(REPORTS_DIR / "speed_comparison.json", "w") as f:
        json.dump(speed_row, f, indent=2)
    print("\nTốc độ:", speed_row)

    # ---- Kiểm tra thiên lệch theo sex và race ----
    def group_bias_table(y_true, y_pred, y_proba, group):
        out = pd.DataFrame({"y_true": y_true.values, "y_pred": y_pred, "y_proba": y_proba, "group": group.values})
        rows = []
        for g, sub in out.groupby("group"):
            rows.append({
                "group": g,
                "n": len(sub),
                "selection_rate(dự đoán >50K)": sub["y_pred"].mean(),
                "true_positive_rate": (sub[sub.y_true == 1]["y_pred"].mean() if (sub.y_true == 1).any() else np.nan),
                "false_positive_rate": (sub[sub.y_true == 0]["y_pred"].mean() if (sub.y_true == 0).any() else np.nan),
            })
        return pd.DataFrame(rows)

    y_pred_gb = gb.predict(X_test)
    bias_sex = group_bias_table(y_test, y_pred_gb, gb_proba, X_test["sex"])
    bias_race = group_bias_table(y_test, y_pred_gb, gb_proba, X_test["race"])
    bias_sex.to_csv(REPORTS_DIR / "bias_by_sex.csv", index=False)
    bias_race.to_csv(REPORTS_DIR / "bias_by_race.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(bias_sex["group"], bias_sex["selection_rate(dự đoán >50K)"], color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Tỉ lệ dự đoán >50K theo giới tính")
    axes[0].set_ylabel("Selection rate")
    axes[1].bar(bias_race["group"], bias_race["selection_rate(dự đoán >50K)"], color="#55A868")
    axes[1].set_title("Tỉ lệ dự đoán >50K theo chủng tộc")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "bias_by_group.png", dpi=120)
    plt.close()

    print("\nThiên lệch theo sex:\n", bias_sex)
    print("\nThiên lệch theo race:\n", bias_race)

    # ---- Lưu model ----
    joblib.dump(gb, MODELS_DIR / "gb_pipeline.joblib")
    print(f"\nĐã lưu model tại {MODELS_DIR / 'gb_pipeline.joblib'}")


if __name__ == "__main__":
    main()
