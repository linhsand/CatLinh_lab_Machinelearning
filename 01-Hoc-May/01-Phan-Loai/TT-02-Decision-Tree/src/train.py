"""
TT-02 — DECISION TREE: Dự đoán nhân viên nghỉ việc & rút ra quy tắc cho phòng Nhân sự
=======================================================================================
Dataset : IBM HR Analytics Employee Attrition (1.470 dòng x 35 cột)
Thuật toán : Decision Tree (CART) — sklearn

Chạy: python src/train.py
Kết quả: models/tree.joblib, models/tree_baseline_overfit.joblib,
         reports/*.png, reports/rules.txt, reports/feature_importance.csv,
         reports/stability_check.txt, reports/RULES_NHAN_SU.md
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def log(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def export_pruned_rules(tree, feature_names, class_labels=(0, 1)):
    """Giống export_text của sklearn nhưng GỘP hai nhánh con cùng là lá và cùng
    dự đoán một lớp thành một dòng "class: X" duy nhất — loại luật thừa
    (hai lá con khác điều kiện nhưng cùng kết luận thì tách ra không có ý nghĩa
    diễn giải, chỉ làm luật dài hơn)."""
    tree_ = tree.tree_
    feat_name = [
        feature_names[i] if i != -2 else "undefined!" for i in tree_.feature
    ]

    def is_leaf(node):
        return tree_.children_left[node] == -1 and tree_.children_right[node] == -1

    def leaf_class(node):
        return class_labels[int(np.argmax(tree_.value[node]))]

    lines = []

    def recurse(node, depth):
        indent = "|   " * depth
        left, right = tree_.children_left[node], tree_.children_right[node]

        if is_leaf(node):
            lines.append(f"{indent}|--- class: {leaf_class(node)}")
            return

        if is_leaf(left) and is_leaf(right) and leaf_class(left) == leaf_class(right):
            # Hai nhánh con là lá và cùng lớp -> luật thừa, gộp lại thành 1 lá
            lines.append(f"{indent}|--- class: {leaf_class(left)}  (gộp — cả 2 nhánh đều dự đoán như nhau)")
            return

        name, thresh = feat_name[node], tree_.threshold[node]
        lines.append(f"{indent}|--- {name} <= {thresh:.2f}")
        recurse(left, depth + 1)
        lines.append(f"{indent}|--- {name} >  {thresh:.2f}")
        recurse(right, depth + 1)

    recurse(0, 0)
    return "\n".join(lines)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # BƯỚC 1 — Đọc dữ liệu & phát hiện cột vô dụng (df.nunique())
    # -----------------------------------------------------------------------
    log("BƯỚC 1: Đọc dữ liệu & kiểm tra df.nunique()")
    df = pd.read_csv(DATA_PATH)
    print(f"Kích thước dữ liệu: {df.shape}")

    nunique = df.nunique()
    useless_cols = nunique[nunique == 1].index.tolist()
    print("Các cột chỉ có 1 giá trị duy nhất (0 thông tin):", useless_cols)

    DROP_COLS = list(set(useless_cols + ["EmployeeNumber"]))
    print("=> Sẽ loại bỏ:", DROP_COLS)

    df = df.drop(columns=DROP_COLS)
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # Kiểm tra 2 biến nhạy cảm (đạo đức) — sẽ xem lại khi phân tích feature importance
    SENSITIVE_COLS = ["Gender", "MaritalStatus"]

    y = df["Attrition"]
    X = df.drop(columns=["Attrition"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    print(f"\nSố cột phân loại: {len(cat_cols)} -> {cat_cols}")
    print(f"Số cột số     : {len(num_cols)}")

    # -----------------------------------------------------------------------
    # BƯỚC 2 — Mã hoá biến phân loại trong Pipeline + chia train/test
    # -----------------------------------------------------------------------
    log("BƯỚC 2: Mã hoá OneHotEncoder trong Pipeline + train/test split")

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="passthrough",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]} mẫu | Test: {X_test.shape[0]} mẫu")
    print(f"Tỉ lệ Attrition=Yes  train: {y_train.mean():.3f}  test: {y_test.mean():.3f}")

    # Tách thêm một tập VALIDATION từ chính X_train (không đụng tới X_test) —
    # dùng riêng cho việc dò max_depth ở BƯỚC 6, để test set chỉ được dùng
    # đúng MỘT lần, cho đánh giá cuối cùng ở BƯỚC 7.
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train
    )
    print(f"(Riêng cho dò max_depth) Train2: {X_tr2.shape[0]} mẫu | Validation: {X_val.shape[0]} mẫu")

    # -----------------------------------------------------------------------
    # BƯỚC 3 — EDA nhanh: tỉ lệ nghỉ theo OverTime, JobRole, YearsAtCompany
    # -----------------------------------------------------------------------
    log("BƯỚC 3: EDA — tỉ lệ nghỉ việc theo OverTime / JobRole / YearsAtCompany")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    ot_rate = df.groupby("OverTime")["Attrition"].mean()
    ot_rate.plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Tỉ lệ nghỉ việc theo OverTime")
    axes[0].set_ylabel("Tỉ lệ Attrition")

    role_rate = df.groupby("JobRole")["Attrition"].mean().sort_values(ascending=False)
    role_rate.plot(kind="bar", ax=axes[1], color="#55A868")
    axes[1].set_title("Tỉ lệ nghỉ việc theo JobRole")
    axes[1].tick_params(axis="x", rotation=75)

    bins = [0, 2, 5, 10, 20, 100]
    labels = ["0-2", "3-5", "6-10", "11-20", "20+"]
    df["YearsBin"] = pd.cut(df["YearsAtCompany"], bins=bins, labels=labels)
    years_rate = df.groupby("YearsBin", observed=True)["Attrition"].mean()
    years_rate.plot(kind="bar", ax=axes[2], color="#C44E52")
    axes[2].set_title("Tỉ lệ nghỉ việc theo YearsAtCompany (nhóm)")
    df.drop(columns=["YearsBin"], inplace=True)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "eda_attrition_rate.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Đã lưu reports/eda_attrition_rate.png")
    print("\nTỉ lệ nghỉ theo OverTime:\n", ot_rate)
    print("\nTỉ lệ nghỉ theo JobRole (giảm dần):\n", role_rate)

    # -----------------------------------------------------------------------
    # BƯỚC 4 — Baseline: DummyClassifier
    # -----------------------------------------------------------------------
    log("BƯỚC 4: Baseline DummyClassifier")

    dummy_pipe = Pipeline([("prep", preprocess), ("clf", DummyClassifier(strategy="most_frequent"))])
    dummy_pipe.fit(X_train, y_train)
    dummy_pred = dummy_pipe.predict(X_test)
    dummy_acc = accuracy_score(y_test, dummy_pred)
    dummy_f1 = f1_score(y_test, dummy_pred)
    print(f"Baseline (luôn đoán 'Ở lại') — Accuracy: {dummy_acc:.4f} | F1: {dummy_f1:.4f}")
    print("=> Đây là bẫy kinh điển: accuracy cao (~84%) nhưng F1 = 0, vô dụng với lớp thiểu số.")

    # -----------------------------------------------------------------------
    # BƯỚC 5 — Cây KHÔNG giới hạn độ sâu -> chứng minh overfit
    # -----------------------------------------------------------------------
    log("BƯỚC 5: Cây max_depth=None — chứng minh overfit")

    overfit_pipe = Pipeline(
        [("prep", preprocess), ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]
    )
    overfit_pipe.fit(X_train, y_train)
    train_acc_full = accuracy_score(y_train, overfit_pipe.predict(X_train))
    test_acc_full = accuracy_score(y_test, overfit_pipe.predict(X_test))
    n_leaves_full = overfit_pipe.named_steps["clf"].get_n_leaves()
    depth_full = overfit_pipe.named_steps["clf"].get_depth()

    print(f"max_depth=None -> độ sâu thực tế: {depth_full}, số lá: {n_leaves_full}")
    print(f"Accuracy TRAIN: {train_acc_full:.4f}")
    print(f"Accuracy TEST : {test_acc_full:.4f}")
    print(f"=> Chênh lệch train-test: {train_acc_full - test_acc_full:.4f}  (OVERFIT rõ rệt)")

    joblib.dump(overfit_pipe, os.path.join(MODELS_DIR, "tree_baseline_overfit.joblib"))

    # -----------------------------------------------------------------------
    # BƯỚC 6 — Đường accuracy train/validation theo max_depth = 1..20
    # -----------------------------------------------------------------------
    log("BƯỚC 6: Vẽ đường accuracy train/validation theo max_depth")
    print("Dùng riêng tập VALIDATION (tách từ train) để dò max_depth — KHÔNG đụng tới test set.")

    depths = list(range(1, 21))
    train_accs, val_accs = [], []
    for d in depths:
        p = Pipeline(
            [
                ("prep", preprocess),
                ("clf", DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        )
        p.fit(X_tr2, y_tr2)
        train_accs.append(accuracy_score(y_tr2, p.predict(X_tr2)))
        val_accs.append(accuracy_score(y_val, p.predict(X_val)))

    overfit_start = None
    gap = [tr - va for tr, va in zip(train_accs, val_accs)]
    for i, g in enumerate(gap):
        if g > 0.08:  # ngưỡng chênh lệch train-validation coi là bắt đầu overfit
            overfit_start = depths[i]
            break

    plt.figure(figsize=(9, 6))
    plt.plot(depths, train_accs, marker="o", label="Train accuracy")
    plt.plot(depths, val_accs, marker="o", label="Validation accuracy")
    if overfit_start:
        plt.axvline(overfit_start, color="red", linestyle="--", label=f"Bắt đầu overfit (depth={overfit_start})")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.title("Train vs Validation Accuracy theo max_depth")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(depths)
    plt.savefig(os.path.join(REPORTS_DIR, "overfit_theo_depth.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Đã lưu reports/overfit_theo_depth.png")
    print(f"Vùng cây bắt đầu overfit rõ (theo validation): khoảng max_depth = {overfit_start}")

    # -----------------------------------------------------------------------
    # BƯỚC 7 — GridSearchCV: max_depth, min_samples_leaf, criterion (scoring=f1)
    # -----------------------------------------------------------------------
    log("BƯỚC 7: GridSearchCV tối ưu theo F1")

    full_pipe = Pipeline(
        [
            ("prep", preprocess),
            ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )

    param_grid = {
        "clf__max_depth": [3, 4, 5, 6],
        "clf__min_samples_leaf": [20, 30, 40, 50],
        "clf__criterion": ["gini", "entropy"],
    }

    grid = GridSearchCV(full_pipe, param_grid, scoring="f1", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Tham số tốt nhất:", grid.best_params_)
    print(f"F1 (CV, train) tốt nhất: {grid.best_score_:.4f}")

    best_pipe = grid.best_estimator_
    best_pred = best_pipe.predict(X_test)
    best_acc = accuracy_score(y_test, best_pred)
    best_f1 = f1_score(y_test, best_pred)

    print(f"\nAccuracy TEST (mô hình tốt nhất): {best_acc:.4f}")
    print(f"F1 TEST        (mô hình tốt nhất): {best_f1:.4f}")
    print(f"So với baseline F1={dummy_f1:.4f}  ->  {'ĐẠT' if best_f1 > dummy_f1 else 'CHƯA ĐẠT'} tiêu chí F1 > baseline")
    print("\nClassification report (test):\n", classification_report(y_test, best_pred, target_names=["Ở lại", "Nghỉ"]))

    ConfusionMatrixDisplay.from_predictions(
        y_test, best_pred, display_labels=["Ở lại", "Nghỉ"], cmap="Blues"
    )
    plt.title("Ma trận nhầm lẫn — Cây tốt nhất (test set)")
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Đã lưu reports/confusion_matrix.png")

    joblib.dump(best_pipe, os.path.join(MODELS_DIR, "tree.joblib"))

    # -----------------------------------------------------------------------
    # BƯỚC 8 — Vẽ cây tốt nhất + export_text (đã lọc luật thừa)
    # -----------------------------------------------------------------------
    log("BƯỚC 8: Vẽ cây quyết định + xuất luật dạng văn bản (đã gộp luật thừa)")

    best_tree = best_pipe.named_steps["clf"]
    feature_names = best_pipe.named_steps["prep"].get_feature_names_out()
    feature_names_clean = [f.split("__")[-1] for f in feature_names]

    plt.figure(figsize=(26, 12))
    plot_tree(
        best_tree,
        feature_names=feature_names_clean,
        class_names=["Ở lại", "Nghỉ"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title(f"Cây quyết định tốt nhất (max_depth={grid.best_params_['clf__max_depth']})")
    plt.savefig(os.path.join(REPORTS_DIR, "cay_quyet_dinh.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Đã lưu reports/cay_quyet_dinh.png")

    # export_text "thô" (giữ lại để đối chiếu/tra cứu nếu cần)
    raw_rules_text = export_text(best_tree, feature_names=list(feature_names_clean))
    # Bản đã lọc: gộp các cặp lá con cùng lớp -> bỏ luật thừa
    rules_text = export_pruned_rules(best_tree, feature_names_clean)

    with open(os.path.join(REPORTS_DIR, "rules.txt"), "w", encoding="utf-8") as f:
        f.write(rules_text)
    print("Đã lưu reports/rules.txt (đã gộp các nhánh lá con trùng lớp)")

    n_raw_leaves = raw_rules_text.count("class:")
    n_pruned_leaves = rules_text.count("class:")
    print(f"Số dòng 'class:' trước khi gộp: {n_raw_leaves} -> sau khi gộp: {n_pruned_leaves}")
    print("\n--- Trích đoạn rules.txt (đã gộp) ---\n")
    print("\n".join(rules_text.splitlines()[:25]))

    # -----------------------------------------------------------------------
    # BƯỚC 9 — Feature importance top 10
    # -----------------------------------------------------------------------
    log("BƯỚC 9: Feature importance (top 10)")

    importances = pd.Series(best_tree.feature_importances_, index=feature_names_clean)
    importances = importances[importances > 0].sort_values(ascending=False)
    top10 = importances.head(10)
    print(top10)

    top10.to_csv(os.path.join(REPORTS_DIR, "feature_importance.csv"), header=["importance"])

    plt.figure(figsize=(9, 6))
    top10.sort_values().plot(kind="barh", color="#4C72B0")
    plt.title("Top 10 Feature Importance — Cây quyết định")
    plt.xlabel("Mức độ quan trọng")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Đã lưu reports/feature_importance.png")

    # Kiểm tra đạo đức: cây có dùng Gender / MaritalStatus làm nút chia không?
    sensitive_used = [f for f in importances.index if any(s in f for s in SENSITIVE_COLS)]
    print(f"\n[ĐẠO ĐỨC] Các nút chia liên quan Gender/MaritalStatus: {sensitive_used if sensitive_used else 'KHÔNG có'}")

    # -----------------------------------------------------------------------
    # BƯỚC 10 — 5 quy tắc cho phòng Nhân sự (tự sinh từ dữ liệu thực tế,
    #           lưu vào reports/RULES_NHAN_SU.md — file này TỰ script tạo ra,
    #           không còn trỏ tới một README.md không tồn tại)
    # -----------------------------------------------------------------------
    log("BƯỚC 10: Sinh 5 quy tắc cho phòng Nhân sự từ số liệu thực tế")

    overall_rate = df["Attrition"].mean()

    rule_masks = {
        "1. Nhân viên MỚI (TotalWorkingYears <= 2 năm kinh nghiệm)":
            df["TotalWorkingYears"] <= 2,
        "2. Nhân viên có làm THÊM GIỜ (OverTime = Yes)":
            df["OverTime"] == "Yes",
        "3. Cấp bậc thấp (JobLevel = 1) VÀ có làm thêm giờ":
            (df["JobLevel"] == 1) & (df["OverTime"] == "Yes"),
        "4. KHÔNG có cổ phần thưởng (StockOptionLevel = 0)":
            df["StockOptionLevel"] == 0,
        "5. Từng làm > 4 công ty trước đó VÀ tuổi <= 37":
            (df["NumCompaniesWorked"] > 4) & (df["Age"] <= 37),
    }

    rule_lines = [
        "# 5 QUY TẮC CHO PHÒNG NHÂN SỰ",
        "",
        f"(File này được sinh tự động bởi `src/train.py` từ dữ liệu thực tế — "
        f"tỉ lệ nghỉ việc trung bình toàn bộ tập dữ liệu là **{overall_rate:.1%}**, "
        "dùng làm mốc so sánh cho từng nhóm bên dưới. Các nhóm này bám theo các nút "
        "chia quan trọng nhất của cây quyết định — xem `cay_quyet_dinh.png` và "
        "`feature_importance.png`.)",
        "",
    ]
    for i, (desc, mask) in enumerate(rule_masks.items(), start=1):
        n = int(mask.sum())
        rate = df.loc[mask, "Attrition"].mean() if n > 0 else float("nan")
        delta = rate - overall_rate if n > 0 else float("nan")
        rule_lines.append(f"## {desc}")
        rule_lines.append(
            f"- Số nhân viên trong nhóm: {n} ({n / len(df):.1%} tổng số)"
        )
        rule_lines.append(
            f"- Tỉ lệ nghỉ việc của nhóm: {rate:.1%} "
            f"(chênh lệch so với trung bình chung: {delta:+.1%})"
        )
        rule_lines.append("")

    rules_md = "\n".join(rule_lines)
    with open(os.path.join(REPORTS_DIR, "RULES_NHAN_SU.md"), "w", encoding="utf-8") as f:
        f.write(rules_md)
    print("Đã lưu reports/RULES_NHAN_SU.md (5 quy tắc, số liệu tính trực tiếp từ dữ liệu)")

    # -----------------------------------------------------------------------
    # BƯỚC 11 — So sánh với Random Forest (để chuẩn bị cho TT-03)
    # -----------------------------------------------------------------------
    log("BƯỚC 11: So sánh nhanh với Random Forest")

    rf_pipe = Pipeline(
        [
            ("prep", preprocess),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    rf_pipe.fit(X_train, y_train)
    rf_pred = rf_pipe.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)

    print(f"Decision Tree  -> Accuracy: {best_acc:.4f} | F1: {best_f1:.4f}")
    print(f"Random Forest  -> Accuracy: {rf_acc:.4f} | F1: {rf_f1:.4f}")
    if rf_f1 > best_f1:
        print(
            f"=> Trên lần chạy này, Random Forest có F1 cao hơn cây đơn "
            f"({rf_f1:.3f} so với {best_f1:.3f}) nhưng KHÔNG còn đọc được luật trực tiếp "
            "như một cây đơn — đánh đổi giữa hiệu năng và khả năng giải thích."
        )
    else:
        print(
            f"=> Trên lần chạy này, cây đơn (F1={best_f1:.3f}) lại NHỈNH HƠN Random Forest "
            f"(F1={rf_f1:.3f}). Random Forest thường ổn định/ít phương sai hơn qua nhiều lần "
            "chia dữ liệu khác nhau, nhưng không phải lúc nào cũng thắng cây đơn về F1 trên "
            "một lần chia cụ thể — vẫn đánh đổi hiệu năng lấy khả năng giải thích khi chọn RF."
        )

    # -----------------------------------------------------------------------
    # MỞ RỘNG 1 — Kiểm tra độ ổn định: 10 cây với 10 random_state khác nhau
    # -----------------------------------------------------------------------
    log("MỞ RỘNG: Kiểm tra độ ỔN ĐỊNH của cây đơn (10 random_state)")

    root_features = []
    level2_features = []  # đặc trưng ở 2 nút con của gốc (nếu có)
    f1_scores = []
    for seed in range(10):
        p = Pipeline(
            [
                ("prep", preprocess),
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=grid.best_params_["clf__max_depth"],
                        min_samples_leaf=grid.best_params_["clf__min_samples_leaf"],
                        criterion=grid.best_params_["clf__criterion"],
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
        p.fit(X_train, y_train)
        tree_seed = p.named_steps["clf"]
        fn = p.named_steps["prep"].get_feature_names_out()
        t = tree_seed.tree_
        root_idx = t.feature[0]
        root_features.append(fn[root_idx].split("__")[-1])

        lvl2 = []
        for child in (t.children_left[0], t.children_right[0]):
            if child != -1 and t.feature[child] != -2:
                lvl2.append(fn[t.feature[child]].split("__")[-1])
        level2_features.append(tuple(sorted(lvl2)))

        f1_scores.append(f1_score(y_test, p.predict(X_test)))

    n_unique_root = len(set(root_features))
    n_unique_lvl2 = len(set(level2_features))

    root_desc = (
        f"luôn là {root_features[0]}" if n_unique_root == 1 else f"đổi ({n_unique_root}/10 giá trị khác nhau)"
    )
    lvl2_desc = (
        f"cũng ổn định (luôn là {level2_features[0]})"
        if n_unique_lvl2 == 1
        else f"đã bắt đầu đổi theo seed ({n_unique_lvl2}/10 tổ hợp khác nhau)"
    )

    stability_report = (
        f"Đặc trưng ở NÚT GỐC qua 10 random_state khác nhau:\n{root_features}\n"
        f"Số đặc trưng gốc khác nhau: {n_unique_root} / 10\n\n"
        f"Đặc trưng ở tầng 2 (2 nút con của gốc) qua 10 random_state:\n{level2_features}\n"
        f"Số tổ hợp tầng-2 khác nhau: {n_unique_lvl2} / 10\n\n"
        f"F1 dao động: min={min(f1_scores):.4f}  max={max(f1_scores):.4f}  "
        f"mean={np.mean(f1_scores):.4f}  std={np.std(f1_scores):.4f}\n\n"
        f"=> Kết luận thực nghiệm trên dữ liệu này: nút GỐC {root_desc}, "
        f"và cấu trúc tầng 2 {lvl2_desc}. F1 dao động "
        f"~{(max(f1_scores) - min(f1_scores)) * 100:.1f} điểm % — cho thấy một cây đơn vẫn nhạy với\n"
        "   cách khởi tạo ngẫu nhiên (dù cấu trúc trên cùng vẫn tương đối bền), và điểm F1 vẫn dao động\n"
        "   đủ để KHÔNG nên tin tuyệt đối vào một cây đơn khi diễn giải nguyên nhân — đây là động lực\n"
        "   cho Random Forest (TT-03).\n"
    )
    print(stability_report)
    with open(os.path.join(REPORTS_DIR, "stability_check.txt"), "w", encoding="utf-8") as f:
        f.write(stability_report)

    # -----------------------------------------------------------------------
    # TỔNG KẾT — lưu file metrics.json để tham chiếu nhanh
    # -----------------------------------------------------------------------
    summary = {
        "n_samples": int(df.shape[0]),
        "attrition_rate": float(y.mean()),
        "dropped_columns": DROP_COLS,
        "baseline": {"accuracy": dummy_acc, "f1": dummy_f1},
        "overfit_full_tree": {
            "depth": depth_full,
            "n_leaves": n_leaves_full,
            "train_accuracy": train_acc_full,
            "test_accuracy": test_acc_full,
        },
        "overfit_start_depth": overfit_start,
        "best_params": grid.best_params_,
        "best_model": {"accuracy": best_acc, "f1": best_f1, "cv_f1": grid.best_score_},
        "random_forest": {"accuracy": rf_acc, "f1": rf_f1},
        "sensitive_features_used_as_split": sensitive_used,
        "stability_unique_root_features": n_unique_root,
        "stability_unique_level2_combos": n_unique_lvl2,
    }

    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_default)

    log("HOÀN TẤT — Xem thư mục reports/ và models/")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_default))

    return summary


if __name__ == "__main__":
    main()