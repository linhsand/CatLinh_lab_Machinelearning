# ============================================================
# TT-05 — SVM Breast Cancer Classification
# ============================================================
# Project:
#   Classification of breast tumors as malignant / benign
#
# Dataset:
#   Breast Cancer Wisconsin (Diagnostic)
#   sklearn.datasets.load_breast_cancer
#
# Important:
#   0 = Malignant
#   1 = Benign
#
# This script:
#   1. Loads the dataset
#   2. Splits train/test using stratification
#   3. Compares SVM with and without scaling
#   4. Compares linear / RBF / polynomial kernels
#   5. Performs GridSearchCV
#   6. Creates C x Gamma heatmap
#   7. Counts support vectors
#   8. Selects a threshold targeting malignant recall >= 0.98
#   9. Evaluates the selected threshold on the test set
#  10. Compares SVM with Logistic Regression
#  11. Saves the best SVM pipeline
#  12. Saves the required report images
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_predict
)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    make_scorer
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# train.py is located in:
# TT-05-SVM/src/train.py
#
# Therefore:
# PROJECT_ROOT = TT-05-SVM

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Automatically create folders if they do not exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("TT-05 — SVM BREAST CANCER CLASSIFICATION")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nModels directory:")
print(MODELS_DIR)

print("\nReports directory:")
print(REPORTS_DIR)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("1. LOADING DATA")
print("=" * 70)

data = load_breast_cancer(as_frame=True)

X = data.data
y = data.target

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nTarget names:")
print(data.target_names)

print("\nTarget distribution:")
print(y.value_counts().sort_index())

print("\nIMPORTANT:")
print("0 = Malignant")
print("1 = Benign")


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("2. TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("\ny_train:")
print(y_train.value_counts())

print("\ny_test:")
print(y_test.value_counts())


# ============================================================
# 5. SVM WITHOUT SCALING
# ============================================================

print("\n" + "=" * 70)
print("3. SVM WITHOUT SCALING")
print("=" * 70)

svm_no_scale = SVC(
    kernel="rbf",
    C=1,
    gamma="scale",
    random_state=42
)

svm_no_scale.fit(X_train, y_train)

y_pred_no_scale = svm_no_scale.predict(X_test)

no_scale_accuracy = accuracy_score(
    y_test,
    y_pred_no_scale
)

no_scale_precision = precision_score(
    y_test,
    y_pred_no_scale,
    pos_label=0
)

no_scale_recall = recall_score(
    y_test,
    y_pred_no_scale,
    pos_label=0
)

no_scale_f1 = f1_score(
    y_test,
    y_pred_no_scale,
    pos_label=0
)

print(f"Accuracy: {no_scale_accuracy:.4f}")
print(f"Precision - Malignant: {no_scale_precision:.4f}")
print(f"Recall - Malignant: {no_scale_recall:.4f}")
print(f"F1 - Malignant: {no_scale_f1:.4f}")


# ============================================================
# 6. SVM WITH STANDARD SCALING
# ============================================================

print("\n" + "=" * 70)
print("4. SVM WITH STANDARD SCALING")
print("=" * 70)

svm_scaled = Pipeline([
    ("scale", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=1,
        gamma="scale",
        random_state=42
    ))
])

svm_scaled.fit(X_train, y_train)

y_pred_scaled = svm_scaled.predict(X_test)

scaled_accuracy = accuracy_score(
    y_test,
    y_pred_scaled
)

scaled_precision = precision_score(
    y_test,
    y_pred_scaled,
    pos_label=0
)

scaled_recall = recall_score(
    y_test,
    y_pred_scaled,
    pos_label=0
)

scaled_f1 = f1_score(
    y_test,
    y_pred_scaled,
    pos_label=0
)

print(f"Accuracy: {scaled_accuracy:.4f}")
print(f"Precision - Malignant: {scaled_precision:.4f}")
print(f"Recall - Malignant: {scaled_recall:.4f}")
print(f"F1 - Malignant: {scaled_f1:.4f}")


# ============================================================
# 7. CREATE REQUIRED IMAGE 1
#    scale_vs_noscale.png
# ============================================================

scale_comparison = pd.DataFrame({
    "Model": [
        "SVM - No Scaling",
        "SVM - StandardScaler"
    ],
    "Accuracy": [
        no_scale_accuracy,
        scaled_accuracy
    ],
    "Precision_Malignant": [
        no_scale_precision,
        scaled_precision
    ],
    "Recall_Malignant": [
        no_scale_recall,
        scaled_recall
    ],
    "F1_Malignant": [
        no_scale_f1,
        scaled_f1
    ]
})

print("\nScale comparison:")
print(scale_comparison.to_string(index=False))

# Plot only the main metrics required for comparison
metrics = [
    "Accuracy",
    "Precision_Malignant",
    "Recall_Malignant",
    "F1_Malignant"
]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    x - width / 2,
    scale_comparison.loc[0, metrics],
    width,
    label="SVM - No Scaling"
)

ax.bar(
    x + width / 2,
    scale_comparison.loc[1, metrics],
    width,
    label="SVM - StandardScaler"
)

ax.set_xlabel("Metric")
ax.set_ylabel("Score")
ax.set_title("SVM: Scaling vs No Scaling")
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=20)
ax.set_ylim(0, 1.05)
ax.legend()

plt.tight_layout()

scale_report_path = REPORTS_DIR / "scale_vs_noscale.png"

plt.savefig(
    scale_report_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print(scale_report_path)


# ============================================================
# 8. COMPARE THREE KERNELS
# ============================================================

print("\n" + "=" * 70)
print("5. KERNEL COMPARISON")
print("=" * 70)

kernels = [
    "linear",
    "rbf",
    "poly"
]

kernel_results = []

for kernel in kernels:

    model = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(
            kernel=kernel,
            C=1,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    kernel_results.append({
        "Kernel": kernel,
        "Accuracy": accuracy_score(
            y_test,
            pred
        ),
        "Precision_Malignant": precision_score(
            y_test,
            pred,
            pos_label=0
        ),
        "Recall_Malignant": recall_score(
            y_test,
            pred,
            pos_label=0
        ),
        "F1_Malignant": f1_score(
            y_test,
            pred,
            pos_label=0
        )
    })

kernel_results = pd.DataFrame(kernel_results)

print(kernel_results.to_string(index=False))


# ============================================================
# 9. CREATE REQUIRED IMAGE 2
#    kernel_comparison.png
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(kernels))
width = 0.20

ax.bar(
    x - 1.5 * width,
    kernel_results["Accuracy"],
    width,
    label="Accuracy"
)

ax.bar(
    x - 0.5 * width,
    kernel_results["Precision_Malignant"],
    width,
    label="Precision - Malignant"
)

ax.bar(
    x + 0.5 * width,
    kernel_results["Recall_Malignant"],
    width,
    label="Recall - Malignant"
)

ax.bar(
    x + 1.5 * width,
    kernel_results["F1_Malignant"],
    width,
    label="F1 - Malignant"
)

ax.set_xlabel("Kernel")
ax.set_ylabel("Score")
ax.set_title("SVM Kernel Comparison")
ax.set_xticks(x)
ax.set_xticklabels(kernels)
ax.set_ylim(0, 1.05)
ax.legend()

plt.tight_layout()

kernel_report_path = REPORTS_DIR / "kernel_comparison.png"

plt.savefig(
    kernel_report_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print(kernel_report_path)


# ============================================================
# 10. GRIDSEARCHCV
# ============================================================

print("\n" + "=" * 70)
print("6. GRIDSEARCHCV")
print("=" * 70)

# The target class is:
# 0 = Malignant
#
# Therefore, the scoring function must explicitly calculate
# recall for class 0.

malignant_recall = make_scorer(
    recall_score,
    pos_label=0
)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("svm", SVC(
        probability=True,
        class_weight="balanced",
        random_state=42
    ))
])

param_grid = {
    "svm__C": [
        0.1,
        1,
        10,
        100
    ],
    "svm__gamma": [
        "scale",
        0.001,
        0.01,
        0.1
    ],
    "svm__kernel": [
        "linear",
        "rbf",
        "poly"
    ]
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring=malignant_recall,
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

print("Best parameters:")
print(grid_search.best_params_)

print("\nBest CV malignant recall:")
print(f"{grid_search.best_score_:.6f}")


# ============================================================
# 11. CREATE REQUIRED IMAGE 3
#     C_gamma_heatmap.png
# ============================================================

cv_results = pd.DataFrame(
    grid_search.cv_results_
)

# Only RBF results are used for C x Gamma heatmap
rbf_results = cv_results[
    cv_results["param_svm__kernel"] == "rbf"
].copy()

rbf_results["gamma"] = (
    rbf_results["param_svm__gamma"].astype(str)
)

rbf_results["C"] = (
    rbf_results["param_svm__C"].astype(float)
)

heatmap_data = rbf_results.pivot(
    index="C",
    columns="gamma",
    values="mean_test_score"
)

print("\nRBF C x Gamma results:")
print(heatmap_data)

fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".4f",
    cmap="YlOrRd",
    ax=ax
)

ax.set_title(
    "SVM RBF - C × Gamma\n"
    "Mean CV Recall for Malignant Class"
)

ax.set_xlabel("Gamma")
ax.set_ylabel("C")

plt.tight_layout()

gamma_report_path = REPORTS_DIR / "C_gamma_heatmap.png"

plt.savefig(
    gamma_report_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print(gamma_report_path)


# ============================================================
# 12. BEST MODEL — TEST SET EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("7. BEST MODEL TEST EVALUATION")
print("=" * 70)

best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(
    X_test
)

best_accuracy = accuracy_score(
    y_test,
    y_pred_best
)

best_precision = precision_score(
    y_test,
    y_pred_best,
    pos_label=0
)

best_recall = recall_score(
    y_test,
    y_pred_best,
    pos_label=0
)

best_f1 = f1_score(
    y_test,
    y_pred_best,
    pos_label=0
)

print(f"Test Accuracy: {best_accuracy:.4f}")
print(f"Test Precision - Malignant: {best_precision:.4f}")
print(f"Test Recall - Malignant: {best_recall:.4f}")
print(f"Test F1 - Malignant: {best_f1:.4f}")


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("8. CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred_best
)

print(cm)

# IMPORTANT:
# 0 = Malignant
# 1 = Benign
#
# Therefore:
#
# cm[0,0] = malignant correctly detected = TP
# cm[0,1] = malignant missed = FN
# cm[1,0] = benign predicted malignant = FP
# cm[1,1] = benign correctly detected = TN

tp = cm[0, 0]
fn = cm[0, 1]
fp = cm[1, 0]
tn = cm[1, 1]

print("\nTP - Malignant correctly detected:", tp)
print("FN - Malignant missed:", fn)
print("FP - Benign predicted malignant:", fp)
print("TN - Benign correctly detected:", tn)

print(
    "\nMalignant cases missed (FN):",
    fn
)


# ============================================================
# 14. SUPPORT VECTORS
# ============================================================

print("\n" + "=" * 70)
print("9. SUPPORT VECTORS")
print("=" * 70)

best_svm = best_model.named_steps["svm"]

n_support = best_svm.n_support_

malignant_support = n_support[0]
benign_support = n_support[1]

total_support = n_support.sum()
total_train = len(X_train)

support_percentage = (
    total_support /
    total_train *
    100
)

print(
    "Support vectors - Malignant:",
    malignant_support
)

print(
    "Support vectors - Benign:",
    benign_support
)

print(
    "Total support vectors:",
    total_support
)

print(
    "Total training samples:",
    total_train
)

print(
    f"Support vectors percentage: "
    f"{support_percentage:.2f}%"
)


# ============================================================
# 15. THRESHOLD SELECTION
#     TARGET MALIGNANT RECALL >= 0.98
# ============================================================

print("\n" + "=" * 70)
print("10. THRESHOLD SELECTION")
print("=" * 70)

# Generate out-of-fold probabilities on training data.
# This avoids selecting the threshold directly from the test set.

oof_proba = cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=5,
    method="predict_proba",
    n_jobs=-1
)

# Probability of class 0 = Malignant
oof_malignant_proba = oof_proba[:, 0]

threshold_results = []

for threshold in np.arange(
    0.05,
    0.51,
    0.01
):

    y_pred_threshold = np.where(
        oof_malignant_proba >= threshold,
        0,
        1
    )

    recall = recall_score(
        y_train,
        y_pred_threshold,
        pos_label=0
    )

    precision = precision_score(
        y_train,
        y_pred_threshold,
        pos_label=0
    )

    threshold_results.append({
        "Threshold": threshold,
        "Recall_Malignant": recall,
        "Precision_Malignant": precision
    })

threshold_df = pd.DataFrame(
    threshold_results
)

valid_thresholds = threshold_df[
    threshold_df["Recall_Malignant"] >= 0.98
].copy()

if len(valid_thresholds) > 0:

    # Select the highest threshold that still
    # achieves the target recall.
    best_threshold_row = valid_thresholds.loc[
        valid_thresholds["Threshold"].idxmax()
    ]

    best_threshold = float(
        best_threshold_row["Threshold"]
    )

    print(
        "Selected threshold:",
        best_threshold
    )

    print(
        "OOF Recall:",
        f"{best_threshold_row['Recall_Malignant']:.6f}"
    )

    print(
        "OOF Precision:",
        f"{best_threshold_row['Precision_Malignant']:.6f}"
    )

else:

    best_threshold = 0.50

    print(
        "No threshold in the tested range "
        "achieved recall >= 0.98."
    )

    print(
        "Using default threshold:",
        best_threshold
    )


# ============================================================
# 16. APPLY SELECTED THRESHOLD TO TEST SET
# ============================================================

print("\n" + "=" * 70)
print("11. THRESHOLD TEST EVALUATION")
print("=" * 70)

test_proba = best_model.predict_proba(
    X_test
)

# Probability of malignant = class 0
test_malignant_proba = test_proba[:, 0]

y_pred_threshold = np.where(
    test_malignant_proba >= best_threshold,
    0,
    1
)

threshold_accuracy = accuracy_score(
    y_test,
    y_pred_threshold
)

threshold_precision = precision_score(
    y_test,
    y_pred_threshold,
    pos_label=0
)

threshold_recall = recall_score(
    y_test,
    y_pred_threshold,
    pos_label=0
)

threshold_f1 = f1_score(
    y_test,
    y_pred_threshold,
    pos_label=0
)

print(
    "Selected threshold:",
    best_threshold
)

print(
    f"Test Accuracy: "
    f"{threshold_accuracy:.4f}"
)

print(
    f"Test Precision - Malignant: "
    f"{threshold_precision:.4f}"
)

print(
    f"Test Recall - Malignant: "
    f"{threshold_recall:.4f}"
)

print(
    f"Test F1 - Malignant: "
    f"{threshold_f1:.4f}"
)


# ============================================================
# 17. THRESHOLD CONFUSION MATRIX
# ============================================================

cm_threshold = confusion_matrix(
    y_test,
    y_pred_threshold
)

tp_threshold = cm_threshold[0, 0]
fn_threshold = cm_threshold[0, 1]
fp_threshold = cm_threshold[1, 0]
tn_threshold = cm_threshold[1, 1]

print("\nThreshold confusion matrix:")
print(cm_threshold)

print(
    "\nTP - Malignant detected:",
    tp_threshold
)

print(
    "FN - Malignant missed:",
    fn_threshold
)

print(
    "FP - Benign predicted malignant:",
    fp_threshold
)

print(
    "TN - Benign correctly detected:",
    tn_threshold
)


# ============================================================
# 18. LOGISTIC REGRESSION COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("12. LOGISTIC REGRESSION COMPARISON")
print("=" * 70)

logreg = Pipeline([
    ("scale", StandardScaler()),
    ("logreg", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

logreg.fit(
    X_train,
    y_train
)

y_pred_logreg = logreg.predict(
    X_test
)

logreg_accuracy = accuracy_score(
    y_test,
    y_pred_logreg
)

logreg_precision = precision_score(
    y_test,
    y_pred_logreg,
    pos_label=0
)

logreg_recall = recall_score(
    y_test,
    y_pred_logreg,
    pos_label=0
)

logreg_f1 = f1_score(
    y_test,
    y_pred_logreg,
    pos_label=0
)

print(
    f"Accuracy: {logreg_accuracy:.4f}"
)

print(
    f"Precision - Malignant: "
    f"{logreg_precision:.4f}"
)

print(
    f"Recall - Malignant: "
    f"{logreg_recall:.4f}"
)

print(
    f"F1 - Malignant: "
    f"{logreg_f1:.4f}"
)


# ============================================================
# 19. FINAL MODEL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "SVM",
        "Logistic Regression"
    ],
    "Accuracy": [
        best_accuracy,
        logreg_accuracy
    ],
    "Precision_Malignant": [
        best_precision,
        logreg_precision
    ],
    "Recall_Malignant": [
        best_recall,
        logreg_recall
    ],
    "F1_Malignant": [
        best_f1,
        logreg_f1
    ]
})

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(index=False)
)


# ============================================================
# 20. SAVE BEST SVM MODEL
# ============================================================

print("\n" + "=" * 70)
print("13. SAVING MODEL")
print("=" * 70)

model_path = (
    MODELS_DIR /
    "svm_pipeline.joblib"
)

joblib.dump(
    best_model,
    model_path
)

print("Model saved:")
print(model_path)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    "\nBest SVM parameters:"
)

print(
    grid_search.best_params_
)

print(
    f"\nBest CV malignant recall: "
    f"{grid_search.best_score_:.4f}"
)

print(
    f"Test malignant recall: "
    f"{best_recall:.4f}"
)

print(
    f"Test malignant precision: "
    f"{best_precision:.4f}"
)

print(
    f"Malignant cases missed: "
    f"{fn}"
)

print(
    f"Support vectors: "
    f"{total_support}/{total_train} "
    f"({support_percentage:.2f}%)"
)

print(
    f"\nSelected threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Threshold test malignant recall: "
    f"{threshold_recall:.4f}"
)

print(
    f"Threshold test malignant precision: "
    f"{threshold_precision:.4f}"
)

print(
    f"Threshold malignant cases missed: "
    f"{fn_threshold}"
)


# ============================================================
# 22. CHECK REQUIRED FILES
# ============================================================

print("\n" + "=" * 70)
print("PROJECT OUTPUTS")
print("=" * 70)

required_files = [
    model_path,
    REPORTS_DIR / "scale_vs_noscale.png",
    REPORTS_DIR / "kernel_comparison.png",
    REPORTS_DIR / "C_gamma_heatmap.png"
]

for file_path in required_files:

    if file_path.exists():
        print("OK  :", file_path)

    else:
        print("ERROR:", file_path)

print("\nTraining completed.")
print("=" * 70)