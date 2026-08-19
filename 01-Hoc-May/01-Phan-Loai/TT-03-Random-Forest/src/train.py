import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

df = pd.read_csv("data/bank-additional-full.csv", sep=";")
print(df.shape)        # kỳ vọng (41188, 21)
print(df["y"].value_counts(normalize=True))   # kỳ vọng ~11.3% "yes"

df["y"] = (df["y"].str.strip().str.lower() == "yes").astype(int)

#chứng minh rò rỉ dữ liệu từ cột duration

# Danh sách cột đặc trưng cơ bản (chưa xử lý categorical/one-hot)
cat_cols = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "day_of_week", "poutcome"]
num_cols = ["age", "campaign", "pdays", "previous",
            "emp.var.rate", "cons.price.idx", "cons.conf.idx",
            "euribor3m", "nr.employed"]

def quick_auc(df, use_duration: bool):
    cols = cat_cols + num_cols + (["duration"] if use_duration else [])
    X = pd.get_dummies(df[cols], columns=cat_cols)   # one-hot nhanh để demo
    y = df["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    rf = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                 class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, proba)

auc_no_duration = quick_auc(df, use_duration=False)
auc_with_duration = quick_auc(df, use_duration=True)
print(f"KHÔNG duration: {auc_no_duration:.4f}")   # kỳ vọng ~0.78
print(f"CÓ duration   : {auc_with_duration:.4f}") # kỳ vọng ~0.95

df["never_contacted"] = (df["pdays"] == 999).astype(int)
print(df["never_contacted"].mean())   # tỉ lệ khách chưa từng được liên hệ trước

for col in ["job", "education", "housing", "loan", "default"]:
    n = (df[col] == "unknown").sum()
    print(f"{col}: {n} dòng 'unknown'")
    

numeric_cols = num_cols + ["never_contacted"]   # KHÔNG có duration

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", numeric_cols),
])

def make_pipeline(model):
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

feature_cols = cat_cols + numeric_cols
X = df[feature_cols]
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

dummy = make_pipeline(DummyClassifier(strategy="stratified", random_state=42))
dummy.fit(X_train, y_train)
dummy_auc = roc_auc_score(y_test, dummy.predict_proba(X_test)[:, 1])

tree = make_pipeline(DecisionTreeClassifier(max_depth=8, class_weight="balanced",
                                             random_state=42))
tree.fit(X_train, y_train)
tree_auc = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])

print(f"Dummy: {dummy_auc:.4f}")   # ~0.50
print(f"Tree : {tree_auc:.4f}")    # ~0.78


rf_pipe = make_pipeline(
    RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=5,          # giảm overfit + giảm kích thước model
        class_weight="balanced_subsample",
        oob_score=True,
        n_jobs=-1,
        random_state=42,
    )
)
rf_pipe.fit(X_train, y_train)

oob_score = rf_pipe.named_steps["model"].oob_score_
rf_proba = rf_pipe.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_proba)

print(f"OOB score : {oob_score:.4f}")   # ~0.86-0.89
print(f"Test AUC  : {rf_auc:.4f}")      # ~0.79-0.81


X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

n_values = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500]
aucs = []
for n in n_values:
    rf = RandomForestClassifier(n_estimators=n, max_features="sqrt",
                                 min_samples_leaf=5,
                                 class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=42)
    rf.fit(X_train_enc, y_train)
    proba = rf.predict_proba(X_test_enc)[:, 1]
    aucs.append(roc_auc_score(y_test, proba))

plt.plot(n_values, aucs, marker="o")
plt.xlabel("n_estimators")
plt.ylabel("ROC-AUC")
plt.title("AUC theo số cây")
plt.savefig("reports/auc_theo_so_cay.png", dpi=150)
plt.show()

# Feature importance mặc định (đã bị thiên lệch)
feature_names = preprocessor.get_feature_names_out()
default_imp = pd.Series(
    rf_pipe.named_steps["model"].feature_importances_, index=feature_names
).sort_values(ascending=False)

# Permutation importance (đáng tin cậy hơn)
result = permutation_importance(
    rf_pipe, X_test, y_test, n_repeats=10,
    scoring="average_precision", random_state=42, n_jobs=-1
)
perm_imp = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)

print("Top 10 theo Gini (mặc định):")
print(default_imp.head(10))
print("\nTop 10 theo permutation:")
print(perm_imp.head(10))

def precision_at_k(y_true, y_proba, k):
    y_true = pd.Series(y_true).reset_index(drop=True)
    idx = np.argsort(y_proba)[-k:]        # k phần tử có xác suất cao nhất
    return y_true.iloc[idx].mean()

def lift_at_k(y_true, y_proba, k):
    base_rate = pd.Series(y_true).mean()   # tỉ lệ yes nếu gọi ngẫu nhiên
    return precision_at_k(y_true, y_proba, k) / base_rate

k = 5000
prec = precision_at_k(y_test, rf_proba, k)
lift = lift_at_k(y_test, rf_proba, k)
print(f"PRECISION@{k} = {prec:.4f}")   # ~0.16
print(f"LIFT@{k}      = {lift:.2f}x")  # ~1.4x


def cumulative_gain(y_true, y_proba, n_points=100):
    y_true = pd.Series(y_true).reset_index(drop=True)
    n = len(y_true)
    order = np.argsort(y_proba)[::-1]          # sắp xếp giảm dần theo xác suất
    y_sorted = y_true.iloc[order].values
    cum_positives = np.cumsum(y_sorted)
    total_positives = y_sorted.sum()

    ks = np.unique(np.linspace(1, n, n_points).astype(int))
    frac_called = ks / n
    frac_captured = cum_positives[ks - 1] / total_positives
    return frac_called, frac_captured

frac_called, frac_captured = cumulative_gain(y_test, rf_proba)

plt.plot(frac_called * 100, frac_captured * 100, label="Random Forest")
plt.plot([0, 100], [0, 100], "--", color="gray", label="Gọi ngẫu nhiên")
plt.xlabel("% khách được gọi")
plt.ylabel("% khách 'yes' bắt được")
plt.legend()
plt.savefig("reports/lift_curve.png", dpi=150)
plt.show()


print(f"Decision Tree đơn : {tree_auc:.4f}")
print(f"Random Forest     : {rf_auc:.4f}")

call_list = df.loc[X_test.index, ["age", "job", "marital", "education"]].copy()
call_list["xac_suat_dong_y"] = rf_proba
call_list = call_list.sort_values("xac_suat_dong_y", ascending=False).head(5000)
call_list.to_csv("outputs/danh_sach_goi_top5000.csv", index_label="customer_index")

joblib.dump(rf_pipe, "models/rf_pipeline.joblib", compress=3)