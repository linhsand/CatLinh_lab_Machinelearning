"""
TT-04 — LOGISTIC REGRESSION
Chẩn đoán nguy cơ bệnh tim — model GIẢI THÍCH ĐƯỢC cho bác sĩ

Script này thực hiện đầy đủ 12 bước trong README.md của bài tập:
  1. Xử lý dòng trùng lặp trước khi chia train/test
  2. One-hot encode biến phân loại
  3. EDA nhanh (lưu ra reports/)
  4. Pipeline: OneHot + StandardScaler + LogisticRegression
  5. Baseline: DummyClassifier
  6. Train + bảng Odds Ratio
  7. So sánh L1 vs L2
  8. Vẽ ROC + Precision-Recall
  9. Chọn ngưỡng theo yêu cầu recall >= 0.90
  10. Kiểm tra đa cộng tuyến (VIF)
  11. So sánh với SVM và Random Forest
  12. Diễn giải cho 1 ca bệnh cụ thể

Chạy: python src/train.py
Toàn bộ output số liệu được in ra màn hình VÀ ghi vào reports/run_log.txt
để tiện đọc lại và đưa vào README / báo cáo.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, classification_report,
    confusion_matrix, recall_score, precision_score
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Lưu ý: KHÔNG dùng warnings.filterwarnings("ignore") toàn cục ở đây nữa.
# Bản trước che luôn cả các cảnh báo hữu ích (vd ConvergenceWarning của
# liblinear khi C rất lớn) — những cảnh báo này có giá trị chẩn đoán, không
# nên bị nuốt âm thầm.
# Chỉ lọc CHÍNH XÁC 1 cảnh báo vô hại và lặp lại rất nhiều lần (mỗi fold CV
# một lần) gây nhiễu log: FutureWarning về việc sklearn >=1.8 sẽ đổi cách
# truyền tham số penalty=. Đây là API sẽ đổi ở phiên bản tương lai, không
# phải lỗi hay rủi ro của chính model, nên an toàn để lọc RIÊNG cảnh báo này.
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*'penalty' was deprecated.*"
)
# Hai cảnh báo dưới đây cũng bị lọc RIÊNG vì lý do tương tự: chúng đến từ nội
# bộ LogisticRegressionCV (mặc định l1_ratios, và cách sklearn sẽ đơn giản
# hoá thuộc tính fit_ trong bản 1.10), lặp lại mỗi fold CV -> rất nhiều dòng
# giống hệt nhau, không nói lên điều gì về chất lượng model hay dữ liệu của
# BÀI TẬP NÀY. Các cảnh báo có ý nghĩa thực sự (hội tụ, dữ liệu...) vẫn hiện.
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*l1_ratios will change.*"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*fitted attributes of LogisticRegressionCV.*"
)

# ---------------------------------------------------------------------------
# Đường dẫn dự án
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
REPORTS.mkdir(exist_ok=True, parents=True)
MODELS.mkdir(exist_ok=True, parents=True)

LOG_LINES = []


def log(msg=""):
    """In ra màn hình đồng thời lưu lại để ghi vào file log / đưa vào README."""
    print(msg)
    LOG_LINES.append(str(msg))


def section(title):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


RANDOM_STATE = 42

# Các cột phân loại được mã hoá bằng số trong dữ liệu gốc nhưng KHÔNG có thứ tự
# thực sự -> bắt buộc one-hot, nếu không model sẽ hiểu nhầm là cp=3 "lớn hơn" cp=1.
CATEGORICAL_COLS = ["cp", "restecg", "slope", "ca", "thal"]
# sex, fbs, exang đã là nhị phân 0/1 sẵn -> để nguyên (one-hot cũng được nhưng
# thừa 1 cột không cần thiết vì nhị phân vốn đã "one-hot" cho chính nó).
BINARY_COLS = ["sex", "fbs", "exang"]
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COL = "target"


# ===========================================================================
# BƯỚC 1-3: ĐỌC DỮ LIỆU, XỬ LÝ TRÙNG LẶP, EDA
# ===========================================================================
def load_and_clean_data():
    section("BƯỚC 1: ĐỌC DỮ LIỆU & XỬ LÝ DÒNG TRÙNG LẶP")

    df = pd.read_csv(DATA_PATH)
    log(f"Kích thước dữ liệu gốc: {df.shape[0]} dòng x {df.shape[1]} cột")

    n_dup = df.duplicated().sum()
    log(f"Số dòng trùng lặp phát hiện được: {n_dup}")

    # QUAN TRỌNG: phải drop_duplicates() TRƯỚC KHI chia train/test.
    # Lý do: nếu để trùng lặp lọt vào cả train và test, model "học thuộc lòng"
    # đúng dòng đó ở tập train rồi "đoán đúng" y hệt ở test -> AUC ảo cao bất
    # thường (rò rỉ dữ liệu - data leakage), không phản ánh khả năng tổng quát
    # hoá thực sự của model trên bệnh nhân mới.
    df = df.drop_duplicates().reset_index(drop=True)
    log(f"Sau khi loại {n_dup} dòng trùng lặp: {df.shape[0]} dòng còn lại")

    log("\nKiểm tra missing values:")
    log(df.isna().sum().to_string())

    log("\nGiá trị lạ ở các cột phân loại đặc biệt (ca, thal):")
    log(f"  ca   value_counts:\n{df['ca'].value_counts().sort_index().to_string()}")
    log(f"  thal value_counts:\n{df['thal'].value_counts().sort_index().to_string()}")
    log(
        "  -> Ghi chú: theo bộ mã UCI gốc, ca hợp lệ trong {0,1,2,3} và thal "
        "hợp lệ trong {1,2,3}. Các giá trị ca=4 hoặc thal=0 là mã 'thiếu dữ "
        "liệu/không xác định' do lỗi nhập liệu lịch sử của bộ dữ liệu. Ta vẫn "
        "giữ lại và coi như một hạng mục riêng khi one-hot (an toàn hơn xoá "
        "dòng vì bộ dữ liệu vốn đã nhỏ), nhưng cần nêu rõ giới hạn này trong "
        "báo cáo."
    )

    log("\nPhân bố nhãn target (0 = không bệnh, 1 = có bệnh):")
    log(df[TARGET_COL].value_counts().to_string())
    log(f"Tỉ lệ lớp dương (bệnh): {df[TARGET_COL].mean():.1%}")

    return df


def run_eda(df):
    section("BƯỚC 3: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)")

    # (a) Tỉ lệ bệnh theo loại đau ngực (cp)
    rate_by_cp = df.groupby("cp")[TARGET_COL].mean()
    log("Tỉ lệ mắc bệnh theo loại đau ngực (cp):")
    log(rate_by_cp.to_string())
    log(
        "  Diễn giải: cp=0 (đau thắt ngực điển hình) có tỉ lệ mắc bệnh thấp "
        "nhất trong khi các dạng đau không điển hình (cp=1,2,3) lại có tỉ lệ "
        "cao hơn - đây là điểm hay gây nhầm lẫn lâm sàng và là lý do bác sĩ "
        "cần model hỗ trợ."
    )

    # (b) Tỉ lệ bệnh theo nhóm tuổi
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 40, 50, 60, 100],
        labels=["<40", "40-50", "50-60", "60+"]
    )
    rate_by_age = df.groupby("age_group", observed=True)[TARGET_COL].mean()
    log("\nTỉ lệ mắc bệnh theo nhóm tuổi:")
    log(rate_by_age.to_string())

    # Vẽ hình
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    rate_by_cp.plot(kind="bar", ax=axes[0], color="#c0392b")
    axes[0].set_title("Tỉ lệ bệnh theo loại đau ngực (cp)")
    axes[0].set_ylabel("Tỉ lệ mắc bệnh")

    sns.boxplot(data=df, x=TARGET_COL, y="thalach", ax=axes[1], palette="Set2")
    axes[1].set_title("Nhịp tim tối đa (thalach) theo target")
    axes[1].set_xlabel("target (0=không bệnh, 1=có bệnh)")

    rate_by_age.plot(kind="bar", ax=axes[2], color="#2980b9")
    axes[2].set_title("Tỉ lệ bệnh theo nhóm tuổi")
    axes[2].set_ylabel("Tỉ lệ mắc bệnh")

    plt.tight_layout()
    fig_path = REPORTS / "eda_overview.png"
    plt.savefig(fig_path, dpi=130)
    plt.close()
    log(f"\nĐã lưu hình EDA: {fig_path.relative_to(ROOT)}")

    df.drop(columns=["age_group"], inplace=True)
    return df


# ===========================================================================
# BƯỚC 4-5: PIPELINE + BASELINE
# ===========================================================================
def build_preprocessor():
    """
    ColumnTransformer gồm 2 nhánh:
      - OneHotEncoder cho biến phân loại đa lớp (cp, restecg, slope, ca, thal)
        -> tránh việc model hiểu nhầm các mã số này có quan hệ thứ tự / khoảng
        cách tuyến tính (vd cp=3 không "gấp 3 lần" cp=1).
      - StandardScaler cho biến số liên tục (age, trestbps, chol, thalach,
        oldpeak) -> đưa các biến có đơn vị đo khác nhau (mg/dl, mmHg, mm...)
        về cùng thang đo (mean=0, std=1). Bắt buộc phải làm để:
          (1) thuật toán tối ưu (gradient descent bên trong LogisticRegression)
              hội tụ nhanh và ổn định hơn,
          (2) SO SÁNH được độ lớn các hệ số / odds ratio với nhau một cách công
              bằng - đây là yêu cầu bắt buộc của README mục 4.1.
      - Biến nhị phân (sex, fbs, exang) giữ nguyên vì đã ở dạng 0/1.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_COLS),
            ("bin", "passthrough", BINARY_COLS),
        ]
    )
    return preprocessor


def get_feature_names(preprocessor):
    """Lấy lại tên cột sau khi ColumnTransformer biến đổi, để gắn vào bảng odds ratio."""
    num_names = NUMERIC_COLS
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))
    bin_names = BINARY_COLS
    return num_names + cat_names + bin_names


def run_baseline(X_train, y_train, X_test, y_test):
    section("BƯỚC 5: BASELINE (DummyClassifier)")
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    acc = dummy.score(X_test, y_test)
    log(
        f"DummyClassifier (luôn đoán lớp đông nhất) — accuracy trên test: {acc:.3f}\n"
        f"  -> Đây là 'sàn' tối thiểu. Bất kỳ model thật nào cũng PHẢI vượt qua "
        f"con số này rõ rệt (và quan trọng hơn, phải có recall/AUC tốt vì "
        f"accuracy dễ bị đánh lừa khi 2 lớp gần cân bằng nhưng chi phí sai sót "
        f"không đối xứng trong y tế)."
    )
    return acc


# ===========================================================================
# BƯỚC 6-7: TRAIN + ODDS RATIO + SO SÁNH L1/L2
# ===========================================================================
def train_logreg(X_train, y_train, preprocessor, penalty="l2"):
    """
    Huấn luyện Logistic Regression với regularization được chọn tự động qua
    cross-validation (LogisticRegressionCV) thay vì chọn tay 1 giá trị C.

    Vì sao dùng CV thay vì set C cố định:
      - Bộ dữ liệu nhỏ (303 dòng) -> phương sai của 1 lần train/test split rất
        lớn. LogisticRegressionCV thử nhiều giá trị C trên nhiều fold rồi lấy
        C tốt nhất trung bình -> ổn định và đáng tin hơn.
      - scoring='roc_auc': ROC-AUC không phụ thuộc vào 1 ngưỡng cắt cụ thể
        (đánh giá khả năng XẾP HẠNG xác suất đúng trên mọi ngưỡng), nên phù hợp
        hơn để CHỌN ĐỘ MẠNH REGULARIZATION của model. Yêu cầu lâm sàng "ưu
        tiên recall >= 90%" được xử lý RIÊNG ở bước 9 (chọn ngưỡng cắt xác
        suất), không trộn chung vào bước chọn C này.
        Lưu ý kỹ thuật quan trọng: nếu dùng scoring='recall' trực tiếp ở đây,
        CV sẽ có xu hướng chọn C khiến model phạt cực mạnh, gần như dự đoán
        MỌI bệnh nhân đều dương tính (recall=1.0 "miễn phí" nhưng vô nghĩa,
        precision rất thấp, các hệ số bị co gần hết về 0 nên odds ratio mất ý
        nghĩa) — đây là một cạm bẫy thực tế đã gặp phải khi thử nghiệm.

    Lưu ý về C: C = 1/lambda (ngược trực giác!). C NHỎ = phạt MẠNH (hệ số bị
    ép về gần 0 nhiều hơn, model đơn giản hơn, dễ underfit). C LỚN = phạt NHẸ
    (model tự do khớp dữ liệu hơn, dễ overfit với bộ dữ liệu nhỏ như thế này).
    """
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    model = LogisticRegressionCV(
        Cs=10,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        penalty=penalty,
        solver=solver,
        scoring="roc_auc",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    return pipe


def odds_ratio_table(pipe, feature_names):
    """
    BẢNG ODDS RATIO — phần "linh hồn" của bài tập này.

    Với mỗi hệ số w của Logistic Regression, Odds Ratio = e^w:
      - OR = 1.0  -> biến này KHÔNG ảnh hưởng tới odds mắc bệnh
      - OR > 1.0  -> mỗi 1 đơn vị tăng thêm của biến (đã chuẩn hoá), ODDS mắc
                     bệnh bị NHÂN thêm OR lần (yếu tố nguy cơ)
      - OR < 1.0  -> mỗi 1 đơn vị tăng thêm, ODDS mắc bệnh GIẢM còn OR lần,
                     tức (1-OR)*100% (yếu tố bảo vệ)

    Vì các biến số đã được StandardScaler chuẩn hoá (mean=0, std=1), "1 đơn vị"
    ở đây nghĩa là "1 độ lệch chuẩn", nên có thể so sánh độ lớn ảnh hưởng giữa
    các biến có đơn vị đo gốc khác nhau (tuổi, mg/dl, mmHg...) một cách công bằng.
    """
    clf = pipe.named_steps["clf"]
    coefs = clf.coef_[0]
    table = pd.DataFrame({
        "dac_trung": feature_names,
        "he_so_w": coefs,
        "odds_ratio": np.exp(coefs),
    }).sort_values("odds_ratio", ascending=False).reset_index(drop=True)
    return table


def compare_l1_l2(X_train, y_train, X_test, y_test, preprocessor, feature_names):
    section("BƯỚC 6-7: HUẤN LUYỆN + BẢNG ODDS RATIO + SO SÁNH L1 vs L2")

    results = {}
    for penalty in ["l2", "l1"]:
        log(f"\n--- Regularization: {penalty.upper()} ---")
        pipe = train_logreg(X_train, y_train, preprocessor, penalty=penalty)
        best_C = pipe.named_steps["clf"].C_[0]
        log(f"C tốt nhất tìm được qua CV: {best_C:.4f}")

        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_proba)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        log(f"Test ROC-AUC = {auc:.3f} | Recall = {rec:.3f} | Precision = {prec:.3f}")

        table = odds_ratio_table(pipe, feature_names)
        n_zero = (np.isclose(table["he_so_w"], 0)).sum()
        log(f"Số hệ số bị đưa CHÍNH XÁC về 0 (bị loại bỏ khỏi model): {n_zero}")
        if penalty == "l1" and n_zero > 0:
            removed = table.loc[np.isclose(table["he_so_w"], 0), "dac_trung"].tolist()
            log(f"Các biến bị L1 loại bỏ: {removed}")

        log("\nBảng Odds Ratio (top 8, sắp theo mức độ ảnh hưởng):")
        log(table.head(8).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        results[penalty] = {
            "pipe": pipe, "table": table, "auc": auc,
            "recall": rec, "precision": prec, "best_C": best_C,
        }

    log(
        "\n>>> KẾT LUẬN L1 vs L2:\n"
        f"    L2: AUC={results['l2']['auc']:.3f}, giữ lại toàn bộ {len(feature_names)} biến, "
        "hệ số co về gần 0 nhưng không loại hẳn biến nào -> phù hợp khi ta tin "
        "rằng mọi biến lâm sàng đều có ít nhiều giá trị chẩn đoán.\n"
        f"    L1: AUC={results['l1']['auc']:.3f}, có xu hướng loại bỏ hẳn một số biến "
        "ít quan trọng -> model đơn giản hơn, dễ giải thích hơn với bác sĩ vì "
        "chỉ còn 'các yếu tố nguy cơ thực sự quan trọng', đánh đổi lại có thể "
        "mất một phần thông tin nếu C bị phạt quá mạnh."
    )
    return results


# ===========================================================================
# BƯỚC 8-9: ROC/PR + CHỌN NGƯỠNG THEO RECALL >= 0.90
# ===========================================================================
def pick_threshold_via_cv(X_train, y_train, preprocessor, penalty="l2",
                           target_recall=0.90, n_splits=5):
    """
    Chọn ngưỡng quyết định BẰNG CROSS-VALIDATION TRÊN TẬP TRAIN — không dò
    trên tập test như bản trước.

    ⚠️ Vì sao bản trước sai (feedback đã chỉ ra và bị trừ điểm 'method'):
    Bản cũ gọi precision_recall_curve() trực tiếp trên (y_test, y_proba_test)
    để TÌM ngưỡng thoả recall>=0.90, rồi lại BÁO CÁO recall/precision cũng
    trên chính 61 dòng test đó. Ngưỡng vì vậy đã được "tối ưu hoá" để khớp
    đúng đặc điểm ngẫu nhiên của tập test này -> con số 0.909/0.811 là ước
    lượng LẠC QUAN (optimistic bias), không phản ánh đúng hiệu năng khi model
    gặp bệnh nhân hoàn toàn mới. Đây là một dạng rò rỉ nhãn qua bước chọn
    ngưỡng, tinh vi hơn rò rỉ dữ liệu ở bước 1 nhưng bản chất tương tự: dùng
    thông tin của tập đánh giá để đưa ra một quyết định (ở đây là ngưỡng cắt)
    rồi lại đánh giá trên chính tập đó.

    Cách làm đúng: tập TEST chỉ được dùng ĐÚNG MỘT LẦN, ở bước đánh giá cuối
    cùng, sau khi mọi lựa chọn (C, ngưỡng, biến...) đã chốt xong bằng dữ liệu
    train. Ở đây ta dùng cross_val_predict trên tập TRAIN để lấy xác suất
    OOF (out-of-fold) — với mỗi dòng train, xác suất dự đoán đến từ 1 model
    KHÔNG được huấn luyện trên chính dòng đó — rồi dò ngưỡng trên các xác
    suất OOF này. Ngưỡng tìm được sau đó mới được áp dụng đúng 1 lần lên tập
    test để có số liệu cuối cùng, không thiên lệch.
    """
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    cv_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    model = LogisticRegressionCV(
        Cs=10, cv=cv_inner, penalty=penalty, solver=solver,
        scoring="roc_auc", max_iter=5000, random_state=RANDOM_STATE,
    )
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", model)])

    cv_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_oof_proba = cross_val_predict(
        pipe, X_train, y_train, cv=cv_outer, method="predict_proba"
    )[:, 1]

    precision, recall, pr_thresh = precision_recall_curve(y_train, y_oof_proba)
    valid = recall[:-1] >= target_recall
    if valid.any():
        candidate_idx = np.where(valid)[0]
        best_idx = candidate_idx[np.argmax(pr_thresh[candidate_idx])]
        chosen_threshold = pr_thresh[best_idx]
        oof_precision, oof_recall = precision[best_idx], recall[best_idx]
    else:
        chosen_threshold = 0.5
        oof_precision, oof_recall = precision[0], recall[0]

    log(
        f"Đã chạy cross_val_predict ({n_splits}-fold, OOF) trên {len(y_train)} "
        f"dòng train để lấy xác suất KHÔNG rò rỉ.\n"
        f"-> Ngưỡng được chọn (trên OOF-train) = {chosen_threshold:.3f}\n"
        f"   Tại ngưỡng này trên OOF-train: recall = {oof_recall:.3f}, "
        f"precision = {oof_precision:.3f}\n"
        f"   (Đây CHỈ để chọn ngưỡng — số liệu báo cáo chính thức sẽ tính "
        f"lại trên tập TEST hoàn toàn chưa dùng tới, ở bước tiếp theo.)"
    )
    return chosen_threshold


def plot_roc_pr(pipe, X_test, y_test, chosen_threshold, target_recall=0.90):
    """
    Vẽ đường ROC/Precision-Recall và báo cáo hiệu năng THỰC SỰ trên tập test.

    QUAN TRỌNG: chosen_threshold được truyền vào từ pick_threshold_via_cv(),
    đã chọn xong TRƯỚC KHI hàm này chạm vào y_test. Tập test ở đây chỉ đóng
    vai trò "giám khảo" — đánh giá đúng 1 lần, không tham gia vào việc chọn
    ngưỡng -> recall/precision báo cáo dưới đây là ước lượng không thiên lệch
    (unbiased) cho hiệu năng khi triển khai trên bệnh nhân mới.
    """
    section("BƯỚC 8-9: ĐƯỜNG ROC/PR & CHỌN NGƯỠNG THEO YÊU CẦU LÂM SÀNG")

    y_proba = pipe.predict_proba(X_test)[:, 1]

    fpr, tpr, roc_thresh = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    precision, recall, pr_thresh = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    log(
        "Ngưỡng lâm sàng KHÔNG được dò trên tập test này — nó đã được chọn "
        "trước đó bằng cross-validation trên tập train (xem log phía trên). "
        "Ở đây tập test chỉ dùng để ĐO hiệu năng tại ngưỡng đã chốt."
    )
    log(f"Test ROC-AUC = {auc:.3f}")
    log(f"Test Average Precision (AP) = {ap:.3f}")
    log(f"Mục tiêu lâm sàng: recall >= {target_recall:.0%}")
    log(f"Ngưỡng đang dùng (chọn từ CV trên train) = {chosen_threshold:.3f}")

    # Đối chiếu với ngưỡng mặc định 0.5 để thấy rõ cái giá phải trả
    y_pred_default = (y_proba >= 0.5).astype(int)
    y_pred_chosen = (y_proba >= chosen_threshold).astype(int)
    chosen_recall = recall_score(y_test, y_pred_chosen)
    chosen_precision = precision_score(y_test, y_pred_chosen)
    log("\nSo sánh 2 ngưỡng (đo trên tập TEST, ngưỡng đều đã chốt từ trước):")
    log(
        f"  Ngưỡng 0.50 (mặc định): recall={recall_score(y_test, y_pred_default):.3f}, "
        f"precision={precision_score(y_test, y_pred_default):.3f}"
    )
    log(
        f"  Ngưỡng {chosen_threshold:.3f} (theo yêu cầu bác sĩ, chọn từ CV-train): "
        f"recall={chosen_recall:.3f}, precision={chosen_precision:.3f}"
    )
    log(
        "  -> Đánh đổi: để không bỏ sót ca bệnh nào (recall cao), ta chấp nhận "
        "nhiều báo động giả hơn (precision giảm). Đây là lựa chọn ĐÚNG về mặt "
        "y khoa vì chi phí bỏ sót 1 ca bệnh tim >> chi phí 1 lần xét nghiệm "
        "kiểm tra thêm. Nếu recall trên test không chạm đúng 90% như mục tiêu "
        "CV-train, đó là dao động tự nhiên do test chỉ có ~60 dòng — hãy nhìn "
        "khoảng dao động CV ở Bước 11 thay vì tin tuyệt đối vào 1 con số lẻ."
    )

    cm = confusion_matrix(y_test, y_pred_chosen)
    log(f"\nMa trận nhầm lẫn tại ngưỡng đã chọn:\n{cm}")
    log(f"(hàng=thực tế [0,1], cột=dự đoán [0,1])")

    # Vẽ hình
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, color="#c0392b", lw=2, label=f"ROC (AUC={auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].set_title("Đường ROC (test)")
    axes[0].legend()

    axes[1].plot(recall, precision, color="#2980b9", lw=2, label=f"PR (AP={ap:.3f})")
    axes[1].axvline(target_recall, color="gray", ls="--", lw=1, label=f"Recall mục tiêu={target_recall:.0%}")
    axes[1].scatter([chosen_recall], [chosen_precision], color="green", zorder=5,
                     label=f"Ngưỡng {chosen_threshold:.2f} (chọn từ CV-train)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Đường Precision-Recall (test)")
    axes[1].legend()

    plt.tight_layout()
    fig_path = REPORTS / "roc_pr_curve.png"
    plt.savefig(fig_path, dpi=130)
    plt.close()
    log(f"\nĐã lưu hình: {fig_path.relative_to(ROOT)}")

    return auc, ap


# ===========================================================================
# BƯỚC 10: VIF - ĐA CỘNG TUYẾN
# ===========================================================================
def check_vif(X_train, preprocessor, feature_names):
    section("BƯỚC 10: KIỂM TRA ĐA CỘNG TUYẾN (VIF)")

    log(
        "VIF (Variance Inflation Factor) đo mức độ 1 biến độc lập có thể được "
        "GIẢI THÍCH bởi các biến độc lập còn lại. VIF cao nghĩa là biến đó mang "
        "thông tin trùng lặp với các biến khác -> hệ số hồi quy trở nên không "
        "ổn định (dấu/độ lớn có thể đảo lộn chỉ vì thêm/bớt 1 biến khác), khiến "
        "việc DIỄN GIẢI odds ratio từng biến riêng lẻ trở nên thiếu tin cậy.\n"
        "Quy ước thường dùng: VIF > 10 là dấu hiệu đa cộng tuyến nghiêm trọng, "
        "VIF > 5 đã đáng chú ý."
    )

    X_transformed = preprocessor.transform(X_train)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_vif = pd.DataFrame(X_transformed, columns=feature_names)
    # thêm cột hằng số (intercept) - bắt buộc cho công thức VIF chuẩn
    X_vif_const = X_vif.copy()
    X_vif_const.insert(0, "const", 1.0)

    vif_data = []
    for i, col in enumerate(X_vif_const.columns):
        if col == "const":
            continue
        try:
            v = variance_inflation_factor(X_vif_const.values, i)
        except Exception:
            v = np.nan
        vif_data.append({"bien": col, "VIF": v})

    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False).reset_index(drop=True)
    log("\nBảng VIF (sắp giảm dần):")
    log(vif_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    high_vif = vif_df[vif_df["VIF"] > 10]
    if len(high_vif) > 0:
        log(f"\n-> Biến có VIF > 10 (đa cộng tuyến nghiêm trọng): {high_vif['bien'].tolist()}")
    else:
        log("\n-> KHÔNG có biến nào VIF > 10. Đa cộng tuyến không phải vấn đề "
            "nghiêm trọng trong bộ dữ liệu này -> các hệ số/odds ratio đủ tin "
            "cậy để diễn giải riêng lẻ.")

    vif_path = REPORTS / "vif_table.csv"
    vif_df.to_csv(vif_path, index=False)
    log(f"Đã lưu: {vif_path.relative_to(ROOT)}")
    return vif_df


# ===========================================================================
# BƯỚC 11: SO SÁNH VỚI SVM & RANDOM FOREST
# ===========================================================================
def compare_models(X_train, y_train, X_test, y_test, preprocessor, chosen_threshold):
    section("BƯỚC 11: SO SÁNH LOGISTIC REGRESSION vs SVM vs RANDOM FOREST")

    log(
        "Mục đích so sánh KHÔNG phải để tìm model 'AUC cao nhất bằng mọi giá', "
        "mà để trả lời câu hỏi: Logistic Regression có phải đánh đổi quá nhiều "
        "hiệu năng để đổi lấy khả năng giải thích được (odds ratio) hay không?"
    )

    models = {
        "Logistic Regression (L2)": LogisticRegression(
            penalty="l2", C=1.0, max_iter=5000, random_state=RANDOM_STATE
        ),
        "SVM (RBF kernel)": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, random_state=RANDOM_STATE
        ),
    }

    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, clf in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_proba)
        y_pred = (y_proba >= chosen_threshold).astype(int)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)

        rows.append({
            "model": name,
            "cv_auc_mean": cv_auc.mean(),
            "cv_auc_std": cv_auc.std(),
            "test_auc": test_auc,
            "recall_at_chosen_thr": rec,
            "precision_at_chosen_thr": prec,
            "co_giai_thich_duoc_truc_tiep": "Có (odds ratio)" if "Logistic" in name else "Không",
        })

    comp_df = pd.DataFrame(rows)
    log("\nBảng so sánh (CV-AUC trên train, 5-fold; test-AUC trên tập test giữ lại):")
    log(comp_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    comp_path = REPORTS / "model_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    log(f"\nĐã lưu: {comp_path.relative_to(ROOT)}")

    log(
        "\n>>> KẾT LUẬN: Nếu Random Forest / SVM chỉ nhỉnh hơn Logistic Regression "
        "một chút về AUC (vài phần trăm, nằm trong khoảng độ lệch chuẩn CV), "
        "thì trong bối cảnh LÂM SÀNG, Logistic Regression vẫn là lựa chọn hợp "
        "lý hơn vì bác sĩ có thể tự tay kiểm chứng lý do model đưa ra quyết "
        "định (qua odds ratio), điều RF/SVM 'hộp đen' không làm được trực tiếp."
    )
    return comp_df


# ===========================================================================
# BƯỚC 12: GIẢI THÍCH CHO 1 CA BỆNH CỤ THỂ
# ===========================================================================
def explain_one_patient(pipe, X_test, y_test, feature_names, chosen_threshold, odds_table):
    section("BƯỚC 12: DIỄN GIẢI KẾT QUẢ CHO 1 CA BỆNH CỤ THỂ")

    # Chọn một bệnh nhân dương tính thật (target=1) mà model dự đoán đúng và
    # có xác suất cao, để minh hoạ rõ nét cách đọc kết quả.
    y_proba_all = pipe.predict_proba(X_test)[:, 1]
    idx_candidates = X_test.index[(y_test.values == 1) & (y_proba_all >= chosen_threshold)]
    if len(idx_candidates) == 0:
        idx_candidates = X_test.index[y_test.values == 1]
    patient_idx = idx_candidates[0]
    patient_row = X_test.loc[[patient_idx]]
    proba = pipe.predict_proba(patient_row)[0, 1]
    pred = int(proba >= chosen_threshold)

    log(f"Bệnh nhân mẫu (index gốc trong dữ liệu = {patient_idx}):")
    log(patient_row.to_string(index=False))
    log(f"\nXác suất model dự đoán mắc bệnh: {proba:.1%}")
    log(f"Ngưỡng quyết định đang dùng: {chosen_threshold:.2f} -> Kết luận model: "
        f"{'CÓ nguy cơ cao, cần tầm soát thêm' if pred == 1 else 'nguy cơ thấp'}")
    log(f"Nhãn thực tế trong dữ liệu: {'có bệnh' if y_test.loc[patient_idx] == 1 else 'không bệnh'}")

    top_risk = odds_table.head(3)
    top_protect = odds_table.tail(3).iloc[::-1]

    explanation_vi = (
        f"\n--- ĐOẠN GIẢI THÍCH BẰNG NGÔN NGỮ Y KHOA (mẫu, dùng cho bước 12) ---\n"
        f"Dựa trên các chỉ số đo được, mô hình ước tính bệnh nhân này có "
        f"{proba:.0%} khả năng mắc bệnh tim, cao hơn ngưỡng cảnh báo "
        f"{chosen_threshold:.0%} mà khoa đang áp dụng để đảm bảo không bỏ sót "
        f"ca bệnh. Ba yếu tố có ảnh hưởng làm TĂNG nguy cơ mạnh nhất trên toàn "
        f"bộ tập dữ liệu (theo bảng odds ratio) là: "
        f"{', '.join(top_risk['dac_trung'].tolist())}. Ngược lại, các yếu tố có "
        f"xu hướng làm GIẢM nguy cơ (bảo vệ) là: "
        f"{', '.join(top_protect['dac_trung'].tolist())}. Đây CHỈ LÀ công cụ hỗ "
        f"trợ sàng lọc ban đầu dựa trên thống kê nhóm, không thay thế chẩn "
        f"đoán lâm sàng trực tiếp của bác sĩ; bác sĩ cần đối chiếu thêm với "
        f"bệnh sử, triệu chứng thực tế và các xét nghiệm chuyên sâu trước khi "
        f"kết luận."
    )
    log(explanation_vi)
    return patient_idx, proba


# ===========================================================================
# KIỂM TRA CÔNG BẰNG THEO GIỚI TÍNH (mục 8 - đạo đức)
# ===========================================================================
def fairness_check(pipe, X_test, y_test, chosen_threshold):
    section("KIỂM TRA ĐẠO ĐỨC: MODEL CÓ ĐỐI XỬ KHÁC NHAU GIỮA NAM/NỮ KHÔNG?")

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= chosen_threshold).astype(int)
    df_check = X_test.copy()
    df_check["y_true"] = y_test.values
    df_check["y_pred"] = y_pred

    rows = []
    for sex_val, sex_name in [(1, "Nam"), (0, "Nữ")]:
        mask = df_check["sex"] == sex_val
        if mask.sum() == 0:
            continue
        rec = recall_score(df_check.loc[mask, "y_true"], df_check.loc[mask, "y_pred"])
        prec_denom = df_check.loc[mask, "y_pred"].sum()
        prec = (
            precision_score(df_check.loc[mask, "y_true"], df_check.loc[mask, "y_pred"])
            if prec_denom > 0 else np.nan
        )
        rows.append({"gioi_tinh": sex_name, "so_luong": int(mask.sum()),
                      "recall": rec, "precision": prec})

    fair_df = pd.DataFrame(rows)
    log(fair_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    log(
        "\nGhi chú đạo đức bắt buộc: bộ dữ liệu Cleveland thu thập từ thập niên "
        "1980 tại Mỹ, cỡ mẫu nhỏ và không đại diện cho dân số Việt Nam hiện nay "
        "(khác biệt về chế độ ăn, gen, môi trường sống, tiêu chuẩn chẩn đoán). "
        "Chênh lệch recall/precision giữa 2 giới ở đây (nếu có) một phần đến từ "
        "cỡ mẫu nhỏ của tập test, KHÔNG nên vội kết luận model 'thiên vị' mà "
        "cần kiểm định lại trên bộ dữ liệu lớn hơn trước khi triển khai thực tế."
    )
    fair_path = REPORTS / "fairness_by_sex.csv"
    fair_df.to_csv(fair_path, index=False)
    return fair_df


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    df = load_and_clean_data()
    df = run_eda(df)

    section("BƯỚC 2 & 4: CHIA TRAIN/TEST + XÂY PIPELINE TIỀN XỬ LÝ")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # stratify=y: giữ đúng tỉ lệ lớp 0/1 ở cả train và test, quan trọng vì bộ
    # dữ liệu vốn đã nhỏ (chỉ ~300 dòng) -> nếu không stratify, 1 lần chia xui
    # có thể làm lệch tỉ lệ lớp đáng kể giữa 2 tập.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    log(f"Train: {X_train.shape[0]} dòng | Test: {X_test.shape[0]} dòng")
    log(f"Tỉ lệ lớp dương - train: {y_train.mean():.1%} | test: {y_test.mean():.1%}")

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    feature_names = get_feature_names(preprocessor)
    log(f"\nSố đặc trưng sau one-hot + scaling: {len(feature_names)}")
    log(f"Danh sách: {feature_names}")

    run_baseline(X_train, y_train, X_test, y_test)

    l1_l2_results = compare_l1_l2(X_train, y_train, X_test, y_test, preprocessor, feature_names)

    # Dùng model L2 (giữ đủ biến, ổn định hơn với bộ dữ liệu nhỏ) làm model
    # chính thức để đi tiếp các bước ROC/PR, ngưỡng, VIF, so sánh, giải thích.
    main_pipe = l1_l2_results["l2"]["pipe"]
    main_table = l1_l2_results["l2"]["table"]

    # Chọn ngưỡng lâm sàng bằng CV trên tập TRAIN (không đụng tới y_test) —
    # xem docstring pick_threshold_via_cv() để hiểu vì sao đây là điểm sửa
    # quan trọng nhất so với bản trước.
    chosen_threshold = pick_threshold_via_cv(
        X_train, y_train, preprocessor, penalty="l2", target_recall=0.90
    )
    auc, ap = plot_roc_pr(main_pipe, X_test, y_test, chosen_threshold, target_recall=0.90)

    check_vif(X_train, preprocessor, feature_names)

    compare_models(X_train, y_train, X_test, y_test, preprocessor, chosen_threshold)

    explain_one_patient(main_pipe, X_test, y_test, feature_names, chosen_threshold, main_table)

    fairness_check(main_pipe, X_test, y_test, chosen_threshold)

    # ---------------------------------------------------------------------
    # Lưu model + báo cáo
    # ---------------------------------------------------------------------
    section("LƯU MODEL & BÁO CÁO")
    model_path = MODELS / "logreg_pipeline.joblib"
    joblib.dump({
        "pipeline": main_pipe,
        "feature_names": feature_names,
        "chosen_threshold": chosen_threshold,
    }, model_path)
    log(f"Đã lưu model: {model_path.relative_to(ROOT)}")

    odds_csv = REPORTS / "odds_ratio.csv"
    main_table.to_csv(odds_csv, index=False)
    log(f"Đã lưu bảng odds ratio: {odds_csv.relative_to(ROOT)}")

    # Hình bảng odds ratio dạng biểu đồ ngang, trực quan cho bác sĩ đọc nhanh
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_df = main_table.copy()
    colors = ["#c0392b" if v > 1 else "#2980b9" for v in plot_df["odds_ratio"]]
    ax.barh(plot_df["dac_trung"], plot_df["odds_ratio"], color=colors)
    ax.axvline(1.0, color="black", lw=1, ls="--")
    ax.set_xlabel("Odds Ratio (đỏ = tăng nguy cơ, xanh = bảo vệ)")
    ax.set_title("Odds Ratio theo từng đặc trưng (Logistic Regression, L2)")
    ax.invert_yaxis()
    plt.tight_layout()
    or_fig_path = REPORTS / "odds_ratio.png"
    plt.savefig(or_fig_path, dpi=130)
    plt.close()
    log(f"Đã lưu hình odds ratio: {or_fig_path.relative_to(ROOT)}")

    summary = {
        "n_samples_after_dedup": int(len(df)),
        "n_features": len(feature_names),
        "test_auc_l2": float(l1_l2_results["l2"]["auc"]),
        "test_auc_l1": float(l1_l2_results["l1"]["auc"]),
        "chosen_threshold": float(chosen_threshold),
        "test_auc": float(auc),
        "test_average_precision": float(ap),
    }
    with open(REPORTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"Đã lưu summary.json: {summary}")

    with open(REPORTS / "run_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))
    log(f"\nĐã ghi toàn bộ log ra: {(REPORTS / 'run_log.txt').relative_to(ROOT)}")


if __name__ == "__main__":
    main()