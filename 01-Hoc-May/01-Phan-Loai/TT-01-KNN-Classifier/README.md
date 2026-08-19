# TT-01 — KNN CLASSIFIER
## Sàng lọc nguy cơ tiểu đường tại phòng khám

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 1](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-01-Gioi-thieu-ML) |
| 🧠 **Nhóm** | Phân loại có giám sát |
| 🔧 **Thuật toán** | K-Nearest Neighbors (KNN) |
| 🏭 **Lĩnh vực** | Y tế · Sàng lọc cộng đồng |
| ⏱ **Thời lượng** | 4–6 giờ |
| 📈 **Độ khó** | ⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   "Cho tôi biết bạn giống ai, tôi nói bạn là ai."

   Điểm mới ❓ → tính khoảng cách tới TẤT CẢ điểm đã biết
                → lấy K điểm gần nhất → bỏ phiếu đa số

        🔵 🔵          K = 5:  4 điểm 🔵, 1 điểm 🔴
       🔵 ❓ 🔴         → kết luận ❓ thuộc nhóm 🔵
        🔵 🔵
```

**Đặc điểm cần nhớ:** KNN là thuật toán **"lười"** — `.fit()` chỉ ghi nhớ dữ liệu,
mọi tính toán dồn vào lúc `.predict()`. Không có "model" nào được học cả.

---

## 2. BÀI TOÁN THỰC TẾ

```
   Phòng khám tuyến huyện có 3 bác sĩ, mỗi ngày 200 bệnh nhân.
   Xét nghiệm tiểu đường đầy đủ tốn 350.000đ và mất 2 ngày chờ kết quả.

   → Cần công cụ SÀNG LỌC NHANH từ các chỉ số đo được ngay tại chỗ
     (huyết áp, BMI, tuổi, tiền sử gia đình) để quyết định
     AI CẦN làm xét nghiệm chuyên sâu, ai chưa cần.

   ⚠️ Bỏ sót người bệnh (FN) NGUY HIỂM hơn nhiều so với
      chỉ định xét nghiệm thừa (FP) → ưu tiên RECALL.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Pima Indians Diabetes |
| **Link** | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |
| **Kích thước** | 768 dòng × 9 cột |
| **Nhãn** | `Outcome` (0/1) — khoảng **35% dương tính** |

**Các cột:** `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`,
`BMI`, `DiabetesPedigreeFunction`, `Age`, `Outcome`

### ⚠️ Bẫy có sẵn: giá trị 0 giả danh dữ liệu thiếu

```
   Glucose = 0        → không thể có người sống với đường huyết 0
   BloodPressure = 0  → vô lý
   BMI = 0            → vô lý
   SkinThickness = 0  → thiếu ~30% dòng
   Insulin = 0        → thiếu ~49% dòng  ← nặng nhất

   → Đây là DỮ LIỆU THIẾU được mã hoá bằng số 0, không phải giá trị thật.
   → Phải chuyển thành NaN rồi điền median, KHÔNG để nguyên.
```

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. Vì sao bài này bắt buộc chuẩn hoá

```
   Insulin: 0 – 846        (biên độ ~846)
   BMI:     18 – 67        (biên độ ~49)

   Khoảng cách Euclid: √( (ΔInsulin)² + (ΔBMI)² )
   → Insulin ÁP ĐẢO hoàn toàn, BMI gần như bị bỏ qua.
   → KHÔNG chuẩn hoá = KNN chỉ nhìn đúng 1 cột.
```

### 4.2. Pipeline chuẩn

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

COT_KHONG_THE_BANG_0 = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
X[COT_KHONG_THE_BANG_0] = X[COT_KHONG_THE_BANG_0].replace(0, np.nan)

pipe = Pipeline([
    ('imp',   SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),                    # ⭐ BẮT BUỘC với KNN
    ('knn',   KNeighborsClassifier(n_neighbors=5)),
])
```

### 4.3. Dò K bằng cross-validation

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

grid = {'knn__n_neighbors': list(range(1, 32, 2)),   # K LẺ để tránh hoà phiếu
        'knn__weights': ['uniform', 'distance'],
        'knn__metric':  ['euclidean', 'manhattan']}

gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                  scoring='recall', n_jobs=-1)      # ⭐ tối ưu RECALL, không phải accuracy
gs.fit(X_train, y_train)
```

### 4.4. Metric

```
   Chính : RECALL (không bỏ sót người bệnh)
   Phụ   : Precision, F1, PR-AUC, ma trận nhầm lẫn
   ❌ KHÔNG dùng accuracy làm metric chính (65% người âm tính)
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Nạp dữ liệu, in describe() → PHÁT HIỆN các cột có min = 0 phi lý
   ☐ 2. Thay 0 → NaN ở 5 cột y sinh, đếm % thiếu mỗi cột
   ☐ 3. EDA: histogram Glucose theo nhóm Outcome (kỳ vọng tách rõ)
   ☐ 4. Chia train/test stratify, random_state=42
   ☐ 5. Baseline: DummyClassifier(strategy='most_frequent')
   ☐ 6. KNN với K=5 mặc định → ghi lại điểm
   ☐ 7. ⚠️ CHẠY THỬ KNN KHÔNG chuẩn hoá → so sánh, chứng minh chuẩn hoá quan trọng
   ☐ 8. GridSearchCV dò K, weights, metric
   ☐ 9. Vẽ đường Recall & Accuracy theo K → giải thích hình dạng
   ☐ 10. Chạm tập TEST 1 lần, báo cáo ma trận nhầm lẫn
   ☐ 11. So sánh KNN với Logistic Regression → cái nào hợp bài này hơn?
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Đã phát hiện và xử lý đúng bẫy "0 = thiếu dữ liệu"
   ☐ Có bảng so sánh CÓ vs KHÔNG chuẩn hoá (chứng minh bằng số)
   ☐ Có biểu đồ Recall/Accuracy theo K, K được chọn có lý do
   ☐ Recall trên tập test > baseline rõ rệt
   ☐ Giải thích được vì sao K=1 cho accuracy 100% trên tập TRAIN
   ☐ Nêu được hạn chế: KNN chậm khi dữ liệu lớn, không giải thích được cá nhân
```

**Mức tham chiếu:** Recall thường đạt ~0,65–0,75 với bộ này. Đây là bộ dữ liệu **khó**,
đừng kỳ vọng 0,9+.

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Không chuẩn hoá | Insulin áp đảo, model gần như vô dụng |
| Để nguyên giá trị 0 | Model học "đường huyết 0 là bình thường" |
| Chọn K chẵn | Hoà phiếu, kết quả phụ thuộc thứ tự dữ liệu |
| Dùng accuracy làm metric | Bỏ sót người bệnh mà vẫn thấy điểm cao |
| Scale trước khi chia train/test | Rò rỉ dữ liệu |

---

## 8. SẢN PHẨM NỘP

```
TT-01-KNN-<HoTen>/
├── README.md              ← có bảng so sánh có/không chuẩn hoá
├── notebooks/knn_diabetes.ipynb
├── src/train.py
├── models/knn_pipeline.joblib
├── reports/{recall_theo_K.png, confusion_matrix.png}
└── requirements.txt
```

> ⚖️ **Lưu ý đạo đức:** ghi rõ trong báo cáo rằng đây là công cụ **hỗ trợ sàng lọc**,
> không thay thế chẩn đoán của bác sĩ. Bộ dữ liệu chỉ thu thập trên phụ nữ Pima
> từ 21 tuổi → **không tổng quát hoá** được cho dân số Việt Nam.

---

## 9. MỞ RỘNG

```
   1. Thử KNN Regressor dự đoán chỉ số Glucose (xem TT-21)
   2. Dùng KNNImputer thay SimpleImputer — điền thiếu bằng chính hàng xóm
   3. Đo thời gian predict khi nhân dữ liệu lên 100× → thấy điểm yếu của KNN
```

**Tham khảo:** [Lý thuyết KNN — Buổi 1](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-01-Gioi-thieu-ML/Tai-Lieu/ly_thuyet_chi_tiet_buoi_01.md)
