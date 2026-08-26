# TT-07 — GRADIENT BOOSTING
## Dự đoán mức thu nhập để chấm điểm hồ sơ vay tiêu dùng

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 6](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-06-Ensemble-EndToEnd) |
| 🧠 **Nhóm** | Phân loại · Ensemble (Boosting) |
| 🔧 **Thuật toán** | Gradient Boosting Classifier |
| 🏭 **Lĩnh vực** | Tài chính · Tín dụng tiêu dùng |
| ⏱ **Thời lượng** | 6–8 giờ |
| 📈 **Độ khó** | ⭐⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   BAGGING (Random Forest — TT-03)     BOOSTING (bài này)
     Cây 1 ─┐                            Cây 1 → sai ─→ Cây 2 → sai ─→ Cây 3
     Cây 2 ─┼─▶ BỎ PHIẾU                   ↓        ↓        ↓
     Cây 3 ─┘                            Kết quả = Cây1 + Cây2 + Cây3
     (học SONG SONG, độc lập)            (học TUẦN TỰ, cây sau sửa lỗi cây trước)
     Giảm VARIANCE                        Giảm BIAS
```

**Cơ chế:** mỗi cây mới học để dự đoán **PHẦN DƯ (residual)** của các cây trước.

```
   Dự đoán ban đầu = giá trị trung bình
   Lặp lại:
     ① Tính residual = thực tế − dự đoán hiện tại
     ② Train 1 cây NÔNG (max_depth 3) để dự đoán residual đó
     ③ Dự đoán mới = dự đoán cũ + learning_rate × cây mới
```

---

## 2. BÀI TOÁN THỰC TẾ

```
   Công ty tài chính tiêu dùng cần ước lượng KHẢ NĂNG TÀI CHÍNH của khách
   trước khi duyệt khoản vay trả góp, nhưng khách thường KHÔNG khai thu nhập
   hoặc khai không chính xác.

   → Dùng thông tin có thể kiểm chứng (nghề nghiệp, học vấn, giờ làm/tuần,
     tình trạng hôn nhân) để dự đoán khách có thu nhập > 50.000$/năm hay không.

   → Kết quả là 1 trong các đầu vào của mô hình chấm điểm tín dụng.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Adult Census Income |
| **Link** | https://archive.ics.uci.edu/dataset/2/adult |
| **Kích thước** | 48.842 dòng × 14 cột |
| **Nhãn** | `income` (`<=50K` / `>50K`) — khoảng **24% là >50K** |

**Các cột:** `age`, `workclass`, `fnlwgt`, `education`, `education-num`,
`marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`,
`capital-loss`, `hours-per-week`, `native-country`

### ⚠️ Bốn bẫy trong dữ liệu

```
   1. Giá trị thiếu ghi bằng ' ?' (có DẤU CÁCH đứng trước!)
      → df.replace(' ?', np.nan) — nếu chỉ replace('?') sẽ KHÔNG bắt được

   2. Mọi giá trị chuỗi đều có dấu cách thừa ở đầu → .str.strip()

   3. education và education-num là CÙNG MỘT THÔNG TIN (một dạng chữ, một dạng số)
      → giữ cả hai là trùng lặp → bỏ 1 cột

   4. fnlwgt là trọng số thống kê dân số, KHÔNG liên quan tới cá nhân
      → nên BỎ, giữ lại chỉ gây nhiễu
```

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. Ba siêu tham số then chốt

```
   learning_rate (η) — mỗi cây đóng góp bao nhiêu
        η nhỏ (0,05) + n_estimators LỚN (500)  = ⭐ TỐT NHẤT, ổn định
        η lớn (0,3)  + n_estimators nhỏ (100)  = nhanh nhưng dễ overfit

   n_estimators — số cây (quan hệ NGƯỢC với learning_rate)

   max_depth = 3 — cây phải NÔNG (weak learner)
        ⚠️ Khác hoàn toàn Random Forest (cây sâu, mạnh)
        Boosting cần cây YẾU để cộng dồn từ từ; cây sâu → overfit ngay
```

> 💡 **Quy tắc vàng:** `learning_rate` NHỎ + `n_estimators` LỚN. Nếu train quá lâu,
> hãy chuyển sang **HistGradientBoostingClassifier** (nhanh hơn 10–50×) hoặc LightGBM.

### 4.2. Code

```python
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=3,
    subsample=0.8,              # Stochastic GB — chống overfit
    validation_fraction=0.1, n_iter_no_change=20,   # dừng sớm
    random_state=42,
)

# Bản NHANH cho dữ liệu lớn (xử lý được cả NaN và biến phân loại)
hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                     early_stopping=True, random_state=42)
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Đọc dữ liệu, .str.strip() mọi cột chuỗi, replace ' ?' → NaN
   ☐ 2. Bỏ fnlwgt và 1 trong 2 cột education/education-num
   ☐ 3. EDA: tỉ lệ >50K theo học vấn, giờ làm/tuần, tình trạng hôn nhân
   ☐ 4. ⚠️ capital-gain lệch cực đoan (91% bằng 0) → xem xét biến đổi log hoặc nhị phân hoá
   ☐ 5. Pipeline: OneHot(cat) + passthrough(num)  [Boosting KHÔNG cần scale]
   ☐ 6. Baseline: DummyClassifier + Decision Tree
   ☐ 7. Gradient Boosting với tham số mặc định
   ☐ 8. ⭐ Vẽ đường TRAIN & VALIDATION loss theo số cây (staged_decision_function)
        → chỉ ra điểm bắt đầu OVERFIT
   ☐ 9. Dò learning_rate × n_estimators (lưới 3×3) → chứng minh quan hệ nghịch
   ☐ 10. So sánh 3 thuật toán trên CÙNG dữ liệu:
         Random Forest (TT-03) vs Gradient Boosting vs AdaBoost (TT-09)
         → bảng: PR-AUC, thời gian train, số tham số
   ☐ 11. Đo thời gian: GradientBoosting vs HistGradientBoosting
   ☐ 12. ⚖️ Kiểm tra THIÊN LỆCH theo `sex` và `race`
```

```python
# Vẽ overfit theo số cây
import numpy as np
from sklearn.metrics import log_loss

test_loss = [log_loss(y_test, p) for p in gb.staged_predict_proba(X_test)]
train_loss = [log_loss(y_train, p) for p in gb.staged_predict_proba(X_train)]
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Xử lý đúng bẫy ' ?' và dấu cách thừa
   ☐ Có biểu đồ train/validation loss theo số cây → chỉ rõ điểm overfit
   ☐ Có bảng lưới learning_rate × n_estimators → chứng minh quan hệ nghịch
   ☐ Có bảng so sánh Bagging vs Boosting vs AdaBoost (kèm thời gian train)
   ☐ PR-AUC > baseline rõ rệt
   ☐ ⚖️ Có phân tích thiên lệch theo giới tính / chủng tộc
   ☐ Giải thích được vì sao Boosting dùng cây NÔNG còn RF dùng cây SÂU
```

**Mức tham chiếu:** ROC-AUC ~0,92–0,93 · Accuracy ~0,87.

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| `replace('?')` không có dấu cách | Không bắt được giá trị thiếu |
| `max_depth` lớn (8–10) | Overfit ngay — sai bản chất boosting |
| `learning_rate` lớn + nhiều cây | Overfit nặng |
| Giữ cả `education` và `education-num` | Trùng lặp thông tin |
| Giữ `fnlwgt` | Nhiễu, không liên quan tới cá nhân |
| Bỏ qua kiểm tra thiên lệch | Rủi ro đạo đức & pháp lý nghiêm trọng |

---

## 8. SẢN PHẨM NỘP

```
TT-07-GradientBoosting-<HoTen>/
├── README.md                       ← có mục "BAGGING vs BOOSTING" và "THIÊN LỆCH"
├── notebooks/gradient_boosting_income.ipynb
├── src/train.py
├── models/gb_pipeline.joblib
├── reports/{loss_theo_so_cay.png, lr_vs_nestimators.png, bias_by_group.png}
└── requirements.txt
```

> ⚖️ **Cảnh báo đạo đức bắt buộc:** bộ dữ liệu này từ điều tra dân số Mỹ 1994,
> chứa **định kiến lịch sử** rõ rệt về giới tính và chủng tộc. Model sẽ **học và
> khuếch đại** các định kiến đó. Bài tập yêu cầu đo và báo cáo mức chênh lệch —
> tuyệt đối không dùng model này cho quyết định thật về con người.

---

## 9. MỞ RỘNG

```
   1. Thử LightGBM và so sánh: nhanh hơn bao nhiêu lần với cùng độ chính xác?
   2. Dùng SHAP để giải thích 3 hồ sơ cụ thể
   3. Thử ràng buộc công bằng: bỏ hẳn cột `sex`, `race` → điểm giảm bao nhiêu?
      Model có còn thiên lệch không? (gợi ý: vẫn có, qua biến thay thế như occupation)
```

**Tham khảo:** [Buổi 6 — Ensemble & Boosting](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-06-Ensemble-EndToEnd/Tai-Lieu/ly_thuyet_chi_tiet_buoi_06.md)
