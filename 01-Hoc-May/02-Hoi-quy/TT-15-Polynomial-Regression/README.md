# TT-15 — POLYNOMIAL REGRESSION
## Quan hệ nhiệt độ ↔ công suất nhà máy điện là ĐƯỜNG CONG, không phải đường thẳng

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao) |
| 🧠 **Nhóm** | Hồi quy phi tuyến (vẫn tuyến tính theo tham số) |
| 🔧 **Thuật toán** | Polynomial Regression (`PolynomialFeatures` + Linear/Ridge) |
| 🏭 **Lĩnh vực** | Năng lượng · Vận hành nhà máy |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   Bậc 1:  ŷ = w₁x + b                          (đường thẳng)
   Bậc 2:  ŷ = w₁x + w₂x² + b                   (parabol)
   Bậc 3:  ŷ = w₁x + w₂x² + w₃x³ + b            (uốn 2 lần)

   MẸO: chỉ cần TẠO THÊM CỘT x², x³, x·y … rồi chạy Linear Regression bình thường.
   → Model vẫn TUYẾN TÍNH theo tham số w, chỉ PHI TUYẾN theo biến x.
```

```
   Bậc 1 (underfit)        Bậc 3 (vừa)          Bậc 15 (overfit)
     ╱                       ╱‾╲                   ╱╲  ╱╲
    ╱ • • •                ╱ • •╲                 ╱•╲╱•╲╱
   ╱• •  •                ╱  •   ╲               ╱ • ╲  ╲
```

---

## 2. BÀI TOÁN THỰC TẾ

```
   Nhà máy nhiệt điện chu trình hỗn hợp cần dự báo CÔNG SUẤT PHÁT (MW)
   theo điều kiện môi trường, để chào giá lên thị trường điện mỗi giờ.

   Chào cao hơn khả năng phát → bị phạt hợp đồng
   Chào thấp hơn            → mất doanh thu

   ⚠️ Quan hệ nhiệt độ → công suất KHÔNG tuyến tính:
      nhiệt độ tăng làm hiệu suất tua-bin khí giảm theo đường CONG.
   → Model tuyến tính sai có hệ thống ở hai đầu dải nhiệt độ.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Combined Cycle Power Plant (UCI) |
| **Link** | https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant |
| **Kích thước** | 9.568 dòng × 4 đặc trưng |
| **Nhãn** | `PE` — công suất phát (MW), khoảng 420–495 |

**Đặc trưng:** `AT` nhiệt độ môi trường · `V` áp suất chân không · `AP` áp suất khí quyển
· `RH` độ ẩm tương đối

### ⚠️ Lưu ý

```
   1. AT và V tương quan cao (r ≈ 0,84) → khi tạo đặc trưng đa thức
      số cột bùng nổ và đa cộng tuyến càng nặng → PHẢI dùng Ridge thay Linear.

   2. Số cột sau PolynomialFeatures tăng RẤT NHANH:
        4 biến, bậc 2 → 15 cột
        4 biến, bậc 3 → 35 cột
        4 biến, bậc 5 → 126 cột
      → bậc cao + ít dữ liệu = overfit chắc chắn.
```

---

## 4. HƯỚNG ĐI ĐÚNG

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('poly',  PolynomialFeatures(degree=2, include_bias=False)),
    ('scale', StandardScaler()),     # ⭐ SAU poly — vì x² có thang đo khác hẳn x
    ('ridge', Ridge(alpha=1.0)),     # ⭐ Ridge, không dùng Linear thuần
])
```

> ⚠️ **Thứ tự cực kỳ quan trọng:** `PolynomialFeatures` TRƯỚC, `StandardScaler` SAU.
> Nếu scale trước rồi mới bình phương, các giá trị âm sau khi bình phương sẽ mất dấu
> và thang đo lại lệch tiếp.

### Chọn bậc bằng đường cong xác thực

```python
from sklearn.model_selection import validation_curve
import numpy as np

train_s, val_s = validation_curve(
    pipe, X_train, y_train, param_name='poly__degree',
    param_range=[1, 2, 3, 4, 5], cv=5, scoring='neg_root_mean_squared_error')
# Vẽ 2 đường: RMSE train và RMSE validation theo bậc
# → điểm validation thấp nhất = bậc tối ưu; sau đó val tăng lên = overfit
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. EDA: scatter AT vs PE → NHÌN THẤY đường cong bằng mắt
   ☐ 2. Baseline: Linear Regression bậc 1 → ghi RMSE
   ☐ 3. ⭐ Vẽ RESIDUAL PLOT của model bậc 1
        → phần dư có hình CHỮ U không? → bằng chứng thiếu bậc phi tuyến
   ☐ 4. Chạy bậc 1 → 5, ghi RMSE train và validation
   ☐ 5. Vẽ đường cong xác thực → chỉ ra bậc tối ưu và điểm bắt đầu overfit
   ☐ 6. Đếm số cột sinh ra ở mỗi bậc → lập bảng (bậc | số cột | RMSE)
   ☐ 7. So sánh Linear vs Ridge ở bậc cao (bậc 4–5) → Ridge cứu được bao nhiêu?
   ☐ 8. Vẽ lại residual plot của model bậc tối ưu → chữ U đã biến mất chưa?
   ☐ 9. So sánh với Random Forest Regressor (TT-17) — cây bắt phi tuyến tự nhiên
   ☐ 10. ✍️ Diễn giải: ở dải nhiệt độ nào công suất giảm nhanh nhất?
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Có scatter plot cho thấy quan hệ cong
   ☐ ⭐ Có residual plot TRƯỚC (hình chữ U) và SAU (phân tán đều)
   ☐ Có đường cong xác thực theo bậc, bậc tối ưu chọn có căn cứ
   ☐ Có bảng: bậc | số cột sinh ra | RMSE train | RMSE test
   ☐ Có so sánh Linear vs Ridge ở bậc cao
   ☐ Giải thích được vì sao phải scale SAU khi tạo đặc trưng đa thức
   ☐ Nêu hạn chế: đa thức bậc cao "phát điên" khi ngoại suy ngoài dải dữ liệu
```

**Mức tham chiếu:** bậc 2–3 thường tối ưu, RMSE ~4,0–4,3 MW (bậc 1 khoảng 4,5–4,6).

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Scale TRƯỚC khi tạo đặc trưng đa thức | Sai thang đo, mất dấu |
| Dùng Linear thuần ở bậc ≥ 3 | Đa cộng tuyến cực nặng, hệ số nổ |
| Chọn bậc theo RMSE **train** | Luôn chọn bậc cao nhất → overfit |
| Ngoại suy ngoài dải dữ liệu | Đa thức bậc cao cho giá trị vô lý |
| Quên `include_bias=False` | Trùng với intercept của model |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-15-Polynomial-<HoTen>/
├── README.md          ← có residual plot trước/sau
├── notebooks/polynomial_power_plant.ipynb
├── src/train.py
├── models/poly_pipeline.joblib
├── reports/{scatter_AT_PE.png, residual_truoc_sau.png, validation_curve.png}
└── requirements.txt
```

**Mở rộng:**
1. Dùng `interaction_only=True` → chỉ tạo tích chéo, không tạo bậc cao. So sánh.
2. Thử **Spline** (`SplineTransformer`) — linh hoạt hơn đa thức và không "phát điên" khi ngoại suy
3. Chứng minh hiện tượng ngoại suy: dự đoán ở AT = 50°C (ngoài dải dữ liệu) bằng bậc 5 → kết quả vô lý cỡ nào?

**Tham khảo:** [Buổi 13 — Polynomial & Bias-Variance](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
