# TT-11 — LINEAR REGRESSION
## Định giá nhà ở — model nền tảng của mọi bài hồi quy

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 1](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-01-Gioi-thieu-ML) + [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao) |
| 🧠 **Nhóm** | Hồi quy tuyến tính |
| 🔧 **Thuật toán** | Linear Regression (OLS) |
| 🏭 **Lĩnh vực** | Bất động sản · Thẩm định giá |
| ⏱ **Thời lượng** | 4–6 giờ |
| 📈 **Độ khó** | ⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
        ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

   HỌC = tìm bộ (w, b) làm SAI SỐ BÌNH PHƯƠNG nhỏ nhất:

        MSE = (1/n) · Σ (y − ŷ)²

   Giá (tỷ)
      8 │              ⭐
        │          ⭐╱
      4 │      ⭐╱          ← đường thẳng "khớp" nhất với dữ liệu
        │  ⭐╱
      0 └──────────────── Diện tích
```

**Vì sao vẫn học đầu tiên năm 2026:** đây là model **duy nhất** mà mỗi hệ số có ý
nghĩa trực tiếp ("thêm 1 m² thì giá tăng X triệu") — thẩm định viên đọc được ngay.

---

## 2. BÀI TOÁN THỰC TẾ

```
   Sàn giao dịch BĐS cần ước tính giá tham chiếu ngay khi chủ nhà đăng tin,
   để: ① cảnh báo tin đăng giá bất thường  ② gợi ý khoảng giá hợp lý

   Yêu cầu: phải GIẢI THÍCH được cho khách "vì sao hệ thống định giá vậy".
   → Bắt buộc dùng model tuyến tính, không dùng hộp đen.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | California Housing |
| **Cách lấy** | `from sklearn.datasets import fetch_california_housing` |
| **Kích thước** | 20.640 dòng × 8 đặc trưng |
| **Nhãn** | `MedHouseVal` — giá trung vị (đơn vị 100.000 USD) |

**Cột:** `MedInc` (thu nhập trung vị), `HouseAge`, `AveRooms`, `AveBedrms`,
`Population`, `AveOccup`, `Latitude`, `Longitude`

> ⚠️ **Không dùng bộ Boston Housing** — đã bị gỡ khỏi scikit-learn từ v1.2 vì
> chứa biến phân biệt chủng tộc. Nếu thấy tài liệu cũ hướng dẫn dùng Boston, hãy bỏ qua.

### ⚠️ Bẫy dữ liệu

```
   1. NHÃN BỊ CẮT NGỌN: MedHouseVal tối đa = 5,0 (500k USD)
      → có ~1.000 căn bị "dồn" vào đúng giá 5,0
      → model không bao giờ dự đoán được nhà đắt hơn → phải nêu trong hạn chế

   2. AveRooms, AveOccup có outlier cực đoan (AveOccup > 1000!)
      → là lỗi dữ liệu ở vùng dân cư đặc biệt → clip theo phân vị 99

   3. Latitude/Longitude là toạ độ — quan hệ với giá là PHI TUYẾN
      → model tuyến tính bắt kém → gợi ý tạo đặc trưng khoảng cách tới trung tâm
```

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. Bốn giả định của hồi quy tuyến tính — phải kiểm tra

```
   ① TUYẾN TÍNH     quan hệ x → y phải gần thẳng    → vẽ scatter, residual plot
   ② ĐỘC LẬP        các quan sát không phụ thuộc nhau
   ③ PHƯƠNG SAI ĐỀU sai số phân tán đều              → residual vs fitted plot
   ④ SAI SỐ CHUẨN   phần dư phân phối chuẩn          → Q-Q plot

   Vi phạm ③ (heteroscedasticity) rất hay gặp với dữ liệu giá:
   nhà đắt thì sai số cũng lớn hơn → gợi ý dự đoán log(giá) thay vì giá.
```

### 4.2. Code khung

```python
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np, pandas as pd

pipe = Pipeline([('scale', StandardScaler()), ('lr', LinearRegression())])
pipe.fit(X_train, y_train)

# Diễn giải hệ số (đã scale nên so sánh được độ lớn)
he_so = pd.Series(pipe['lr'].coef_, index=X.columns).sort_values(key=abs, ascending=False)
```

### 4.3. Metric

```
   RMSE  ← chính, cùng đơn vị với giá, dễ giải thích cho nghiệp vụ
   MAE   ← ít nhạy outlier hơn
   R²    ← % biến động được giải thích
   MAPE  ← sai số theo %, so sánh được giữa phân khúc giá
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Nạp dữ liệu, describe() → phát hiện outlier AveOccup, AveRooms
   ☐ 2. Phát hiện nhãn bị cắt ngọn ở 5,0 (đếm số dòng)
   ☐ 3. EDA: scatter MedInc vs giá · heatmap tương quan · bản đồ giá theo toạ độ
   ☐ 4. Baseline: DummyRegressor(strategy='mean') → RMSE bao nhiêu?
   ☐ 5. Linear Regression cơ bản → RMSE, MAE, R²
   ☐ 6. ⭐ Vẽ RESIDUAL PLOT (phần dư vs giá dự đoán)
        → có hình phễu không? (dấu hiệu phương sai không đều)
   ☐ 7. Vẽ Q-Q plot kiểm tra phân phối phần dư
   ☐ 8. Thử dự đoán log(giá) thay vì giá → residual plot cải thiện không?
   ☐ 9. Kiểm tra ĐA CỘNG TUYẾN bằng VIF (AveRooms ↔ AveBedrms tương quan cao)
   ☐ 10. Feature engineering: rooms_per_household, khoảng cách tới (SF, LA)
   ☐ 11. Bảng hệ số đã chuẩn hoá → diễn giải 3 yếu tố ảnh hưởng mạnh nhất
   ☐ 12. So sánh với Ridge (TT-12) và Random Forest Regressor (TT-17)
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Đã phát hiện & nêu vấn đề nhãn bị cắt ngọn ở 5,0
   ☐ Có residual plot + Q-Q plot + nhận xét về giả định
   ☐ Có kiểm tra VIF
   ☐ Có bảng hệ số đã chuẩn hoá + diễn giải bằng ngôn ngữ nghiệp vụ
   ☐ RMSE tốt hơn baseline rõ rệt
   ☐ Có thử nghiệm log-transform và kết luận
   ☐ Nêu hạn chế: quan hệ toạ độ–giá là phi tuyến, model tuyến tính bắt kém
```

**Mức tham chiếu:** R² ~0,58–0,61 · RMSE ~0,72–0,75 (đơn vị 100k USD).
Random Forest đạt R² ~0,80 — chênh lệch này chính là **cái giá của tính giải thích được**.

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Dùng bộ Boston Housing | Bộ đã bị gỡ vì lý do đạo đức |
| Bỏ qua nhãn bị cắt ngọn | Dự đoán sai toàn bộ phân khúc cao cấp |
| Không kiểm tra giả định | Kết luận sai về hệ số |
| Không chuẩn hoá rồi so sánh hệ số | So sánh vô nghĩa (khác đơn vị) |
| Bỏ qua đa cộng tuyến | Hệ số nhảy loạn, diễn giải sai |
| Kết luận NHÂN QUẢ từ hệ số | "Tăng số phòng LÀM giá tăng" — chưa chứng minh |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-11-LinearRegression-<HoTen>/
├── README.md          ← có phần kiểm tra 4 giả định
├── notebooks/linear_regression_housing.ipynb
├── src/train.py
├── models/lr_pipeline.joblib
├── reports/{residual_plot.png, qq_plot.png, he_so.png, ban_do_gia.png}
└── requirements.txt
```

**Mở rộng:**
1. Dùng `statsmodels.OLS` để có **p-value** và khoảng tin cậy cho từng hệ số
2. Thêm đặc trưng tương tác (`MedInc × HouseAge`) → R² cải thiện bao nhiêu?
3. Hồi quy có trọng số (WLS) để xử lý phương sai không đều

**Tham khảo:** [Buổi 13 — Regression nâng cao](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
