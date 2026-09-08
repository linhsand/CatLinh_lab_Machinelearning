# TT-20 — SVR (SUPPORT VECTOR REGRESSION)
## Dự đoán cường độ chịu nén bê tông trước khi đổ móng

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao) |
| 🧠 **Nhóm** | Hồi quy phi tuyến bằng kernel |
| 🔧 **Thuật toán** | SVR (Support Vector Regression) |
| 🏭 **Lĩnh vực** | Xây dựng · Vật liệu · Kiểm định chất lượng |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

SVM (TT-05) dùng cho hồi quy — nhưng mục tiêu **ngược lại**:

```
   SVM phân loại : tìm lề RỘNG NHẤT giữa 2 lớp
   SVR hồi quy   : tìm "ống" hẹp bề rộng ε chứa CÀNG NHIỀU điểm càng tốt

              ╱‾‾‾‾‾‾‾‾  ← biên trên (+ε)
        •  ╱ • •  •
         ╱  •  •  •      ← đường hồi quy
       ╱ •  •   •
     ╱________________   ← biên dưới (−ε)

   Điểm NẰM TRONG ống → sai số = 0, KHÔNG bị phạt (ε-insensitive loss)
   Chỉ điểm NGOÀI ống mới bị phạt → những điểm đó là support vectors.
```

**Ưu điểm:** không bị outlier kéo lệch mạnh như MSE, vì sai số nhỏ hơn ε bị bỏ qua hoàn toàn.

---

## 2. BÀI TOÁN THỰC TẾ

```
   Trạm trộn bê tông cần biết cường độ chịu nén (MPa) của mẻ trộn.
   Kiểm định thật: đúc mẫu, chờ ĐỦ 28 NGÀY rồi mới nén thử.

   → Nếu mẻ không đạt, công trình đã đổ móng xong từ lâu → phải đục phá, cực đắt.
   → Cần dự đoán NGAY từ tỉ lệ phối trộn để điều chỉnh trước khi đổ.

   ⚠️ Bài toán này ưu tiên KHÔNG ĐÁNH GIÁ CAO QUÁ (dự báo cường độ cao hơn thực tế
      → nguy hiểm kết cấu). Nên xem xét dự báo phân vị thấp thay vì trung bình.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Concrete Compressive Strength (UCI) |
| **Link** | https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength |
| **Kích thước** | 1.030 dòng × 8 đặc trưng |
| **Nhãn** | `Concrete compressive strength` (MPa), khoảng 2–83 |

**Đặc trưng (kg/m³ trừ cột cuối):** `Cement`, `Blast Furnace Slag`, `Fly Ash`,
`Water`, `Superplasticizer`, `Coarse Aggregate`, `Fine Aggregate`, `Age` (ngày)

### ⚠️ Lưu ý

```
   1. `Age` từ 1 đến 365 ngày, lệch phải rất mạnh (nhiều mẫu 28 ngày)
      → quan hệ cường độ ~ log(Age) → cân nhắc thêm cột log(Age)

   2. Tỉ lệ NƯỚC/XI MĂNG là yếu tố quyết định trong ngành xây dựng
      → BẮT BUỘC tạo đặc trưng: water_cement_ratio = Water / Cement
      → đây là kiến thức miền, model không tự nghĩ ra được

   3. Bộ chỉ 1.030 dòng → SVR chạy nhanh, rất hợp.
      (SVR có độ phức tạp ~O(n²)–O(n³), không dùng cho dữ liệu > 50.000 dòng)
```

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. ⭐ SVR bắt buộc chuẩn hoá CẢ X LẪN y

```
   Khác với hầu hết model khác: SVR nhạy với thang đo của cả ĐẦU RA.
   ε mặc định = 0,1 — với y tính bằng MPa (2–83) thì ε=0,1 là cực nhỏ
   → gần như mọi điểm nằm ngoài ống → mất hết lợi ích của ε-insensitive.

   → Dùng TransformedTargetRegressor để scale y tự động.
```

```python
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

svr = Pipeline([('scale', StandardScaler()),
                ('svr', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1))])

model = TransformedTargetRegressor(regressor=svr, transformer=StandardScaler())
```

### 4.2. Ba siêu tham số

| | Ý nghĩa | Ảnh hưởng |
|---|---|---|
| `C` | Mức phạt điểm ngoài ống | Lớn → bám sát dữ liệu, dễ overfit |
| `epsilon` | Bề rộng ống | Lớn → ít support vector, model đơn giản hơn |
| `gamma` | Tầm ảnh hưởng 1 điểm (RBF) | Lớn → ranh giới ngoằn ngoèo, overfit |

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. EDA: scatter từng đặc trưng vs cường độ; heatmap tương quan
   ☐ 2. ⭐ Tạo đặc trưng miền: water_cement_ratio, tong_chat_ket_dinh, log(Age)
   ☐ 3. Baseline: DummyRegressor + Linear Regression
   ☐ 4. ⚠️ Chạy SVR KHÔNG scale → ghi lại (kết quả sẽ rất tệ)
   ☐ 5. SVR có scale cả X và y → so sánh, chứng minh
   ☐ 6. So sánh 3 kernel: linear / rbf / poly (degree 2, 3)
   ☐ 7. GridSearchCV: C ∈ {1,10,100,1000} × gamma ∈ {scale,0.01,0.1,1}
        × epsilon ∈ {0.01,0.1,0.5}
   ☐ 8. Vẽ heatmap RMSE theo (C, gamma)
   ☐ 9. Đếm support vectors: model.n_support_ → chiếm % bao nhiêu dữ liệu?
        (ít SV = model gọn, tổng quát hoá tốt)
   ☐ 10. Đo hiệu quả của đặc trưng miền (water/cement): RMSE trước vs sau
   ☐ 11. So sánh SVR vs XGBoost (TT-19) vs Random Forest — bảng RMSE + thời gian
   ☐ 12. ⚠️ Đo thời gian train khi nhân dữ liệu lên 10× (lặp lại dữ liệu)
         → chứng minh SVR không mở rộng được
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Có bảng so sánh CÓ/KHÔNG chuẩn hoá (cả X và y)
   ☐ ⭐ Có đặc trưng miền water_cement_ratio + đo được mức cải thiện
   ☐ Có so sánh 3 kernel
   ☐ Có heatmap C × gamma
   ☐ Báo cáo số support vectors và tỉ lệ %
   ☐ Có thí nghiệm thời gian train theo kích thước dữ liệu
   ☐ R² test > 0,88
   ☐ Nêu hạn chế: SVR không giải thích được từng dự đoán, không mở rộng được
```

**Mức tham chiếu:** R² ~0,88–0,92 · RMSE ~4,5–5,5 MPa (XGBoost thường tốt hơn chút).

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Chỉ scale X, quên scale y | ε=0,1 vô nghĩa với y đơn vị MPa |
| Dùng SVR cho dữ liệu lớn | Train hàng giờ, có khi không hội tụ |
| `gamma` quá lớn | Overfit nặng |
| Bỏ qua kiến thức miền | Mất đặc trưng mạnh nhất (tỉ lệ nước/xi măng) |
| Dự báo cường độ cao hơn thực tế | Rủi ro AN TOÀN KẾT CẤU — cần phân tích riêng |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-20-SVR-<HoTen>/
├── README.md          ← có mục "ĐẶC TRƯNG TỪ KIẾN THỨC MIỀN"
├── notebooks/svr_concrete.ipynb
├── src/train.py
├── models/svr_pipeline.joblib
├── reports/{scale_vs_noscale.png, kernel_comparison.png, C_gamma_heatmap.png, thoi_gian_train.png}
└── requirements.txt
```

**Mở rộng:**
1. `LinearSVR` — nhanh hơn rất nhiều, chính xác kém bao nhiêu?
2. Phân tích an toàn: đếm số mẫu bị dự báo CAO HƠN thực tế > 5 MPa (rủi ro kết cấu)
3. Dùng `GradientBoostingRegressor(loss='quantile', alpha=0.1)` để dự báo cận dưới an toàn

**Tham khảo:** [Buổi 4 — SVM](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-04-LogReg-SVM-Metrics/Tai-Lieu) · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
