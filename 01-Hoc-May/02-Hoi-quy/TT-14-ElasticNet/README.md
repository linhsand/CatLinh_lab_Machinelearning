# TT-14 — ELASTICNET
## Dự báo tiêu thụ năng lượng toà nhà khi các biến thiết kế dính chặt nhau

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao) |
| 🧠 **Nhóm** | Hồi quy có regularization kết hợp |
| 🔧 **Thuật toán** | ElasticNet (L1 + L2) |
| 🏭 **Lĩnh vực** | Năng lượng · Thiết kế xây dựng |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   ElasticNet = trộn CẢ HAI hình phạt:

        Loss = MSE + λ·[ ρ·Σ|wᵢ|  +  (1−ρ)/2·Σwᵢ² ]
                        └─ L1 ─┘      └─── L2 ───┘

        ρ (l1_ratio) = 1   → giống hệt LASSO
        ρ = 0              → giống hệt RIDGE
        ρ = 0,5            → cân bằng cả hai
```

**Nó sinh ra để vá đúng điểm yếu của Lasso:** khi 2 biến tương quan 0,95, Lasso chọn
ngẫu nhiên 1 biến và bỏ hẳn biến kia. ElasticNet **giữ cả nhóm** biến tương quan
(hiệu ứng gom nhóm — grouping effect) mà vẫn loại được biến vô dụng.

| | Ridge | Lasso | **ElasticNet** |
|---|---|---|---|
| Đưa hệ số về 0 | ❌ | ✅ | ✅ |
| Giữ nhóm biến tương quan | ✅ | ❌ | ✅ |
| Khi số biến > số mẫu | Kém | Tối đa n biến | ✅ Tốt nhất |

---

## 2. BÀI TOÁN THỰC TẾ

```
   Công ty thiết kế cần ước tính TẢI SƯỞI và TẢI LÀM MÁT của toà nhà
   NGAY TỪ BẢN VẼ, trước khi xây, để chọn công suất điều hoà.

   Chọn thừa công suất → lãng phí đầu tư + tốn điện vận hành
   Chọn thiếu công suất → toà nhà không đủ mát → phải cải tạo, cực đắt

   ⚠️ Đặc thù dữ liệu: các biến thiết kế DÍNH CHẶT nhau về mặt hình học
      Diện tích tường ↔ Diện tích mái ↔ Chiều cao ↔ Diện tích sàn
      (đổi 1 cái là các cái kia đổi theo — ràng buộc vật lý)
   → Lasso sẽ bỏ oan biến quan trọng. ElasticNet là lựa chọn đúng.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Energy Efficiency (UCI) |
| **Link** | https://archive.ics.uci.edu/dataset/242/energy+efficiency |
| **Kích thước** | 768 dòng × 8 đặc trưng |
| **Nhãn** | `Y1` = tải sưởi · `Y2` = tải làm mát (**2 bài hồi quy**) |

**Đặc trưng:** `X1` độ gọn tương đối, `X2` diện tích bề mặt, `X3` diện tích tường,
`X4` diện tích mái, `X5` chiều cao tổng, `X6` hướng nhà, `X7` diện tích kính,
`X8` phân bố kính

### ⚠️ Lưu ý dữ liệu

```
   1. X1, X2, X4, X5 tương quan gần như HOÀN HẢO (|r| > 0,95)
      → chính là lý do chọn bộ này cho ElasticNet.

   2. X6 (hướng nhà) và X8 (phân bố kính) là biến PHÂN LOẠI mã hoá bằng số
      → phải one-hot, không để dạng số có thứ tự.

   3. Bộ chỉ có 768 dòng → dùng cross-validation, đừng tin 1 lần chia train/test.
```

---

## 4. HƯỚNG ĐI ĐÚNG

```python
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

pipe = Pipeline([
    ('scale', StandardScaler()),                       # ⭐ BẮT BUỘC
    ('en', ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0], # dò cả tỉ lệ trộn
        alphas=np.logspace(-4, 1, 100),
        cv=5, max_iter=50000, random_state=42)),
])
pipe.fit(X_train, y_train)
print("alpha:", pipe['en'].alpha_, "| l1_ratio:", pipe['en'].l1_ratio_)
```

> 💡 `l1_ratio` gần 1 → dữ liệu ưa Lasso (ít biến thật sự quan trọng).
> `l1_ratio` gần 0 → dữ liệu ưa Ridge (nhiều biến cùng đóng góp).
> Con số máy chọn ra chính là **câu trả lời về bản chất dữ liệu**.

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Nạp dữ liệu, tính ma trận tương quan + VIF → xác nhận đa cộng tuyến nặng
   ☐ 2. One-hot X6, X8; chuẩn hoá các biến số
   ☐ 3. Baseline: DummyRegressor + Linear Regression
   ☐ 4. Chạy 3 model trên CÙNG dữ liệu: Ridge · Lasso · ElasticNet
   ☐ 5. ⭐ BẢNG SO SÁNH: mỗi model giữ bao nhiêu biến? RMSE bao nhiêu?
   ☐ 6. ⭐ Kiểm chứng HIỆU ỨNG GOM NHÓM:
        • Lasso giữ X1 hay X4 hay X5? (nó sẽ chọn 1, bỏ phần còn lại)
        • ElasticNet giữ mấy biến trong nhóm đó?
   ☐ 7. Vẽ heatmap RMSE theo lưới (alpha × l1_ratio)
   ☐ 8. Làm CẢ HAI nhãn Y1 và Y2 → so sánh: biến nào quan trọng cho sưởi,
        biến nào cho làm mát? (kết quả thường KHÁC NHAU — có ý nghĩa kỹ thuật)
   ☐ 9. Kiểm tra ổn định: bootstrap 100 lần → hệ số ElasticNet dao động bao nhiêu?
   ☐ 10. ✍️ Đề xuất 3 thay đổi thiết kế giúp giảm tải năng lượng
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Có bảng VIF chứng minh đa cộng tuyến
   ☐ Có bảng so sánh Ridge / Lasso / ElasticNet (số biến giữ + RMSE)
   ☐ ⭐ Chỉ rõ được hiệu ứng gom nhóm: Lasso bỏ biến nào mà ElasticNet giữ
   ☐ Có heatmap alpha × l1_ratio
   ☐ Làm đủ cả 2 nhãn Y1 và Y2, có so sánh
   ☐ Giải thích được ý nghĩa của l1_ratio mà máy chọn
   ☐ RMSE tốt hơn baseline rõ rệt
```

**Mức tham chiếu:** R² ~0,90–0,92 cho Y1 (tải sưởi). Đây là bộ dữ liệu mô phỏng
nên quan hệ rất sạch — đừng kỳ vọng dữ liệu thật cũng đẹp như vậy.

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Quên chuẩn hoá | Phạt bất công giữa các biến khác đơn vị |
| Chỉ dò `alpha`, cố định `l1_ratio` | Bỏ lỡ điểm tối ưu thật |
| Để X6, X8 dạng số | Model hiểu nhầm "hướng 4 > hướng 2" |
| Tin 1 lần chia train/test với 768 dòng | Phương sai lớn → dùng cross-validation |
| `max_iter` mặc định | Cảnh báo không hội tụ |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-14-ElasticNet-<HoTen>/
├── README.md          ← có mục "HIỆU ỨNG GOM NHÓM"
├── notebooks/elasticnet_energy.ipynb
├── src/train.py
├── models/{elasticnet_Y1.joblib, elasticnet_Y2.joblib}
├── reports/{vif_table.csv, so_sanh_3_model.png, heatmap_alpha_l1ratio.png}
└── requirements.txt
```

**Mở rộng:**
1. Thêm đặc trưng tương tác (`X3 × X7`) → ElasticNet có tự loại bớt không?
2. So sánh với Gradient Boosting Regressor (TT-18) — mất tính giải thích để đổi lấy bao nhiêu % RMSE?
3. Dự đoán đồng thời Y1 và Y2 bằng `MultiTaskElasticNet` — có tốt hơn 2 model riêng không?

**Tham khảo:** [Buổi 13 — Regularization](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
