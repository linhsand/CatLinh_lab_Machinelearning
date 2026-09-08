# TT-17 — RANDOM FOREST REGRESSOR
## Dự đoán giá vé máy bay để tư vấn khách "nên mua bây giờ hay chờ"

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao) |
| 🧠 **Nhóm** | Hồi quy · Ensemble (Bagging) |
| 🔧 **Thuật toán** | Random Forest Regressor |
| 🏭 **Lĩnh vực** | Du lịch · Hàng không · OTA |
| ⏱ **Thời lượng** | 6–8 giờ |
| 📈 **Độ khó** | ⭐⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

Nhiều cây hồi quy (TT-16) học trên các mẫu bootstrap khác nhau, kết quả cuối = **TRUNG
BÌNH** dự đoán của tất cả cây (thay vì bỏ phiếu như bản phân loại).

```
   Cây 1 → 3,2 triệu ─┐
   Cây 2 → 3,5 triệu ─┼─▶ TRUNG BÌNH = 3,37 triệu
   Cây 3 → 3,4 triệu ─┘

   Lợi ích kép:
     ① Hàm dự đoán MƯỢT hơn cây đơn (không còn bậc thang thô)
     ② Ổn định — đổi chút dữ liệu không làm kết quả nhảy
```

⚠️ **Vẫn giữ nguyên điểm yếu của cây: KHÔNG NGOẠI SUY ĐƯỢC.** Giá vé vượt ngoài
khoảng đã thấy trong tập train sẽ bị "kẹp trần".

---

## 2. BÀI TOÁN THỰC TẾ

```
   Đại lý vé trực tuyến muốn hiển thị: "Giá hiện tại 3,2 triệu —
   dự báo tuần sau 3,8 triệu → NÊN MUA NGAY."

   Giá vé phụ thuộc phi tuyến và có nhiều tương tác:
     • Số ngày còn lại tới chuyến bay (giá tăng vọt sát ngày)
     • Hãng bay, hạng vé, số điểm dừng
     • Giờ khởi hành (sáng sớm rẻ hơn)

   → Quan hệ nhiều tương tác chéo → cây/ensemble hợp hơn model tuyến tính.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Flight Price Prediction |
| **Link** | https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction |
| **Kích thước** | ~300.000 dòng × 11 cột |
| **Nhãn** | `price` (Rupee Ấn Độ) |

**Cột:** `airline`, `flight`, `source_city`, `departure_time`, `stops`, `arrival_time`,
`destination_city`, `class` (Economy/Business), `duration`, `days_left`, `price`

### ⚠️ Lưu ý

```
   1. BỎ cột `flight` (mã chuyến bay) — hàng nghìn giá trị duy nhất,
      one-hot sẽ tạo hàng nghìn cột và cây sẽ học thuộc mã chuyến.

   2. `class` là biến MẠNH NHẤT (Business đắt gấp ~6 lần Economy).
      → Cân nhắc tách thành 2 model riêng cho 2 hạng vé.

   3. `days_left` có quan hệ PHI TUYẾN rõ rệt: giá gần như phẳng từ 50→20 ngày,
      rồi tăng vọt trong 7 ngày cuối. Đây là thứ cây bắt rất tốt.

   4. Dữ liệu Ấn Độ 2022 → KHÔNG áp dụng trực tiếp cho thị trường Việt Nam.
```

---

## 4. HƯỚNG ĐI ĐÚNG

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=300,
    max_features=1.0,        # ⭐ hồi quy thường dùng nhiều đặc trưng hơn phân loại
    min_samples_leaf=2,
    n_jobs=-1, random_state=42, oob_score=True,
)
```

> 💡 Khác biệt so với bản phân loại: `max_features` mặc định cho **hồi quy** là `1.0`
> (dùng hết đặc trưng), còn **phân loại** là `'sqrt'`. Vẫn nên dò thử `0.3`–`1.0`.

### Dự đoán khoảng tin cậy từ rừng

```python
import numpy as np
# Lấy dự đoán của TỪNG cây → có phân phối → suy ra khoảng
du_doan_tung_cay = np.stack([cay.predict(X_test) for cay in rf.estimators_])
khoang_thap = np.percentile(du_doan_tung_cay, 10, axis=0)
khoang_cao  = np.percentile(du_doan_tung_cay, 90, axis=0)
```

→ Cho phép hiển thị **"giá dự kiến 3,2–3,9 triệu"** thay vì một con số cứng — đúng
nhu cầu nghiệp vụ hơn nhiều.

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Nạp dữ liệu, bỏ cột `flight` và cột index thừa
   ☐ 2. EDA: giá theo `days_left` (vẽ đường) → thấy rõ đoạn tăng vọt
   ☐ 3. EDA: boxplot giá theo `class` và theo `airline`
   ☐ 4. Pipeline: OneHotEncoder cho biến phân loại (cây KHÔNG cần scale)
   ☐ 5. Baseline: DummyRegressor + Linear Regression + 1 cây đơn (TT-16)
   ☐ 6. Random Forest → báo cáo `oob_score_`
   ☐ 7. Vẽ RMSE theo n_estimators = 10..500 → tìm điểm bão hoà
   ☐ 8. Permutation importance (KHÔNG dùng feature_importances_ mặc định —
        nó thiên vị biến nhiều mức như airline)
   ☐ 9. ⭐ Vẽ PDP (Partial Dependence Plot) cho `days_left`
        → định lượng: mua sớm 1 tuần tiết kiệm bao nhiêu tiền?
   ☐ 10. ⭐ Tính khoảng dự báo 10–90% từ các cây, đo % ca giá thật rơi trong khoảng
   ☐ 11. ⚠️ THÍ NGHIỆM NGOẠI SUY: dự đoán cho `days_left = 100`
         (ngoài dải train) → kết quả bị kẹp trần thế nào?
   ☐ 12. So sánh với XGBoost Regressor (TT-19)
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Đã bỏ cột `flight`, giải thích lý do
   ☐ Báo cáo oob_score_
   ☐ Có biểu đồ RMSE theo số cây
   ☐ Dùng permutation importance, không dùng importance mặc định
   ☐ ⭐ Có PDP cho days_left + kết luận bằng con số tiền cụ thể
   ☐ ⭐ Có khoảng dự báo 10–90% + tỉ lệ phủ thực tế
   ☐ Có thí nghiệm ngoại suy + giải thích hạn chế
   ☐ R² test > 0,95 (bộ này quan hệ khá mạnh) — nhưng phải kiểm tra không rò rỉ
```

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Giữ cột `flight` | Hàng nghìn cột one-hot, cây học thuộc mã chuyến |
| Dùng `feature_importances_` mặc định | Thiên vị biến nhiều mức → kết luận sai |
| Kỳ vọng ngoại suy | Giá ngoài dải train bị kẹp trần |
| Chỉ báo 1 con số giá | Nghiệp vụ cần KHOẢNG giá |
| Áp dụng thẳng cho thị trường VN | Dữ liệu Ấn Độ, cấu trúc giá khác hẳn |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-17-RandomForestRegressor-<HoTen>/
├── README.md          ← có PDP + khoảng dự báo
├── notebooks/rf_regressor_flight.ipynb
├── src/train.py
├── models/rf_reg.joblib
├── reports/{gia_theo_days_left.png, pdp_days_left.png, rmse_theo_so_cay.png, khoang_du_bao.png}
└── requirements.txt
```

**Mở rộng:**
1. Tách 2 model riêng cho Economy và Business → tổng RMSE có tốt hơn 1 model chung không?
2. Dùng `ExtraTreesRegressor` (chia ngẫu nhiên hoàn toàn) → nhanh hơn, chính xác tương đương?
3. Xây quy tắc tư vấn: "nếu giá dự báo tuần sau cao hơn hiện tại > 10% → khuyên mua ngay"
   rồi đo hiệu quả trên dữ liệu lịch sử

**Tham khảo:** [Buổi 3 — Random Forest](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-03-Feature-Eng-Tree/Tai-Lieu) · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
