# TT-19 — XGBOOST REGRESSOR
## Dự báo nhu cầu thuê xe đạp công cộng theo giờ để điều phối xe

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao) |
| 🧠 **Nhóm** | Hồi quy · Boosting · Dữ liệu có yếu tố thời gian |
| 🔧 **Thuật toán** | XGBoost Regressor |
| 🏭 **Lĩnh vực** | Vận tải công cộng · Chia sẻ phương tiện |
| ⏱ **Thời lượng** | 7–9 giờ |
| 📈 **Độ khó** | ⭐⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

XGBoost = Gradient Boosting (TT-18) + regularization trong hàm mục tiêu + tối ưu
kỹ thuật. Với dữ liệu bảng, đây là thuật toán **mặc định nên thử** năm 2026.

```
   Hàm mục tiêu = Σ Loss(y, ŷ) + Σ Ω(cây)
                                  └─ Ω = γ·T + ½λ·Σw² ─┘
                                     T = số lá · w = giá trị lá
   → Phạt cây có NHIỀU LÁ và lá có giá trị LỚN
   → Chống overfit ngay trong công thức, không chỉ dựa vào cắt tỉa
```

---

## 2. BÀI TOÁN THỰC TẾ

```
   Hệ thống xe đạp công cộng có 500 trạm. Mỗi sáng phải điều xe tải
   chở xe từ trạm thừa sang trạm thiếu.

   Dự báo THIẾU  → khách tới không có xe → mất doanh thu, mất khách
   Dự báo THỪA   → điều xe tải vô ích → tốn chi phí vận hành

   ⚠️ SAI SỐ KHÔNG CÂN XỨNG: thiếu xe tệ hơn thừa xe
      → cân nhắc hàm loss bất đối xứng, hoặc dự báo phân vị 70% thay vì trung vị.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Bike Sharing Dataset (UCI) |
| **Link** | https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset |
| **File** | `hour.csv` — 17.379 dòng × 17 cột (2 năm, theo giờ) |
| **Nhãn** | `cnt` = tổng lượt thuê trong giờ đó |

**Cột:** `dteday`, `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`,
`weathersit`, `temp`, `atemp`, `hum`, `windspeed`, `casual`, `registered`, `cnt`

### 🚨 Hai bẫy nghiêm trọng

```
   BẪY 1 — RÒ RỈ TRỰC TIẾP
     cnt = casual + registered  (đúng bằng tổng hai cột kia!)
     → Giữ `casual` và `registered` làm đặc trưng → R² = 1,0
     → PHẢI BỎ cả hai cột.

   BẪY 2 — CHIA DỮ LIỆU NGẪU NHIÊN
     Đây là dữ liệu THEO THỜI GIAN → chia ngẫu nhiên là nhìn thấy tương lai.
     ✅ Chia theo thời gian: năm 1 train · 9 tháng đầu năm 2 validation ·
        3 tháng cuối test.
```

### Đặc trưng chu kỳ

```
   `hr` (0–23): giờ 23 và giờ 0 KỀ NHAU thực tế nhưng cách xa về giá trị
   → mã hoá sin/cos:
        hr_sin = sin(2π·hr/24)     hr_cos = cos(2π·hr/24)
   → làm tương tự cho `mnth` (chu kỳ 12) và `weekday` (chu kỳ 7)
```

---

## 4. HƯỚNG ĐI ĐÚNG

```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, reg_alpha=0.1, min_child_weight=3,
    objective='reg:squarederror',
    early_stopping_rounds=100, eval_metric='rmse',
    tree_method='hist', n_jobs=-1, random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)
```

**Xử lý nhãn lệch phải (nhiều giờ đêm có cnt rất nhỏ):**
```python
import numpy as np
y_log = np.log1p(y_train)                       # train trên log
du_doan = np.expm1(model.predict(X_test))       # đổi ngược
du_doan = np.clip(du_doan, 0, None)             # ⭐ số lượt thuê không thể ÂM
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Nạp hour.csv, BỎ `casual`, `registered`, `dteday` (đã tách thành yr/mnth/hr)
   ☐ 2. ⭐ Chứng minh rò rỉ: chạy 1 lần CÓ giữ casual/registered → R² ≈ 1,0
        → viết nhận xét, sau đó bỏ đi
   ☐ 3. Chia THEO THỜI GIAN (không shuffle)
   ☐ 4. EDA: cnt trung bình theo giờ (2 đỉnh: 8h và 17–18h) · theo mùa · theo thời tiết
   ☐ 5. Mã hoá sin/cos cho hr, mnth, weekday
   ☐ 6. Baseline: dự báo naive "bằng cùng giờ tuần trước" → RMSE bao nhiêu?
        ⚠️ Baseline này rất mạnh, nhiều model phức tạp không thắng nổi
   ☐ 7. XGBoost + early stopping trên tập validation
   ☐ 8. So sánh có/không dùng log1p cho nhãn
   ☐ 9. Dò siêu tham số bằng RandomizedSearchCV (nhanh hơn Grid nhiều)
   ☐ 10. Feature importance (gain) + SHAP summary
   ☐ 11. ⭐ Vẽ dự báo vs thực tế trên 2 tuần cuối → nhìn model sai ở đâu
   ☐ 12. Phân tích lỗi: model sai nhiều nhất ở giờ nào, điều kiện thời tiết nào?
   ☐ 13. So sánh với Random Forest (TT-17) và Gradient Boosting (TT-18)
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Đã bỏ casual/registered + có phần chứng minh rò rỉ
   ☐ Chia dữ liệu THEO THỜI GIAN
   ☐ Có mã hoá chu kỳ sin/cos
   ☐ ⭐ Có baseline naive "cùng giờ tuần trước" và XGBoost phải THẮNG nó
   ☐ Có early stopping (báo cáo số cây thực tế dùng)
   ☐ Có biểu đồ dự báo vs thực tế 2 tuần cuối
   ☐ Có phân tích lỗi theo giờ/thời tiết
   ☐ Dự đoán được clip về ≥ 0
```

**Mức tham chiếu:** RMSE ~40–55 lượt/giờ · R² ~0,93–0,95 trên tập test theo thời gian.

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Giữ `casual` + `registered` | R² = 1,0 — rò rỉ tuyệt đối |
| Chia ngẫu nhiên | Nhìn thấy tương lai → điểm ảo |
| Để `hr` dạng số thường | Mất tính chu kỳ 23→0 |
| Không clip dự đoán về ≥ 0 | Báo "âm 5 lượt thuê" |
| Bỏ qua baseline naive | Không biết model có thật sự hữu ích không |
| Dùng `gpu_hist` trên máy không GPU | Lỗi hoặc chậm hơn `hist` |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-19-XGBoostRegressor-<HoTen>/
├── README.md          ← có mục "CHỨNG MINH RÒ RỈ" và so sánh với baseline naive
├── notebooks/xgboost_bike_demand.ipynb
├── src/{features.py, train.py}
├── models/xgb_bike.json
├── reports/{cnt_theo_gio.png, du_bao_vs_thuc_te.png, shap_summary.png, phan_tich_loi.png}
└── requirements.txt
```

**Mở rộng:**
1. Hàm loss BẤT ĐỐI XỨNG: phạt dự báo thiếu nặng gấp 2 lần dự báo thừa (custom objective)
2. Dự báo nhiều bước: 1 giờ tới, 6 giờ tới, 24 giờ tới — độ chính xác giảm thế nào?
3. Thêm dữ liệu thời tiết dự báo (thay vì thời tiết thực tế) → sát thực tế vận hành hơn

**Tham khảo:** [Buổi 9 — Time Series](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-09-TimeSeries/Tai-Lieu/ly_thuyet_chi_tiet_buoi_09.md) · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
