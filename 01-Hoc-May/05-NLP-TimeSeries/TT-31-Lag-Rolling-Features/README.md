# TT-31 — LAG + ROLLING FEATURES
## Dự báo phụ tải điện ngày mai để lập kế hoạch phát điện

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 9](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-09-TimeSeries) |
| 🧠 **Nhóm** | Chuỗi thời gian · **Feature engineering** |
| 🔧 **Kỹ thuật** | Lag features · Rolling statistics · Mã hoá chu kỳ |
| 🏭 **Lĩnh vực** | Điện lực · Vận hành hệ thống |
| ⏱ **Thời lượng** | 7–9 giờ |
| 📈 **Độ khó** | ⭐⭐⭐ |

---

## 1. KỸ THUẬT NÀY LÀ GÌ

Biến bài toán **chuỗi thời gian** thành bài toán **bảng thông thường** để dùng được
Random Forest / XGBoost — những thuật toán mạnh nhất cho dữ liệu bảng.

```
   Ngày   phu_tai   lag_1    lag_7    trung_binh_7
   01/01   1250      NaN      NaN        NaN
   02/01   1310     1250      NaN        NaN
   ...
   08/01   1400     1380     1250       1322      ← trung bình 7 ngày TRƯỚC ĐÓ

   → Sau khi có các cột này, mỗi dòng ĐỘC LẬP → dùng được mọi thuật toán ML.
```

**Vì sao đây là kỹ thuật quan trọng nhất trong dự báo thực tế:** nó cho phép đưa
**biến ngoài** (thời tiết, ngày lễ, giá điện) vào model — thứ mà ARIMA (TT-30) rất khó làm.

---

## 2. BÀI TOÁN THỰC TẾ

```
   Trung tâm Điều độ Hệ thống điện phải công bố kế hoạch huy động nguồn
   cho NGÀY MAI trước 16h hôm nay.

   Dự báo THIẾU → phải mua điện giá cao trên thị trường tức thời, hoặc SA THẢI PHỤ TẢI
   Dự báo THỪA → nhà máy chạy không tải, đốt nhiên liệu vô ích

   Sai 1% phụ tải của một hệ thống trung bình ≈ hàng trăm triệu đồng/ngày.

   → Đây là bài toán mà độ chính xác quy đổi TRỰC TIẾP ra tiền.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Individual Household Electric Power Consumption (UCI) |
| **Link** | https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption |
| **Kích thước** | 2.075.259 dòng (đo mỗi PHÚT, 2006–2010) |
| **Nhãn** | `Global_active_power` (kW) |

**Nguồn thay thế:** ElectricityLoadDiagrams20112014 (UCI, 370 khách hàng) hoặc dữ liệu
phụ tải công bố của EVN.

### ⚠️ Bốn bước chuẩn bị bắt buộc

```
   1. GỘP VỀ THEO GIỜ hoặc THEO NGÀY
      2 triệu dòng theo phút quá chi tiết và nhiễu → resample('h').mean()

   2. GIÁ TRỊ THIẾU ghi bằng dấu '?'
      → na_values='?' khi đọc; có ~1,25% dòng thiếu

   3. Ghép cột Date + Time thành một cột datetime rồi set làm index

   4. Có LỖ HỔNG thời gian → reindex đầy đủ, đánh dấu đoạn thiếu
```

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. 🚨 QUY TẮC SỐ 1 — luôn `.shift(1)` TRƯỚC `.rolling()`

```python
# ❌ SAI — RÒ RỈ:
df['tb_7'] = df['phu_tai'].rolling(7).mean()
#   cửa sổ tại thời điểm t BAO GỒM CHÍNH yₜ → dùng đáp án để đoán đáp án

# ✅ ĐÚNG:
df['tb_7'] = df['phu_tai'].shift(1).rolling(7).mean()
#   cửa sổ chỉ gồm t−1 … t−7 → chỉ dùng thông tin ĐÃ BIẾT
```

> Đây là lỗi rò rỉ **tinh vi nhất** trong toàn bộ mảng chuỗi thời gian. Rất nhiều
> người làm sai mà không phát hiện, chỉ thấy "model tốt bất thường".

### 4.2. Bộ đặc trưng đầy đủ

```python
import numpy as np

# Lag — giá trị quá khứ
for lag in [1, 2, 3, 7, 14, 30]:
    df[f'lag_{lag}'] = df['phu_tai'].shift(lag)

# Rolling — thống kê cửa sổ quá khứ
for w in [7, 14, 30]:
    s = df['phu_tai'].shift(1)
    df[f'tb_{w}']    = s.rolling(w).mean()
    df[f'std_{w}']   = s.rolling(w).std()
    df[f'max_{w}']   = s.rolling(w).max()
    df[f'min_{w}']   = s.rolling(w).min()

# Lịch + mã hoá chu kỳ
df['thu']    = df.index.dayofweek
df['thang']  = df.index.month
df['cuoi_tuan'] = (df['thu'] >= 5).astype(int)
df['thang_sin'] = np.sin(2*np.pi*df['thang']/12)
df['thang_cos'] = np.cos(2*np.pi*df['thang']/12)
```

### 4.3. ⚠️ Nhược điểm phải xử lý: cây KHÔNG ngoại suy

```
   Phụ tải có xu hướng TĂNG dần qua các năm.
   XGBoost chỉ dự đoán trong khoảng giá trị đã thấy lúc train
   → dự báo năm sau sẽ ĐI NGANG ở mức cao nhất đã gặp.

   Cách chữa: dự đoán CHÊNH LỆCH thay vì giá trị tuyệt đối
        y' = yₜ − yₜ₋₁      (hoặc yₜ − yₜ₋₃₆₅)
        rồi cộng ngược lại khi báo cáo.
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Đọc dữ liệu với na_values='?', ghép datetime, resample về theo NGÀY
   ☐ 2. Xử lý thiếu + lỗ hổng thời gian, ghi lại % dữ liệu bị ảnh hưởng
   ☐ 3. EDA: phụ tải theo mùa · theo thứ trong tuần · so cuối tuần vs ngày thường
   ☐ 4. Tạo lag + rolling + đặc trưng lịch (LUÔN có .shift(1))
   ☐ 5. ⚠️ Xoá các dòng đầu bị NaN — KHÔNG điền bằng mean (là bịa quá khứ)
   ☐ 6. Chia theo thời gian: 2006–2008 train · 2009 val · 2010 test
   ☐ 7. ⭐ 2 BASELINE: naive (hôm qua) và seasonal naive (thứ này tuần trước)
   ☐ 8. Ridge · Random Forest · XGBoost trên cùng bộ đặc trưng → bảng so sánh
   ☐ 9. ⭐ THÍ NGHIỆM RÒ RỈ: cố tình tạo rolling KHÔNG shift
        → MAE giảm bao nhiêu một cách giả tạo? Viết nhận xét.
   ☐ 10. Feature importance → lag nào quan trọng nhất? (thường lag_1 và lag_7)
   ☐ 11. ⭐ THÍ NGHIỆM NGOẠI SUY: dự báo tuyệt đối vs dự báo chênh lệch
         → so sánh trên năm cuối
   ☐ 12. Vẽ dự báo vs thực tế 60 ngày cuối
   ☐ 13. Dùng TimeSeriesSplit để đánh giá ổn định (KHÔNG dùng KFold thường)
   ☐ 14. Quy đổi sai số ra TIỀN: MAE × giá điện × số ngày
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Mọi rolling feature đều có .shift(1) — kiểm tra lại từng dòng code
   ☐ Chia theo thời gian, dùng TimeSeriesSplit
   ☐ ⭐ Có 2 baseline và model phải thắng seasonal naive
   ☐ ⭐ Có thí nghiệm chứng minh rò rỉ khi quên shift
   ☐ ⭐ Có thí nghiệm ngoại suy (tuyệt đối vs chênh lệch)
   ☐ Có bảng so sánh ≥ 3 thuật toán
   ☐ Có biểu đồ dự báo vs thực tế
   ☐ ⭐ Có quy đổi sai số ra tiền
```

**Mức tham chiếu:** MAPE ~5–8% cho dự báo phụ tải ngày (dữ liệu hộ gia đình đơn lẻ
nhiễu hơn phụ tải hệ thống nhiều).

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| `rolling()` không `.shift(1)` | **Rò rỉ tương lai** — lỗi nghiêm trọng nhất |
| Điền NaN đầu chuỗi bằng mean | Bịa ra dữ liệu quá khứ không tồn tại |
| Dùng KFold thường | Train chứa dữ liệu SAU test |
| Để `thang` dạng số 1–12 | Tháng 12 và tháng 1 bị coi là cách xa nhau |
| Kỳ vọng cây ngoại suy trend | Dự báo đi ngang ở mức cao nhất đã thấy |
| Bỏ qua ngày lễ | Sai lớn vào Tết, lễ dài ngày |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-31-LagRolling-<HoTen>/
├── README.md          ← ⭐ có thí nghiệm rò rỉ + ngoại suy + quy đổi tiền
├── notebooks/lag_rolling_power.ipynb
├── src/{features.py, train.py}
├── models/xgb_power.json
├── reports/{eda_mua_vu.png, thi_nghiem_ro_ri.png, ngoai_suy.png, du_bao_60_ngay.png}
└── requirements.txt
```

**Mở rộng:**
1. Thêm dữ liệu **thời tiết** (nhiệt độ là yếu tố quyết định phụ tải điều hoà)
2. Dự báo **nhiều bước**: 1 ngày / 3 ngày / 7 ngày tới — sai số tăng thế nào?
3. Dự báo **phân vị** (`GradientBoostingRegressor(loss='quantile')`) → cho khoảng
   phụ tải, giúp điều độ lập kế hoạch dự phòng

**Tham khảo:** [Buổi 9 — Lag & Rolling Features](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-09-TimeSeries/Tai-Lieu/ly_thuyet_chi_tiet_buoi_09.md)
