# TT-16 — DECISION TREE REGRESSOR
## Định giá cước chuyến xe — bảng giá dạng LUẬT cho tổng đài

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao) |
| 🧠 **Nhóm** | Hồi quy phi tuyến bằng cây |
| 🔧 **Thuật toán** | Decision Tree Regressor |
| 🏭 **Lĩnh vực** | Vận tải · Logistics · Gọi xe |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

Giống cây phân loại (TT-02) nhưng lá trả về **SỐ TRUNG BÌNH** thay vì nhãn, và tiêu
chí chia là **giảm MSE** thay vì giảm Gini.

```
              Quãng đường > 5 km?
             ╱                  ╲
          KHÔNG                  CÓ
           ╱                      ╲
     Giờ cao điểm?          Quãng đường > 15 km?
      ╱        ╲                ╱          ╲
   45.000đ   62.000đ        180.000đ    310.000đ
   (lá = TRUNG BÌNH giá của các chuyến rơi vào nhánh này)
```

⚠️ **Đặc điểm phải hiểu:** cây hồi quy cho ra hàm **BẬC THANG**, không phải đường
mượt. Số giá trị đầu ra khác nhau **bằng đúng số lá**. Cây 8 lá chỉ báo được 8 mức giá.

```
   Dữ liệu thật        Cây dự đoán
      ╱                  ┌─┐
     ╱                ┌──┘ │
    ╱              ┌──┘    │     ← bậc thang, không mượt
   ╱            ───┘       │
```

---

## 2. BÀI TOÁN THỰC TẾ

```
   Hãng taxi cần BẢNG GIÁ ƯỚC TÍNH để tổng đài báo cho khách qua điện thoại
   trước khi điều xe.

   Yêu cầu nghiệp vụ:
     ① Tổng đài viên phải TRA ĐƯỢC nhanh, không dùng máy tính phức tạp
     ② Giá báo phải GIẢI THÍCH ĐƯỢC nếu khách thắc mắc
     ③ Sai số chấp nhận ±15%

   → Cây nông (4–5 tầng) in ra thành BẢNG TRA là sản phẩm phù hợp nhất.
   → Model chính xác hơn nhưng không tra được bằng tay thì vô dụng ở đây.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | NYC Taxi Trip Record Data (chính thức, cập nhật hàng tháng) |
| **Link** | https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page |
| **File gợi ý** | `yellow_tripdata_YYYY-MM.parquet` (1 tháng ≈ 3 triệu chuyến) |
| **Nhãn** | `total_amount` hoặc `fare_amount` |

**Cột dùng được:** `trip_distance`, `passenger_count`, `PULocationID`, `DOLocationID`,
`tpep_pickup_datetime` (→ giờ, thứ), `payment_type`

### ⚠️ Bốn bước làm sạch bắt buộc

```
   1. LẤY MẪU: 3 triệu dòng quá nặng → lấy ngẫu nhiên 200.000 dòng để luyện tập
   2. Loại chuyến vô lý:
        fare_amount <= 0      (huỷ, hoàn tiền)
        trip_distance <= 0    hoặc > 100 miles
        passenger_count == 0
   3. ⚠️ RÒ RỈ: KHÔNG dùng cột `tip_amount`, `tolls_amount`, `total_amount`
      làm đặc trưng khi dự đoán `fare_amount` — chúng chỉ biết SAU chuyến đi
   4. Tạo đặc trưng thời gian: giờ trong ngày, thứ trong tuần, có phải giờ cao điểm
```

---

## 4. HƯỚNG ĐI ĐÚNG

```python
from sklearn.tree import DecisionTreeRegressor, export_text

tree = DecisionTreeRegressor(
    max_depth=5,               # ⭐ đủ nông để in thành bảng tra
    min_samples_leaf=500,      # mỗi mức giá phải dựa trên ≥500 chuyến thật
    criterion='squared_error',
    random_state=42,
)
tree.fit(X_train, y_train)
print(export_text(tree, feature_names=list(X.columns)))
```

**Metric:**
```
   MAE   ← chính (cùng đơn vị tiền, dễ giải thích: "sai trung bình 12.000đ")
   MAPE  ← sai số theo % — khớp với yêu cầu nghiệp vụ "±15%"
   RMSE  ← phụ, nhạy với chuyến giá cực cao
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Tải 1 tháng dữ liệu, lấy mẫu 200.000 dòng
   ☐ 2. Làm sạch theo 4 bước ở mục 3, ghi lại số dòng bị loại mỗi bước
   ☐ 3. Tạo đặc trưng thời gian (giờ, thứ, cao điểm)
   ☐ 4. EDA: scatter quãng đường vs cước · giá trung bình theo giờ trong ngày
   ☐ 5. Baseline: DummyRegressor(mean) và công thức tuyến tính thủ công
        (vd: 12.000 + 8.500 × km) → MAE bao nhiêu?
   ☐ 6. Cây KHÔNG giới hạn độ sâu → so MAE train vs test → CHỨNG MINH overfit
   ☐ 7. Vẽ MAE train/test theo max_depth = 1..20 → chọn điểm tối ưu
   ☐ 8. ⭐ Vẽ hàm dự đoán theo quãng đường → NHÌN THẤY hình BẬC THANG
   ☐ 9. Cây max_depth=5 → xuất export_text và vẽ cây
   ☐ 10. ⭐ Chuyển cây thành BẢNG TRA CƯỚC (CSV) cho tổng đài
   ☐ 11. Kiểm tra: bao nhiêu % chuyến có sai số trong ±15%?
   ☐ 12. So sánh với Random Forest Regressor (TT-17) và Linear Regression
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Đã làm sạch đủ 4 bước, nêu số dòng loại bỏ
   ☐ Không dùng cột gây rò rỉ (tip, tolls, total)
   ☐ Có biểu đồ MAE train/test theo độ sâu
   ☐ ⭐ Có biểu đồ hàm bậc thang
   ☐ ⭐ Có file BẢNG TRA CƯỚC xuất ra CSV
   ☐ Báo cáo % chuyến đạt sai số ±15%
   ☐ Giải thích được vì sao cây KHÔNG ngoại suy được (chuyến 200 km sẽ ra sao?)
```

**Mức tham chiếu:** MAE ~$2–3 với cây độ sâu 5–8 (dữ liệu NYC tính bằng USD).

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Dùng `total_amount` làm đặc trưng | Rò rỉ — nó CHỨA cả fare cần dự đoán |
| `max_depth=None` | Overfit, cây hàng nghìn lá, không tra được |
| Kỳ vọng cây ngoại suy | Chuyến 200 km sẽ nhận đúng giá của lá "xa nhất" đã thấy |
| Quên loại chuyến giá âm | Kéo lệch toàn bộ trung bình |
| Không lấy mẫu | Notebook treo vì 3 triệu dòng |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-16-DecisionTreeRegressor-<HoTen>/
├── README.md          ← có mục "VÌ SAO CÂY KHÔNG NGOẠI SUY ĐƯỢC"
├── notebooks/tree_regressor_taxi.ipynb
├── src/train.py
├── models/tree_reg.joblib
├── outputs/bang_tra_cuoc.csv     ← ⭐ sản phẩm bàn giao cho tổng đài
├── reports/{ham_bac_thang.png, mae_theo_depth.png, cay_quyet_dinh.png}
└── requirements.txt
```

**Mở rộng:**
1. Thử `criterion='absolute_error'` → ít nhạy outlier hơn, nhưng chậm hơn nhiều
2. Chứng minh tính KHÔNG ỔN ĐỊNH: train 10 cây với 10 mẫu con khác nhau → bảng tra có giống nhau không?
3. Dự đoán **khoảng giá** thay vì 1 con số: dùng `GradientBoostingRegressor(loss='quantile')` cho phân vị 10% và 90%

**Tham khảo:** [Buổi 3 — Tree](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-03-Feature-Eng-Tree/Tai-Lieu) · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
