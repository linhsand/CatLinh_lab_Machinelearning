# TT-18 — GRADIENT BOOSTING REGRESSOR
## Thẩm định giá nhà tự động (AVM) với 80 đặc trưng

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao) |
| 🧠 **Nhóm** | Hồi quy · Ensemble (Boosting) |
| 🔧 **Thuật toán** | Gradient Boosting Regressor |
| 🏭 **Lĩnh vực** | Bất động sản · Ngân hàng (định giá tài sản thế chấp) |
| ⏱ **Thời lượng** | 7–9 giờ |
| 📈 **Độ khó** | ⭐⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   Dự đoán ban đầu = giá TRUNG BÌNH toàn bộ nhà
   Lặp lại N lần:
     ① residual = giá thật − dự đoán hiện tại
     ② train 1 cây NÔNG (depth 3) để dự đoán chính residual đó
     ③ dự đoán mới = dự đoán cũ + η × cây mới

   Ví dụ số:
     TB = 500tr · nhà A giá thật 700tr → residual = +200tr
     Cây 1 học "nhà 3 phòng ngủ thì residual ≈ +200"
     Dự đoán mới = 500 + 0,1×200 = 520tr   (gần 700 hơn)
     → residual mới = 180 → cây 2 sửa tiếp → ...
```

---

## 2. BÀI TOÁN THỰC TẾ

```
   Ngân hàng cần định giá tài sản thế chấp NHANH khi duyệt vay mua nhà.
   Thuê thẩm định viên: 2–3 triệu/căn, mất 3–5 ngày.
   Hệ thống AVM (Automated Valuation Model) trả kết quả trong 2 giây.

   ⚠️ YÊU CẦU NGHIỆP VỤ NGHIÊM NGẶT:
      • Sai số trung vị < 10% (chuẩn ngành AVM quốc tế)
      • Phải báo KHOẢNG GIÁ + mức tin cậy
      • Nếu độ tin cậy thấp → tự động chuyển cho thẩm định viên (human-in-the-loop)
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Ames Housing (House Prices — Advanced Regression) |
| **Link** | https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data |
| **Kích thước** | 1.460 dòng × 79 đặc trưng + `SalePrice` |
| **Nhãn** | `SalePrice` (USD) |

**Vì sao chọn bộ này:** đây là bộ dữ liệu **bẩn thật** — 19 cột có giá trị thiếu,
trộn lẫn biến số/thứ tự/danh mục, nhãn lệch phải mạnh. Rèn đúng kỹ năng thực chiến.

### ⚠️ Ba nhóm vấn đề phải xử lý

```
   1. GIÁ TRỊ THIẾU CÓ Ý NGHĨA — không phải lỗi!
      PoolQC = NaN  → nhà KHÔNG CÓ bể bơi (không phải thiếu dữ liệu)
      Alley  = NaN  → không có ngõ sau
      → Điền bằng chuỗi 'None', KHÔNG xoá cột, KHÔNG điền mode.
      ⚠️ Đây là bẫy khiến rất nhiều người mất điểm.

   2. BIẾN THỨ TỰ bị mã hoá dạng chữ
      ExterQual: Ex > Gd > TA > Fa > Po  (có thứ tự rõ ràng!)
      → Dùng OrdinalEncoder theo đúng thứ tự, KHÔNG one-hot (mất thông tin thứ tự)

   3. NHÃN LỆCH PHẢI mạnh (vài căn siêu đắt)
      → Train trên log1p(SalePrice), dự đoán xong dùng expm1() để đổi ngược
      → Đây cũng là metric chính thức của cuộc thi (RMSE trên log)
```

---

## 4. HƯỚNG ĐI ĐÚNG

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

y_log = np.log1p(y)          # ⭐ huấn luyện trên thang log

gbr = GradientBoostingRegressor(
    n_estimators=1000, learning_rate=0.03, max_depth=3,
    subsample=0.8, max_features='sqrt',
    validation_fraction=0.1, n_iter_no_change=50,   # dừng sớm
    random_state=42,
)
gbr.fit(X_train, y_log_train)

gia_du_doan = np.expm1(gbr.predict(X_test))         # đổi ngược về VND/USD
```

### Dự báo KHOẢNG bằng hồi quy phân vị

```python
mo_hinh = {}
for q in [0.1, 0.5, 0.9]:
    mo_hinh[q] = GradientBoostingRegressor(loss='quantile', alpha=q,
                                           n_estimators=500, learning_rate=0.05,
                                           max_depth=3, random_state=42
                                           ).fit(X_train, y_log_train)
# → khoảng giá [phân vị 10%, phân vị 90%] cho mỗi căn nhà
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Thống kê giá trị thiếu từng cột, PHÂN LOẠI: thiếu thật vs "không có tiện ích"
   ☐ 2. Xử lý riêng 2 nhóm đó (điền 'None' vs điền median)
   ☐ 3. Mã hoá biến thứ tự bằng OrdinalEncoder ĐÚNG THỨ TỰ (ít nhất 5 cột chất lượng)
   ☐ 4. One-hot các biến danh mục thuần
   ☐ 5. Kiểm tra độ lệch của SalePrice (skew) → áp dụng log1p
   ☐ 6. Baseline: DummyRegressor + Linear Regression + Ridge
   ☐ 7. Gradient Boosting + early stopping
   ☐ 8. ⭐ Vẽ đường train/validation loss theo số cây → chỉ điểm overfit
   ☐ 9. Feature engineering: TotalSF = 1stFlr + 2ndFlr + Basement,
        TuoiNha = YrSold − YearBuilt, DaSuaChua = (YearRemodAdd != YearBuilt)
        → RMSE cải thiện bao nhiêu?
   ☐ 10. Hồi quy phân vị 10/50/90 → khoảng giá
   ☐ 11. ⭐ Tính MEDIAN APE (sai số phần trăm trung vị) — chuẩn ngành AVM
   ☐ 12. Cơ chế human-in-the-loop: nếu khoảng dự báo rộng > 25% → chuyển thẩm định viên.
         Bao nhiêu % hồ sơ tự động được?
   ☐ 13. So sánh thời gian train: GradientBoosting vs HistGradientBoosting
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Phân biệt đúng "thiếu thật" và "không có tiện ích" (nêu số cột mỗi loại)
   ☐ Có ít nhất 5 cột được mã hoá thứ tự đúng
   ☐ Có dùng log1p cho nhãn + đổi ngược khi báo cáo
   ☐ Có biểu đồ train/validation loss theo số cây
   ☐ ⭐ Median APE < 12%
   ☐ ⭐ Có khoảng dự báo 10–90% + tỉ lệ phủ thực tế
   ☐ Có bảng human-in-the-loop (% tự động vs % chuyển người)
   ☐ Có đo hiệu quả của feature engineering (trước/sau)
```

**Mức tham chiếu:** RMSE trên thang log ~0,12–0,13 · Median APE ~8–10%.

### ✅ Kết quả thực đo (`notebooks/gbr_regressor_house.ipynb`, xem `reports/tom_tat.json`)

| Tiêu chí | Kết quả |
|---|---|
| Giá trị thiếu | 17 cột "không có tiện ích" (điền `None`/0) · 2 cột thiếu thật (`LotFrontage` theo median-Neighborhood, `Electrical` theo mode) |
| Biến thứ tự (`OrdinalEncoder`) | 18 cột (yêu cầu tối thiểu 5) |
| RMSE (thang log) | Gradient Boosting: ~0,135 · HistGradientBoosting: ~0,136 · Ridge: ~0,134 |
| ⭐ Median APE | **~5,4%** (chuẩn ngành AVM: < 10–12%) |
| ⭐ Khoảng dự báo 10–90% | Thô (chưa hiệu chỉnh): độ phủ thực tế chỉ ~67% → **quá tự tin**. Sau **split-conformal calibration**: độ phủ ~79%, sát mức danh nghĩa 80% |
| Human-in-the-loop | Ngưỡng độ rộng khoảng > 25% giá dự đoán → ~42% hồ sơ tự động duyệt, ~58% chuyển thẩm định viên (MAE nhóm tự động thấp hơn hẳn nhóm chuyển người, xác nhận ngưỡng lọc đúng) |
| Feature engineering (`TotalSF`, `TuoiNha`, `DaSuaChua`) | Median APE cải thiện từ ~5,80% → ~5,50% |
| Số cây tối ưu (từ đường loss train/validation) | 540 / 1000 (sau mốc này validation loss chững lại — early stopping đúng hướng) |

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Xoá cột có nhiều NaN (PoolQC…) | Mất thông tin "nhà không có bể bơi" |
| One-hot biến thứ tự | Mất quan hệ Ex > Gd > TA |
| Không log1p nhãn | Vài căn siêu đắt chi phối toàn bộ loss |
| Quên `expm1` khi báo cáo | Báo giá sai đơn vị (log thay vì USD) |
| `max_depth` lớn | Sai bản chất boosting, overfit |
| Chỉ báo 1 con số | Nghiệp vụ thẩm định cần khoảng + độ tin cậy |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-18-Gradient-Boosting-Regressor/
├── README.md              ← có Median APE + bảng human-in-the-loop (kết quả thực đo ở mục 6)
├── data/train.csv         ← Ames Housing that (qua OpenML, xem ghi chu trong notebook)
├── notebooks/
│   └── gbr_regressor_house.ipynb   ← toan bo pipeline, giai thich tung buoc bang markdown
├── models/gbr_pipeline.joblib      ← pipeline GBR (co FE) + 3 model quantile 10/50/90 + preprocessor
├── reports/{missing_analysis.png, skew_saleprice.png, loss_theo_so_cay.png, khoang_gia.png,
│            ape_distribution.png, so_sanh_feature_engineering.csv, so_sanh_models.csv,
│            human_in_the_loop.csv, tom_tat.json}
└── requirements.txt
```

**Mở rộng:**
1. So sánh với XGBoost (TT-19) và LightGBM — chênh lệch RMSE và thời gian
2. Dùng SHAP giải thích 1 căn nhà cụ thể → "giá cao vì diện tích + khu vực"
3. Xếp chồng (stacking): GBR + Ridge + RandomForest → meta-model có tốt hơn không?

**Tham khảo:** [Buổi 6 — Boosting](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-06-Ensemble-EndToEnd/Tai-Lieu/ly_thuyet_chi_tiet_buoi_06.md) · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
