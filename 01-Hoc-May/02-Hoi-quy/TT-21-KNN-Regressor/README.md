# TT-21 — KNN REGRESSOR
## Định giá nhà theo "các căn tương tự trong khu vực" — đúng cách thẩm định viên làm

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao) |
| 🧠 **Nhóm** | Hồi quy dựa trên khoảng cách |
| 🔧 **Thuật toán** | KNN Regressor |
| 🏭 **Lĩnh vực** | Bất động sản · Thẩm định giá |
| ⏱ **Thời lượng** | 4–6 giờ |
| 📈 **Độ khó** | ⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   Giống KNN phân loại (TT-01) nhưng lá trả về TRUNG BÌNH thay vì bỏ phiếu:

        ŷ = trung bình giá của K căn nhà GẦN NHẤT (giống nhất)

   Với weights='distance', căn càng gần càng có trọng số cao:

        ŷ = Σ(yᵢ/dᵢ) / Σ(1/dᵢ)
```

**Điểm hay của bài này:** đây chính xác là **phương pháp so sánh (comparable sales)**
mà thẩm định viên bất động sản đã dùng hàng chục năm — tìm 3–5 căn tương tự vừa bán
trong khu, lấy giá trung bình rồi điều chỉnh. KNN là phiên bản tự động hoá của nó.

---

## 2. BÀI TOÁN THỰC TẾ

```
   Sàn BĐS muốn hiển thị: "Căn này ước tính 4,2 tỷ — dựa trên 5 căn tương tự
   đã bán gần đây:" rồi LIỆT KÊ 5 căn đó ra cho khách xem.

   → Đây là ưu thế RIÊNG của KNN: nó không chỉ cho con số,
     mà chỉ ra ĐƯỢC CĂN CỨ là những căn nhà cụ thể nào.
   → Model khác (XGBoost) cho số chính xác hơn nhưng KHÔNG làm được điều này.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | California Housing (dùng lại từ TT-11 để SO SÁNH trực tiếp) |
| **Cách lấy** | `from sklearn.datasets import fetch_california_housing` |
| **Kích thước** | 20.640 dòng × 8 đặc trưng |
| **Nhãn** | `MedHouseVal` (đơn vị 100.000 USD) |

> ⭐ Dùng **cùng bộ dữ liệu với TT-11 (Linear Regression)** là có chủ đích: học viên
> so sánh trực tiếp 2 triết lý — model tham số (học ra công thức) vs model phi tham số
> (ghi nhớ dữ liệu).

### ⚠️ Đặc thù quan trọng: KNN rất hợp với toạ độ

```
   Linear Regression bắt kém quan hệ Latitude/Longitude → giá (phi tuyến, cục bộ).
   KNN thì NGƯỢC LẠI: "hàng xóm gần nhau thì giá giống nhau" là giả định
   HOÀN TOÀN ĐÚNG với bất động sản.
   → Kỳ vọng KNN THẮNG Linear Regression trên bộ này.
```

---

## 4. HƯỚNG ĐI ĐÚNG

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scale', StandardScaler()),                 # ⭐ BẮT BUỘC
    ('knn', KNeighborsRegressor(n_neighbors=10, weights='distance',
                                metric='minkowski', p=2, n_jobs=-1)),
])
```

### Lấy ra "các căn tương tự" để hiển thị cho khách

```python
khoang_cach, chi_so = pipe['knn'].kneighbors(X_test_scaled[[0]], n_neighbors=5)
cac_can_tuong_tu = X_train.iloc[chi_so[0]]
gia_cac_can      = y_train.iloc[chi_so[0]]
```

### Trọng số cho toạ độ — kỹ thuật nâng cao

```
   Mặc định mọi đặc trưng có trọng số bằng nhau sau khi chuẩn hoá.
   Nhưng với BĐS, VỊ TRÍ quan trọng hơn số phòng.
   → Thử NHÂN Latitude/Longitude với hệ số 2–3 sau khi scale
     → ép KNN ưu tiên tìm hàng xóm ĐỊA LÝ trước.
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Nạp dữ liệu (giống TT-11), xử lý outlier AveOccup/AveRooms
   ☐ 2. Baseline: DummyRegressor + Linear Regression (lấy lại kết quả TT-11)
   ☐ 3. ⚠️ KNN KHÔNG scale → ghi lại (sẽ rất tệ vì Population áp đảo)
   ☐ 4. KNN có scale, K=5 mặc định
   ☐ 5. Vẽ RMSE train/test theo K = 1..50
        ⚠️ K=1 cho RMSE train = 0 → giải thích vì sao
   ☐ 6. So sánh weights='uniform' vs 'distance'
   ☐ 7. So sánh metric: euclidean vs manhattan
   ☐ 8. ⭐ THÍ NGHIỆM TRỌNG SỐ VỊ TRÍ: nhân Lat/Lon × {1, 2, 3, 5}
        → RMSE thay đổi thế nào? Rút ra kết luận về vai trò vị trí.
   ☐ 9. ⭐ Với 3 căn nhà bất kỳ, in ra 5 "căn tương tự" mà KNN dùng
        → kiểm tra bằng mắt: chúng có thật sự giống nhau không?
   ☐ 10. Bảng so sánh cuối: Linear (TT-11) vs KNN (TT-21) vs RandomForest
         → RMSE · thời gian train · thời gian dự đoán 1 căn · giải thích được không
   ☐ 11. ⚠️ Đo thời gian dự đoán khi tập train tăng 1× / 5× / 10×
         → chứng minh KNN chậm dần khi dữ liệu lớn
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Có bảng so sánh CÓ/KHÔNG chuẩn hoá
   ☐ Có biểu đồ RMSE theo K + giải thích hiện tượng K=1
   ☐ Có so sánh uniform vs distance
   ☐ ⭐ Có thí nghiệm trọng số vị trí + kết luận
   ☐ ⭐ Có ví dụ in ra 5 căn tương tự cho ít nhất 3 căn nhà
   ☐ ⭐ Có bảng so sánh 3 thuật toán kèm THỜI GIAN DỰ ĐOÁN
   ☐ KNN phải THẮNG Linear Regression về RMSE (nếu không, xem lại chuẩn hoá)
   ☐ Nêu hạn chế: chậm khi dữ liệu lớn, không ngoại suy, cần lưu toàn bộ dữ liệu
```

**Mức tham chiếu:** KNN (K≈10, distance) đạt R² ~0,70–0,75 — **cao hơn hẳn** Linear
Regression (~0,60) trên bộ này, đúng như kỳ vọng lý thuyết.

### ✅ Kết quả thực đo (`notebooks/knn_regressor_housing.ipynb`, xem `reports/tom_tat.json`)

| Bước | RMSE | R² |
|---|---|---|
| Baseline Dummy (dự đoán trung bình) | 1,150 | — |
| Baseline Linear Regression | 0,679 | 0,651 |
| KNN K=5 **không** chuẩn hoá | 1,066 | 0,141 (⚠️ tệ hơn cả Linear Regression) |
| KNN K=5 có chuẩn hoá | 0,591 | 0,736 |
| KNN chuẩn (K=10, `weights='distance'`) — `models/knn_pipeline.joblib` | 0,571 | 0,754 |
| ⭐ **KNN + trọng số vị trí ×50** (chọn bằng 5-fold CV) — `models/knn_vi_tri_pipeline.joblib` | **0,436** | **0,856** |

**Bảng so sánh Linear vs KNN vs Random Forest** (`reports/so_sanh_models.csv`):

| | RMSE | R² | Train (ms) | Dự đoán 1 căn (ms) | Giải thích được |
|---|---|---|---|---|---|
| Linear Regression | 0,679 | 0,651 | ~3,5 | ~0,46 | Có (hệ số) |
| KNN (K=10, distance) | 0,571 | 0,754 | ~16 | ~0,39 | Có (5 căn tương tự) |
| **KNN + vị trí ×50** | **0,436** | **0,856** | ~31 | ~1,2 | Có (5 căn tương tự) |
| Random Forest (300 cây) | 0,495 | 0,815 | ~2.600 | ~54 | Một phần (feature importance) |

*(Thời gian đo trên máy cá nhân, dao động giữa các lần chạy; bản "KNN + vị trí" tính cả bước scale + nhân trọng số trong pipeline.)*

**Phát hiện chính:**
- **Trọng số vị trí** (thử ×1/2/3/5 theo README rồi mở rộng tới ×500): đường RMSE hình chữ U — giảm mạnh tới ×5, đáy ở **×50** (chọn bằng CV trên train, không nhìn test), rồi tăng lại khi KNN gần như chỉ còn nhìn toạ độ. KNN *chỉ* dùng toạ độ đạt RMSE 0,526 — tệ hơn bản tối ưu → vị trí quyết định "khu nào", các đặc trưng còn lại "phân xử trong khu". Kết quả: KNN **vượt cả Random Forest** mà vẫn chỉ ra được 5 căn tương tự.
- `K=1` cho RMSE train = 0 (điểm train là hàng xóm của chính nó); K tối ưu = 10 theo RMSE test. Với `weights='distance'`, RMSE train = 0 với **mọi** K.
- `distance` tốt hơn `uniform` ở mọi K, rõ nhất khi K lớn (K=50: 0,587 vs 0,593).
- `metric='manhattan'` tốt hơn `euclidean` (0,552 vs 0,571).
- Thời gian dự đoán: `algorithm='brute'` tăng gần tuyến tính theo n (1× → 20× dữ liệu: ~1 → ~3,2 ms); KD-Tree (`auto`, 8 chiều) gần như không đổi (~0,4 ms).
- **Không ngoại suy:** tăng `MedInc` lên gấp 3 lần max trong train, KNN đi ngang ở 5,0 (= max giá train) còn Linear tiếp tục tăng (`reports/ngoai_suy.png`).
- Tiêu chí "KNN phải thắng Linear Regression" đạt rõ ràng (0,571 và 0,436 so với 0,679).

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Quên chuẩn hoá | `Population` (hàng nghìn) áp đảo mọi đặc trưng khác |
| Chọn K theo RMSE train | K=1 luôn cho train = 0 → chọn sai |
| Dùng KNN cho dữ liệu rất lớn | Dự đoán chậm không chấp nhận được |
| Kỳ vọng ngoại suy | Nhà đắt hơn mọi căn trong train sẽ bị kẹp trần |
| Quên `n_jobs=-1` | Dự đoán chậm không cần thiết |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-21-KNN-Regressor/
├── README.md          ← có bảng so sánh Linear vs KNN vs RF (mục 6)
├── notebooks/knn_regressor_housing.ipynb  ← toàn bộ pipeline; mỗi bước có markdown "Logic" + "Giải thích code" + "Đọc kết quả"
├── src/{features.py, train.py}            ← nạp/lọc outlier + hàm nhân trọng số vị trí dùng chung; train.py train 2 model
├── models/{knn_pipeline.joblib,           ← StandardScaler + KNeighborsRegressor (K=10, distance)
│           knn_vi_tri_pipeline.joblib}    ← + bước nhân Latitude/Longitude ×50 (bản tốt nhất)
├── reports/{rmse_theo_K.png, uniform_vs_distance.png, trong_so_vi_tri.png, can_tuong_tu_vi_du.png,
│            thoi_gian_predict.png, ngoai_suy.png, so_sanh_scale.csv, so_sanh_models.csv,
│            tom_tat.json, train_metrics.json}
└── requirements.txt
```

**Mở rộng:**
1. Dùng `BallTree`/`KDTree` với `algorithm='kd_tree'` → tăng tốc tìm hàng xóm
2. `RadiusNeighborsRegressor` — lấy mọi căn trong bán kính X km thay vì K căn cố định
   (sát thực tế thẩm định hơn)
3. Kết hợp: dùng KNN tạo đặc trưng "giá trung bình 10 căn lân cận" rồi đưa vào XGBoost

**Tham khảo:** [Buổi 1 — KNN](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-01-Gioi-thieu-ML/Tai-Lieu/ly_thuyet_chi_tiet_buoi_01.md) · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
