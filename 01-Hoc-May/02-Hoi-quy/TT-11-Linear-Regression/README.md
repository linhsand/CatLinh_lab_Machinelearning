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

> **Kết quả chính (test 4.128 căn):** model cuối là Linear Regression + feature engineering, đạt
> **RMSE 0,6662** (≈66.600 USD) và **R² 0,6614**. So với baseline đoán giá trung bình
> (RMSE 1,1449), sai số giảm **42%**. Random Forest đạt R² 0,8279, tức **0,167 điểm R²**
> là cái giá của tính giải thích được. Có 3 vấn đề lớn: nhãn bị cắt ngọn ở 5,0 (4,81% dữ
> liệu, RMSE nhóm này 1,64), phương sai không đều (Breusch-Pagan p ≈ 1e-25), và đa cộng
> tuyến trong nhóm biến vị trí sau FE (VIF tới 19,5).

---

## 1. BÀI TOÁN

Sàn giao dịch BĐS cần ước tính **giá tham chiếu** ngay khi chủ nhà đăng tin, để
① cảnh báo tin đăng giá bất thường và ② gợi ý khoảng giá hợp lý. Hệ thống **phải giải
thích được** "vì sao định giá vậy", nên dùng model tuyến tính
`ŷ = w₁x₁ + … + wₙxₙ + b` (học bằng cách cực tiểu MSE). Mỗi hệ số là một câu giải thích
đọc được.

## 2. DỮ LIỆU

| | |
|---|---|
| **Tên** | California Housing — `sklearn.datasets.fetch_california_housing()` (không dùng Boston Housing: đã bị gỡ vì có biến phân biệt chủng tộc) |
| **Kích thước** | 20.640 khu dân cư × 8 đặc trưng |
| **Nhãn** | `MedHouseVal` — giá nhà trung vị, đơn vị **100.000 USD** |
| **Chia** | `train_test_split` 80/20, `random_state=42` → 16.512 train / 4.128 test |

**3 bẫy dữ liệu, đã phát hiện và đo được** (`reports/data_describe.csv`):

| Bẫy | Số đo | Xử lý |
|---|---|---|
| Nhãn **bị cắt ngọn** ở 5,0 | **992 / 20.640 dòng (4,81%)** dồn đúng vào 5,0 | Nêu trong hạn chế, đo riêng sai số nhóm này (mục 4.3) |
| Outlier cực đoan | `AveOccup` max **1.243** người/hộ (p99 = 5,39); `AveRooms` max **141,9** phòng/hộ (p99 = 10,36) | Clip ở p99, ngưỡng **học từ train** (transformer `ClipOutliers` trong Pipeline) |
| Toạ độ–giá phi tuyến | `ban_do_gia.png`: giá cao tập trung thành cụm quanh SF Bay và LA/San Diego | Thêm khoảng cách tới SF/LA (mục 4.5) |

`AveRooms` và `AveBedrms` có tương quan cặp **0,848**, nên bài kiểm tra đa cộng tuyến bằng VIF (mục 4.4).

---

## 3. KẾT QUẢ CHÍNH — `reports/model_comparison.csv`

| Model (test, đơn vị 100k USD) | RMSE | MAE | R² | MAPE |
|---|---:|---:|---:|---:|
| Baseline `DummyRegressor(mean)` | 1,1449 | 0,9061 | −0,0002 | 62,9% |
| LR cơ bản (8 cột, clip outlier) | 0,6918 | 0,5052 | 0,6348 | 30,8% |
| LR dự đoán log(giá) | 0,7610 | 0,4915 | 0,5581 | 27,6% |
| **LR + feature engineering** ✅ model cuối | **0,6662** | **0,4855** | **0,6614** | **29,8%** |
| Ridge (TT-12, alpha=1) | 0,6662 | 0,4855 | 0,6614 | 29,8% |
| Random Forest (TT-17, 200 cây) | 0,4749 | 0,3036 | 0,8279 | 17,2% |

**Diễn giải cho nghiệp vụ:**
* **Vượt baseline rõ rệt.** Model cuối sai điển hình (RMSE) **~66.600 USD**, so với ~114.500 USD
  khi chỉ đoán giá trung bình, tức giảm 42%. Sai số tuyệt đối trung bình (MAE) là **≈48.500 USD**.
* **R² 0,6614.** Model giải thích được 66% biến động giá giữa các khu, cao hơn mức tham chiếu
  0,58–0,61 của đề nhờ clip outlier và FE. 34% còn lại đến từ các yếu tố không có trong dữ liệu
  (trường học, an ninh…) và từ quan hệ phi tuyến.
* **MAPE 29,8%.** Sai số theo % lớn ở phân khúc rẻ: cùng sai ~50.000 USD, căn 100k sai 50%,
  còn căn 400k chỉ sai 12,5%. Vì vậy khoảng giá gợi ý không nên là "±30%" đồng loạt (xem 4.2).
* **Ridge trùng LR** (khác nhau ở chữ số thứ 7). Với 16.512 mẫu và 12 đặc trưng, alpha=1 gần như
  không phạt gì, nên không chữa được đa cộng tuyến ở 4.4.
* **Random Forest** giảm thêm 29% RMSE (0,4749), nhưng không cho ra câu giải thích dạng
  "+X USD cho mỗi đơn vị". Bài vẫn chọn LR vì ràng buộc nghiệp vụ.
* Không có siêu tham số nào được tinh chỉnh (alpha=1, 200 cây là giá trị cố định), nên việc so
  sánh trên test không làm lạc quan kết quả.

---

## 4. KIỂM TRA GIẢ ĐỊNH & PHÂN TÍCH

### 4.1. Bốn giả định — kết luận

| Giả định | Bằng chứng (số đo) | Kết luận |
|---|---|---|
| ① Tuyến tính | MedInc–giá gần thẳng (`eda_medinc_scatter.png`). Nhưng phần dư trung bình theo 5 nhóm giá dự đoán có **hình chữ U**: +0,19 / −0,13 / −0,20 / −0,12 / +0,03 | **Vi phạm một phần**: đoán thiếu ở 2 đầu, thừa ở giữa |
| ② Độc lập | Các khu lân cận có giá và sai số tương quan theo không gian. Bài không kiểm định trực tiếp | Có khả năng vi phạm |
| ③ Phương sai đều | Std phần dư theo nhóm giá dự đoán: **0,50 → 0,47 → 0,55 → 0,66 → 0,75**. Breusch-Pagan **p = 1,1e-25** | **Vi phạm rõ** (hình phễu) |
| ④ Phần dư chuẩn | Q-Q (`qq_plot.png`) bám đường chéo ở thân, lệch ở 2 đuôi. Skew 0,84, excess kurtosis 5,2 | **Vi phạm ở đuôi**, chủ yếu do nhãn cắt ngọn |

### 4.2. Residual plot — `reports/residual_plot.png`, `phan_du_theo_nhom_gia.csv`, `chan_doan_phan_du.csv`

* **Đường chéo thẳng ở bên phải** là các căn có nhãn bị gán cứng 5,0, nên phần dư = 5,0 − ŷ
  giảm tuyến tính theo ŷ. Đây là lỗi của **nhãn**, không phải của model.
* **Hình phễu:** nhóm giá dự đoán cao nhất phân tán gấp **1,5 lần** nhóm thấp nhất (std 0,75
  so với 0,50). Hệ quả nghiệp vụ: khoảng giá gợi ý phải **rộng hơn ở phân khúc đắt**, không dùng
  một biên sai số chung.
* **Dự đoán vô lý:** LR cơ bản cho 0,48% dự đoán **giá âm**, model cuối 1,11%; 0,85% dự đoán
  vượt 5,0 (max 7,0). Chỉ cần chặn dự đoán về [0; 5] (nhãn không thể nằm ngoài khoảng này) là
  RMSE model cuối giảm 0,6662 → **0,6557**. Nên thêm bước này khi triển khai.

### 4.3. Nhãn bị cắt ngọn ở 5,0

Trên 4.128 căn test, sai số của nhóm bị cắt ngọn lớn hơn hẳn phần còn lại:

| | RMSE nhóm cắt ngọn | RMSE phần còn lại | Model đoán thấp hơn TB |
|---|---:|---:|---:|
| LR cơ bản | **1,640** | 0,613 | +1,09 (≈109.000 USD) |
| LR + FE | 1,591 | 0,589 | +1,05 |

Giá thật của nhóm này **≥ 500.000 USD** (có thể cao hơn nhiều), nên mức đoán thấp thật sự còn
lớn hơn số đo. Model **không định giá được nhà > 500k**. Khi triển khai, căn nào dự đoán ≥ ~4,5
nên được gắn cờ "cần thẩm định riêng".

### 4.4. Đa cộng tuyến — `reports/vif.csv`, `vif_feature_engineering.csv`

| Đặc trưng | VIF (8 cột gốc) | VIF (12 cột sau FE — bộ hệ số được diễn giải) |
|---|---:|---:|
| Longitude | 9,31 | **19,51** |
| Latitude | 9,74 | **15,41** |
| DistLA_km | — | **12,99** |
| DistSF_km | — | **12,91** |
| AveRooms | 2,56 | 7,31 |
| BedrmsRatio | — | 6,95 |
| AveBedrms | 1,30 | 5,51 |
| MedInc | 2,37 | 2,58 |
| HouseAge / Population / AveOccup | 1,25 / 1,17 / 1,08 | 1,38 / 1,19 / 1,11 |

* Trên 8 cột gốc, không VIF nào vượt 10. `AveRooms`/`AveBedrms` dù tương quan cặp 0,848 cũng chỉ
  có VIF 2,6/1,3, vì tương quan cặp cao không đồng nghĩa VIF cao. Toạ độ sát ngưỡng (9,7/9,3).
* **Sau FE, 4 biến vị trí vượt 10.** Các biến khoảng cách được tính từ chính Latitude/Longitude,
  nên hệ số của nhóm này "chia phần" cho nhau và có thể đổi độ lớn khi thêm/bớt một biến. Hệ quả
  được nêu ở 4.6.

### 4.5. Log-target & feature engineering

**Log-target thắng một nửa, thua một nửa:**
* Tốt hơn ở **MAE** (0,4915 so với 0,5052) và **MAPE** (27,6% so với 30,8%): đa số căn, nhất là căn
  rẻ, được đoán sát hơn.
* Tệ hơn ở **RMSE** (0,7610) và **R²** (0,5581). Nguyên nhân là `expm1` thổi phồng vài dự đoán:
  max **15,9** (1,59 triệu USD), 1,5% dự đoán vượt 5,0 (`residual_plot_log.png`). Khi chặn về
  [0; 5], RMSE còn **0,6668**, tốt hơn LR cơ bản chặn tương tự (0,6780).
* Log-target **không** chữa được phễu khi nhìn ở thang giá gốc: std nhóm cao nhất vẫn là 0,90.
* **Kết luận:** không dùng log-target cho model cuối. Lý do không phải vì nó "kém", mà vì hệ số trên
  log là %-thay đổi, khó giải thích hơn "+X USD", và RMSE thô tệ hơn. Đây là một lựa chọn đánh đổi.

**Feature engineering** (`FeatureEngineer`, kế thừa `ClipOutliers`):
* `BedrmsRatio = AveBedrms / AveRooms`. Dữ liệu đã cho sẵn `AveRooms` là số phòng/hộ, nên "rooms
  per household" của đề chính là `AveRooms`.
* `DistSF_km`, `DistLA_km` (khoảng cách Haversine), và `DistNearestCity_km = min(...)`.
* Hiệu quả: RMSE 0,6918 → **0,6662** (−3,7%), R² 0,6348 → **0,6614** (+0,027). Cải thiện thật
  nhưng nhỏ, vì một hàm tuyến tính của khoảng cách vẫn không vẽ được hình dạng các cụm nóng.

### 4.6. Hệ số chuẩn hoá — diễn giải nghiệp vụ — `reports/he_so.csv`, `he_so.png`

Hệ số trên thang đã chuẩn hoá (`StandardScaler`). Mỗi hệ số là mức giá thay đổi (đơn vị 100k USD)
khi đặc trưng tăng **1 độ lệch chuẩn**, với các đặc trưng khác giữ nguyên:

| # | Đặc trưng | Hệ số | Diễn giải | Độ tin cậy |
|---|---|---:|---|---|
| 1 | Longitude | −0,90 | Đi về phía Đông (xa bờ biển Thái Bình Dương) → giá giảm | ⚠️ VIF 19,5 |
| 2 | **MedInc** | **+0,70** | Thu nhập khu vực cao hơn 1 độ lệch chuẩn (**≈19.000 USD/năm**) ↔ giá cao hơn **≈70.000 USD** | ✅ VIF 2,6 |
| 3 | Latitude | −0,68 | Đi về phía Bắc → giá giảm (ngoài cụm SF Bay) | ⚠️ VIF 15,4 |
| 4 | AveOccup | −0,28 | Đông người/hộ hơn ↔ giá thấp hơn ≈28.000 USD | ✅ VIF 1,1 |
| 5 | BedrmsRatio | +0,28 | Không diễn giải riêng: được tính từ AveBedrms/AveRooms nên hệ số chia phần với 2 biến đó | ⚠️ VIF 7,0 |

* **Cách nói với khách:** "Vị trí là yếu tố lớn nhất" (đọc **cả nhóm** Longitude/Latitude/khoảng
  cách như một khối, không đọc riêng "−0,90"). Kế đến là "thu nhập khu vực: mỗi 19.000 USD thu nhập
  trung vị cao hơn ứng với giá cao hơn ~70.000 USD".
* **Không kết luận nhân quả.** Hệ số chỉ thể hiện tương quan: "tăng thu nhập LÀM giá tăng" là
  chưa chứng minh. Các yếu tố ẩn đi cùng vị trí và thu nhập (trường học, hạ tầng) có thể mới là
  nguyên nhân.

---

## 5. TIÊU CHÍ HOÀN THÀNH

```
   ☑ Đã phát hiện & nêu vấn đề nhãn bị cắt ngọn ở 5,0   → 992 dòng (4,81%), RMSE nhóm này 1,64 (mục 4.3)
   ☑ Có residual plot + Q-Q plot + nhận xét về giả định  → mục 4.1–4.2 (BP p≈1e-25, skew 0,84)
   ☑ Có kiểm tra VIF                                     → mục 4.4 (gốc max 9,7; sau FE max 19,5)
   ☑ Có bảng hệ số đã chuẩn hoá + diễn giải nghiệp vụ    → mục 4.6 (MedInc +0,70 ≈ +70.000 USD)
   ☑ RMSE tốt hơn baseline rõ rệt                        → 1,1449 → 0,6662 (−42%)
   ☑ Có thử nghiệm log-transform và kết luận             → mục 4.5
   ☑ Nêu hạn chế: toạ độ–giá phi tuyến                   → mục 6
```

**Các bước đề yêu cầu:**
```
   ☑ 1. describe() → outlier AveOccup/AveRooms     ☑ 7. Q-Q plot
   ☑ 2. Đếm nhãn cắt ngọn ở 5,0                    ☑ 8. Thử log(giá)
   ☑ 3. EDA: scatter · heatmap · bản đồ giá        ☑ 9. VIF
   ☑ 4. Baseline DummyRegressor(mean)              ☑ 10. Feature engineering
   ☑ 5. LR cơ bản → RMSE, MAE, R²                  ☑ 11. Bảng hệ số + 3 yếu tố mạnh nhất
   ☑ 6. Residual plot                              ☑ 12. So sánh Ridge & Random Forest
```

---

## 6. HẠN CHẾ

1. **Nhãn cắt ngọn ở 5,0** (4,81%): model không định giá được nhà > 500.000 USD. RMSE nhóm này
   (1,64) gấp 2,7 lần phần còn lại.
2. **Toạ độ–giá phi tuyến:** FE chỉ thêm +0,027 R², phần dư còn dạng chữ U, trong khi Random Forest
   đạt R² 0,83.
3. **Phương sai không đều:** sai số ở phân khúc đắt lớn gấp 1,5 lần phân khúc rẻ, nên khoảng giá
   gợi ý cần co giãn theo mức giá.
4. **Đa cộng tuyến sau FE:** không diễn giải được riêng hệ số từng biến vị trí. Ridge với alpha=1
   không đủ mạnh để ổn định chúng (TT-12 nên tinh chỉnh alpha bằng CV).
5. **Dự đoán ngoài khoảng hợp lệ:** 1,11% dự đoán âm. Cần chặn về [0; 5] khi triển khai.
6. **Đơn vị dữ liệu là khu dân cư (block group), không phải từng căn nhà:** giá là trung vị của khu,
   nên khi áp cho một căn cụ thể sai số sẽ lớn hơn số đo ở đây.

---

## 7. SẢN PHẨM & CÁCH CHẠY

```
TT-11-Linear-Regression/
├── README.md
├── notebooks/linear_regression_housing.ipynb   ← giải thích từng bước + output thật
├── src/train.py                                ← toàn bộ pipeline, tái lập reports/
├── models/lr_pipeline.joblib                   ← Pipeline(FeatureEngineer → StandardScaler → LinearRegression)
├── reports/
│   ├── residual_plot.png, qq_plot.png (+ _log)     ← giả định ③④
│   ├── he_so.png, he_so.csv                        ← hệ số chuẩn hoá
│   ├── ban_do_gia.png, eda_*.png                   ← EDA
│   ├── model_comparison.csv, vif.csv, vif_feature_engineering.csv
│   ├── phan_du_theo_nhom_gia.csv, chan_doan_phan_du.csv   ← định lượng phễu / cắt ngọn / dự đoán vô lý
│   └── tom_tat.json                                ← toàn bộ số liệu
└── requirements.txt
```

```bash
pip install -r requirements.txt
python src/train.py        # ~1 phút; ghi lại toàn bộ reports/ và models/
```

```python
import sys, joblib; sys.path.insert(0, "src")   # class FeatureEngineer nằm trong src/train.py
model = joblib.load("models/lr_pipeline.joblib")
gia = model.predict(X_8_cot_goc).clip(0, 5)      # đơn vị 100k USD; >= 4,5 → gắn cờ thẩm định riêng
```

**Tham khảo:** [Buổi 13 — Regression nâng cao](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
