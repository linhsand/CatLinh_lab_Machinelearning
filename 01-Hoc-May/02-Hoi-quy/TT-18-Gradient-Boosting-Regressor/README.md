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

> **Kết quả chính (test 292 căn; mọi siêu tham số chọn bằng 5-fold CV trên 1.168 căn train):**
> * **Gradient Boosting** (`lr=0,03`, `max_depth=3`, 2.362 cây): RMSE_log test **0,125**, **Median APE 4,95%**
>   (chuẩn AVM < 10–12%). RMSE_log CV **0,119 ± 0,016**.
> * **So với Linear Regression:** trên test GB chỉ nhỉnh hơn (0,125 so với 0,127), nhưng trên CV **GB thắng cả 5/5
>   fold** (0,119 so với 0,158). Linear trông tốt trên test chủ yếu nhờ cách chia; nó vỡ ở fold có căn ngoại lệ (Id
>   1299 bị đoán gấp ~11 lần giá thật).
> * Bản cũ "GB thua Linear" (0,135 so với 0,127) vì GB **dừng quá sớm** (540 cây, early stopping trên 117 căn).
> * Khoảng 10–90%: độ phủ thô **65,8%** → sau split-conformal **77,1%**. Ngưỡng 25% cho **61,3% hồ sơ tự động duyệt**.

---

## 1. THUẬT TOÁN & BÀI TOÁN

```
   Dự đoán ban đầu = trung bình;  lặp N lần:
     ① residual = giá thật − dự đoán hiện tại
     ② train 1 cây NÔNG để dự đoán residual
     ③ dự đoán mới = dự đoán cũ + learning_rate × cây mới
```

**Bài toán:** ngân hàng cần định giá tài sản thế chấp nhanh khi duyệt vay. Thẩm định viên tốn 2–3 triệu/căn và 3–5
ngày. AVM phải trả kết quả trong vài giây, với yêu cầu: **Median APE < 10%**, có **khoảng giá + độ tin cậy**, và
**chuyển hồ sơ khó cho người** (human-in-the-loop).

## 2. DỮ LIỆU & CÁCH ĐÁNH GIÁ

| | |
|---|---|
| **Nguồn** | Ames Housing (Kaggle House Prices), 1.460 căn × 79 đặc trưng. Xem `data/DATA_SOURCE.md` |
| **Nhãn** | `SalePrice` (USD), skew **1,88** → huấn luyện trên `log1p` (skew 0,12), báo cáo bằng `expm1` |
| **Chia** | 80/20 (`random_state=42`) → 1.168 train / 292 test |
| **Chọn siêu tham số** | **5-fold CV trên train**: Ridge `alpha`, GB `learning_rate` × `max_depth`, số cây (đường loss trung bình 5 fold) |
| **Metric** | RMSE trên thang log (metric cuộc thi), **Median APE** (chuẩn AVM), MAE (USD), P90 APE |

**Giá trị thiếu** (19 cột, `reports/phan_loai_gia_tri_thieu.csv`):

| Nhóm | Số cột | Ví dụ | Xử lý |
|---|---:|---|---|
| "Không có tiện ích" | **17** | `PoolQC` 99,5%, `MiscFeature` 96,3%, `Alley` 93,8%, `Fence` 80,8%, các cột `Garage*`/`Bsmt*` | 15 cột chữ → `'None'`, `GarageYrBlt`/`MasVnrArea` → 0 |
| Thiếu thật | **2** | `LotFrontage` (17,7%), `Electrical` (1 dòng) | median theo `Neighborhood` / mode, **học từ train** rồi áp sang test |

Bản cũ tính median/mode trên toàn bộ dữ liệu trước khi chia (rò rỉ nhẹ từ test). Bản này dùng `TrueMissingFiller`
fit trên train.

**Mã hoá:** **18 cột thứ tự** qua `OrdinalEncoder` đúng thứ tự (10 cột thang `None < Po < Fa < TA < Gd < Ex`, cộng
`BsmtExposure`, `BsmtFinType1/2`, `GarageFinish`, `Functional`, `LandSlope`, `PavedDrive`, `Utilities`), 25 cột
one-hot, 36 cột số. Model tuyến tính thì scale thêm cột số và cột thứ tự.

---

## 3. KẾT QUẢ

### 3.1. Ridge: scale + chọn `alpha` bằng CV — `reports/ridge_chon_alpha_cv.csv`

Bản cũ dùng `Ridge(alpha=10)` trên dữ liệu **chưa scale**, nên hình phạt L2 tác động lên `LotArea` (hàng chục nghìn)
và `OverallQual` (1–10) hoàn toàn khác nhau. Sau khi scale, CV chọn **alpha ≈ 31,6** (RMSE_log CV 0,1394). Đường CV
phẳng trong khoảng 10–100.

### 3.2. Tuning GB: `learning_rate` × `max_depth` — `reports/gb_tuning_cv.png/.csv`

5-fold CV, early stopping trong từng fold (tối đa 3.000 cây), `subsample=0,8`, `max_features='sqrt'`:

| max_depth \ lr | 0,01 | **0,03** | 0,05 | 0,1 |
|---|---:|---:|---:|---:|
| 2 | 0,1293 | 0,1248 | 0,1263 | 0,1251 |
| **3** | 0,1247 | **0,1218** | 0,1226 | 0,1286 |
| 4 | 0,1240 | 0,1232 | 0,1234 | 0,1253 |
| 5 | 0,1251 | 0,1233 | 0,1232 | 0,1252 |

* CV chọn **lr = 0,03, depth = 3**, đúng cấu hình bản cũ đặt cố định. Giờ nó có căn cứ.
* Cả 16 ô nằm trong **0,1218–0,1293**, chênh nhau **ít hơn một độ lệch chuẩn** giữa các fold (~0,015). Với 1.168
  căn, CV chỉ loại được vùng tệ (`lr=0,1, depth=3`; `lr=0,01, depth=2`), không phân biệt chắc chắn các cấu hình tốt.
* Cây sâu (depth 5) có RMSE train thấp hơn hẳn (0,047 so với 0,072) nhưng CV không tốt hơn, đúng tinh thần "cây
  nông" của boosting.

### 3.3. ⭐ Đường loss theo số cây — trung bình 5 fold — `reports/loss_theo_so_cay.png/.csv`

| Số cây | 100 | 300 | 540 (bản cũ) | 1.000 | **2.362** | 3.000 |
|---|---:|---:|---:|---:|---:|---:|
| MSE_log validation (TB 5 fold) | 0,0254 | 0,0167 | 0,0153 | 0,0146 | **0,01435** | 0,01437 |

* Loss train giảm đơn điệu. Loss validation giảm nhanh tới ~500 cây rồi **gần như phẳng**, và chỉ tăng 0,2% sau cực
  tiểu: **không có overfit rõ** trong 3.000 cây, nhờ `subsample` và `max_features` (stochastic GB).
* Bản cũ chọn số cây trên **một** tập validation 117 căn và dừng ở 540, khi đó còn kém cực tiểu ~6% MSE. Đây là
  nguyên nhân chính khiến GB cũ thua Linear trên test (3.5).

### 3.4. Feature engineering — `reports/so_sanh_feature_engineering.csv`

| | GB: CV / test / Median APE | Ridge: CV / test |
|---|---|---|
| Trước FE | 0,1189 / 0,1229 / 5,26% | 0,1394 / 0,1370 |
| Sau FE (`TotalSF`, `TuoiNha`, `DaSuaChua`) | 0,1187 / 0,1253 / 4,95% | 0,1396 / 0,1369 |

**FE gần như không có tác dụng.** Chênh lệch CV 0,0002 nhỏ hơn 1/50 độ lệch chuẩn, còn test và Median APE đi ngược
chiều nhau. `TotalSF` là tổng ba cột đã có (GB tự cộng gộp được qua các lần chia, Ridge vốn là tổng có trọng số).
`TuoiNha` gần trùng `YearBuilt`, vì `YrSold` chỉ trải 2006–2010. Bài vẫn giữ FE vì CV không tệ hơn và các cột này dễ
giải thích, nhưng **không coi đây là nguồn cải thiện**.

### 3.5. ⭐ So sánh công bằng: GB có thật sự thắng Linear? — `reports/so_sanh_models.csv`, `so_sanh_models_theo_fold.csv`, `sai_so_theo_nhom_gia.csv`

| Model | RMSE_log CV (± std) | RMSE_log test | MAE test (USD) | Median APE test | P90 APE | GB thắng (fold) |
|---|---:|---:|---:|---:|---:|---:|
| Dummy (trung bình) | 0,390 ± 0,018 | 0,433 | 59.931 | 27,1% | 64,8% | 5/5 |
| Linear Regression | 0,158 ± 0,029 | 0,127 | 15.153 | 5,83% | 18,7% | **5/5** |
| Ridge (alpha 31,6, scale) | 0,140 ± 0,029 | 0,137 | 16.364 | 6,66% | 18,8% | 5/5 |
| GB cấu hình cũ (early stop 10%) | 0,121 ± 0,016 | 0,133 | 15.415 | 5,53% | 18,0% | 3/5 |
| **GB (lr 0,03, d 3, 2.362 cây)** | **0,119 ± 0,016** | **0,125** | **14.290** | **4,95%** | **17,0%** | — |
| Blend (GB + Ridge) / 2 | 0,123 ± 0,020 | 0,127 | 14.509 | 5,26% | 18,0% | 3/5 |

RMSE_log từng fold:

| | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---:|---:|---:|---:|---:|
| Linear | 0,136 | 0,160 | **0,211** | 0,130 | 0,153 |
| GB | 0,129 | 0,131 | 0,135 | 0,104 | 0,095 |

1. **Vì sao bản cũ "GB thua Linear" (0,135 so với 0,127)?**
   (a) GB cũ dừng quá sớm (3.3); chỉ riêng việc lấy số cây từ đường loss 5 fold đã đưa RMSE test từ 0,133 xuống
   0,125.
   (b) Test 292 căn **thuận lợi cho Linear**: RMSE test của Linear (0,127) tốt hơn hẳn CV của chính nó (0,158).
2. **Trên CV, GB thắng Linear ở cả 5 fold**, thấp hơn trung bình 25%. Linear **vỡ ở fold 3** vì căn **Id 1299**
   (5.642 sqft, chất lượng 10/10, bán `Partial` với giá 160.000 USD): Linear ngoại suy theo diện tích và đoán **gấp
   ~11 lần**. 4 căn sai nhất chiếm 68% tổng bình phương sai số của Linear ở fold đó. Cây bị chặn trong dải giá đã
   thấy, nên không bao giờ ra con số vô lý như vậy.
3. **Nhưng trên test khoảng cách nhỏ**, nên không bán GB như "vượt trội tuyệt đối". Lợi thế thật của GB là **ổn định
   hơn** (std giữa fold 0,016 so với 0,029) và **Median APE thấp hơn** (4,95% so với 5,83%). Theo nhóm giá, GB tốt
   nhất ở phân khúc 119–235 nghìn USD, còn ở nhóm rẻ nhất (< 119 nghìn) Linear nhỉnh hơn (0,176 so với 0,194).
4. **Blend GB + Ridge không giúp** trên CV (0,123), vì Ridge kéo blend xuống ở fold 3.

### 3.6. ⭐ Khoảng giá 10–90% + split-conformal — `reports/khoang_gia.png`

Quantile GB dùng **cùng** `lr`, `depth` và số cây đã chọn (bản cũ lấy số cây ở `lr=0,03` nhưng train quantile ở
`lr=0,05`). Train trên 85% train, giữ **15% (176 căn) làm tập calib**.

| | Độ phủ test |
|---|---:|
| [q10, q90] thô | **65,8%** (calib: 61,9%) |
| Sau split-conformal (nới mỗi đầu 0,035 log ≈ 3,5% giá) | **77,1%** |

Quantile GB **quá tự tin**, và conformal kéo độ phủ về sát 80%. Phần hụt 2,9 điểm % nằm trong biên dao động của 176
mẫu calib (sai số chuẩn ~3 điểm %). Độ rộng trung bình **24,9% giá**.

### 3.7. ⭐ Median APE & human-in-the-loop — `reports/ape_distribution.png`, `human_in_the_loop.csv`, `human_in_the_loop_quet_nguong.csv`

**Median APE = 4,95%.** Ngưỡng **25%** (khoảng rộng hơn 25% giá dự đoán thì chuyển người):

| Nhóm | Số căn | Tỷ lệ | MAE (USD) | Median APE |
|---|---:|---:|---:|---:|
| Tự động duyệt (AVM) | 179 | **61,3%** | 12.119 | 4,57% |
| Chuyển thẩm định viên | 113 | 38,7% | 17.729 | 6,53% |

Quét ngưỡng để ngân hàng tự chọn điểm cân bằng:

| Ngưỡng độ rộng | 15% | 20% | **25%** | 30% | 35% | 40% |
|---|---:|---:|---:|---:|---:|---:|
| % tự động | 6,8 | 34,9 | **61,3** | 77,4 | 88,0 | 93,8 |
| % hồ sơ tự động có APE > 20% | 0,0 | 2,0 | **3,4** | 5,3 | 5,8 | 6,2 |

Nới từ 25% lên 30% thì tự động thêm 16 điểm %, đổi lại tỷ lệ "định giá lệch > 20% mà không ai kiểm tra" tăng từ 3,4%
lên 5,3%.

### 3.8. GB vs HistGradientBoosting — `reports/gb_vs_hist_gb.csv`

Với 1.168 dòng, `HistGradientBoosting` **không nhanh hơn** GB, thậm chí chậm hơn, vì lợi thế gộp bin chỉ rõ khi có
hàng trăm nghìn dòng. HGB cũng kém chính xác hơn (RMSE_log ~0,14 so với 0,125) do không có `subsample`/`max_features`
và chưa được tune riêng.

---

## 4. TIÊU CHÍ HOÀN THÀNH

```
   ☑ Phân biệt "thiếu thật" / "không có tiện ích"     → 17 / 2 cột (mục 2)
   ☑ ≥ 5 cột mã hoá thứ tự đúng                        → 18 cột
   ☑ log1p cho nhãn + đổi ngược khi báo cáo            → skew 1,88 → 0,12
   ☑ Đường train/validation loss theo số cây           → 3.3 (trung bình 5 fold)
   ☑ ⭐ Median APE < 12%                               → 4,95%
   ☑ ⭐ Khoảng 10–90% + độ phủ thực tế                 → 65,8% → 77,1% (conformal)
   ☑ Bảng human-in-the-loop                            → 3.7 (kèm quét ngưỡng)
   ☑ Đo hiệu quả feature engineering                   → 3.4 (gần như không có tác dụng)
```

Mức tham chiếu của đề (RMSE_log 0,12–0,13, Median APE 8–10%): đạt (CV 0,119, test 0,125; Median APE 4,95%).

## 5. HẠN CHẾ

1. **Test nhỏ (292 căn).** Kết luận so sánh model dựa chủ yếu vào CV; một lần test có thể lệch như đã thấy với Linear.
2. **Tuning trong nhiễu.** Các cấu hình GB chênh nhau ít hơn một độ lệch chuẩn giữa các fold.
3. **Ngoại lệ chưa xử lý.** Id 1299 là căn rất lớn bán giá rẻ bất thường (`Partial`). Cây chịu được,
   model tuyến tính thì không. Một AVM thật nên có luật chặn riêng cho các giao dịch bất thường.
4. **Độ phủ conformal 77,1%**, hơi dưới 80%, do tập calib chỉ có 176 căn.
5. Dữ liệu Ames 2006–2010 không áp dụng trực tiếp cho thị trường khác hay thời điểm khác.

## 6. SẢN PHẨM & CÁCH CHẠY

```
TT-18-Gradient-Boosting-Regressor/
├── README.md
├── data/train.csv, data/DATA_SOURCE.md
├── notebooks/gbr_regressor_house.ipynb   ← gọi hàm từ src/train.py, có "Đọc kết quả" từng bước
├── src/train.py                          ← toàn bộ pipeline, đường dẫn tính từ __file__
├── models/gbr_pipeline.joblib            ← GB + filler thiếu thật + 3 quantile + hệ số conformal
├── reports/
│   ├── phan_loai_gia_tri_thieu.csv, missing_analysis.png, skew_saleprice.png
│   ├── ridge_chon_alpha_cv.csv, gb_tuning_cv.png/.csv, loss_theo_so_cay.png/.csv
│   ├── so_sanh_feature_engineering.csv, so_sanh_models.csv, so_sanh_models_theo_fold.csv
│   ├── sai_so_theo_nhom_gia.csv, top5_sai_so_lon.csv
│   ├── khoang_gia.png, ape_distribution.png, human_in_the_loop.csv, human_in_the_loop_quet_nguong.csv
│   ├── gb_vs_hist_gb.csv
│   └── tom_tat.json
└── requirements.txt
```

```bash
pip install -r requirements.txt
python src/train.py
```

**Tham khảo:** [Buổi 6 — Boosting](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-06-Ensemble-EndToEnd/Tai-Lieu/ly_thuyet_chi_tiet_buoi_06.md) · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
