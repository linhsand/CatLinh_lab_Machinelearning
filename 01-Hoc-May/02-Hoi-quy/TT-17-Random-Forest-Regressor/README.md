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

> **Kết quả chính (test 60.031 vé; mọi siêu tham số chọn trên train bằng OOB / CV / validation):**
> * **Random Forest** `max_features=0,7`, `min_samples_leaf=5`, 100 cây: **R² test 0,9857**, `oob_score_` 0,9860,
>   MAE 1.206 Rupee. Model nén 77 MB.
> * Cấu hình của đề `(1,0; 2)` là **tốt nhất theo OOB**, nhưng nặng gấp đôi (139 MB). Bản chọn gọn kém 0,5% RMSE
>   nhưng **kém 10% MAE**: đó là đánh đổi có chủ đích, có ghi rõ.
> * **XGBoost sau khi tune thắng RF:** RMSE thấp hơn 13% (2.341 so với 2.712), R² 0,989, model 31 MB.
> * **PDP:** vé Economy đặt trước 30 ngày thay vì 7 ngày **rẻ ~5.140 Rupee (−50%)**; Business chỉ cần tránh tuần cuối.

---

## 1. THUẬT TOÁN & BÀI TOÁN

Nhiều cây hồi quy học trên các mẫu **bootstrap** khác nhau, mỗi lần chia chỉ xét một phần đặc trưng; dự đoán là
**trung bình** của các cây. Mỗi cây bỏ sót ~37% dòng train (**out-of-bag**), nên có sẵn một ước lượng ngoài mẫu
(`oob_score_`) mà bài này dùng để **chọn siêu tham số**. Vẫn giữ điểm yếu của cây: **không ngoại suy**.

**Bài toán:** OTA muốn hiển thị *"giá hiện tại X, dự báo tuần sau Y → nên mua ngay / chờ"*. Giá phụ thuộc phi tuyến
và có tương tác (số ngày còn lại × hạng vé × hãng × tuyến).

## 2. DỮ LIỆU & CÁCH ĐÁNH GIÁ

| | |
|---|---|
| **Nguồn** | [Flight Price Prediction (Kaggle)](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction), vé nội địa Ấn Độ 2022. Xem `data/DATA_SOURCE.md` |
| **Kích thước** | 300.153 vé × 9 đặc trưng (sau khi bỏ `Unnamed: 0` và `flight`), không có giá trị thiếu |
| **Nhãn** | `price` (Rupee): Economy trung bình 6.572, Business 52.540 (gấp 8,0 lần) |
| **Bỏ cột** | `flight`: 1.561 mã chuyến; one-hot sẽ thành hơn 1.500 cột và cây học thuộc giá từng mã |
| **Mã hoá** | `OneHotEncoder` (dense) cho 7 biến phân loại → 35 cột, cộng `duration`, `days_left` = **37 cột**. Cây không cần scale |
| **Chia** | 80/20 → 240.122 train / 60.031 test |
| **Chọn siêu tham số** | RF: **OOB**. Cây đơn: **5-fold CV**. XGBoost: **15% train làm validation** + early stopping. Test chỉ để báo cáo |

---

## 3. KẾT QUẢ

### 3.1. EDA — `reports/gia_theo_days_left.png`, `boxplot_class_airline.png`

* Giá trung bình theo `days_left` **đỉnh ở ngày 2 (30.211)** rồi giảm dần về ~19.000 khi còn 20–49 ngày, còn ngày 1
  lại thấp (21.592). Hình dạng này khác mô tả của đề, và chứa lẫn khác biệt về cơ cấu chuyến (PDP ở 3.6 tách riêng).
* `class` áp đảo. `Vistara` và `Air_India` đắt nhất vì đó là **hai hãng duy nhất bán Business**.

### 3.2. Baseline: train vs test — `reports/baseline_train_test.csv`, `cay_don_chon_depth_cv.csv`

| Model | MAE train | MAE test | RMSE train | RMSE test | R² test |
|---|---:|---:|---:|---:|---:|
| Dummy | 19.757 | 19.769 | 22.696 | 22.704 | 0,000 |
| Linear Regression | 4.574 | 4.553 | 6.752 | 6.762 | 0,911 |
| 1 cây không giới hạn | **58** | 1.162 | **606** | 3.532 | 0,976 |
| 1 cây `max_depth=20` (5-fold CV) | 802 | 1.419 | 2.077 | 3.446 | 0,977 |

Cây không giới hạn **overfit rõ** (MAE test gấp 20 lần train). Cắt độ sâu bằng CV giảm RMSE nhưng tăng MAE.

### 3.3. Tuning RF bằng OOB — `reports/rf_tuning_oob.png/.csv`

RMSE-OOB, 100 cây (mỗi cấu hình chấm trên các dòng mà từng cây không thấy, chỉ dùng train):

| min_samples_leaf \ max_features | 0,3 | 0,5 | 0,7 | 1,0 |
|---|---:|---:|---:|---:|
| 1 | 2.820 | 2.787 | 2.759 | 2.743 |
| 2 | 2.724 | 2.686 | 2.664 | **2.662** (tốt nhất) |
| 5 | 2.804 | 2.726 | **2.685 ✔ chọn** | 2.680 |
| 10 | 2.992 | 2.872 | 2.800 | 2.776 |

* Cấu hình gợi ý của đề `(1,0; 2)` **là tốt nhất** theo OOB, nên giờ nó có căn cứ.
* **Quy tắc chọn:** lấy cấu hình **ít nút nhất** trong số các cấu hình cách RMSE-OOB tốt nhất ≤ 1%. Kết quả là
  **`(0,7; 5)`**: +0,86% RMSE-OOB, nhưng **~37.800 nút/cây thay vì ~81.000**. Lý do: model phục vụ trên web OTA (bộ
  nhớ, tốc độ), và phải lưu được vào repo. Bản cũ lưu model 300 cây nặng **1,75 GB**, vượt xa giới hạn 100 MB/file của
  GitHub, nên repo thực chất không có model.

### 3.4. RMSE-OOB theo số cây — `reports/rmse_theo_so_cay.png/.csv`

| Số cây | 10 | 30 | 50 | **100** | 150 | 200 | 300 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RMSE-OOB | 4.264 | 2.718 | 2.700 | **2.685** | 2.682 | 2.678 | 2.676 |

Bão hoà sớm: **100 cây** là số nhỏ nhất trong 0,5% so với mức tốt nhất. 300 cây (như đề) chỉ tốt hơn 0,34% mà tốn
gấp 3 lần. Bản cũ vẽ đường này trên **test**; bản này dùng OOB.

**Model cuối:** `oob_score_` **0,9860**, sát R² test **0,9857**. MAE train 990 so với test 1.206.

### 3.5. Permutation importance (gộp theo biến gốc) — `reports/permutation_importance.png/.csv`

| Biến | Permutation (R² giảm) | Hạng | Impurity mặc định (gộp) | Hạng |
|---|---:|---:|---:|---:|
| class | 1,738 | 1 | 0,878 | 1 |
| duration | 0,104 | 2 | 0,045 | 2 |
| airline | 0,045 | 3 | 0,022 | 3 |
| destination_city | 0,044 | 4 | 0,010 | **7** |
| source_city | 0,043 | 5 | 0,010 | **6** |
| days_left | 0,029 | 6 | 0,014 | **4** |
| arrival_time | 0,014 | 7 | 0,005 | 8 |
| departure_time | 0,012 | 8 | 0,004 | 9 |
| stops | 0,011 | 9 | 0,012 | **5** |

Impurity đẩy `days_left` và `stops` lên trên, đồng thời dìm hai cột thành phố. Permutation, vốn xáo **cả cột gốc**
trên 20.000 dòng test, cho thấy tuyến bay quan trọng hơn `days_left`, còn `stops` gần như thừa khi đã có `duration`.

### 3.6. ⭐ PDP cho `days_left` — `reports/pdp_days_left.png/.csv`

| days_left | 1 | 7 | 14 | 30 | 49 |
|---|---:|---:|---:|---:|---:|
| Tất cả | 29.652 | 23.858 | 22.832 | 19.705 | 19.319 |
| Economy | 14.839 | 10.378 | 9.555 | **5.238** | 4.894 |
| Business | 62.355 | **53.618** | 52.144 | 51.643 | 51.165 |

* PDP giảm **đơn điệu**, không có "dip" ở ngày 1 như biểu đồ thô (dip đó do cơ cấu chuyến).
* **Economy:** đặt trước 30 ngày thay vì 7 ngày tiết kiệm **~5.140 Rupee (−50%)**, chủ yếu trong đoạn 14–30 ngày.
* **Business:** tiết kiệm nằm ở **tuần cuối** (đặt trước 7 ngày thay vì 1 ngày: −8.737). Từ 14 ngày trở ra gần phẳng.
* **Quy tắc tư vấn:** *Economy còn > 14 ngày thì nên đặt sớm; Business chỉ cần tránh tuần cuối.*

### 3.7. ⭐ Khoảng dự báo 10–90% từ các cây — `reports/khoang_du_bao.csv`

| | Độ phủ thực tế | Độ rộng TB |
|---|---:|---:|
| Tất cả | **76,6%** (danh nghĩa 80%) | 2.396 Rupee |
| Economy | 74,5% | ~1.570 (~24% giá) |
| Business | 81,4% | ~4.230 (~8% giá) |

Phân vị giữa các cây đo **độ bất đồng giữa các cây**, không đo nhiễu của từng vé, nên không có bảo đảm độ phủ, và độ
phủ phụ thuộc siêu tham số: `(1,0; 2)` cho 86%, `(0,7; 5)` cho 76,6%. Muốn đúng 80% thì cần hiệu chỉnh (split-conformal,
như TT-18).

### 3.8. ⚠️ Ngoại suy — `reports/ngoai_suy_days_left.csv`

Chuyến mẫu Air_India Delhi → Kolkata, Economy, 1 điểm dừng:

| days_left | 1 | 20 | 49 | 60 | 100 | 365 |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 15.315 | 6.614 | 6.905 | **6.905** | **6.905** | **6.905** |
| Linear Regression | 8.521 | 6.042 | 2.259 | 824 | **−4.393** | **−38.961** |

RF **kẹp trần**: mọi `days_left ≥ 49` cho cùng một giá. Linear còn tệ hơn: nó ngoại suy ra **giá âm**. Hệ thống phải
từ chối hoặc gắn cảnh báo với vé đặt trước quá 7 tuần.

### 3.9. So sánh với XGBoost (cũng tune) — `reports/so_sanh_models.csv`, `xgb_tuning_val.csv`

XGBoost: lưới `max_depth` {6, 8, 10, 12} × `learning_rate` {0,1; 0,05}, early stopping (100 vòng) trên 15% train.
Kết quả chọn **depth 10, lr 0,05, 1.345 vòng** (RMSE_val 2.373). Cấu hình cũ hard-code (depth 8, 300 vòng) dừng quá
sớm: riêng depth 8 cần ~1.900–3.900 vòng. Depth 6 chạm trần 4.000 vòng mà chưa hội tụ.

| Model (test) | MAE | RMSE | R² | Fit | Model nén |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 4.553 | 6.762 | 0,911 | 2,5 s | — |
| 1 cây `max_depth=20` (CV) | 1.419 | 3.446 | 0,977 | 5 s | — |
| **RF (0,7; 5), 100 cây — chọn** | 1.206 | 2.712 | 0,9857 | 16 s | **77 MB** |
| RF (1,0; 2), 100 cây — OOB tốt nhất | **1.093** | 2.699 | 0,9859 | 40 s | 139 MB |
| **XGBoost depth 10, 1.345 vòng** | 1.097 | **2.341** | **0,9894** | 42 s | **31 MB** |

* **Cái giá của RF gọn:** RMSE kém 0,5% nhưng **MAE kém 10%**. Lá ≥ 5 mẫu làm mịn dự đoán, nên vé Economy rẻ (chiếm
  đa số) bị lệch nhiều hơn. Quy tắc chọn theo RMSE-OOB không thấy được điều này. Nếu MAE là metric nghiệp vụ, nên dùng
  `(1,0; 2)` và lưu model ngoài Git.
* **XGBoost sau khi tune thắng RF:** RMSE thấp hơn 13%, MAE ngang RF tốt nhất, model nhỏ nhất. Bản cũ kết luận "RF
  thắng XGBoost" chỉ vì XGB bị hard-code 300 vòng.
* RF vẫn có ưu điểm: ít siêu tham số nhạy, có OOB miễn phí, có khoảng dự báo từ các cây. Nhưng xét dự đoán điểm,
  **XGBoost là lựa chọn tốt hơn** cho bài này.

### 3.10. Mở rộng — `reports/mo_rong.csv`

| (cùng tham số RF đã chọn) | MAE | RMSE |
|---|---:|---:|
| RF 1 model chung | 1.206 | 2.712 |
| RF tách 2 model Economy / Business | 1.204 | 2.718 |
| ExtraTrees | 1.339 | 2.818 |

**Tách theo hạng vé không giúp gì**: rừng chung đã tự tách theo `class` ngay ở gốc. **ExtraTrees kém hơn 4%** và không
nhanh hơn ở đây.

---

## 4. TIÊU CHÍ HOÀN THÀNH

```
   ☑ Bỏ cột flight, giải thích lý do                  → mục 2 (1.561 mã chuyến)
   ☑ Báo cáo oob_score_                               → 0,9860 (R² test 0,9857)
   ☑ Biểu đồ RMSE theo số cây                         → 3.4 (dùng OOB, không dùng test)
   ☑ Permutation importance, không dùng mặc định      → 3.5
   ☑ ⭐ PDP days_left + số tiền cụ thể                → 3.6
   ☑ ⭐ Khoảng 10–90% + tỉ lệ phủ                     → 3.7 (76,6%)
   ☑ Thí nghiệm ngoại suy                             → 3.8
   ☑ R² test > 0,95, không rò rỉ                      → 0,9857; chỉ dùng thông tin biết trước khi mua vé
```

## 5. HẠN CHẾ

1. **RF gọn đổi 10% MAE lấy kích thước model** (3.9). Đây là quyết định triển khai, không phải tối ưu độ chính xác.
2. **Khoảng dự báo chưa hiệu chỉnh** (76,6%, Economy chỉ 74,5%).
3. **Không có chuỗi giá theo thời gian của cùng một chuyến**, nên chưa backtest được quy tắc "giá tuần sau cao hơn
   10% → mua ngay" (mở rộng 3 của đề). PDP chỉ cho hiệu ứng trung bình.
4. **Dữ liệu Ấn Độ, thu thập 11/02–31/03/2022**, `days_left` chỉ 1–49. Không áp dụng thẳng cho thị trường Việt Nam hay mùa cao
   điểm.
5. Thời gian fit được đo khi máy chạy song song tác vụ khác, nên chỉ có ý nghĩa tương đối.

## 6. SẢN PHẨM & CÁCH CHẠY

```
TT-17-Random-Forest-Regressor/
├── README.md
├── data/Clean_Dataset.csv, data/DATA_SOURCE.md
├── notebooks/rf_regressor_flight.ipynb   ← gọi hàm từ src/train.py, có "Đọc kết quả" từng bước
├── src/train.py                          ← toàn bộ pipeline, đường dẫn tính từ __file__
├── models/rf_reg.joblib                  ← Pipeline(OneHot → RF 100 cây, mf 0,7, leaf 5), nén zlib, 77 MB
├── reports/
│   ├── gia_theo_days_left.png, boxplot_class_airline.png
│   ├── baseline_train_test.csv, cay_don_chon_depth_cv.csv
│   ├── rf_tuning_oob.png/.csv, rmse_theo_so_cay.png/.csv
│   ├── permutation_importance.png/.csv, pdp_days_left.png/.csv
│   ├── khoang_du_bao.csv, ngoai_suy_days_left.csv
│   ├── xgb_tuning_val.csv, so_sanh_models.csv, mo_rong.csv
│   └── tom_tat.json
├── requirements.txt
└── .gitignore                            ← bỏ qua *.log của nbconvert
```

```bash
pip install -r requirements.txt
python src/train.py        # ~40 phút trên máy 20 luồng (tuning RF 16 cấu hình + XGBoost 8 cấu hình)
```

**Tham khảo:** [Buổi 3 — Random Forest](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-03-Feature-Eng-Tree/Tai-Lieu) · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
