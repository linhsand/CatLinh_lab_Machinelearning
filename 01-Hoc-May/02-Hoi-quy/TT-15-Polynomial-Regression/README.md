# TT-15 — POLYNOMIAL REGRESSION
## Quan hệ nhiệt độ ↔ công suất nhà máy điện là ĐƯỜNG CONG, không phải đường thẳng

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao) |
| 🧠 **Nhóm** | Hồi quy phi tuyến (vẫn tuyến tính theo tham số) |
| 🔧 **Thuật toán** | Polynomial Regression (`PolynomialFeatures` + Linear/Ridge) |
| 🏭 **Lĩnh vực** | Năng lượng · Vận hành nhà máy |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐ |

> **Kết quả chính (test 1.914 giờ):**
> * Đa thức **bậc 5 + Ridge** (bậc chọn bằng 5-fold CV trên train) đạt **RMSE 4,05 MW, R² 0,943**, so với bậc 1
>   (đường thẳng) RMSE 4,50, R² 0,930.
> * Cải thiện chỉ **−10%**, vì bài toán vốn gần tuyến tính: AT–PE có r = −0,948, và đường thẳng đã giảm RMSE 74%
>   so với Dummy.
> * Nhưng cải thiện **tập trung ở ngày nóng > 27 °C (−20%)**, nơi đường thẳng đoán thiếu có hệ thống 1,7 MW.
> * Random Forest tốt hơn hẳn (RMSE 3,23, R² 0,964). Đa thức bậc 5 **ngoại suy vô lý**: ở 60 °C dự báo 683 MW,
>   vượt cả mức tối đa từng đo là 496 MW.

---

## 1. THUẬT TOÁN & BÀI TOÁN

```
   Bậc 1:  ŷ = w₁x + b          Bậc 2:  ŷ = w₁x + w₂x² + b          Bậc 3:  … + w₃x³
   MẸO: tạo thêm cột x², x³, x·y … rồi chạy Linear/Ridge → vẫn TUYẾN TÍNH theo tham số w
```

**Bài toán:** nhà máy nhiệt điện chu trình hỗn hợp cần dự báo **công suất phát PE (MW)** mỗi giờ từ điều kiện môi
trường để chào giá lên thị trường điện. Chào cao hơn khả năng phát thì bị phạt; chào thấp hơn thì mất doanh thu.
Nhiệt độ tăng làm hiệu suất tua-bin khí giảm theo đường **cong**.

## 2. DỮ LIỆU & CÁCH ĐÁNH GIÁ

| | |
|---|---|
| **Nguồn** | [Combined Cycle Power Plant (UCI)](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant). `src/train.py` tự tải về `data/` |
| **Kích thước** | 9.568 giờ × 4 đặc trưng: `AT` nhiệt độ (1,8–37,1 °C), `V` áp suất chân không, `AP` áp suất khí quyển, `RH` độ ẩm |
| **Nhãn** | `PE` công suất phát: 420,3–495,8 MW, trung bình 454,4 MW, độ lệch chuẩn 17,1 |
| **Chia** | 80/20, `random_state=42` → 7.654 train / 1.914 test |
| **Chọn bậc / alpha** | 5-fold CV **trên train** (`validation_curve`, `RidgeCV`); test chỉ để báo cáo |

Tương quan với PE: AT = **−0,948**, V = −0,870, AP = 0,518, RH = 0,390. AT–V = **0,844**, nên khi tạo đặc trưng đa
thức, đa cộng tuyến sẽ rất nặng.

**Pipeline:** `PolynomialFeatures(include_bias=False) → StandardScaler → Ridge`. Phải scale **sau** khi tạo đa thức:
các cột sinh ra có thang đo chênh nhau khủng khiếp (AT ≈ 20 nhưng AT⁵ ≈ 3·10⁶). Nếu không chuẩn hoá lại, phạt L2
của Ridge sẽ đè lên các cột bậc thấp còn gần như không chạm các cột bậc cao. `include_bias=False` vì intercept đã
có trong model.

---

## 3. KẾT QUẢ

### 3.1. Chọn bậc bằng cross-validation — `reports/validation_curve.png/.csv`, `kiem_tra_bac_cao.png/.csv`

| Bậc | 1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Số cột | 4 | 14 | 34 | 69 | **125** | 209 | 329 | 494 |
| CV Ridge(alpha=1) | 4,572 | 4,309 | 4,256 | 4,220 | **4,201** | 4,187 | 4,174 | 4,163 |
| CV Linear thuần | 4,572 | 4,268 | 4,178 | 4,124 | **4,054** ± 0,09 | 4,056 ± 0,16 | 4,104 | **4,789 ± 0,49** |

* **Bậc 5 được chọn** (cực tiểu CV trên lưới 1–5 của đề). Lưới gốc dừng đúng ở 5, nên bài mở rộng thêm tới bậc 8
  để kiểm tra:
  * **Linear thuần** đạt tốt nhất ở bậc 5. Bậc 6 ngang bằng nhưng dao động giữa các fold gấp đôi, bậc 7 kém đi, và
    **bậc 8 vỡ** (+18% RMSE, dao động gấp 5 lần). Overfit thật chỉ xuất hiện rất muộn, vì 7.654 dòng ≫ 125 cột.
  * **Ridge(1)** vẫn giảm tới bậc 8 nhưng mỗi bậc chỉ bớt ~0,013 MW (0,3%), trong khi số cột tăng 1,6 lần.
  * → Dừng ở bậc 5 là hợp lý: lợi ích sau đó gần 0.
* **Lợi ích giảm dần rất nhanh:** bậc 1 → 2 giảm 0,26 MW CV; 2 → 3 giảm 0,05; 3 → 5 chỉ giảm thêm 0,05. **Bậc 3 đã giữ
  ~85%** lợi ích của bậc 5 với 34 cột thay vì 125. Mức tham chiếu của đề ("bậc 2–3 thường tối ưu") khớp nếu ưu tiên
  model gọn.

### 3.2. Bảng bậc | số cột | RMSE (Linear thuần) — `reports/bang_bac_so_cot_rmse.csv`

| Bậc | Số cột | RMSE train | RMSE test | \|hệ số\| lớn nhất |
|---:|---:|---:|---:|---:|
| 1 | 4 | 4,571 | 4,503 | 14,8 |
| 2 | 14 | 4,261 | 4,231 | 91,6 |
| 3 | 34 | 4,161 | 4,107 | 23.554 |
| 4 | 69 | 4,088 | 4,062 | 1.901.197 |
| 5 | 125 | 3,982 | 3,937 | **1.261.832.125** |

RMSE train ≈ test ở mọi bậc nên **không overfit theo RMSE**. Triệu chứng thật của bậc cao là **hệ số nổ** lên 1,26 tỷ ở
bậc 5: đa cộng tuyến giữa AT, AT², AT·V… làm nghiệm OLS cực kỳ nhạy.

### 3.3. Linear vs Ridge ở bậc cao — `reports/so_sanh_linear_ridge_bac_cao.csv`, `danh_doi_alpha.csv`

| Bậc | Linear: RMSE test / \|hệ số\|max | Ridge (alpha = 0,001): RMSE test / \|hệ số\|max | Hệ số nhỏ hơn |
|---:|---|---|---:|
| 4 | 4,062 / 1,90e6 | 4,091 / 231 | 8.246 lần |
| 5 | 3,937 / 1,26e9 | 4,052 / 206 | **6,1 triệu lần** |

**Đánh đổi alpha ở bậc 5** (5-fold CV trên train):

| | Linear | 1e-8 | 1e-5 | **1e-3 (model)** | 0,1 | 1 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RMSE CV | **4,054** | 4,083 | 4,093 | **4,133** | 4,175 | 4,201 | 4,251 |
| So với Linear | — | +0,7% | +1,0% | **+2,0%** | +3,0% | +3,6% | +4,9% |
| \|hệ số\| lớn nhất | 1,26e9 | 4,75e4 | 2.320 | **206** | 23,9 | 9,5 | 4,7 |

* Trong dải dữ liệu, CV **không cần** regularization: alpha càng nhỏ RMSE càng tốt. RidgeCV dừng ở alpha = 0,001
  chỉ vì đó là **biên dưới** của lưới `logspace(-3, 4)`. Đây là điểm cần nói rõ.
* Giữ alpha = 0,001 là **đánh đổi có chủ đích**: chấp nhận RMSE CV kém 2% (+0,08 MW) để hệ số nhỏ hơn 6 triệu lần.
  Với hệ số cỡ 1e9, sai số đo hay làm tròn nhỏ ở đầu vào cũng bị khuếch đại, và mô hình ngoại suy càng mất kiểm soát.
  Ridge "cứu" ở đây nghĩa là cứu **độ ổn định**, không phải cứu RMSE.

### 3.4. ⭐ Residual plot TRƯỚC / SAU — `reports/residual_truoc_sau.png`

| | Bậc 1 | Bậc 5 Ridge |
|---|---:|---:|
| "Độ cong" phần dư (R² của parabol fit vào phần dư) | 0,063 | **0,003** (−95%) |
| Độ lệch phần dư TB theo 5 dải nhiệt độ | **+0,98 / −0,87 / −1,10 / −0,93 / +1,72** | −0,26 / −0,29 / +0,38 / −0,17 / +0,12 |

Bậc 1 có **hình chữ U rõ ràng**: đoán thiếu ở hai đầu nhiệt độ, đoán thừa ở giữa. Bậc 5 xoá gần hết độ lệch có hệ
thống này (|lệch| ≤ 0,4 MW ở mọi dải).

### 3.5. ⭐ Vì sao cải thiện so với bậc 1 NHỎ — và nó có đáng không? — `reports/cai_thien_tong_the.csv`, `cai_thien_theo_nhiet_do.csv/.png`

| Model (test) | RMSE (MW) | % công suất TB | MAE (MW) | Giảm RMSE so với bậc 1 |
|---|---:|---:|---:|---:|
| Dummy (trung bình) | 17,03 | 3,75% | 14,81 | — |
| Bậc 1 (đường thẳng) | 4,50 | 0,99% | 3,60 | — |
| **Poly bậc 5 + Ridge** | **4,05** | **0,89%** | **3,18** | **−10,0%** |
| Spline Ridge (6 nút) | 4,15 | 0,91% | 3,27 | −7,8% |
| Random Forest (300 cây) | 3,23 | 0,71% | 2,31 | −28,3% |

1. **Cải thiện nhỏ vì bài toán vốn gần tuyến tính.** Đường thẳng đã giảm RMSE 74% so với Dummy (17,0 → 4,5). Lấy RF
   (3,23) làm mốc "đạt được", đường thẳng đã đi **91%** quãng đường. Phần cong còn lại ít (độ cong phần dư chỉ
   0,063), nên đa thức chỉ còn 0,45 MW để ăn.
2. **Nhưng cải thiện KHÔNG đều, tập trung ở ngày nóng:**

   | Dải AT (test) | 2,3–12,2 °C | 12,2–17,8 | 17,8–22,9 | 22,9–27,1 | **27,1–37,1 °C** |
   |---|---:|---:|---:|---:|---:|
   | PE trung bình | 478,7 | 464,9 | 450,8 | 441,2 | **434,9** |
   | RMSE bậc 1 | 4,87 | 4,57 | 3,92 | 4,33 | **4,76** |
   | RMSE poly bậc 5 | 4,53 (−7%) | 4,28 (−6%) | 3,65 (−7%) | 3,92 (−9%) | **3,81 (−20%)** |
   | RMSE Random Forest | 3,85 | 3,30 | 2,79 | 3,10 | 3,00 |

   Lợi ích lớn nhất (**−20%**) rơi vào **ngày nóng**: công suất thấp nhất nhưng nhu cầu điện thường cao nhất. Ở dải này
   bậc 1 lại **đoán thiếu 1,7 MW có hệ thống**, tức chào giá thấp hơn khả năng phát.
3. **Quy ra vận hành:** MAE giảm 0,42 MW, tức **~3.660 MWh/năm** sai lệch dự báo ít hơn (dự báo theo giờ, 8.760 giờ/năm).
   Random Forest giảm ~11.230 MWh/năm. Giá trị tiền phụ thuộc cơ chế phạt lệch của thị trường điện.
4. **Có đáng không?** Đáng nếu cần **mô hình minh bạch**: vẫn là hồi quy tuyến tính, công thức đóng, dự báo tức thì,
   và sửa được độ lệch có hệ thống ở hai đầu. Nhưng chỉ là bước **khiêm tốn**: nếu chấp nhận hộp đen, Random Forest
   giảm RMSE thêm 20% so với đa thức, và tốt hơn đa thức ở **mọi** dải nhiệt độ.

### 3.6. Diễn giải: ở dải nhiệt độ nào công suất giảm nhanh nhất? — `reports/do_doc_theo_nhiet_do.csv`

Đạo hàm dPE/dAT của đa thức bậc 5 (AT → PE) trên toàn bộ dữ liệu:

| AT | 5 °C | **~16 °C** | 20 °C | 35 °C |
|---|---:|---:|---:|---:|
| dPE/dAT (MW/°C) | −1,72 | **−2,59 (dốc nhất)** | −2,43 | −0,15 |

Nhà máy **nhạy nhất với biến động nhiệt độ ở vùng 10–20 °C**: mỗi 1 °C tăng làm mất ~2,5 MW. Khi dự báo thời tiết vùng
này sai 2 °C, dự báo công suất sẽ lệch ~5 MW, lớn hơn cả RMSE của model. Ở trên 30 °C, công suất gần như đã chạm đáy
và ít nhạy hơn.

### 3.7. Mở rộng — `reports/interaction_only_vs_full.csv`, `spline_vs_polynomial.csv`

| Model (test) | Số cột | RMSE | R² |
|---|---:|---:|---:|
| Bậc 2 đầy đủ (có x²) | 14 | 4,232 | 0,9383 |
| Bậc 2 `interaction_only` (chỉ tích chéo) | 10 | 4,284 | 0,9367 |
| Spline (6 nút, bậc 3) | — | 4,153 | 0,9405 |
| Poly bậc 5 Ridge | 125 | 4,052 | 0,9434 |

Bỏ các số hạng bình phương làm RMSE kém đi 0,05: độ cong là của **từng biến** (chủ yếu AT), không chỉ do tương tác.

### 3.8. ⚠️ Ngoại suy — `reports/ngoai_suy_AT.csv/.png`

V, AP, RH giữ ở giá trị trung bình. Dữ liệu train chỉ có AT ∈ [1,8; 37,1] °C và PE ∈ [420,3; 495,8] MW.

| AT | 1,8 °C | 37,1 °C | **45 °C** | **50 °C** | **60 °C** |
|---|---:|---:|---:|---:|---:|
| Bậc 1 | 489,8 | 419,7 | 404,0 | 394,1 | 374,2 |
| **Poly bậc 5** | 479,0 | 424,2 | 418,9 | **442,3 ↑** | **683,0 ↑↑** |
| Spline (`extrapolation="constant"`) | 486,6 | 433,2 | 433,2 | 433,2 | 433,2 |

Ra khỏi dải dữ liệu, đa thức bậc 5 **đổi chiều**: nóng hơn mà công suất *tăng*, và ở 60 °C dự báo 683 MW, vượt xa mức
tối đa từng đo (496 MW). Đó là điều vô lý về mặt vật lý. Đường thẳng ngoại suy hợp lý hơn, spline giữ hằng số. Khi
triển khai cần **chặn đầu vào** về dải đã thấy, hoặc dùng spline.

---

## 4. TIÊU CHÍ HOÀN THÀNH

```
   ☑ Scatter plot cho thấy quan hệ cong                      → reports/scatter_AT_PE.png
   ☑ ⭐ Residual plot TRƯỚC (chữ U) và SAU (phân tán đều)     → 3.4: độ cong 0,063 → 0,003
   ☑ Đường cong xác thực theo bậc, bậc chọn có căn cứ        → 3.1 (CV trên train, kiểm tra tới bậc 8)
   ☑ Bảng bậc | số cột | RMSE train | RMSE test              → 3.2
   ☑ So sánh Linear vs Ridge ở bậc cao                       → 3.3
   ☑ Giải thích vì sao scale SAU khi tạo đa thức             → mục 2
   ☑ Hạn chế: đa thức bậc cao "phát điên" khi ngoại suy      → 3.8: 683 MW ở 60 °C
```

**Các bước đề yêu cầu:**
```
   ☑ 1. EDA scatter AT vs PE           ☑ 6. Bảng bậc | số cột | RMSE
   ☑ 2. Baseline bậc 1                 ☑ 7. Linear vs Ridge bậc 4–5
   ☑ 3. Residual plot bậc 1            ☑ 8. Residual plot bậc tối ưu
   ☑ 4. Bậc 1 → 5, RMSE train/val      ☑ 9. So sánh Random Forest
   ☑ 5. Đường cong xác thực            ☑ 10. Dải nhiệt độ giảm nhanh nhất
   ☑ Mở rộng: interaction_only · SplineTransformer · demo ngoại suy
```

## 5. HẠN CHẾ

1. **Cải thiện nhỏ** (−10% RMSE so với bậc 1). Random Forest tốt hơn đa thức ở mọi dải nhiệt độ.
2. **alpha nằm ở biên lưới**. Mức phạt 0,001 là đánh đổi ổn định ↔ chính xác có chủ đích, không phải tối ưu của CV.
3. **Ngoại suy nguy hiểm** (3.8). Mô hình chỉ dùng được trong AT ∈ [1,8; 37,1] °C.
4. **Một lần chia ngẫu nhiên** trên dữ liệu chuỗi thời gian theo giờ. Các giờ liền kề rất giống nhau nên test có thể
   lạc quan. Chia theo thời gian (train quá khứ, test tương lai) sẽ thực tế hơn.
5. Quy đổi MWh/năm giả định dự báo cả 8.760 giờ với sai số như tập test.

## 6. SẢN PHẨM & CÁCH CHẠY

```
TT-15-Polynomial-Regression/
├── README.md
├── data/Folds5x2_pp.xlsx                 ← tự tải từ UCI khi chạy lần đầu
├── notebooks/polynomial_power_plant.ipynb
├── src/train.py
├── models/poly_pipeline.joblib           ← PolynomialFeatures(5) → StandardScaler → RidgeCV (alpha 0,001)
├── reports/
│   ├── scatter_AT_PE.png, residual_truoc_sau.png
│   ├── validation_curve.png/.csv, kiem_tra_bac_cao.png/.csv, bang_bac_so_cot_rmse.csv
│   ├── so_sanh_linear_ridge_bac_cao.csv, danh_doi_alpha.csv
│   ├── cai_thien_tong_the.csv, cai_thien_theo_nhiet_do.csv/.png
│   ├── so_sanh_random_forest.csv, spline_vs_polynomial.csv, interaction_only_vs_full.csv
│   ├── do_doc_theo_nhiet_do.csv, ngoai_suy_AT.csv/.png
│   └── tom_tat.json
└── requirements.txt
```

```bash
pip install -r requirements.txt
python src/train.py
```

**Tham khảo:** [Buổi 13 — Regression nâng cao](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
