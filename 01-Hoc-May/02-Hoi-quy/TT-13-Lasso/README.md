# TT-13 — LASSO REGRESSION (L1)
## Chọn ra các chỉ số xét nghiệm quan trọng nhất trong 200 chỉ số

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao) |
| 🧠 **Nhóm** | Hồi quy + **CHỌN ĐẶC TRƯNG tự động** |
| 🔧 **Thuật toán** | Lasso (phạt L1) |
| 🏭 **Lĩnh vực** | Y tế · Xét nghiệm · Nghiên cứu lâm sàng |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐ |

> **Kết quả chính:**
> * Trên 200 biến (10 thật + 190 nhiễu), Linear Regression overfit nặng: R² train 0,81 nhưng **R² test −0,36**.
> * **LassoCV** (alpha = 7,92, chọn bằng 5-fold CV) chỉ giữ **5/200 biến** và đạt **R² test 0,455**
>   (RMSE 53,7). Kết quả này tốt hơn Ridge (R² 0,364), dù Ridge giữ cả 200 biến.
> * **Chấm điểm chọn biến:** giữ 4/10 biến thật (`bmi, bp, s3, s5`) và chỉ giữ nhầm **1/190** biến nhiễu.
> * **Đề xuất:** bộ 4 xét nghiệm, **180.000đ/bệnh nhân**, rẻ hơn 51,4% so với bộ 10 chỉ số gốc.

---

## 1. THUẬT TOÁN & BÀI TOÁN

```
   Ridge phạt:  λ·Σ wᵢ²   → CO hệ số về gần 0
   Lasso phạt:  λ·Σ |wᵢ|  → ĐẨY hệ số về ĐÚNG 0  → vừa hồi quy vừa CHỌN BIẾN (model thưa)
```

**Bài toán:** bệnh viện muốn xây bộ xét nghiệm sàng lọc tiến triển tiểu đường. Có thể đo 200 chỉ số,
mỗi chỉ số tốn 30.000–200.000đ, và bệnh nhân không thể lấy 200 ống máu. Cần chọn ít chỉ số mà vẫn đủ
tốt để dự đoán.

## 2. DỮ LIỆU & THIẾT KẾ THÍ NGHIỆM

| | |
|---|---|
| **Gốc** | `sklearn.datasets.load_diabetes()`: 442 bệnh nhân × 10 chỉ số (`age, sex, bmi, bp, s1–s6`), nhãn = mức tiến triển bệnh sau 1 năm |
| **Mở rộng** | + 190 cột nhiễu Gaussian thuần tuý (`chi_so_nhieu_000…189`, `seed=42`) → **200 cột, chỉ 10 cột có tín hiệu** |
| **Chia** | 80/20, `random_state=42` → **353 train / 89 test**; p = 200 gần bằng n = 353 |
| **Chọn alpha** | 5-fold CV trên train (LassoCV, lưới `logspace(-4, 1, 100)`); tập test chỉ dùng để báo cáo |

⭐ Vì biết trước 10 cột nào là thật, việc chọn biến **chấm điểm được khách quan**.

---

## 3. KẾT QUẢ

### 3.1. Baseline: Linear Regression trên 200 biến → overfit

| | RMSE | R² |
|---|---:|---:|
| Train | 33,99 | 0,8099 |
| Test | **84,78** | **−0,3566** |

R² test âm nghĩa là model **tệ hơn cả việc đoán giá trị trung bình**. RMSE test gấp 2,5 lần train. Với
p ≈ n, OLS đủ bậc tự do để khớp luôn nhiễu của 190 cột vô nghĩa.

### 3.2. LassoCV — `reports/rmse_theo_alpha.png/.csv`

Đường RMSE lấy thẳng từ `mse_path_` của LassoCV, tức đúng các fold đã dùng để chọn alpha. **Không dùng
tập test.** Code `assert` rằng cực tiểu của đường này trùng alpha được chọn.

| alpha | 0,0001 (≈ OLS) | 0,0105 | 0,977 | **7,92 (chọn)** | 10 |
|---|---:|---:|---:|---:|---:|
| RMSE train | 33,99 | 33,99 | 38,96 | 56,28 | 57,01 |
| RMSE CV (5 fold) | 104,77 | 103,41 | 67,31 | **57,91** | 58,32 |
| Số biến còn lại | 200 | 197 | 133 | **5** | 4 |

* Ở alpha nhỏ, CV gấp **3 lần** train (105 so với 34): overfit. Tăng alpha, Lasso loại dần biến nhiễu
  và RMSE CV giảm đều. Đến alpha = 7,92, train và CV gần nhau (56,3 so với 57,9), tức **hết overfit**.
* Ở alpha = 10, CV tăng lại và Lasso bắt đầu loại cả biến thật. Alpha chọn nằm gần biên trên của lưới,
  nhưng đường CV đã đi lên lại, nên cực tiểu nằm trong lưới.
* Bản trước có quét alpha trên **test** và lưu `lasso_alpha_best_on_test`, dù "chỉ để minh hoạ". Bước này
  đã **xoá**: hình và JSON giờ không dùng tập test.

### 3.3. ⭐ Chấm điểm chọn biến — `reports/chon_bien_score.csv/.png`

| Loại | Tổng | Lasso giữ | Tỉ lệ | Danh sách |
|---|---:|---:|---:|---|
| Biến **thật** | 10 | **4** | 40% (recall) | `bmi, bp, s3, s5` |
| Biến **nhiễu** | 190 | **1** | **0,5%** (false positive) | `chi_so_nhieu_009` |

* **Loại nhiễu rất tốt:** 189/190 cột nhiễu bị loại; precision = 4/5 = 80%.
* **Recall 4/10** thấp hơn mức tham chiếu 6–9/10 của đề. Alpha được chọn để **dự đoán tốt nhất**, không
  phải để tìm đủ biến thật. Với 353 dòng, các biến tín hiệu yếu (`age, sex`) hoặc trùng thông tin bị loại
  cùng nhiễu. Ví dụ, s1–s2 có r = 0,90; s4 tương quan với s3 (r = −0,74) và với s5 (r = 0,62), tức phần lớn
  thông tin của s4 đã có trong 2 biến được giữ.
* 4 biến giữ lại đều có ý nghĩa lâm sàng: khối cơ thể, huyết áp, HDL cholesterol, log triglyceride.

### 3.4. Coefficient path — `reports/lasso_path.png`

Ở alpha rất nhỏ cả 200 hệ số khác 0. Khi alpha tăng, 190 đường xám (nhiễu) rơi về 0 gần như đồng loạt và
rất sớm, vì không có tín hiệu "bảo vệ" chúng khỏi phạt. 10 đường màu (biến thật) tồn tại lâu hơn. Tại alpha
LassoCV chỉ còn `bmi, bp, s3, s5` khác 0, khớp với bảng chấm điểm.

### 3.5. Ridge vs Lasso (test 89 dòng) — `reports/ridge_vs_lasso.csv/.png`

| Model | Số biến giữ | RMSE test | MAE test | R² test |
|---|---:|---:|---:|---:|
| Linear Regression | 200/200 | 84,78 | 70,76 | −0,357 |
| Ridge (RidgeCV, alpha = 351) | 200/200 (chỉ co nhỏ) | 58,05 | 48,54 | 0,364 |
| **Lasso (LassoCV, alpha = 7,92)** ✅ | **5/200** | **53,73** | **44,26** | **0,455** |
| Debiased Lasso (OLS trên 5 biến Lasso chọn) | 5/200 | 53,74 | 44,27 | 0,455 |

* **Lasso thắng Ridge** 4,3 RMSE, vì 190/200 biến là nhiễu thuần tuý. Ridge buộc phải chia một phần trọng
  số cho mọi biến nhiễu. Ở TT-12 thì ngược lại: mọi biến đều có tín hiệu nên Ridge và Lasso ngang nhau.
* **Debiased lasso ≈ Lasso** (53,74 so với 53,73): giá trị của Lasso ở bài này nằm ở **chọn biến**, còn
  việc co nhẹ hệ số giữ lại gần như không ảnh hưởng đến dự đoán.
* Mọi model chỉ được báo cáo trên test, không model nào được chọn bằng test.

### 3.6. ⚠️ Thí nghiệm biến tương quan — `reports/thi_nghiem_tuong_quan.csv`

Nhân đôi `bmi` → `bmi_dup = bmi + N(0; 0,01)` (r = **0,9806**), chạy LassoCV trên 5 cách chia train/test:

| seed | hệ số bmi | hệ số bmi_dup | Giữ | Tổng hai hệ số |
|---:|---:|---:|---|---:|
| 0 | 26,13 | 0 | chỉ bmi | 26,1 |
| 1 | 24,10 | 0,61 | cả hai | 24,7 |
| 2 | 18,53 | 4,90 | cả hai | 23,4 |
| 3 | 24,92 | 0 | chỉ bmi | 24,9 |
| 4 | 16,15 | 6,21 | cả hai | 22,4 |

* **Lựa chọn không ổn định:** 2/5 seed chỉ giữ `bmi`, 3/5 giữ **cả hai** với tỉ lệ chia khác nhau. Hệ số
  riêng của `bmi` dao động 16,2–26,1 (±25%).
* Thực nghiệm khác lý thuyết thường nói ("chọn 1 bỏ 1"): Lasso **không** luôn loại hẳn một biến, nó chia
  hệ số không ổn định.
* **Tổng** hai hệ số ổn định hơn nhiều (22,4–26,1, ±8%). Tác động gộp của nhóm được ước lượng tốt, phân
  chia thì không. Đây là cùng hiện tượng với TT-12 (Ridge, TV/Facebook).
* Hệ quả y tế: không được kết luận "chỉ số bị loại là vô dụng". Nó có thể chỉ trùng thông tin với chỉ số
  được giữ, và chỉ số bị loại có khi lại rẻ hơn. Đây là động lực của ElasticNet (TT-14).

### 3.7. ✍️ Đề xuất bộ xét nghiệm — `reports/de_xuat_bo_xet_nghiem.csv`

Chi phí giả định: `age, sex` = 0đ (nhân khẩu học); `bmi` 20.000đ; `bp` 30.000đ; `s1–s6` 40.000–70.000đ
(xét nghiệm máu). 190 chỉ số nhiễu được gán ngẫu nhiên 30.000–200.000đ.

| So với | Chi phí | Đề xuất | Tiết kiệm |
|---|---:|---|---:|
| Bộ 200 chỉ số (giả lập) | 22.105.930đ | 286.566đ (5 chỉ số Lasso chọn, gồm 1 nhiễu) | 98,7% |
| **Bộ 10 chỉ số thật** | **370.000đ** | **180.000đ** (`bmi` · `bp` · `s3` · `s5`) | **51,4%** |

**Đề xuất:** đo **4 chỉ số `bmi, bp, s3, s5`, 180.000đ/bệnh nhân**.
* Con số 98,7% bị **thổi phồng**: 22,1 triệu gần như toàn bộ là chi phí ngẫu nhiên gán cho 190 chỉ số nhiễu
  giả lập. So sánh có ý nghĩa là với bộ 10 chỉ số thật: tiết kiệm **51,4% (190.000đ)**.
* Bỏ `chi_so_nhieu_009` (false positive, hệ số rất nhỏ) là quyết định **rà soát lâm sàng**, không dựa trên
  kết quả test. Thực tế ta không biết trước biến nào là nhiễu, nên bác sĩ phải duyệt từng chỉ số.
* Giới hạn: R² 0,455, tức bộ 4 chỉ số giải thích chưa tới một nửa biến động. Đây là công cụ **sàng lọc**,
  không phải chẩn đoán.

---

## 4. VÌ SAO L1 ĐƯA HỆ SỐ VỀ ĐÚNG 0 CÒN L2 THÌ KHÔNG

```
      RIDGE: Σwᵢ² ≤ t  (hình TRÒN)        LASSO: Σ|wᵢ| ≤ t  (hình THOI)
          ╭───╮                                ◇
         │  ●  │                              ╱ ● ╲     ● = điểm elip MSE chạm vùng ràng buộc
          ╰───╯                              ╲   ╱
      biên trơn, không có góc trên trục     GÓC nhọn nằm ĐÚNG trên trục → wᵢ = 0
```

* **Hình học:** nghiệm là điểm elip MSE (quanh nghiệm OLS) chạm vùng ràng buộc lần đầu khi phình ra. Hình
  thoi có góc nhọn nằm trên trục toạ độ, và với đa số vị trí của elip thì điểm chạm rơi vào một góc, tức
  một hệ số = 0. Hình tròn trơn, nên điểm chạm hầu như không nằm đúng trên trục.
* **Đại số:** đạo hàm của `λ|w|` là `λ·sign(w)`, một lực **không đổi** kéo w về 0 dù w nhỏ đến đâu. Nếu
  tương quan của biến với phần dư nhỏ hơn λ thì nghiệm tối ưu là đúng w = 0. Đạo hàm của `λw²` là `2λw`,
  lực này **yếu dần** khi w nhỏ nên không bao giờ đẩy w tới đúng 0.
* **Bằng chứng ở bài này:** Lasso đưa 195/200 hệ số về đúng 0, còn Ridge giữ 200/200 khác 0 (3.5).

## 5. TIÊU CHÍ HOÀN THÀNH

```
   ☑ ⭐ Bảng chấm điểm: giữ đúng mấy/10 biến thật, nhầm mấy biến nhiễu → 3.3: 4/10 thật, 1/190 nhiễu
   ☑ Coefficient path của Lasso                                   → 3.4
   ☑ Bảng so sánh Ridge vs Lasso (số biến + RMSE)                 → 3.5: 200 vs 5 biến, RMSE 58,0 vs 53,7
   ☑ Thí nghiệm biến tương quan + nhận xét                        → 3.6
   ☑ Giải thích VÌ SAO hình thoi đưa hệ số về 0                   → 4
   ☑ ✍️ Đề xuất bộ xét nghiệm + con số tiết kiệm                   → 3.7: 180.000đ, −51,4%
```

**Các bước đề yêu cầu:**
```
   ☑ 1. load_diabetes + 190 cột nhiễu       ☑ 6. RMSE train/CV theo alpha
   ☑ 2. Linear Regression 200 cột (overfit) ☑ 7. Ridge vs Lasso
   ☑ 3. LassoCV                             ☑ 8. Thí nghiệm biến tương quan (5 seed)
   ☑ 4. Chấm điểm chọn biến                 ☑ 9. Debiased lasso
   ☑ 5. Coefficient path                    ☑ 10. Đề xuất bộ xét nghiệm + chi phí
```

## 6. HẠN CHẾ

1. **Một lần chia train/test** cho bảng chấm điểm. Với 353 dòng, bộ biến được chọn có thể đổi khi chia khác
   (3.6 cho thấy điều này). Stability Selection (chạy Lasso trên nhiều mẫu con, giữ biến được chọn > 60% số
   lần) sẽ cho kết quả đáng tin hơn.
2. **Lasso không ổn định với biến tương quan** (3.6). Kết luận ở mức "nhóm chỉ số" đáng tin hơn ở mức "từng chỉ số".
3. **Tối ưu dự đoán ≠ tìm đúng biến:** alpha CV cho recall chỉ 40%. Muốn tìm đủ biến thật phải chấp nhận
   alpha nhỏ hơn và nhiều false positive hơn.
4. **Chi phí là giả định**, và 190 chỉ số nhiễu là giả lập. Con số tiết kiệm thực tế phụ thuộc bảng giá bệnh viện.
5. **Không phải nhân quả:** chỉ số được giữ là chỉ số *dự đoán* tốt, chưa chắc là nguyên nhân gây tiến triển bệnh.

## 7. SẢN PHẨM & CÁCH CHẠY

```
TT-13-Lasso/
├── README.md
├── notebooks/lasso_feature_selection.ipynb   ← giải thích từng bước + output thật
├── src/train.py                              ← tái lập toàn bộ reports/ và models/
├── models/lasso_pipeline.joblib              ← Pipeline(StandardScaler → LassoCV), alpha = 7,92
├── reports/
│   ├── chon_bien_score.csv/.png      ← chấm điểm chọn biến
│   ├── lasso_path.png                ← coefficient path
│   ├── rmse_theo_alpha.png/.csv      ← RMSE train / CV theo alpha (không dùng test)
│   ├── ridge_vs_lasso.csv/.png
│   ├── thi_nghiem_tuong_quan.csv
│   ├── de_xuat_bo_xet_nghiem.csv
│   └── tom_tat.json
└── requirements.txt
```

```bash
pip install -r requirements.txt
python src/train.py      # vài chục giây
```

**Tham khảo:** [Buổi 13 — Regularization & chọn biến](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
