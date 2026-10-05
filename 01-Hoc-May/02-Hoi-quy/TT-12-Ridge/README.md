# TT-12 — RIDGE REGRESSION (L2)
## Phân bổ ngân sách quảng cáo đa kênh khi các kênh chạy cùng lúc

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao) |
| 🧠 **Nhóm** | Hồi quy có regularization |
| 🔧 **Thuật toán** | Ridge (phạt L2) |
| 🏭 **Lĩnh vực** | Marketing · Truyền thông |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐ |

> **Kết quả chính:**
> * Ridge (alpha = 4,71, chọn bằng 5-fold CV) giảm dao động hệ số TV/Facebook **~93%** so với OLS, và
>   hệ số Facebook không còn ra âm (OLS: 39% số lần). Ridge không mất độ chính xác: RMSE test 383,2
>   (OLS 383,6, baseline Dummy 651,8).
> * Nhưng **ổn định ≠ đúng**. Dữ liệu tự sinh nên biết hệ số thật, và Ridge xếp Facebook là kênh hiệu
>   quả nhất (3,59/đơn vị) trong khi thật ra nó kém nhất (1,80).
> * Phân bổ ngân sách theo hệ số Ridge chuẩn hoá (cách cũ) sẽ **mất 4,5% doanh thu**.
> * **Đề xuất:** giữ cơ cấu hiện tại (TV 912,7 tr · Facebook 547,0 tr · Google 540,3 tr / 2 tỷ) và chạy
>   A/B test để tách TV và Facebook.

---

## 1. THUẬT TOÁN & BÀI TOÁN

```
   Linear Regression:  min  MSE                    → w = (XᵀX)⁻¹Xᵀy
   Ridge:              min  MSE + λ·Σwᵢ²           → w = (XᵀX + λI)⁻¹Xᵀy
```

Khi hai cột của X gần song song, XᵀX gần suy biến, nên nghịch đảo của nó rất nhạy với nhiễu và hệ
số OLS "nhảy loạn". Cộng thêm λI giữ ma trận khả nghịch ổn định. Ridge **co** hệ số về gần 0 nhưng
không bao giờ đúng 0 (xem 4.3).

**Bài toán:** CMO hỏi "kênh nào tạo ra doanh thu, nên dồn tiền vào đâu?" (ngân sách 2 tỷ/tháng). Các
kênh được tăng/giảm **cùng nhau**, nên ngân sách tương quan rất cao.

## 2. DỮ LIỆU

Tự sinh 500 kỳ theo gợi ý của đề, nhưng **làm khó hơn** để thấy rõ vấn đề. Bộ gốc của đề (nhiễu
Facebook std 15, nhiễu doanh thu std 50) cho R² ≈ 0,99, sạch tới mức hệ số OLS vẫn ổn định.

```python
tv = U(50, 500);  fb = 0,6·tv + N(0, 3);  gg = U(20, 300)        # 1 đơn vị = 1 triệu VND
doanh_thu = 3,2·tv + 1,8·fb + 2,5·gg + N(0, 400)                  # ← HỆ SỐ THẬT (ROI) đã biết
```

| | |
|---|---|
| Tương quan TV ↔ Facebook | **0,9992** (`reports/correlation_matrix.csv`) |
| Chia | 80/20, `random_state=42` → 400 train / 100 test |
| Lựa chọn `alpha` | **5-fold CV trên train** (RidgeCV, chấm bằng MSE); tập test chỉ dùng để báo cáo |

---

## 3. KẾT QUẢ

### 3.1. Đa cộng tuyến — `reports/vif_table.csv`

| Kênh | VIF |
|---|---:|
| TV | **642,08** |
| Facebook | **642,07** |
| Google | 1,00 |

VIF 642 gấp 64 lần ngưỡng 10, nghĩa là 99,84% phương sai của TV được giải thích bởi Facebook và ngược lại.

### 3.2. Độ chính xác dự đoán (test 100 kỳ) — `reports/so_sanh_mo_hinh.csv`

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| **Baseline Dummy (mean)** | 651,78 | 533,67 | −0,0006 |
| Linear Regression | 383,57 | 312,91 | 0,6535 |
| **Ridge (alpha = 4,71)** ✅ | 383,16 | 312,10 | 0,6542 |
| Lasso (TT-13, LassoCV) | 383,30 | 312,69 | 0,6540 |
| ElasticNet (TT-14, ElasticNetCV) | 382,94 | 311,81 | 0,6546 |

* Mọi model tuyến tính giảm RMSE **41%** so với baseline và giải thích 65% biến động doanh thu. 35% còn
  lại là nhiễu (std 400 khi sinh dữ liệu).
* 4 model chênh nhau dưới 0,7 RMSE (0,2%), không phân biệt được trên 100 dòng test. Ridge **không đánh
  đổi** độ chính xác. Lasso/ElasticNet không hơn vì cả 3 kênh đều có tác động thật, không có biến nào để loại.

### 3.3. Chọn alpha bằng CV — `reports/rmse_theo_alpha.png/.csv`

| alpha | 0,001 (≈ OLS) | 1,15 | **4,71 (chọn)** | 19,3 | 79,1 | 324 | 1000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RMSE train | 405,46 | 405,46 | 405,49 | 405,78 | 409,93 | 446,18 | 534,51 |
| RMSE CV (5 fold) | 410,23 | 409,49 | **409,39** | 409,73 | 415,56 | 462,20 | 558,34 |

* Đường CV **rất phẳng** từ 0,001 tới ~20: alpha tốt nhất chỉ hơn OLS 0,84 RMSE (0,2%). Từ ~80 trở lên,
  cả train lẫn CV tăng vọt, tức là **underfit**. Train và CV chỉ chênh ~4–5, nên overfit không phải vấn
  đề ở đây. Vấn đề thật là **phương sai của từng hệ số**, thứ mà RMSE không nhìn thấy (3.4).
* Code `assert` cực tiểu của đường CV trùng alpha RidgeCV. RidgeCV được đặt `scoring="neg_mean_squared_error"`
  để cùng thước đo; mặc định của nó (R²) sẽ chọn 6,25.
* Bản trước tính thêm một "alpha tốt nhất trên **test**" (89) và ghi vào JSON. Bước này đã **xoá** vì là
  rò rỉ: hình và JSON giờ không dùng tập test.

### 3.4. ⭐ Độ ổn định hệ số: 100 lần lấy ngẫu nhiên 80% dữ liệu — `reports/bootstrap_he_so.png/.csv`

| Hệ số chuẩn hoá | std OLS | std Ridge | Giảm | Khoảng OLS | Khoảng Ridge | % lần ÂM (OLS / Ridge) |
|---|---:|---:|---:|---|---|---|
| TV | 200,9 | 13,8 | **93,2%** | 42,6 … 885,4 | 259,1 … 322,5 | 0% / 0% |
| Facebook | 199,7 | 12,8 | **93,6%** | **−331,0** … 513,4 | 236,4 … 298,8 | **39%** / 0% |
| Google | 8,2 | 8,1 | 1,1% | 191,9 … 235,1 | 190,0 … 232,4 | 0% / 0% |

Với OLS, chỉ cần đổi 20% dữ liệu là có **39%** khả năng mô hình báo "Facebook làm *giảm* doanh thu".
Ridge thu hẹp dao động ~93% và chỉ tác động lên 2 kênh đa cộng tuyến, còn Google gần như giữ nguyên.

### 3.5. ⭐ ROI thang gốc — hệ số có ĐÚNG không? — `reports/roi_kenh.csv`

ROI = doanh thu tăng thêm khi chi thêm **1 đơn vị**, bằng hệ số chuẩn hoá ÷ độ lệch chuẩn của kênh.
Khoảng 95% lấy từ 100 lần lấy mẫu con ở 3.4.

| ROI | **Thật** | OLS | Khoảng OLS | Ridge | Khoảng Ridge | Ridge chứa giá trị thật? |
|---|---:|---:|---|---:|---|---|
| TV | **3,20** | 2,75 | [0,67; 6,58] | 2,23 | [2,04; 2,40] | ❌ |
| Facebook | **1,80** | 2,76 | [−3,86; 6,19] | **3,59** | [3,09; 3,73] | ❌ |
| Google | 2,50 | 2,59 | [2,45; 2,82] | 2,56 | [2,43; 2,79] | ✅ |
| **Gói TV+Facebook** (62,5% TV, như lịch sử) | 2,68 | 2,75 | [2,61; 2,77] | 2,74 | [2,59; 2,75] | ✅ |
| Chênh lệch gói − Google | 0,18 | 0,17 | | 0,18 | **[−0,15; 0,27]** | ✅ |

* **Ridge "chắc chắn nhưng sai":** khoảng của Ridge cho TV và Facebook hẹp nhưng **không chứa giá trị
  thật**. Ridge xếp Facebook hiệu quả nhất, trong khi thật ra nó kém nhất.
* **Vì sao:** Ridge kéo hai hệ số *chuẩn hoá* về gần bằng nhau (coefficient path, 4.2). Facebook có độ
  lệch chuẩn nhỏ hơn TV (77 so với 129), nên cùng hệ số chuẩn hoá thì ROI thang gốc của Facebook lớn hơn
  ~1,67 lần. Ridge chia theo **thang đo**, không theo hiệu quả thật. Khi hai kênh tương quan 0,999, dữ
  liệu **không chứa thông tin** để tách chúng: OLS thừa nhận điều đó bằng khoảng rất rộng, còn Ridge che
  nó bằng một mặc định (chia đều).
* **Dữ liệu xác định tốt** ROI của **gói TV+Facebook** và của **Google**: cả OLS lẫn Ridge đều đúng ở mức này.
* ⚠️ Khoảng lấy từ lấy mẫu con 80% không hoàn lại nên hẹp hơn khoảng tin cậy thật (khoảng 2 lần). Kết
  luận vẫn đứng vững khi nhân đôi độ rộng.

### 3.6. ✍️ Đề xuất phân bổ ngân sách 2 tỷ VND/tháng — `reports/phan_bo_ngan_sach.csv`

| Phương án | TV | Facebook | Google | DT tăng thêm theo Ridge | **DT theo hệ số thật** |
|---|---|---|---|---:|---:|
| Cơ cấu hiện tại (trung bình lịch sử) | 912,7 tr (45,6%) | 547,0 tr (27,3%) | 540,3 tr (27,0%) | 5.381 | **5.256** |
| Cách cũ: tỉ lệ theo hệ số Ridge chuẩn hoá | 737,9 tr (36,9%) | 713,1 tr (35,7%) | 549,0 tr (27,5%) | **5.611** | 5.017 (**−4,5%**) |
| **✅ Đề xuất** | **912,7 tr (45,6%)** | **547,0 tr (27,3%)** | **540,3 tr (27,0%)** | 5.381 | 5.256 |

**Đề xuất cho CMO:**
1. **Giữ cơ cấu hiện tại.** Chênh lệch ROI giữa gói TV+Facebook và Google là 0,18, khoảng
   [−0,15; 0,27] **chứa 0**, nên chưa đủ bằng chứng để dịch tiền giữa hai nhóm. Quy tắc trong code: chỉ
   dịch tối đa 10 điểm %, và chỉ khi khoảng không chứa 0.
2. **Không dùng hệ số Ridge để chia tiền giữa TV và Facebook.** Cách cũ (bản trước của bài) dồn thêm
   ~166 triệu sang Facebook. Theo chính mô hình thì có vẻ tốt nhất (5.611), nhưng thực tế mất 239 (−4,5%).
   Mô hình tự đánh giá mình tốt ở đúng chỗ nó sai.
3. **Chạy geo-test / A/B** để tách TV và Facebook, ví dụ giảm 30% Facebook ở một nhóm vùng trong 4–8 tuần
   mà giữ TV. Chỉ có biến thiên **độc lập** giữa hai kênh mới cho ra ROI riêng.

---

## 4. VÌ SAO RIDGE ỔN ĐỊNH HƠN — VÀ GIỚI HẠN CỦA NÓ

### 4.1. Cơ chế
Với VIF = 642, XᵀX gần suy biến: OLS phải "chọn" cách chia phần phương sai chung giữa TV và Facebook
theo một tỉ lệ cực nhạy với nhiễu. Chỉ cần đổi 20% dữ liệu là tỉ lệ đảo ngược (Facebook từ −331 tới
+513). Ridge cộng λI vào XᵀX trước khi nghịch đảo: với cùng một tổng `w_TV + w_FB`, phạt
`w_TV² + w_FB²` nhỏ nhất khi **hai hệ số bằng nhau**. Vì vậy nghiệm bị "ghim" vào cách chia đều và
hết nhạy với nhiễu.

### 4.2. Coefficient path — `reports/coefficient_path.png`
* Ở alpha nhỏ, TV ≈ 350 và Facebook ≈ 210 (nghiệm OLS). Khi alpha tăng, hai đường **hội tụ** (≈ 280 ở
  alpha ≈ 10, trùng hẳn từ alpha ≈ 30–100), rồi cùng co về 0 ở alpha rất lớn.
* Google gần như phẳng (≈ 213) tới alpha ≈ 10–30 rồi mới co, vì nó không phải "chia phần" với biến nào.

### 4.3. Vì sao Ridge KHÔNG đưa hệ số về đúng 0
Đạo hàm của số hạng phạt `λw²` là `2λw`. Lực kéo về 0 **tỉ lệ với chính w**, nên khi w đã nhỏ thì lực
cũng gần 0 và không đủ đẩy w xuống đúng 0. Về hình học, đường đồng mức `Σwᵢ² = c` là hình tròn **trơn**,
không có góc trên trục toạ độ, nên điểm tiếp xúc với elip MSE hầu như không rơi đúng vào `wᵢ = 0`.
Lasso (`Σ|wᵢ|`, TT-13) có đạo hàm không đổi `λ·sign(w)` và đường đồng mức hình thoi có góc nhọn trên
trục, nên mới đưa hệ số về đúng 0. Trên coefficient path, không đường nào chạm 0 kể cả ở alpha = 10⁴.

### 4.4. Giới hạn
Ridge chữa **triệu chứng** (hệ số ổn định, không đổi dấu) nhưng **không tạo ra thông tin** mà dữ liệu
không có. Cách chia đều của Ridge chỉ là một giả định, không phải một phát hiện.

---

## 5. TIÊU CHÍ HOÀN THÀNH

```
   ☑ Có bảng VIF chứng minh đa cộng tuyến                 → 3.1: VIF TV/Facebook = 642
   ☑ ⭐ Có biểu đồ so sánh ĐỘ ỔN ĐỊNH hệ số (bootstrap)    → 3.4: std giảm 93%, Facebook âm 39% → 0%
   ☑ Có coefficient path                                  → 4.2
   ☑ Có đường RMSE theo alpha, alpha chọn bằng CV         → 3.3: alpha 4,71 (5-fold CV, không dùng test)
   ☑ Giải thích được vì sao Ridge KHÔNG đưa hệ số về 0    → 4.3
   ☑ ✍️ Có đề xuất phân bổ ngân sách bằng con số cụ thể   → 3.6: 912,7 / 547,0 / 540,3 triệu
   ☑ Nêu hạn chế: hệ số KHÔNG phải nhân quả               → mục 6
```

**Các bước đề yêu cầu:**
```
   ☑ 1. Dữ liệu có đa cộng tuyến (r = 0,9992)      ☑ 6. Coefficient path
   ☑ 2. Tương quan + VIF                           ☑ 7. RMSE train/CV theo alpha
   ☑ 3. Linear Regression + hệ số                  ☑ 8. So sánh RMSE test (+ baseline Dummy)
   ☑ 4. Bootstrap 100 lần × 80%                    ☑ 9. So sánh Lasso / ElasticNet
   ☑ 5. RidgeCV dò alpha                           ☑ 10. Đề xuất phân bổ ngân sách
```

## 6. HẠN CHẾ

1. **Không phải nhân quả.** Hệ số đo tương quan trên dữ liệu quan sát. Khẳng định "tăng TV *làm* tăng doanh
   thu" cần thí nghiệm (A/B, geo-test).
2. **Không tách được TV và Facebook** khi chúng tương quan 0,999. Ridge cho con số ổn định nhưng sai (3.5).
3. **ROI không đổi** (tuyến tính) chỉ đúng trong vùng chi tiêu đã quan sát. Mô hình bỏ qua **bão hoà** và
   **adstock** (hiệu ứng trễ). Ngân sách 2 tỷ lớn hơn tổng chi trung bình lịch sử (~600 triệu/kỳ), nên con
   số doanh thu tuyệt đối ở 3.6 là ngoại suy; chỉ nên đọc **so sánh tương đối** giữa các phương án.
4. **Khoảng tin cậy xấp xỉ** (lấy mẫu con 80%), hẹp hơn khoảng thật.
5. Dữ liệu tự sinh. Ưu điểm là biết đáp án để kiểm tra; nhược điểm là thực tế có nhiều kênh hơn, có mùa vụ
   và yếu tố ngoài quảng cáo.

## 7. SẢN PHẨM & CÁCH CHẠY

```
TT-12-Ridge/
├── README.md
├── notebooks/ridge_marketing.ipynb   ← gọi lại các hàm trong src/train.py, có nhận xét bằng số thật
├── src/train.py
├── models/ridge_pipeline.joblib      ← Pipeline(StandardScaler → RidgeCV), alpha = 4,71
├── reports/
│   ├── vif_table.csv, correlation_matrix.csv
│   ├── bootstrap_he_so.png/.csv, coefficient_path.png, rmse_theo_alpha.png/.csv
│   ├── so_sanh_mo_hinh.csv, roi_kenh.csv, phan_bo_ngan_sach.csv
│   └── tom_tat.json
└── requirements.txt
```

```bash
pip install -r requirements.txt
python src/train.py      # vài giây
```

**Tham khảo:** [Buổi 13 — Regularization](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
