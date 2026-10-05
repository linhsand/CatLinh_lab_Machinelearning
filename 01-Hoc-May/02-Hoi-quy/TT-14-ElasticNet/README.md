# TT-14 — ELASTICNET
## Dự báo tải sưởi / làm mát toà nhà khi các biến thiết kế dính chặt nhau

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao) |
| 🧠 **Nhóm** | Hồi quy có regularization kết hợp |
| 🔧 **Thuật toán** | ElasticNet (L1 + L2) |
| 🏭 **Lĩnh vực** | Năng lượng · Thiết kế xây dựng |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐ |

> **Kết quả chính:**
> * ElasticNet dự báo tải sưởi Y1 với **R² test 0,921** (RMSE 2,88 kWh/m²) và tải làm mát Y2 với **R² 0,895**
>   (RMSE 3,12), so với baseline Dummy R² ≈ 0 (RMSE 10,24 / 9,67).
> * Ridge, Lasso và ElasticNet ngang nhau vì n = 614 ≫ p = 14, nên CV chọn mức phạt gần 0. Hiệu ứng gom nhóm
>   chỉ lộ ra khi tăng alpha: tại alpha = 0,0104, Lasso bỏ 1/4 biến của nhóm hình học, ElasticNet giữ đủ 4/4.
> * **Khuyến nghị thiết kế** (mỗi toà 220,5 m² sàn): **1 tầng thay vì 2 tầng** giảm tổng tải **35 kWh/m²**
>   (Y1 −57%, Y2 −51%, ~6,4 triệu đ/năm). Kế đến là chọn đúng hình khối (10–25 kWh/m²), rồi đến giảm kính
>   40% → 10% (9 kWh/m²). Hướng nhà và cách phân bố kính **gần như không ảnh hưởng** (< 1 kWh/m²), nên kiến
>   trúc sư được tự do chọn.

---

## 1. THUẬT TOÁN & BÀI TOÁN

```
   Loss = MSE + λ·[ ρ·Σ|wᵢ|  +  (1−ρ)/2·Σwᵢ² ]       ρ = l1_ratio:  1 → Lasso,  0 → Ridge
```

ElasticNet vá điểm yếu của Lasso: thành phần L2 kéo hệ số các biến tương quan lại gần nhau (**hiệu ứng gom
nhóm**) thay vì để L1 bỏ hẳn một biến.

**Bài toán:** công ty thiết kế cần ước tính **tải sưởi (Y1)** và **tải làm mát (Y2)** ngay từ bản vẽ, để
① chọn công suất điều hoà đúng (thừa thì lãng phí, thiếu thì phải cải tạo) và ② biết **quyết định thiết kế
nào** đáng tối ưu nhất.

## 2. DỮ LIỆU

| | |
|---|---|
| **Nguồn** | [Energy Efficiency (UCI)](https://archive.ics.uci.edu/dataset/242/energy+efficiency). `src/train.py` tự tải và cache vào `data/` |
| **Kích thước** | 768 toà nhà mô phỏng (Ecotect) × 8 đặc trưng, 2 nhãn Y1, Y2 (kWh/m²) |
| **Chia** | 80/20, `random_state=42` → 614 train / 154 test, làm **riêng cho Y1 và Y2** |
| **Chọn siêu tham số** | 5-fold CV trên train (RidgeCV / LassoCV / ElasticNetCV với lưới 100 alpha × 7 l1_ratio); test chỉ để báo cáo |

**Đặc điểm cấu trúc, quyết định cách đọc kết quả:**

| Phát hiện | Bằng chứng | Hệ quả |
|---|---|---|
| Đây là **thí nghiệm giai thừa cân bằng** | 12 hình khối × 4 hướng (X6) × 16 tổ hợp kính (X7, X8) = 768 | So sánh trung bình theo mức = ước lượng sạch tác động từng quyết định (mục 4) |
| Đa cộng tuyến **hoàn hảo** trong nhóm hình học | `X2 = X3 + 2·X4` đúng tuyệt đối → VIF X2, X3, X4 = **∞**; X1 = 105,5; X5 = 31,2; X7 = 1,0. \|r\| > 0,8 giữa mọi cặp trong {X1, X2, X4, X5} | Không đọc riêng hệ số từng biến hình học |
| **X8 = 0 ⇔ X7 = 0** (không kính) | Kiểm tra trên cả 768 dòng | Cột one-hot `X8_1…X8_5` đo "**có kính**", không phải "phân bố kính" |
| X6, X8 là biến phân loại | Mã hoá 2–5 / 0–5 | One-hot (`drop_first`) → 14 cột |

---

## 3. KẾT QUẢ MÔ HÌNH

### 3.1. So sánh model (test 154 toà) — `reports/so_sanh_3_model_Y1.csv`, `_Y2.csv`

| Model | alpha | l1_ratio | Biến giữ | **Y1** RMSE | **Y1** R² | **Y2** RMSE | **Y2** R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline Dummy (mean) | — | — | — | 10,238 | −0,006 | 9,666 | −0,008 |
| Linear Regression | — | — | 14/14 | 2,872 | 0,9209 | 3,120 | 0,8950 |
| Ridge (RidgeCV) | 0,272 / 0,192 | — | 14/14 | 2,876 | 0,9207 | 3,121 | 0,8949 |
| Lasso (LassoCV) | 0,00129 / 0,00072 | 1,0 | 14/14 | 2,875 | 0,9207 | 3,120 | 0,8949 |
| **ElasticNet (ElasticNetCV)** ✅ | 0,00045 / 0,00032 | 0,1 / 0,1 | 14/14 | 2,876 | 0,9207 | 3,121 | 0,8949 |

* Mọi model tuyến tính giảm RMSE **72% (Y1) / 68% (Y2)** so với Dummy. Sai số điển hình ~2,9–3,1 kWh/m²
  trên tải trung bình 22,3 / 24,6 kWh/m².
* 4 model chênh nhau dưới 0,005 RMSE. Với 614 mẫu và 14 biến, model không overfit, nên CV chọn alpha gần 0
  và cả ba phương pháp hội tụ về OLS. Y2 khó dự đoán hơn Y1 một chút.

### 3.2. Heatmap RMSE **cross-validation** theo alpha × l1_ratio — `reports/heatmap_alpha_l1ratio_Y*.png`, `heatmap_cv_Y*.csv`

Lấy thẳng từ `mse_path_` của ElasticNetCV (5 fold trên train). **Không dùng tập test.** Code `assert` ô tốt
nhất trùng lựa chọn của CV.

| | Ô tốt nhất (RMSE-CV) | alpha ≈ 0,1 | alpha = 10, l1 = 0,1 | alpha = 10, l1 = 1,0 (Lasso) |
|---|---|---|---|---|
| Y1 | **2,829** (alpha 0,00045, l1 0,1) | 3,01–3,15 | 8,28 | **10,05** (≈ Dummy) |
| Y2 | **3,232** (alpha 0,00032, l1 0,1) | 3,32–3,51 | 7,83 | 9,47 |

Ở mức phạt lớn, Lasso thuần (l1 = 1) tệ nhất vì loại sạch cả nhóm hình học, còn l1 nhỏ (gần Ridge) giữ được
tín hiệu. Bản trước vẽ heatmap này trên **test** và đánh dấu "ô tốt nhất trên test". Đó là cùng lỗi chọn trên
test đã sửa ở TT-12/13, nên đã được thay.

### 3.3. ⭐ Hiệu ứng gom nhóm — `reports/hieu_ung_gom_nhom_theo_alpha_Y*.csv/.png`

Ở alpha do CV chọn, cả Lasso và ElasticNet đều giữ 4/4 biến {X1, X2, X4, X5}. Khi ép alpha tăng:

| Nhãn | alpha Lasso bắt đầu bỏ biến trong nhóm | Lasso giữ (nhóm / tổng) | ElasticNet (l1 = 0,5) giữ |
|---|---:|---|---|
| Y1 | 0,0104 | **3/4** (11/14) | **4/4** (14/14) |
| Y2 | 0,0227 | **3/4** (12/14) | **4/4** (14/14) |

Lasso giảm dần xuống 1/4 biến trong nhóm ở alpha ≈ 0,1–0,24, trong khi ElasticNet vẫn giữ 3–4/4. Đây là bằng
chứng thực nghiệm của hiệu ứng gom nhóm: với X2 tương quan −0,992 với X1, Lasso dồn "trách nhiệm" cho một biến,
còn ElasticNet chia sẻ.

### 3.4. Ổn định hệ số: 100 lần lấy mẫu 80% — `reports/bootstrap_elasticnet_Y*.csv/.png`

| | Biến hình học X1, X2, X4, X5 | X7 (kính) | X8_* (có kính) | X3 | X6_* (hướng) |
|---|---|---|---|---|---|
| Hệ số biến thiên (Y1) | 3,8–5,7% | 2,5% | 6,7–7,8% | 9,6% | **176–593%** |
| Hệ số biến thiên (Y2) | 4,4–7,4% | 3,7% | 13,6–19,1% | 63% | **48–190%** |

Hệ số các biến thiết kế chính rất ổn định. Hệ số hướng nhà dao động quanh 0, nghĩa là hướng nhà **không có ảnh
hưởng đáng tin cậy**. Điều này khớp với mục 4.

### 3.5. Mô hình tuyến tính sai ở đâu? — `reports/phan_du_theo_nhom_Y*.csv`

Phần dư trung bình (thực tế − dự báo) trên test, theo chiều cao × diện tích kính, đơn vị kWh/m²:

| Y1 | Không kính | 10% | 25% | 40% |
|---|---:|---:|---:|---:|
| 1 tầng (3,5 m) | +1,80 | +0,89 | −0,17 | −0,80 |
| 2 tầng (7 m) | **−3,49** | +0,68 | −0,38 | +1,13 |

* Mô hình cộng tính giả định "kính thêm một lượng tải như nhau cho mọi nhà". Thực tế, kính 0 → 40% làm tăng Y1
  **+14,3** ở nhà 2 tầng nhưng chỉ **+8,0** ở nhà 1 tầng (`reports/tuong_tac_chieu_cao_x_kinh.csv`). Kết quả
  là model dự báo **thừa 3,5 kWh/m²** cho nhà 2 tầng không kính (Y2: −2,09).
* R² 0,92 che mất sai lệch có hệ thống này. Khi dùng để **định cỡ điều hoà**, nên thêm tương tác `X5 × X7`
  (TT-15) hoặc dùng model cây (TT-16/17).

### 3.6. ⚠️ Vì sao KHÔNG dùng bảng |hệ số| để ra quyết định thiết kế

Top hệ số ElasticNet (chuẩn hoá): X5 +7,32 / +7,21 · X1 −6,22 / −7,18 · X4 −3,64 / −3,83 · X2 −3,48 / −3,99 ·
X7 +2,31 / +1,81 (Y1 / Y2).
* **X1 âm** đọc ra là "nhà gọn hơn → tải thấp hơn". Nhưng trong dữ liệu, mọi nhà gọn (X1 ≥ 0,76) đều là nhà
  2 tầng và có tải sưởi trung bình **cao gấp 2,3 lần** (31,3 so với 13,3 kWh/m²). Hệ số X1 chỉ có nghĩa khi "giữ nguyên X2, X4, X5", một điều bất khả
  thi về mặt hình học. Bản trước của bài khuyến nghị "tăng độ gọn X1", trái với dữ liệu.
* Nhóm **X8_\*** (≈ 1,6 cho Y1) bị xếp hạng 6–10 và bị đọc thành "phân bố kính quan trọng". Thực ra nó đo
  **có kính hay không**. Phân bố kính 1–5 chỉ chênh **0,9 kWh/m²**.

→ Căn cứ đúng để ra quyết định là **so sánh trung bình theo mức** trên thiết kế giai thừa (mục 4).

---

## 4. ⭐ ĐÒN BẨY THIẾT KẾ — `reports/don_bay_thiet_ke.csv/.png`

Vì dữ liệu là thiết kế giai thừa **cân bằng**, ở hai bên của mỗi so sánh các quyết định khác được phân bố như
nhau. Hiệu số trung bình vì vậy là tác động của riêng quyết định đó. Nhóm hình học chỉ có 12 hình khối nên
so sánh ở mức **hình khối**.

**Quy đổi (giả định):** mọi toà có cùng thể tích 771,75 m³, tức **220,5 m² sàn**. Coi Y là tải năm. Điều hoà /
bơm nhiệt **COP = 3**. Giá điện **2.500đ/kWh**. Thứ tự các đòn bẩy **không** phụ thuộc giả định này.

| # | Quyết định | Tải sưởi Y1 | Tải làm mát Y2 | Giảm tổng tải | Điện/toà/năm | ≈ Tiền/toà/năm |
|---|---|---|---|---:|---:|---:|
| 1 | **1 tầng (3,5 m) thay vì 2 tầng (7 m)**, cùng thể tích | 31,28 → 13,34 (**−57%**) | 33,10 → 16,07 (**−51%**) | **35,0 kWh/m²** | ~2.570 kWh | **~6,4 triệu đ** |
| 2a | Hình khối tốt nhất trong nhóm **2 tầng** (X1 0,79 → 0,82) | 38,61 → 25,56 (−34%) | 40,24 → 28,03 (−30%) | 25,3 | ~1.860 kWh | ~4,6 triệu đ |
| 2b | Hình khối tốt nhất trong nhóm **1 tầng** (X1 0,64 → 0,74) | 16,62 → 11,89 (−28%) | 20,23 → 14,81 (−27%) | 10,2 | ~750 kWh | ~1,9 triệu đ |
| 3 | **Kính 40% → 10%** diện tích sàn | 25,41 → 20,36 (−20%) | 26,91 → 22,94 (−15%) | 9,0 | ~660 kWh | ~1,7 triệu đ |
| | (40% → 25%; 25% → 10%) | | | (4,7; 4,4) | | (0,86; 0,80 triệu đ) |
| 4 | Phân bố kính tốt nhất (X8 1 → 3) | 23,03 → 22,68 | 25,18 → 24,66 | 0,9 | ~60 kWh | ~0,16 triệu đ |
| 5 | Hướng nhà tốt nhất (X6 5 → 3) | 22,28 → 22,38 | 24,95 → 24,31 | 0,5 | ~40 kWh | ~0,10 triệu đ |

**Khuyến nghị cho công ty thiết kế:**
1. **Quyết định ở giai đoạn ý tưởng (chiều cao, hình khối) chiếm gần hết đòn bẩy.** Chuyển 2 tầng xuống 1 tầng
   có tác dụng gấp **~4 lần** so với cắt kính 40% → 10%. Đổi lại, nhà 1 tầng cần gấp đôi diện tích đất (mái
   220,5 m² so với 110–147 m²). Đây là đánh đổi **năng lượng vs giá đất**, cần đưa cho chủ đầu tư quyết định
   bằng con số: ~6,4 triệu đ/năm, tức ~130 triệu đ trong vòng đời 20 năm (chưa chiết khấu).
2. **Đã chốt chiều cao thì mô phỏng từng hình khối**, đừng suy từ X1. Trong nhóm 2 tầng, hai hình khối có X1
   gần nhau (0,79 và 0,82) mà đã chênh **25 kWh/m²**.
3. **Kính: giảm bước nào cũng có lợi** (mỗi bước ~4,5 kWh/m²). Tác động của kính ở nhà 2 tầng mạnh **gần gấp
   đôi** nhà 1 tầng (+14,3 so với +8,0 khi tăng 0 → 40%), nên nhà cao cần hạn chế kính chặt hơn.
4. **Hướng nhà và cách phân bố kính: tự do.** Chênh lệch dưới 1 kWh/m² (< 2% tổng tải trung bình 46,9). Kiến
   trúc sư chọn theo view, thẩm mỹ, quy hoạch mà không phải trả giá năng lượng đáng kể.
5. **Định cỡ điều hoà:** dùng model với sai số ±3 kWh/m² (RMSE). Nhà 2 tầng nhiều kính có thể bị dự báo thiếu
   ~1 kWh/m², nên cộng biên an toàn ~5% cho nhóm này (mục 3.5).

---

## 5. TIÊU CHÍ HOÀN THÀNH

```
   ☑ Ma trận tương quan + VIF chứng minh đa cộng tuyến     → mục 2: VIF ∞ (X2, X3, X4), 105,5 (X1)
   ☑ One-hot X6, X8                                       → 14 cột
   ☑ Baseline Dummy + Linear                              → 3.1
   ☑ Ridge / Lasso / ElasticNet trên cùng dữ liệu, CV     → 3.1
   ☑ Bảng so sánh số biến giữ + RMSE                      → 3.1
   ☑ ⭐ Kiểm chứng hiệu ứng gom nhóm                       → 3.3: Lasso 3/4 vs ElasticNet 4/4
   ☑ Heatmap theo lưới alpha × l1_ratio                   → 3.2 (cross-validation, không dùng test)
   ☑ Làm cả Y1 và Y2                                      → toàn bài
   ☑ Bootstrap ổn định hệ số                              → 3.4
   ☑ ✍️ Đề xuất thay đổi thiết kế bằng con số              → mục 4
```

## 6. HẠN CHẾ

1. **Dữ liệu mô phỏng**: 12 hình khối, một vùng khí hậu (Athens, Hy Lạp), vật liệu cố định. Thứ tự đòn bẩy đáng
   tin, con số tuyệt đối cần mô phỏng lại cho khí hậu Việt Nam.
2. **Quy đổi tiền** dựa trên giả định COP 3 và giá điện 2.500đ/kWh, và coi Y là tải năm.
3. **Mô hình tuyến tính cộng tính** bỏ qua tương tác chiều cao × kính và phi tuyến của kính. R² 0,92 không có
   nghĩa là đúng đều ở mọi nhóm (3.5).
4. **Không tách được tác động riêng** của X1, X2, X3, X4, X5 (đa cộng tuyến hoàn hảo). Chỉ so sánh được giữa các
   hình khối có sẵn.
5. **Một lần chia train/test**. Siêu tham số chọn bằng CV, nhưng con số test đến từ một lần chia 154 toà.

## 7. SẢN PHẨM & CÁCH CHẠY

```
TT-14-ElasticNet/
├── README.md
├── data/ENB2012_data.xlsx                 ← tự tải từ UCI khi chạy lần đầu
├── notebooks/elasticnet_energy.ipynb      ← giải thích từng bước + output thật
├── src/train.py
├── models/elasticnet_Y1.joblib, elasticnet_Y2.joblib
├── reports/
│   ├── vif_table.csv, correlation_matrix.csv
│   ├── so_sanh_3_model_Y*.csv/.png, heatmap_alpha_l1ratio_Y*.png, heatmap_cv_Y*.csv
│   ├── hieu_ung_gom_nhom_theo_alpha_Y*.csv/.png, bootstrap_elasticnet_Y*.csv/.png
│   ├── phan_du_theo_nhom_Y*.csv, tuong_tac_chieu_cao_x_kinh.csv
│   ├── don_bay_thiet_ke.csv/.png          ← khuyến nghị thiết kế
│   └── tom_tat.json
└── requirements.txt
```

```bash
pip install -r requirements.txt
python src/train.py
```

**Tham khảo:** [Buổi 13 — Regularization](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
