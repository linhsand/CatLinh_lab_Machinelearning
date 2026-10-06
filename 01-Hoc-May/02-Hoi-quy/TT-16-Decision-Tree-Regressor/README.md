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

> **Kết quả chính (test 37.645 chuyến, chỉ dùng 1 lần để báo cáo):**
> * **Cây tổng đài** `max_depth=5, min_samples_leaf=500` cho ra **bảng tra 28 dòng**: **MAE $2,20**, **65,0%** chuyến
>   sai số trong ±15%. Baseline: Dummy MAE $11,08 (13,6%); công thức tay $3 + $3,5 × mile MAE $3,72 (33,6%).
> * Độ sâu tối ưu được chọn bằng **5-fold CV trên train** (không chọn trên test): **depth=11**, MAE-CV $1,90.
>   Random Forest chọn `max_depth` bằng OOB.
> * Cây sâu (MAE $1,89, 71,1%) và RF (MAE $1,75, 74,2%) **thắng không nhiều**: ở 83% số chuyến (< 5 mile) RF chỉ
>   sát hơn **$0,24–0,41/chuyến**, nhưng cần 1.223 lá hoặc ~1,19 triệu lá, nên không tra tay được.
> * Điểm yếu thật của cây d=5 là **chuyến > 20 mile** (1% số chuyến, MAE $13,67). Nên vá bằng luật đồng giá sân bay,
>   không cần model phức tạp hơn.

---

## 1. THUẬT TOÁN & BÀI TOÁN

Giống cây phân loại (TT-02) nhưng lá trả về **số trung bình** thay vì nhãn, và tiêu chí chia là **giảm MSE** thay vì
giảm Gini. Kết quả là hàm **bậc thang**: số mức giá khác nhau đúng bằng số lá.

**Bài toán:** hãng taxi cần **bảng giá ước tính** để tổng đài báo cho khách qua điện thoại trước khi điều xe.
① tổng đài viên phải **tra được bằng tay**; ② giá báo phải **giải thích được**; ③ sai số chấp nhận **±15%**.
Vì vậy sản phẩm là một cây **nông**, in ra thành bảng tra.

## 2. DỮ LIỆU & CÁCH ĐÁNH GIÁ

| | |
|---|---|
| **Nguồn** | [NYC Yellow Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), tháng 1/2023 (3.066.766 chuyến). `src/train.py` tự tải về `data/` |
| **Mẫu** | 200.000 dòng ngẫu nhiên (`random_state=42`) → **188.224** dòng sau làm sạch |
| **Nhãn** | `fare_amount` (USD), trung bình $18,47 |
| **Đặc trưng** | `trip_distance`, `passenger_count`, `PULocationID`, `DOLocationID`, `payment_type`, `hour`, `dow`, `is_peak` |
| **Chia** | 80/20 → 150.579 train / 37.645 test |
| **Chọn siêu tham số** | Cây: **5-fold CV trên train**. RF: **sai số OOB** trên train. Test chỉ dùng một lần ở bảng so sánh |

**Làm sạch** (loại 11.776 dòng, 5,9%):

| Bước | Số dòng loại |
|---|---:|
| `fare_amount ≤ 0` (huỷ / hoàn tiền) | 1.691 |
| `trip_distance ≤ 0` hoặc `> 100` mile | 2.790 |
| `passenger_count == 0` hoặc thiếu | 7.291 |
| Giờ đón ngoài tháng 1/2023 (lỗi nhập liệu) | 4 |

**Chống rò rỉ:** không đọc `tip_amount`, `tolls_amount`, `total_amount`, vì các cột này chỉ biết **sau** chuyến đi.
`is_peak` = thứ 2–6, 7–9h hoặc 16–19h (28,8% số chuyến).

**Metric:** MAE (đơn vị USD, dễ nói với khách), **% chuyến trong ±15%** (khớp yêu cầu ③), RMSE và R² là phụ.

---

## 3. KẾT QUẢ

### 3.1. EDA — `reports/eda_scatter_distance_fare.png`, `eda_avg_fare_by_hour.png`

* Cước gần tỷ lệ thuận với quãng đường, **cộng một đường ngang đậm đặc ở đúng $70**: giá đồng giá JFK ↔ Manhattan.
  Một công thức "hệ số × quãng đường" không biểu diễn được vùng này, còn cây thì tách được.
* Cước trung bình thấp nhất lúc 2h ($16,19), cao nhất lúc **5h ($27,16)** do nhiều chuyến ra sân bay sớm. "Giờ đông
  khách" và "giờ cước cao" là hai chuyện khác nhau.

### 3.2. Baseline & overfit

| Model (test) | MAE | % trong ±15% |
|---|---:|---:|
| Dummy (luôn đoán $18,47) | $11,08 | 13,6% |
| Công thức tay $3,00 + $3,50 × mile (không fit) | $3,72 | 33,6% |
| Cây **không giới hạn** độ sâu | $2,43 (train $0,006) | — |

Cây không giới hạn có **118.031 lá, sâu 47 tầng**: MAE test gấp ~400 lần MAE train. Nó học thuộc tập train, và
không thể in thành bảng tra.

### 3.3. Chọn độ sâu bằng CV trên train — `reports/mae_theo_depth.png/.csv`

| max_depth | 3 | 5 | 8 | 10 | **11** | 12 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Số lá | 8 | 31 | 217 | 702 | **1.223** | 2.084 | 8.858 | 43.263 |
| MAE train | 2,72 | 2,16 | 1,88 | 1,75 | **1,68** | 1,61 | 1,33 | 0,70 |
| MAE-CV (± std 5 fold) | 2,72 | 2,18 | 1,95 | 1,909 | **1,900 ± 0,020** | 1,905 | 1,98 | 2,20 |

* Nếu chọn depth theo **MAE test thấp nhất** rồi báo cáo MAE test của chính cây đó, con số sẽ lạc quan vì test đã
  được "nhìn" lúc chọn. Vì vậy depth được chọn bằng **5-fold CV trên train**: kết quả là **depth=11**. Test chỉ xuất
  hiện ở 3.5.
* **Đáy rất phẳng:** depth 10–13 chênh nhau ít hơn một độ lệch chuẩn giữa các fold. Theo quy tắc "1 độ lệch chuẩn",
  depth=10 (702 lá) cũng tốt ngang về mặt thống kê.
* Sau depth 11, MAE train vẫn giảm còn MAE-CV tăng lại: overfit.

### 3.4. ⭐ Cây tổng đài & bảng tra — `outputs/bang_tra_cuoc.csv`, `reports/cay_quyet_dinh_text.txt`, `cay_quyet_dinh.png`, `ham_bac_thang.png`

`DecisionTreeRegressor(max_depth=5, min_samples_leaf=500)`, nghĩa là mỗi mức giá dựa trên **≥ 500 chuyến thật**.

* **28 lá = 28 dòng bảng tra**, giá từ $7,28 đến $111,97. Lá nhỏ nhất có đúng 500 chuyến.
* **Gần như mọi nút đều chia theo `trip_distance`**, nên bảng tra thực chất là "bảng giá theo mile". Chỉ một nút chia
  theo `DOLocationID` (nhánh 16–22,6 mile): lá **$70,56** (5.792 chuyến) chính là vùng đồng giá sân bay ở 3.1.
* Hàm dự đoán theo quãng đường là **bậc thang**: 26 mức giá trên 400 điểm quét 0,1–20 mile. Có một lá lạ: chuyến
  ≤ 0,16 mile có giá **$19,25** (chuyến chờ lâu hoặc lỗi GPS), nên cần ghi chú khi bàn giao.
* Trích `export_text`, dạng luật để tổng đài đọc:

```
|--- trip_distance <= 8.22
|   |--- trip_distance <= 2.71
|   |   |--- trip_distance <= 1.50
|   |   |   |--- trip_distance <= 1.01
|   |   |   |   |--- trip_distance >  0.16  → $7.28
|   ...
|--- trip_distance >  8.22  ...  trip_distance > 16.02
|   |--- DOLocationID >  34  → $70.56   (vùng đồng giá sân bay)
|   |--- DOLocationID <= 34  → $82.02
```

**Kiểm tra ±15%:** **65,0%** chuyến test có giá báo lệch trong ±15%. MAE $2,20 nằm trong mức tham chiếu của đề
($2–3 với cây độ sâu 5–8).

### 3.5. So sánh model — `reports/so_sanh_models.csv`, `rf_chon_depth_oob.csv`

RF chọn `max_depth` ∈ {8, 12, 16, None} bằng MAE OOB: **16** thắng ($1,764). `None` kém hơn ($1,798).

| Model (test) | Siêu tham số chọn bằng | Số lá | MAE | RMSE | R² | % trong ±15% |
|---|---|---:|---:|---:|---:|---:|
| **Cây tổng đài** d=5, leaf ≥ 500 | nghiệp vụ (tra tay) | **28** | $2,20 | 5,04 | 0,912 | 65,0% |
| Cây sâu d=11 | CV trên train | 1.223 | $1,89 | 4,61 | 0,926 | 71,1% |
| Random Forest 100 cây, d=16 | OOB trên train | ~1,19 triệu | **$1,75** | **4,03** | **0,944** | **74,2%** |
| Linear Regression | — | 9 hệ số | $2,37 | 4,87 | 0,918 | 60,1% |

Linear Regression **thua cả cây d=5**: một đường thẳng không biểu diễn được vùng đồng giá $70 và các bậc giá.

### 3.6. ⚖️ Đánh đổi 1 — thêm tầng cây mua được bao nhiêu? — `reports/duong_bien_so_la_mae.png/.csv`

Giữ `min_samples_leaf=500`, quét depth bằng 5-fold CV trên train:

| depth | 3 | 4 | **5** | 6 | 8 | 12 |
|---|---:|---:|---:|---:|---:|---:|
| Số lá (dòng bảng tra) | 8 | 15 | **28** | 48 | 101 | 213 |
| MAE-CV | $2,72 | $2,32 | **$2,23** | $2,15 | $2,07 | $2,05 |
| % trong ±15% (CV) | 53,9 | 62,8 | **64,3** | 67,0 | 68,9 | 70,1 |

* **Lợi ích giảm rất nhanh.** depth 3 → 4 giảm $0,39; 4 → 5 giảm $0,09; 5 → 6 chỉ giảm $0,08 mà số dòng gần gấp đôi.
  Từ depth 8 trở đi, MAE gần như đứng yên dù số lá tăng gấp đôi.
* **Với leaf ≥ 500, đường biên chững ở ~$2,05.** Muốn xuống tới $1,90 như cây sâu thì phải chấp nhận các lá nhỏ (tổ
  hợp vùng đón/trả hiếm), đúng loại mức giá mà nghiệp vụ không muốn tin.
* **depth=5 là điểm cuối còn vừa một trang A4** (≤ ~40 dòng). depth=6 (48 dòng, +2,7 điểm %) là phương án dự phòng
  nếu tổng đài chấp nhận bảng hai trang.

### 3.7. ⚖️ Đánh đổi 2 — cây sâu / RF thắng ở ĐÂU? — `reports/chenh_lech_theo_quang_duong.png/.csv`

| Dải (mile) | Tỷ trọng | Cước TB | Cây d=5: MAE / % ±15 | Cây sâu d=11 | RF | Đóng góp vào chênh MAE với RF |
|---|---:|---:|---:|---:|---:|---:|
| 0–1 | 21,6% | $7,52 | $1,86 / 49% | $1,57 / 55% | $1,46 / 61% | 19% |
| 1–2 | 33,3% | $11,13 | $1,57 / 61% | $1,41 / 68% | $1,33 / 71% | 18% |
| 2–5 | 28,0% | $18,15 | $2,24 / 69% | $1,92 / 77% | $1,83 / 79% | 26% |
| 5–10 | 8,8% | $33,39 | $3,11 / 84% | $2,99 / 85% | $2,71 / 87% | 8% |
| 10–20 | 7,2% | $61,29 | $3,24 / 92% | $2,70 / 94% | $2,37 / 96% | 14% |
| 20–100 | 1,0% | $86,88 | **$13,67** / 71% | $8,44 / 85% | $6,68 / 90% | 16% |

1. **Thắng nhỏ về tiền.** Với 83% số chuyến (< 5 mile), RF chỉ sát hơn **$0,24–0,41/chuyến**, ít hơn một nấc đồng
   hồ taxi NYC ($0,70 cho mỗi 1/5 mile).
2. **Khoảng cách % ±15% trông lớn hơn tác động thật.** Ở chuyến 0–1 mile (cước $7,5), ±15% chỉ còn ±$1,1, nên sai
   $1,9 đã là trượt. Metric % rất khắt khe với chuyến ngắn, trong khi khách đi chuyến $7 hiếm khi khiếu nại vì chênh
   $1–2.
3. **Điểm yếu thật là chuyến > 20 mile.** Chỉ 1% số chuyến nhưng chiếm 16% khoảng chênh: mọi chuyến > 22,6 mile rơi
   vào một lá $111,97. Cách vá rẻ hơn là thêm hai luật vào đầu bảng tra: "JFK ↔ Manhattan: đồng giá $70" và
   "> 20 mile: báo giá theo đồng hồ". Không cần đổi model.
4. **Quy ra 1.000 cuộc gọi:** cây d=5 báo đúng ±15% cho ~650 khách, cây sâu ~711, RF ~742. RF thêm **~92 khách**,
   nhưng đổi lại:

   | | Cây d=5 | Cây sâu d=11 | Random Forest |
   |---|---|---|---|
   | ① Tra tay | ✅ 28 dòng | ❌ 1.223 dòng | ❌ cần phần mềm |
   | ② Giải thích cho khách | ✅ "đi 2–2,4 mile → $14,70" | ⚠️ luật dài 11 điều kiện | ❌ trung bình 100 cây |
   | Ổn định khi train lại | ✅ 28 lá ở cả 10 lần lấy mẫu lại | — | ✅ |

**Kết luận:** với bài toán *tổng đài tra tay*, giữ **cây depth=5** (cộng hai luật vá chuyến xa). RF chỉ đáng dùng nếu
bài toán đổi thành "hệ thống điều xe tự tính giá", khi đó yêu cầu ①② không còn và được thêm ~9 điểm %.

### 3.8. Mở rộng — `reports/so_sanh_criterion.csv`, `instability_10_cay.csv`, `khoang_gia_quantile.csv`

* **`criterion='absolute_error'`** (cùng depth=5, leaf ≥ 500): MAE **$2,03**, **68,9%** trong ±15% (+3,9 điểm, bảng
  vẫn 28 dòng). Đổi lại, huấn luyện chậm hơn ~10 lần (3,5 s so với 0,3 s). Đây là cải thiện rẻ nhất trong bài, vì
  MAE chính là metric nghiệp vụ.
* **Độ bất ổn:** 10 cây trên 10 mẫu con 80% **đều có đúng 28 lá**. Hệ số biến thiên của giá dự đoán là 0,1–2% ở
  2/5/10/20 mile, nhưng **10,7% ở 1 mile** (giá nhảy giữa ~$7,3 và ~$8,9 vì ngưỡng 1,01 mile bị dịch). Với leaf ≥ 500,
  cây nông ổn định hơn nhiều so với tiếng "không ổn định" của cây quyết định. Riêng các ngưỡng sát nhau vẫn dao động.
* **Khoảng giá** (`GradientBoostingRegressor(loss='quantile')`, phân vị 10–90%): 2 mile $11,6–17,1; 10 mile
  $38,9–49,4; ở 20 mile khoảng giá co về **$70,0** vì phần lớn chuyến ~20 mile là đồng giá sân bay.

### 3.9. ⚠️ Vì sao cây KHÔNG ngoại suy được — `reports/ngoai_suy_quang_duong.csv`

| Quãng đường | 60 mile | 81,5 (xa nhất đã thấy) | **100** | **124 (~200 km)** |
|---|---:|---:|---:|---:|
| Cây d=5 | $111,97 | $111,97 | **$111,97** | **$111,97** |
| Linear Regression | $227 | $306 | $375 | $463 |

Mỗi lá chỉ lưu **một con số** (trung bình các chuyến train rơi vào lá), không lưu công thức. Vì vậy mọi quãng đường
vượt ngưỡng chia cuối (22,6 mile) đều nhận cùng một giá. Chuyến 200 km sẽ được báo $111,97, ngang chuyến 23 mile.
Linear Regression thì vẫn kéo dài đường thẳng ra ngoài dải dữ liệu. Khi triển khai, bảng tra **phải ghi rõ phạm vi
áp dụng**, và chuyến xa thì báo giá thủ công.

---

## 4. TIÊU CHÍ HOÀN THÀNH

```
   ☑ Làm sạch đủ 4 bước, nêu số dòng loại bỏ           → mục 2: 11.776 dòng (5,9%)
   ☑ Không dùng cột rò rỉ (tip, tolls, total)          → mục 2
   ☑ Biểu đồ MAE train/validation theo độ sâu          → 3.3 (chọn bằng CV trên train, không dùng test)
   ☑ ⭐ Biểu đồ hàm bậc thang                          → 3.4: 26 mức giá
   ☑ ⭐ BẢNG TRA CƯỚC CSV                              → outputs/bang_tra_cuoc.csv, 28 dòng
   ☑ % chuyến đạt sai số ±15%                          → 65,0%
   ☑ Giải thích vì sao cây KHÔNG ngoại suy             → 3.9
   ☑ (thêm) Đánh đổi chính xác ↔ tra tay              → 3.6, 3.7
```

**Các bước đề yêu cầu:**
```
   ☑ 1. Lấy mẫu 200.000 dòng          ☑ 7. MAE theo max_depth 1..20 (CV)
   ☑ 2. Làm sạch 4 bước               ☑ 8. Hàm bậc thang
   ☑ 3. Đặc trưng thời gian           ☑ 9. Cây depth=5 + export_text + vẽ cây
   ☑ 4. EDA                           ☑ 10. Bảng tra cước CSV
   ☑ 5. Baseline Dummy + công thức    ☑ 11. % trong ±15%
   ☑ 6. Cây không giới hạn → overfit  ☑ 12. So sánh RF & Linear
   ☑ Mở rộng: absolute_error · độ bất ổn 10 cây · khoảng giá quantile
```

## 5. HẠN CHẾ

1. **Mã vùng được coi như số.** `PULocationID`/`DOLocationID` là mã danh mục, nên ngưỡng `DOLocationID <= 34` không
   có nghĩa địa lý, chỉ tình cờ tách được vùng sân bay. Nên thay bằng cờ `di_san_bay` (JFK = 132, LGA = 138, EWR = 1).
2. **% ±15% khắt khe với chuyến ngắn** (3.7), nên nếu dùng làm KPI cho tổng đài, nên kèm sai số tuyệt đối.
3. **Chỉ 1 tháng (1/2023).** Biểu giá NYC thay đổi theo năm và phụ phí theo mùa. Bảng tra cần huấn luyện lại định kỳ.
4. **Một lần chia ngẫu nhiên.** Chia theo thời gian (train tuần 1–3, test tuần 4) sẽ sát cách dùng thật hơn.
5. Cây d=5 **không ngoại suy** (3.9). Chỉ dùng trong dải ≤ ~22 mile, ngoài dải này phải báo giá thủ công.

## 6. SẢN PHẨM & CÁCH CHẠY

```
TT-16-Decision-Tree-Regressor/
├── README.md
├── data/yellow_tripdata_2023-01.parquet   ← tự tải từ NYC TLC khi chạy lần đầu
├── notebooks/tree_regressor_taxi.ipynb
├── src/train.py
├── models/tree_reg.joblib                 ← cây tổng đài depth=5, min_samples_leaf=500
├── outputs/bang_tra_cuoc.csv              ← ⭐ sản phẩm bàn giao cho tổng đài
├── reports/
│   ├── eda_scatter_distance_fare.png, eda_avg_fare_by_hour.png
│   ├── mae_theo_depth.png/.csv            ← CV trên train
│   ├── cay_quyet_dinh.png, cay_quyet_dinh_text.txt, ham_bac_thang.png
│   ├── so_sanh_models.csv, rf_chon_depth_oob.csv
│   ├── duong_bien_so_la_mae.png/.csv, chenh_lech_theo_quang_duong.png/.csv
│   ├── so_sanh_criterion.csv, instability_10_cay.csv, khoang_gia_quantile.csv
│   ├── ngoai_suy_quang_duong.csv
│   └── tom_tat.json
└── requirements.txt
```

```bash
pip install -r requirements.txt
python src/train.py
```

**Tham khảo:** [Buổi 3 — Tree](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-03-Feature-Eng-Tree/Tai-Lieu) · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
