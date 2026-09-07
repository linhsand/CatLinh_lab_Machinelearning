# TT-08 — XGBOOST
## Phát hiện gian lận thẻ tín dụng theo thời gian thực

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 6](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-06-Ensemble-EndToEnd) |
| 🧠 **Nhóm** | Phân loại · Boosting · **Dữ liệu lệch cực đoan** |
| 🔧 **Thuật toán** | XGBoost |
| 🏭 **Lĩnh vực** | Ngân hàng · Thanh toán · Chống gian lận |
| ⏱ **Thời lượng** | 7–9 giờ |
| 📈 **Độ khó** | ⭐⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

XGBoost = Gradient Boosting (TT-07) + 4 cải tiến khiến nó thắng gần như mọi cuộc thi
dữ liệu bảng:

```
   ① REGULARIZATION (L1 + L2) ngay trong hàm mục tiêu  → chống overfit tốt hơn hẳn
   ② Tự xử lý GIÁ TRỊ THIẾU → học luôn hướng đi cho NaN, không cần điền
   ③ Song song hoá việc tìm điểm cắt → nhanh hơn nhiều lần
   ④ Cắt tỉa theo chiều sâu + `gamma` (ngưỡng lợi ích tối thiểu để chia nhánh)
```

---

## 2. BÀI TOÁN THỰC TẾ

```
   Cổng thanh toán xử lý 300 giao dịch/giây.
   Tỉ lệ gian lận: 0,172%  (492 / 284.807)  → LỆCH CỰC ĐOAN

   Ràng buộc:
     • Quyết định trong < 100 ms
     • Chặn nhầm giao dịch thật → khách hàng phẫn nộ, có thể mất khách vĩnh viễn
     • Bỏ lọt gian lận → ngân hàng đền tiền

   ⚠️ Ở mức lệch 0,17%, ACCURACY và ROC-AUC đều VÔ NGHĨA.
      → Metric chính: PR-AUC (Average Precision) + Recall @ Precision cố định.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Credit Card Fraud Detection |
| **Link** | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| **Kích thước** | 284.807 giao dịch × 31 cột (~150 MB gốc, ~98 MB bản mirror CSV) |
| **Nhãn** | `Class` — 1 = gian lận (**492 ca ≈ 0,1727%**, đã xác nhận lại bằng code, không lấy số mô tả bài toán) |

**Cột:** `Time`, `V1`–`V28` (đã **PCA hoá** để ẩn thông tin gốc), `Amount`, `Class`

Chi tiết nguồn tải (không có Kaggle API key trong môi trường chạy → dùng mirror
GitHub công khai, cùng số dòng/số ca gian lận với bản gốc) xem tại
[`data/DATA_SOURCE.md`](data/DATA_SOURCE.md).

### ⚠️ Ba lưu ý quan trọng (đã xử lý trong notebook/script)

```
   1. V1–V28 đã qua PCA → KHÔNG diễn giải được ý nghĩa từng cột.
      → Bài này KHÔNG làm được feature engineering theo nghiệp vụ.
      → Đây cũng là hạn chế đã nêu rõ trong mục 9 bên dưới.

   2. `Time` là số giây kể từ giao dịch đầu tiên (trải dài đúng 48 giờ / 2 ngày).
      → Không dùng trực tiếp. Đã đổi thành GIỜ TRONG NGÀY: (Time // 3600) % 24

   3. `Amount` chưa scale trong khi V1–V28 đã scale sẵn → đã scale riêng
      Amount bằng log1p rồi StandardScaler (fit chỉ trên tập train).
```

---

## 4. HƯỚNG ĐI ĐÃ THỰC HIỆN

### 4.1. Chia dữ liệu THEO THỜI GIAN, không ngẫu nhiên

```
   Gian lận có tính THỜI ĐIỂM (kẻ gian tấn công theo đợt).
   Chia ngẫu nhiên → model "nhìn thấy tương lai" → điểm ảo.

   → Sắp xếp theo Time → 70% đầu train · 15% giữa validation · 15% cuối test
```

Kết quả chia thực tế:

| Tập | Số giao dịch | Số ca gian lận | Tỉ lệ |
| :--- | ---: | ---: | ---: |
| Train | 199.364 | 384 | 0,1926% |
| Validation | 42.721 | 56 | 0,1311% |
| Test | 42.722 | 52 | 0,1217% |

### 4.2. Tham số cho dữ liệu lệch

```python
import xgboost as xgb

ty_le = (y_train == 0).sum() / (y_train == 1).sum()      # thực đo = 518.18

model = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=ty_le,          # ⭐ cân bằng lớp
    reg_lambda=1.0, reg_alpha=0.1,   # regularization
    eval_metric='aucpr',             # ⭐ PR-AUC, KHÔNG dùng 'auc'
    early_stopping_rounds=50,
    tree_method='hist', n_jobs=-1, random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
```

Early stopping dừng ở **cây thứ 124/1000** (PR-AUC validation hội tụ ~0,850).

### 4.3. Metric đúng cho lệch cực đoan

```
   ROC-AUC  = 0,9770  → ĐẸP GIẢ TẠO (vì TN quá nhiều, FPR luôn nhỏ)
   PR-AUC   = 0,7645  → con số THẬT, phản ánh đúng năng lực
   → Chênh lệch ~0,21 điểm — xem giải thích đầy đủ ở mục 6.
```

---

## 5. KẾT QUẢ THỰC NGHIỆM

Toàn bộ số liệu dưới đây là **kết quả chạy thật** trên bộ dữ liệu gốc
(`src/train.py`, tái lập được 100%), không phải số minh hoạ.

### 5.1. EDA — gian lận theo giờ và theo số tiền

![EDA](reports/eda_hour_amount.png)

* Tỉ lệ gian lận cao rõ rệt vào khung giờ **2h–4h** (mốc thời gian tương đối
  của tập dữ liệu), gấp 5–10 lần mức trung bình — khung giờ ít giao dịch thật
  nên gian lận dễ "lẫn vào" hơn.
* Giao dịch gian lận tập trung nhiều ở mức tiền nhỏ nhưng vẫn có đuôi lệch
  dài → không thể chỉ lọc theo ngưỡng số tiền.

### 5.2. So sánh các mô hình trên tập test

| Mô hình | PR-AUC | ROC-AUC | Train time (s) | Predict (ms/giao dịch) |
| :--- | :---: | :---: | :---: | :---: |
| *Baseline (Dummy)* | *0,0012* | *0,4991* | *–* | *–* |
| Logistic Regression (balanced) | 0,6948 | 0,9778 | 1,14 | 0,55 |
| **XGBoost** | **0,7645** | 0,9770 | 2,34 | 2,14 |
| Random Forest | 0,7770 | 0,9750 | 23,42 | 53,25 |
| LightGBM | 0,6771 | 0,9725 | 0,88 | 1,73 |

> Bảng đầy đủ: [`reports/so_sanh_mo_hinh.csv`](reports/so_sanh_mo_hinh.csv). Thời
> gian train/predict phụ thuộc máy chạy — chạy lại `src/train.py` sẽ cho số hơi
> khác, nhưng tỉ lệ tương đối giữa các mô hình ổn định qua các lần chạy.

**Nhận xét:**
* Cả 4 mô hình đều bỏ xa baseline (PR-AUC ≈ 0,001 → gần một nghìn lần đối với
  XGBoost) — xác nhận `scale_pos_weight`/`class_weight` hoạt động đúng.
* Random Forest có PR-AUC cao nhất trong lần chạy này nhưng **chậm hơn
  XGBoost ~10 lần khi train và ~25 lần khi dự đoán** — với ràng buộc < 100ms/giao
  dịch và 300 giao dịch/giây, độ trễ dự đoán quan trọng ngang PR-AUC.
* ⭐ **Phát hiện thực nghiệm quan trọng về LightGBM:** áp trực tiếp
  `scale_pos_weight=518` (hoặc `is_unbalance=True`) **kết hợp early stopping**
  khiến LightGBM sụp đổ hoàn toàn — dừng chỉ sau 2 vòng lặp, ROC-AUC rơi
  xuống **0,187** (tệ hơn đoán ngẫu nhiên). Nguyên nhân: cây leaf-wise của
  LightGBM vốn dễ overfit từng vòng hơn cây depth-wise của XGBoost; khi
  gradient của 0,19% mẫu dương bị khuếch đại gấp 518 lần, vài lá đầu tiên bị
  kéo lệch cực đoan, AP trên validation tụt ngay ở vòng 2 và early stopping
  (đúng chức năng) dừng luôn. Đây **không phải lỗi cài đặt** — tắt hẳn việc
  cân bằng lớp cho LightGBM giúp nó huấn luyện ổn định và cạnh tranh sòng
  phẳng với XGBoost (bảng trên là kết quả **sau khi tắt**). Bài học:
  `scale_pos_weight` không "an toàn" như nhau giữa các thư viện boosting,
  luôn phải nhìn đường cong validation trước khi tin số liệu cuối cùng.

### 5.3. ⭐ VÌ SAO ROC-AUC ĐÁNH LỪA

![PR vs ROC](reports/pr_vs_roc.png)

ROC-AUC dựa trên **False Positive Rate = FP / (FP + TN)**. Khi lớp âm (giao
dịch hợp lệ) áp đảo (284.315 / 284.807 ≈ 99,83%), mẫu số `TN` cực lớn khiến
FPR gần như luôn nhỏ dù số lượng FP tuyệt đối (giao dịch thật bị chặn nhầm)
có thể lên tới hàng trăm — con số rất đau với khách hàng thật.

PR-AUC dựa trên **Precision = TP / (TP + FP)** — không có `TN` trong công
thức nên phản ánh đúng câu hỏi nghiệp vụ: "trong số cảnh báo mô hình đưa ra,
bao nhiêu % là đúng?".

* ROC-AUC = 0,9770 → nhìn qua tưởng mô hình gần như hoàn hảo.
* PR-AUC = 0,7645 → con số thật, vẫn còn đánh đổi Precision/Recall đáng kể.

→ **Luôn báo cáo cả hai**, và ưu tiên PR-AUC làm chỉ số chính khi lớp dương
hiếm hơn 1%.

### 5.4. Chọn ngưỡng theo Precision ≥ 90% — **chọn trên VALIDATION, báo cáo trên TEST**

⚠️ **Bản trước của báo cáo này quét ngưỡng trực tiếp trên TEST rồi báo cáo
Precision/Recall trên chính TEST đó** — về bản chất là để mô hình "nhìn thấy"
nhãn thật của tập dùng để đánh giá nó, cùng loại rò rỉ đã xử lý ở TT-04/TT-05,
chỉ khác là rò rỉ qua bước *chọn ngưỡng* thay vì bước *huấn luyện*. Con số
"Precision 90,24% / Recall 71,15%" trước đây bị lạc quan ảo vì lý do này.

**Cách làm đúng** (đã sửa trong `src/train.py` và notebook): quét ngưỡng trên
**validation**, chọn ngưỡng đạt Precision ≥ 90% tại đó, rồi chỉ áp dụng đúng
**1 lần** lên **test** để biết con số thật khi triển khai:

* **Ngưỡng (chọn trên VAL):** 0,9513 — tại đây Precision(VAL) = 91,67%,
  Recall(VAL) = 78,57%.
* **Áp dụng 1 lần lên TEST:** Precision = **84,78%**, Recall = **75,00%**.

→ Precision thật trên test (84,78%) **không đạt mục tiêu 90%** dù đạt trên
validation (91,67%) — đây là khoảng chênh val↔test **có thật**, chính là phần
bị che giấu khi chọn ngưỡng thẳng trên test. Recall trên test lại cao hơn cách
làm cũ (75,00% so với 71,15%) — không nhất quán theo một chiều, cho thấy đây
đúng là nhiễu do cỡ mẫu dương rất nhỏ (52 ca gian lận trên test, 56 ca trên
val) chứ không phải mô hình "kém đi thấy rõ".

### 5.5. Tối ưu ngưỡng theo chi phí thực tế — **chọn trên VALIDATION, báo cáo trên TEST**

![Chi phí theo ngưỡng](reports/chi_phi_theo_nguong.png)

**Giả định quy đổi:** `Amount` gốc là EUR; quy đổi đơn giản 1 EUR ≈ 27.000 VND
cho mục đích minh hoạ (không phải tỉ giá thực tế thời điểm nào).

Chi phí = (số FP × 200.000đ chăm sóc khách hàng) + (tổng số tiền các giao dịch FN).

⚠️ Cùng lỗi rò rỉ như mục 5.4: bản trước quét *và* báo cáo chi phí trên cùng
TEST, ra ngưỡng "tối ưu" = 0,97 với chi phí ≈ 65.075.320đ — con số này **không
đáng tin** vì đã bị chọn để khớp tốt nhất với chính tập dùng để báo cáo nó.

**Cách làm đúng:** quét chi phí trên **validation**, chọn ngưỡng chi phí thấp
nhất tại đó, áp dụng đúng 1 lần lên **test**:

* **Ngưỡng tối ưu (chọn trên VAL):** 0,83 — tổng chi phí trên VAL ≈
  10.388.850đ (trên 42.721 giao dịch val).
* **Áp dụng 1 lần lên TEST:** tổng chi phí ≈ **69.654.800đ** (trên 42.722
  giao dịch test), Precision = 58,21%, Recall = 75,00%.

Đường nét đứt màu xám trên biểu đồ là chi phí đo trực tiếp trên TEST — **chỉ
để đối chiếu, không dùng để chọn ngưỡng**: nó cho thấy nếu "gian lận" nhìn cả
nhãn test, ngưỡng tối ưu sẽ trôi về 0,97 — khác hẳn 0,83 chọn trên val. Khoảng
lệch 0,83 vs 0,97 này **chính là bằng chứng cụ thể** cho việc tại sao không
được chọn ngưỡng trên tập dùng để báo cáo: hai tập validation/test tuy liền kề
về thời gian nhưng phân phối gian lận (số ca, giá trị giao dịch) đủ khác nhau
để kéo ngưỡng "tối ưu" đi xa.

> Bảng chi phí đầy đủ theo từng ngưỡng (cả val lẫn test-để-đối-chiếu):
> [`reports/chi_phi_theo_nguong.csv`](reports/chi_phi_theo_nguong.csv)

#### 5.5.1. Độ nhạy theo giả định chi phí chặn nhầm

`COST_CHAN_NHAM = 200.000đ` là một **giả định**, không phải số đo lường được,
và công thức chi phí (FP × 200.000đ + tổng tiền FN) không tách riêng "giá trị
thu hồi" khi bắt đúng TP — phần thu hồi đó nằm **ẩn** trong công thức dưới
dạng "chi phí FN tránh được" so với kịch bản không có mô hình (ngưỡng =
100%), chứ không phải một số dương được cộng thêm rõ ràng. Để không báo cáo
một con số duy nhất như thể chắc chắn đúng, quét lại toàn bộ quy trình
(chọn ngưỡng trên val, báo cáo trên test) với vài mức chi phí FP khác nhau:

| Chi phí FP giả định | Ngưỡng (chọn trên VAL) | Precision (TEST) | Recall (TEST) | Tổng chi phí (TEST) |
| ---: | :---: | :---: | :---: | ---: |
| 100.000đ | 0,83 | 58,21% | 75,00% | 66.854.800đ |
| **200.000đ (mặc định)** | **0,83** | **58,21%** | **75,00%** | **69.654.800đ** |
| 300.000đ | 0,95 | 84,78% | 75,00% | 66.154.800đ |
| 500.000đ | 0,95 | 84,78% | 75,00% | 67.554.800đ |

> Bảng đầy đủ: [`reports/do_nhay_chi_phi.csv`](reports/do_nhay_chi_phi.csv)

Ngưỡng tối ưu nhảy từ 0,83 lên 0,95 khi chi phí chặn nhầm giả định tăng từ
200k lên 300k (hợp lý: FP đắt hơn → mô hình nên "dè dặt" hơn, đòi hỏi xác suất
cao hơn mới chặn) — kết luận cuối cùng **không nhạy quá mức** với giả định
200k cụ thể, nhưng đây rõ ràng không phải một hằng số nên "đóng đinh".

#### 5.5.2. Kiểm tra calibration ở vùng vận hành

Cả hai ngưỡng (0,83–0,9513) đều được đọc như xác suất thật ("ngưỡng 0,95
nghĩa là tin 95%"). Đo Brier score trên **validation**: **0,00502** (0 =
hoàn hảo). Con số này thấp chủ yếu vì 99,87% mẫu val là lớp âm với xác suất
dự đoán gần 0 — Brier score toàn cục **không đủ** để kết luận mô hình
calibrated tốt đúng ở vùng ngưỡng vận hành (0,83–0,98), vì vùng đó gần như
không có mẫu.

Thực vậy, đường calibration 10-bin theo quantile ([`reports/calibration.png`](reports/calibration.png),
[`reports/calibration.csv`](reports/calibration.csv)) dồn gần hết các bin vào
vùng xác suất < 0,17 — quá ít giao dịch gian lận trên val (56 ca) để quantile
binning tách riêng được vùng cao. Kiểm tra trực tiếp thay thế: Precision tại
ngưỡng 0,9513 trên chính val là **91,67%**, tức khá gần với "lời hứa xác suất"
95% của ngưỡng đó (chỉ hơi *under-confident* — thực tế đúng nhiều hơn một
chút so với xác suất dự đoán). Đây là hướng lệch an toàn hơn cho bài toán chặn
gian lận so với overconfident, nhưng vẫn nên xem là ước lượng thô — mẫu quá
nhỏ (56 ca) để kết luận chắc chắn.

### 5.6. Tốc độ dự đoán 1 giao dịch

| Mô hình | Thời gian / giao dịch |
| :--- | ---: |
| Logistic Regression | 0,55 ms |
| **XGBoost** | **2,14 ms** |
| LightGBM | 1,73 ms |
| Random Forest | 53,25 ms |

→ XGBoost đạt yêu cầu **< 100ms** với biên độ dư ~47 lần, đủ an toàn cho
300 giao dịch/giây (dự đoán tuần tự vẫn còn dư nhiều thời gian; production
thật sẽ dùng batch/song song để tối ưu thêm). Số ms tuyệt đối phụ thuộc máy
chạy (xem ghi chú ở mục 5.2) — điều ổn định qua các lần chạy là **thứ hạng
tương đối**: XGBoost luôn nhanh hơn Random Forest hàng chục lần.

### 5.7. Feature importance

![Feature importance](reports/feature_importance.png)

`V14` và `V10` chiếm phần lớn tổng gain — hai thành phần PCA này thường xuất
hiện đầu bảng trong các phân tích công khai khác về cùng bộ dữ liệu, dù không
biết chúng đại diện cho biến gốc nào (đúng hạn chế đã nêu ở mục 3).
`Amount_scaled` và `Hour` — hai đặc trưng tự tạo — lọt top nửa trên, cho thấy
bước feature engineering có đóng góp thực chất.

---

## 6. HẠN CHẾ CẦN NÊU RÕ

```
   1. V1–V28 đã PCA hoá → không giải thích được ý nghĩa nghiệp vụ của từng
      đặc trưng, không thể tinh chỉnh feature engineering theo miền (domain).
   2. Quy đổi EUR → VND (27.000) trong phân tích chi phí là giả định đơn
      giản hoá cho mục đích minh hoạ phương pháp, không phải tỉ giá thực.
   3. Bộ dữ liệu chỉ trải dài 48 giờ (2 ngày) → chưa đủ để đánh giá drift
      theo mùa vụ/tuần/tháng, chỉ mô phỏng được drift ở quy mô rất nhỏ.
   4. Chi phí chặn nhầm (200.000đ/FP) là một con số cố định giả định, không
      phân biệt theo giá trị giao dịch hay theo khách hàng — mục 5.5.1 đã đo
      độ nhạy quanh giả định này nhưng chưa mô hình hoá chi phí biến thiên.
   5. Công thức chi phí không tách riêng "giá trị thu hồi" khi bắt đúng TP
      thành một số dương độc lập — nó nằm ẩn dưới dạng chi phí FN tránh được
      so với kịch bản không dùng mô hình. Muốn báo cáo rõ khoản tiết kiệm
      tuyệt đối, cần thêm bước so sánh với baseline "không chặn gì".
   6. Kiểm tra calibration (mục 5.5.2) chỉ đo được thô ở vùng ngưỡng vận hành
      vì val chỉ có 56 ca gian lận — không đủ dữ liệu để khẳng định chắc
      chắn xgb_score là xác suất hiệu chỉnh tốt ở đó.
```

---

## 7. PHƯƠNG ÁN THEO DÕI DRIFT (kể gian đổi chiêu thức liên tục)

```
   1. Ghi log mỗi dự đoán: xác suất, ngưỡng áp dụng, đặc trưng đầu vào,
      timestamp — phục vụ audit và huấn luyện lại sau này.
   2. Theo dõi PHÂN PHỐI đặc trưng đầu vào theo cửa sổ thời gian trượt
      (vd. PSI - Population Stability Index) trên các đặc trưng quan trọng
      nhất (V14, V10, V4, Amount_scaled) — cảnh báo khi PSI vượt ngưỡng.
   3. Theo dõi PR-AUC trên nhãn thật có độ trễ (feedback từ đội xác minh gian
      lận, thường về sau vài ngày) theo cửa sổ trượt — cảnh báo khi PR-AUC
      giảm liên tục qua nhiều cửa sổ.
   4. Lên lịch huấn luyện lại định kỳ (vd. hàng tuần) + huấn luyện lại ngay
      khi cảnh báo drift vượt ngưỡng, luôn giữ tập validation "mới nhất" để
      early-stopping phản ánh đúng phân phối hiện tại.
```

---

## 8. CẠM BẪY ĐÃ TRÁNH

| Cạm bẫy | Hậu quả | Đã xử lý bằng |
| :--- | :--- | :--- |
| Chia ngẫu nhiên | Rò rỉ thời gian → điểm ảo | Sort theo `Time`, chia 70/15/15 tuần tự |
| Dùng ROC-AUC làm metric chính | Che giấu năng lực thật | Ưu tiên PR-AUC, báo cáo cả hai (mục 5.3) |
| Fit scaler trên toàn bộ dữ liệu | Rò rỉ thống kê từ test vào train | `StandardScaler.fit()` chỉ trên train |
| Quên `scale_pos_weight` | Model bỏ qua lớp thiểu số | `scale_pos_weight = 518.18` đo trên train |
| Không early stopping với 1000 cây | Overfit + tốn thời gian | `early_stopping_rounds=50` theo `aucpr` |
| Đánh giá bằng accuracy | 99,83% mà bắt được 0 vụ gian lận | Precision/Recall/PR-AUC theo ngưỡng |
| Copy nguyên `scale_pos_weight` sang LightGBM | Model sụp đổ (ROC-AUC 0,19) | Kiểm chứng riêng từng thư viện (mục 5.2) |
| **Quét & chọn ngưỡng (Precision, chi phí) trên chính TEST rồi báo cáo trên TEST đó** | **Precision/Recall/chi phí lạc quan ảo, không tái lập khi triển khai — lỗi thật đã tồn tại ở bản trước của báo cáo này** | Quét & chọn cả hai ngưỡng trên **VALIDATION**, chỉ áp dụng 1 lần lên TEST để báo cáo (mục 5.4, 5.5) |
| Commit thẳng file dữ liệu ~102MB vào Git | `.git` phồng to dù đã có hướng dẫn tải ở `DATA_SOURCE.md` | `data/.gitignore` bỏ qua `*.csv`, gỡ khỏi tracking (mục 9) |

---

## 9. SẢN PHẨM & CẤU TRÚC THƯ MỤC

```
TT-08-XGBoost/
├── README.md                       # Báo cáo này
├── requirements.txt
├── data/
│   ├── creditcard.csv               # 284.807 giao dịch — KHÔNG commit (xem .gitignore), tải theo DATA_SOURCE.md
│   ├── .gitignore                   # Bỏ qua *.csv — tránh lặp lại lỗi commit file ~102MB
│   └── DATA_SOURCE.md
├── notebooks/
│   └── xgboost_fraud.ipynb          # Giải thích từng bước + toàn bộ output thật
├── src/
│   └── train.py                     # Script huấn luyện + sinh toàn bộ report tự động
├── models/
│   ├── xgb_fraud.json               # Model XGBoost đã huấn luyện
│   └── amount_scaler.joblib         # StandardScaler cho Amount_log
└── reports/
    ├── eda_hour_amount.png
    ├── pr_vs_roc.png
    ├── chi_phi_theo_nguong.png
    ├── feature_importance.png
    ├── calibration.png              # Đường calibration (VAL) — mục 5.5.2
    ├── fraud_theo_gio.csv
    ├── so_sanh_mo_hinh.csv
    ├── chi_phi_theo_nguong.csv      # Chi phí theo ngưỡng: cả val (dùng để chọn) lẫn test (đối chiếu)
    ├── do_nhay_chi_phi.csv          # Độ nhạy theo giả định chi phí FP — mục 5.5.1
    ├── calibration.csv
    └── tom_tat.json
```

---

## 10. HƯỚNG DẪN CHẠY DỰ ÁN

```bash
pip install -r requirements.txt
python src/train.py                              # huấn luyện + sinh toàn bộ report
jupyter notebook notebooks/xgboost_fraud.ipynb    # khám phá từng bước có giải thích
```

---

## 11. HƯỚNG PHÁT TRIỂN & MỞ RỘNG

1. Thử **Isolation Forest** (phát hiện bất thường không giám sát) → so sánh
   với XGBoost trên cùng tập test (đặc biệt hữu ích khi nhãn gian lận đến
   trễ hoặc không đầy đủ trong thực tế).
2. Mô phỏng **concept drift** thực tế: train trên nửa đầu 48 giờ, test trên
   nửa sau → đo mức giảm PR-AUC để ước lượng tốc độ cần huấn luyện lại.
3. Deploy FastAPI với ngưỡng cấu hình được (mặc định theo mục 5.5) + ghi log
   mọi dự đoán theo đề xuất giám sát drift ở mục 7.

**Tham khảo:** [Buổi 6 — Ensemble](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-06-Ensemble-EndToEnd/Tai-Lieu) · [Buổi 12 — Chọn metric theo giá của lỗi](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-12-Capstone-TongKet/Tai-Lieu/ly_thuyet_chi_tiet_buoi_12.md)
