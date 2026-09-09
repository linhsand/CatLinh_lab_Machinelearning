# TT-13 — LASSO REGRESSION (L1)
## Chọn ra 10 chỉ số xét nghiệm quan trọng nhất trong 200 chỉ số

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao) |
| 🧠 **Nhóm** | Hồi quy + **CHỌN ĐẶC TRƯNG tự động** |
| 🔧 **Thuật toán** | Lasso (phạt L1) |
| 🏭 **Lĩnh vực** | Y tế · Xét nghiệm · Nghiên cứu lâm sàng |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   Ridge phạt:  λ·Σ wᵢ²      → CO về gần 0
   Lasso phạt:  λ·Σ |wᵢ|     → ĐẨY VỀ ĐÚNG 0  ⭐

   Vì sao trị tuyệt đối lại đưa được về đúng 0?
   Hình dạng vùng ràng buộc:

      RIDGE (hình TRÒN)          LASSO (hình THOI)
          ╭───╮                       ◇
         │  ●  │                     ╱ ● ╲        ● = nghiệm tối ưu
          ╰───╯                     ╲   ╱
      Điểm chạm hiếm khi           Điểm chạm HAY RƠI VÀO GÓC
      nằm trên trục                mà GÓC nằm trên TRỤC → w = 0
```

→ Lasso vừa hồi quy vừa **chọn biến**, cho ra model **thưa (sparse)**.

---

## 2. BÀI TOÁN THỰC TẾ

```
   Bệnh viện muốn xây bộ xét nghiệm sàng lọc tiến triển bệnh tiểu đường.
   Có thể đo 200 chỉ số sinh hoá, nhưng:
     • Mỗi chỉ số tốn 30.000–200.000đ
     • Bệnh nhân không thể lấy 200 ống máu

   → Cần chọn ~10 chỉ số ĐỦ TỐT để dự đoán, bỏ 190 chỉ số còn lại.
   → Đây chính xác là việc Lasso sinh ra để làm.

   💰 Giá trị: giảm chi phí xét nghiệm từ ~5 triệu xuống ~500 nghìn/bệnh nhân.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Khởi động** | `sklearn.datasets.load_diabetes()` — 442 dòng × 10 đặc trưng |
| **Nâng cao** | Tự mở rộng lên 200 đặc trưng bằng cách thêm biến NHIỄU |

```python
from sklearn.datasets import load_diabetes
import numpy as np, pandas as pd

X, y = load_diabetes(return_X_y=True, as_frame=True)

# Thêm 190 cột NHIỄU thuần tuý → mô phỏng "200 chỉ số xét nghiệm"
rng = np.random.default_rng(42)
nhieu = pd.DataFrame(rng.normal(size=(len(X), 190)),
                     columns=[f'chi_so_nhieu_{i:03d}' for i in range(190)])
X_full = pd.concat([X, nhieu], axis=1)      # 200 cột, chỉ 10 cột có tín hiệu thật
```

> ⭐ Thiết kế này cho phép **chấm điểm khách quan**: Lasso có tìm lại đúng 10 cột
> gốc và loại 190 cột nhiễu không? Đây là bài tập hiếm hoi có **đáp án đúng biết trước**.

---

## 4. HƯỚNG ĐI ĐÚNG

```python
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

pipe = Pipeline([
    ('scale', StandardScaler()),               # ⭐ BẮT BUỘC
    ('lasso', LassoCV(alphas=np.logspace(-4, 1, 100), cv=5,
                      max_iter=50000, random_state=42)),
])
pipe.fit(X_train, y_train)

he_so = pipe['lasso'].coef_
so_bien_giu = (he_so != 0).sum()
print(f"Lasso giữ lại {so_bien_giu}/200 biến")
```

**So sánh 3 loại phạt:**

| | Ridge (L2) | Lasso (L1) | ElasticNet |
|---|---|---|---|
| Đưa hệ số về đúng 0 | ❌ | ✅ | ✅ |
| Nhóm biến tương quan cao | Giữ **cả nhóm** | Chọn **1, bỏ phần còn lại** ⚠️ | Giữ cả nhóm |
| Khi p > n | Kém | ⚠️ Chỉ chọn tối đa n biến | ✅ Tốt |

> ⚠️ **Điểm yếu của Lasso:** với 2 biến tương quan 0,95, Lasso chọn **ngẫu nhiên 1**
> và bỏ hẳn biến kia. Trong y tế điều này nguy hiểm: biến bị bỏ có thể là biến
> rẻ tiền/dễ đo hơn. → Đó là lý do ra đời ElasticNet (TT-14).

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Nạp load_diabetes, mở rộng thành 200 cột (10 thật + 190 nhiễu)
   ☐ 2. Baseline: Linear Regression trên 200 cột → quan sát overfit nặng
   ☐ 3. LassoCV dò alpha
   ☐ 4. ⭐ CHẤM ĐIỂM CHỌN BIẾN:
        • Bao nhiêu trong 10 biến THẬT được giữ?   (recall chọn biến)
        • Bao nhiêu biến nhiễu bị giữ nhầm?         (false positive)
   ☐ 5. Vẽ coefficient path của Lasso → thấy các hệ số lần lượt "rơi" về 0
   ☐ 6. Vẽ RMSE train/test theo alpha
   ☐ 7. So sánh Ridge vs Lasso trên cùng dữ liệu:
        • Ridge giữ bao nhiêu biến? (đáp án: cả 200, chỉ co nhỏ)
        • RMSE cái nào tốt hơn?
   ☐ 8. ⚠️ THÍ NGHIỆM BIẾN TƯƠNG QUAN: nhân đôi 1 cột thật (thêm nhiễu nhỏ)
        → Lasso giữ cột nào? Chạy lại với seed khác → có đổi không?
   ☐ 9. Train lại Linear Regression CHỈ trên các biến Lasso chọn
        → RMSE so với Lasso đầy đủ? (kỹ thuật "debiased lasso")
   ☐ 10. ✍️ Đề xuất bộ xét nghiệm cuối cùng + ước tính chi phí tiết kiệm
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ ⭐ Có bảng chấm điểm: giữ đúng mấy/10 biến thật, giữ nhầm mấy biến nhiễu
   ☐ Có coefficient path của Lasso
   ☐ Có bảng so sánh Ridge vs Lasso (số biến giữ lại + RMSE)
   ☐ Có thí nghiệm biến tương quan + nhận xét về tính không ổn định của Lasso
   ☐ Giải thích được VÌ SAO hình thoi đưa hệ số về 0 còn hình tròn thì không
   ☐ ✍️ Đề xuất bộ xét nghiệm + con số tiết kiệm chi phí
```

**Mức tham chiếu:** Lasso thường giữ 8–15 biến, trong đó bắt lại được 6–9 trên 10
biến thật. Không đạt 10/10 là bình thường — bộ diabetes vốn có tín hiệu yếu.

---

## 6.1. KẾT QUẢ CHẠY THỰC TẾ

**Toàn bộ số liệu dưới đây là kết quả chạy thật** (`notebooks/lasso_feature_selection.ipynb`,
tái lập bằng `src/train.py`), `random_state=42`.

**Baseline overfit (Linear Regression trên 200 biến, p≈n):**

| | RMSE | R² |
|---|---|---|
| Train | 33,99 | 0,8099 |
| Test | 84,78 | **-0,3566** (âm — tệ hơn dự đoán bằng trung bình!) |

**⭐ Bảng chấm điểm chọn biến** (LassoCV, alpha tối ưu = 7,92483, giữ 5/200 biến):

| Loại | Tổng số | Số được giữ | Tỷ lệ | Danh sách |
|---|---|---|---|---|
| Biến THẬT | 10 | **4** | 40% (recall) | `bmi, bp, s3, s5` |
| Biến NHIỄU | 190 | **1** | 0,5% (false positive) | `chi_so_nhieu_009` |

Recall 4/10 thấp hơn mức tham chiếu (6–9/10) vì `alpha` được chọn hoàn toàn bằng
cross-validation (tối ưu khả năng tổng quát hoá, không tối ưu recall) nên loại
luôn vài biến thật có tín hiệu yếu (age, sex, s1, s2, s4, s6) cùng với nhiễu.
Tỷ lệ false positive cực thấp (0,5%) cho thấy Lasso rất hiệu quả trong việc loại
nhiễu.

**So sánh Ridge vs Lasso (cùng dữ liệu 200 biến):**

| Model | Số biến giữ | RMSE test | R² test |
|---|---|---|---|
| Ridge (alpha≈351) | **200/200** (chỉ co nhỏ) | 58,05 | 0,3640 |
| Lasso (alpha≈7,92) | **5/200** | **53,73** (tốt hơn) | **0,4551** |

Khác với TT-12 (Ridge và Lasso cho RMSE gần bằng nhau), ở đây Lasso vượt trội hơn
Ridge vì 190/200 biến là nhiễu thuần tuý — khả năng loại bỏ hẳn biến vô dụng mang
lại lợi thế dự đoán rõ rệt.

**Thí nghiệm biến tương quan** (nhân đôi `bmi` → `bmi_dup`, r=0,9806, chạy 5 seed
train/test khác nhau):

| seed | hệ số bmi | hệ số bmi_dup | biến được giữ |
|---|---|---|---|
| 0 | 26,13 | 0,00 | chỉ `bmi` |
| 1 | 24,10 | 0,61 | cả hai |
| 2 | 18,53 | 4,90 | cả hai |
| 3 | 24,92 | 0,00 | chỉ `bmi` |
| 4 | 16,15 | 6,21 | cả hai |

→ **Lựa chọn không ổn định** giữa các seed — đúng cảnh báo của README, xác nhận
bằng thực nghiệm.

**Debiased lasso:** Linear Regression chỉ trên 5 biến Lasso chọn cho RMSE = 53,74
(gần như giống hệt RMSE Lasso đầy đủ = 53,73) → phần lớn giá trị của Lasso ở bài
này nằm ở việc **chọn biến**, không phải ở việc co hệ số.

**✍️ Đề xuất bộ xét nghiệm cuối cùng:**

| Bộ xét nghiệm | Số chỉ số | Chi phí/bệnh nhân |
|---|---|---|
| Đầy đủ (200 chỉ số) | 200 | 22.105.930 VND |
| Đề xuất (Lasso chọn) | 5 (`s5, s3, bp, bmi` + 1 nhiễu cần rà soát lâm sàng) | 286.566 VND |
| **Tiết kiệm** | | **98,7% (≈21,8 triệu VND/bệnh nhân)** |

Lưu ý: `chi_so_nhieu_009` là false positive (hệ số rất nhỏ) — cần bác sĩ rà soát
lại ý nghĩa lâm sàng trước khi đưa vào bộ xét nghiệm thật, không nên tin tuyệt
đối vào kết quả thuật toán.

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Quên chuẩn hoá | Biến đơn vị lớn bị phạt oan → bị loại nhầm |
| `max_iter` mặc định | Cảnh báo không hội tụ → tăng lên 50.000 |
| Tin tuyệt đối vào biến Lasso chọn | Với biến tương quan, lựa chọn KHÔNG ổn định |
| Dùng Lasso khi p >> n mà cần giữ nhóm biến | Dùng ElasticNet thay thế |
| Kết luận "biến bị loại = vô dụng" | Có thể chỉ do tương quan với biến khác |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-13-Lasso/
├── README.md          ← có bảng chấm điểm chọn biến (mục 6.1)
├── notebooks/lasso_feature_selection.ipynb
├── src/train.py
├── models/lasso_pipeline.joblib
├── reports/
│   ├── chon_bien_score.csv, chon_bien_score.png   ← bảng/biểu đồ chấm điểm chọn biến
│   ├── lasso_path.png                             ← coefficient path
│   ├── rmse_theo_alpha.png                        ← RMSE train/test theo alpha
│   ├── ridge_vs_lasso.csv, ridge_vs_lasso.png      ← so sánh Ridge vs Lasso
│   ├── thi_nghiem_tuong_quan.csv                  ← thí nghiệm biến tương quan
│   ├── de_xuat_bo_xet_nghiem.csv                  ← bộ xét nghiệm đề xuất + chi phí
│   └── tom_tat.json                               ← tổng hợp toàn bộ số liệu
└── requirements.txt
```

**Mở rộng:**
1. **Stability Selection**: chạy Lasso 100 lần trên các mẫu bootstrap, giữ biến
   được chọn > 60% số lần → khắc phục tính không ổn định
2. So sánh với `SelectKBest(f_regression)` và `RFE` — cách nào chọn biến tốt hơn?
3. Áp dụng cho phân loại: `LogisticRegression(penalty='l1')` — xem TT-04

**Tham khảo:** [Buổi 13 — Regularization & chọn biến](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
