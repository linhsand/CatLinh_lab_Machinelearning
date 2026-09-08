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

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   Linear Regression tối thiểu hoá:        MSE
   Ridge tối thiểu hoá:                    MSE + λ·Σwᵢ²
                                                 └─ phạt hệ số LỚN ─┘

   Tác dụng: CO các hệ số về gần 0 (nhưng KHÔNG bao giờ đúng bằng 0)

        λ = 0     → giống hệt Linear Regression
        λ nhỏ     → co nhẹ
        λ lớn     → mọi hệ số ≈ 0 → model dự đoán gần như hằng số (underfit)
```

**Ridge sinh ra để chữa ĐA CỘNG TUYẾN:** khi 2 biến tương quan cao, Linear Regression
cho hệ số nhảy loạn (một biến +500, biến kia −480). Ridge chia đều ảnh hưởng cho cả hai
→ hệ số ổn định, diễn giải được.

---

## 2. BÀI TOÁN THỰC TẾ

```
   Doanh nghiệp chạy quảng cáo 6 kênh CÙNG LÚC: TV, Facebook, Google,
   TikTok, báo giấy, radio. Ngân sách 2 tỷ/tháng.

   Câu hỏi của CMO: "Kênh nào đang tạo ra doanh thu? Nên dồn tiền vào đâu?"

   ⚠️ VẤN ĐỀ: các kênh thường được tăng/giảm ngân sách CÙNG NHAU
      (chiến dịch lớn thì bơm tiền tất cả kênh)
      → ngân sách các kênh TƯƠNG QUAN RẤT CAO (đa cộng tuyến)
      → Linear Regression cho hệ số vô lý: "TikTok âm 300 triệu doanh thu"
      → Ridge ổn định hoá hệ số → phân bổ ngân sách đáng tin hơn.
```

---

## 3. BỘ DỮ LIỆU

| Lựa chọn | Nguồn | Ghi chú |
|----------|-------|---------|
| **Advertising** (khởi động) | https://www.kaggle.com/datasets/ashydv/advertising-dataset | 200 dòng × 4 cột (TV, Radio, Newspaper, Sales) — nhỏ, dễ hiểu |
| **Marketing Mix** | https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data | Nhiều kênh hơn |
| **Tự sinh** (khuyến nghị cho bài này) | Code bên dưới | ⭐ Chủ động tạo đa cộng tuyến để thấy rõ tác dụng Ridge |

```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 500
tv = rng.uniform(50, 500, n)
fb = tv * 0.6 + rng.normal(0, 15, n)      # ⭐ TƯƠNG QUAN CAO với TV (r ≈ 0,95)
gg = rng.uniform(20, 300, n)
doanh_thu = 3.2*tv + 1.8*fb + 2.5*gg + rng.normal(0, 50, n)
df = pd.DataFrame({'TV': tv, 'Facebook': fb, 'Google': gg, 'DoanhThu': doanh_thu})
```

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. ⚠️ Ridge BẮT BUỘC chuẩn hoá

```
   Phạt λ·Σwᵢ² áp dụng như nhau cho MỌI hệ số.
   Nếu TV tính bằng TRIỆU và Google tính bằng NGHÌN
   → hệ số của Google tự nhiên lớn hơn → bị phạt nặng hơn một cách BẤT CÔNG.
   → PHẢI StandardScaler trước.
```

### 4.2. Dò λ (trong sklearn gọi là `alpha`)

```python
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)),
])
pipe.fit(X_train, y_train)
print("alpha tối ưu:", pipe['ridge'].alpha_)
```

### 4.3. ⭐ Biểu đồ đường co hệ số (coefficient path) — sản phẩm quan trọng nhất

```python
from sklearn.linear_model import Ridge
alphas = np.logspace(-3, 4, 100)
paths = np.array([Ridge(alpha=a).fit(X_scaled, y).coef_ for a in alphas])
# Vẽ: trục x = log(alpha), trục y = giá trị hệ số
# → thấy các hệ số CO DẦN về 0 nhưng không chạm 0
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Tạo/nạp dữ liệu có đa cộng tuyến rõ rệt
   ☐ 2. Tính ma trận tương quan + VIF → chứng minh có đa cộng tuyến (VIF > 10)
   ☐ 3. Chạy Linear Regression → ghi lại hệ số
   ☐ 4. ⭐ THÍ NGHIỆM ỔN ĐỊNH: bootstrap 100 lần, mỗi lần lấy 80% dữ liệu
        → vẽ phân phối hệ số của Linear vs Ridge
        → Linear DAO ĐỘNG MẠNH, Ridge ỔN ĐỊNH  ← đây là bằng chứng thuyết phục nhất
   ☐ 5. RidgeCV dò alpha
   ☐ 6. Vẽ coefficient path
   ☐ 7. Vẽ đường RMSE train/test theo alpha → tìm điểm cân bằng bias-variance
   ☐ 8. So sánh RMSE: Linear vs Ridge (trên tập test)
   ☐ 9. So sánh với Lasso (TT-13) và ElasticNet (TT-14)
   ☐ 10. ✍️ Đề xuất phân bổ ngân sách dựa trên hệ số Ridge
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Có bảng VIF chứng minh đa cộng tuyến
   ☐ ⭐ Có biểu đồ so sánh ĐỘ ỔN ĐỊNH hệ số (bootstrap) giữa Linear và Ridge
   ☐ Có coefficient path
   ☐ Có đường RMSE theo alpha, alpha được chọn bằng cross-validation
   ☐ Giải thích được vì sao Ridge KHÔNG đưa hệ số về đúng 0
   ☐ ✍️ Có đề xuất phân bổ ngân sách bằng con số cụ thể
   ☐ Nêu hạn chế: hệ số hồi quy KHÔNG phải quan hệ nhân quả — cần A/B test để khẳng định
```

---

## 7. VÌ SAO RIDGE ỔN ĐỊNH HƠN

```
   Khi 2 biến gần như song song (TV <-> Facebook, VIF ~ 640 trong bộ dữ liệu
   mô phỏng của bài này): ma trận (XᵀX) gần suy biến -> nghịch đảo của nó cực
   kỳ nhạy với nhiễu nhỏ trong mẫu -> hệ số OLS "nhảy loạn" khi dữ liệu
   thay đổi nhẹ (chỉ cần bỏ 20% dữ liệu là hệ số Facebook có thể đổi dấu).

   Ridge cộng thêm λ·I vào (XᵀX) trước khi nghịch đảo:
        (XᵀX + λI)⁻¹ Xᵀy
   λ·I giữ ma trận khả nghịch ổn định -> nghiệm không còn nhạy cảm với
   nhiễu -> hệ số ổn định qua nhiều lần lấy mẫu lại (bootstrap).
```

**Bằng chứng thực nghiệm (chạy thật trong `notebooks/ridge_marketing.ipynb`):**

| | Linear Regression (OLS) | Ridge |
|---|---|---|
| Độ lệch chuẩn hệ số TV qua 100 lần bootstrap | 200,9 | 11,1 (giảm 94,5%) |
| Độ lệch chuẩn hệ số Facebook qua 100 lần bootstrap | 199,7 | 10,2 (giảm 94,9%) |
| Hệ số Facebook có thể ra ÂM? | Có — 39% số lần bootstrap | Không — 0% số lần bootstrap |
| RMSE trên test | 383,57 | 382,97 (không tệ hơn) |

**Vì sao hệ số không bao giờ đúng bằng 0:** số hạng phạt của Ridge là
`Σwᵢ²` — đường đồng mức của nó là hình tròn/elip **trơn**, không có góc nhọn
nằm trên trục toạ độ, nên điểm tối ưu (giao giữa đường đồng mức MSE và đường
đồng mức phạt) hầu như không bao giờ rơi đúng vào `wᵢ = 0`. Đây là điểm khác
biệt căn bản với Lasso (TT-13), dùng phạt `Σ|wᵢ|` có đường đồng mức hình
thoi với góc nhọn nằm đúng trên trục — khiến nghiệm dễ rơi đúng vào 0.

---

## 8. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Quên chuẩn hoá | Phạt bất công giữa các biến khác đơn vị |
| Chọn alpha bằng tập test | Rò rỉ → alpha không tối ưu thật |
| Kỳ vọng Ridge đưa hệ số về 0 | Nhầm với Lasso — Ridge chỉ CO về gần 0 |
| Kết luận nhân quả | Marketing mix cần thí nghiệm thật (geo-test, A/B) |
| Bỏ qua hiệu ứng trễ (adstock) | Quảng cáo TV hôm nay ảnh hưởng doanh thu tuần sau |

---

## 9. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-12-Ridge-<HoTen>/
├── README.md          ← có mục "VÌ SAO RIDGE ỔN ĐỊNH HƠN"
├── notebooks/ridge_marketing.ipynb
├── src/train.py
├── models/ridge_pipeline.joblib
├── reports/{vif_table.csv, bootstrap_he_so.png, coefficient_path.png, rmse_theo_alpha.png}
└── requirements.txt
```

**Mở rộng:**
1. Thêm **adstock** (hiệu ứng trễ): `x_t + 0.5·x_{t-1} + 0.25·x_{t-2}`
2. Thêm **hiệu ứng bão hoà**: dùng `log(1+x)` thay vì x (chi gấp đôi không cho doanh thu gấp đôi)
3. So sánh với Bayesian Ridge để có khoảng tin cậy cho từng hệ số

**Tham khảo:** [Buổi 13 — Regularization](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
