# TT-04 — LOGISTIC REGRESSION
## Chẩn đoán nguy cơ bệnh tim — model GIẢI THÍCH ĐƯỢC cho bác sĩ

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 4](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-04-LogReg-SVM-Metrics) |
| 🧠 **Nhóm** | Phân loại tuyến tính có giám sát |
| 🔧 **Thuật toán** | Logistic Regression |
| 🏭 **Lĩnh vực** | Y tế · Tim mạch |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   Linear Regression cho ra số bất kỳ (−∞, +∞) → không dùng làm xác suất được.
   Logistic Regression bọc thêm hàm SIGMOID để ép về khoảng (0, 1):

        z = w₁x₁ + w₂x₂ + ... + b          ← phần tuyến tính
        p = σ(z) = 1 / (1 + e⁻ᶻ)           ← ép thành xác suất

          1 ┤      ╭──────
            │     ╱
        0.5 ┤    ╱
            │   ╱
          0 ┤──╯
            └──────────── z
```

**Vì sao ngành y tế và tài chính vẫn ưa dùng:** mỗi hệ số `w` có ý nghĩa rõ ràng
qua **odds ratio** — bác sĩ đọc được "tăng 1 đơn vị chỉ số này thì nguy cơ nhân lên
bao nhiêu lần". Không model nào giải thích tự nhiên như vậy.

---

## 2. BÀI TOÁN THỰC TẾ

```
   Khoa Tim mạch cần công cụ đánh giá nguy cơ NGAY tại phòng khám,
   dựa trên các chỉ số đo được trong 15 phút.

   Yêu cầu bắt buộc từ bác sĩ:
     ① Phải giải thích được: "vì sao bệnh nhân này bị xếp nguy cơ cao?"
     ② Ưu tiên KHÔNG BỎ SÓT (recall cao) — bỏ sót bệnh tim có thể chết người
     ③ Cho ra XÁC SUẤT, không chỉ nhãn 0/1 → bác sĩ tự quyết theo ngưỡng
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Heart Disease (UCI — Cleveland) |
| **Link** | https://archive.ics.uci.edu/dataset/45/heart+disease |
| **Bản Kaggle tiện hơn** | https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset |
| **Kích thước** | 303 dòng × 14 cột (bản Kaggle: 1.025 dòng đã nhân bản) |
| **Nhãn** | `target` (0 = không bệnh, 1 = có bệnh) — khá cân bằng ~54/46 |

**Các cột:** `age`, `sex`, `cp` (kiểu đau ngực), `trestbps` (huyết áp nghỉ),
`chol` (cholesterol), `fbs` (đường huyết đói), `restecg`, `thalach` (nhịp tim tối đa),
`exang` (đau thắt ngực khi gắng sức), `oldpeak`, `slope`, `ca`, `thal`

### ⚠️ Ba lưu ý về dữ liệu

```
   1. Bản Kaggle 1.025 dòng là bản NHÂN BẢN từ 303 dòng gốc
      → có DÒNG TRÙNG LẶP giữa train và test → RÒ RỈ!
      → Phải df.drop_duplicates() trước khi chia dữ liệu.

   2. cp, restecg, slope, thal là biến PHÂN LOẠI được mã hoá bằng SỐ
      → nếu để nguyên, model hiểu nhầm là có thứ tự (cp=3 > cp=1)
      → phải one-hot encode.

   3. ca và thal có giá trị lạ (0 hoặc 4) ở một số bản → kiểm tra value_counts().
```

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. Đọc hệ số bằng ODDS RATIO — phần quan trọng nhất

```
   Odds Ratio = e^w

   OR = 1,0  → không ảnh hưởng
   OR = 2,5  → tăng 1 đơn vị biến này, ODDS mắc bệnh tăng 2,5 lần
   OR = 0,4  → tăng 1 đơn vị, odds GIẢM còn 40% (yếu tố bảo vệ)
```

```python
import numpy as np, pandas as pd

odds = pd.DataFrame({
    'dac_trung': ten_cot,
    'he_so_w':   model.coef_[0],
    'odds_ratio': np.exp(model.coef_[0]),
}).sort_values('odds_ratio', ascending=False)
```

> ⚠️ Chỉ so sánh được độ lớn hệ số khi **đã chuẩn hoá** các biến số. Nếu không,
> hệ số của `chol` (đơn vị mg/dl) và `oldpeak` (đơn vị mm) không so được với nhau.

### 4.2. Regularization — chọn L1 hay L2

| | L2 (Ridge) | L1 (Lasso) |
|---|---|---|
| Tác dụng | Co hệ số về gần 0 | **Đưa hệ số về đúng 0** |
| Dùng khi | Mặc định, giữ mọi biến | Muốn CHỌN biến, bỏ bớt |
| `penalty=` | `'l2'` (mặc định) | `'l1'` + `solver='liblinear'` hoặc `'saga'` |

```python
from sklearn.linear_model import LogisticRegressionCV
model = LogisticRegressionCV(Cs=10, cv=5, penalty='l2',
                             scoring='recall', max_iter=2000, random_state=42)
# C nhỏ = phạt MẠNH ; C lớn = phạt nhẹ  (C = 1/λ, ngược với trực giác!)
```

### 4.3. Chọn ngưỡng theo yêu cầu lâm sàng

```
   Ngưỡng mặc định 0,5 KHÔNG phù hợp cho sàng lọc y tế.
   Bác sĩ yêu cầu recall ≥ 0,90 → hạ ngưỡng xuống ~0,35–0,40.

   → Vẽ precision_recall_curve, tìm ngưỡng thấp nhất đạt recall 0,90
   → Báo cáo cái giá phải trả: precision giảm bao nhiêu?
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. ⚠️ drop_duplicates() TRƯỚC KHI chia train/test (nếu dùng bản Kaggle)
   ☐ 2. Xác định đúng biến phân loại (cp, restecg, slope, thal, ca) → one-hot
   ☐ 3. EDA: tỉ lệ bệnh theo cp, theo nhóm tuổi, boxplot thalach theo target
   ☐ 4. Pipeline: OneHot(cat) + StandardScaler(num) + LogisticRegression
   ☐ 5. Baseline: DummyClassifier
   ☐ 6. Train, in bảng ODDS RATIO xếp hạng
   ☐ 7. So sánh L1 vs L2: L1 loại bỏ mấy biến? Điểm có giảm không?
   ☐ 8. Vẽ đường ROC và Precision-Recall
   ☐ 9. Chọn ngưỡng đạt recall ≥ 0,90, ghi lại precision tương ứng
   ☐ 10. Kiểm tra ĐA CỘNG TUYẾN bằng VIF → biến nào VIF > 10?
   ☐ 11. So sánh với SVM (TT-05) và Random Forest (TT-03)
   ☐ 12. ✍️ Viết 1 đoạn giải thích cho 1 bệnh nhân cụ thể
```

```python
# Kiểm tra đa cộng tuyến
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame({'bien': X.columns,
                    'VIF': [variance_inflation_factor(X.values, i)
                            for i in range(X.shape[1])]})
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Đã xử lý dòng trùng lặp (nêu rõ số dòng bị loại)
   ☐ Biến phân loại được one-hot đúng cách (không để dạng số có thứ tự)
   ☐ Có BẢNG ODDS RATIO xếp hạng, giải thích được 3 yếu tố nguy cơ hàng đầu
   ☐ Có so sánh L1 vs L2
   ☐ Có đường ROC + PR, ngưỡng được chọn theo yêu cầu recall ≥ 0,90
   ☐ Có kiểm tra VIF
   ☐ Giải thích được kết quả cho 1 ca bệnh cụ thể bằng ngôn ngữ y khoa dễ hiểu
```

**Mức tham chiếu:** ROC-AUC ~0,85–0,92 (bộ nhỏ nên phương sai lớn — hãy báo cáo
kèm độ lệch chuẩn qua cross-validation).

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Không loại dòng trùng (bản Kaggle) | Rò rỉ → AUC ~1,0 giả tạo |
| Để `cp`, `thal` dạng số | Model hiểu nhầm thứ tự → sai hoàn toàn |
| Không chuẩn hoá | Không so sánh được độ lớn hệ số |
| Hiểu C ngược | `C` nhỏ = phạt MẠNH (C = 1/λ) |
| Bỏ qua đa cộng tuyến | Hệ số nhảy loạn, diễn giải sai |
| Kết luận NHÂN QUẢ từ odds ratio | "Cholesterol cao GÂY RA bệnh tim" — chưa chứng minh được |

---

## 8. SẢN PHẨM NỘP

```
TT-04-LogisticRegression-<HoTen>/
├── README.md                       ← có bảng odds ratio + diễn giải y khoa
├── notebooks/logistic_heart.ipynb
├── src/train.py
├── models/logreg_pipeline.joblib
├── reports/{odds_ratio.png, roc_pr_curve.png, vif_table.csv}
└── requirements.txt
```

> ⚖️ **Đạo đức bắt buộc ghi trong báo cáo:** đây là công cụ **hỗ trợ**, không thay
> thế bác sĩ. Bộ dữ liệu từ thập niên 1980 tại Mỹ → **không đại diện** cho bệnh nhân
> Việt Nam hiện nay. Kiểm tra xem model có đối xử khác nhau giữa nam/nữ không (`sex`).

---

## 9. MỞ RỘNG

```
   1. Tính khoảng tin cậy 95% cho odds ratio (dùng statsmodels.Logit)
   2. Xây NOMOGRAM — biểu đồ tính điểm nguy cơ bằng tay cho bác sĩ dùng offline
   3. So sánh hiệu chuẩn xác suất (calibration curve) giữa LogReg và Random Forest
      → LogReg thường cho xác suất "thật" hơn
```

**Tham khảo:** [Buổi 4 — LogReg, SVM & Metrics](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-04-LogReg-SVM-Metrics/Tai-Lieu)
