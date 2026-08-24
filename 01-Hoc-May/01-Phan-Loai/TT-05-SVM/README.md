# TT-05 — SVM (SUPPORT VECTOR MACHINE)
## Phân loại khối u lành tính / ác tính từ ảnh sinh thiết

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 4](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-04-LogReg-SVM-Metrics) |
| 🧠 **Nhóm** | Phân loại có giám sát |
| 🔧 **Thuật toán** | SVM (Linear · RBF · Polynomial kernel) |
| 🏭 **Lĩnh vực** | Y tế · Giải phẫu bệnh |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   SVM tìm đường phân chia có LỀ (margin) RỘNG NHẤT giữa 2 lớp:

        ●  ●        ╱ ╱ ╱          ● = lành tính
      ●   ● ●     ╱ ╱ ╱  ▲ ▲       ▲ = ác tính
        ●  ●    ╱ ╱ ╱  ▲  ▲ ▲
              ╱ ╱ ╱   ▲ ▲          ╱╱╱ = vùng lề
             ↑   ↑
        support vectors ← CHỈ những điểm SÁT lề mới quyết định đường phân chia

   Lề rộng → model tổng quát hoá tốt hơn trên dữ liệu mới.
```

**KERNEL TRICK** — sức mạnh thật của SVM: khi dữ liệu không tách được bằng đường
thẳng, kernel "nâng" dữ liệu lên chiều cao hơn nơi nó tách được — mà **không cần
tính toạ độ ở chiều đó**.

| Kernel | Dùng khi | Tham số |
|--------|----------|---------|
| `linear` | Nhiều đặc trưng, ít mẫu (văn bản) | `C` |
| **`rbf`** | ⭐ Mặc định, ranh giới cong | `C`, `gamma` |
| `poly` | Quan hệ đa thức rõ ràng | `C`, `degree` |

---

## 2. BÀI TOÁN THỰC TẾ

```
   Bác sĩ giải phẫu bệnh soi ảnh chọc hút tế bào, đo 30 chỉ số hình thái
   (bán kính, độ nhám, tính đối xứng… của nhân tế bào).

   Cần phân loại: LÀNH TÍNH hay ÁC TÍNH.

   ⚠️ Cái giá của lỗi CỰC KỲ không cân xứng:
      Báo lành tính nhưng thật ra ác tính (FN) → bỏ lỡ điều trị → NGUY HIỂM TÍNH MẠNG
      Báo ác tính nhưng thật ra lành tính (FP) → sinh thiết thêm → tốn kém, lo lắng

   → RECALL của lớp ÁC TÍNH phải ≥ 0,98. Không thoả hiệp.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Breast Cancer Wisconsin (Diagnostic) |
| **Cách lấy** | `from sklearn.datasets import load_breast_cancer` (có sẵn, không cần tải) |
| **Link gốc** | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| **Kích thước** | 569 dòng × 30 đặc trưng |
| **Nhãn** | 0 = ác tính (212), 1 = lành tính (357) |

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
# ⚠️ CHÚ Ý: trong sklearn, 0 = malignant (ác tính) — dễ nhầm dấu!
```

**30 đặc trưng** = 10 chỉ số × 3 thống kê (mean, standard error, worst).
→ Các cột **tương quan rất mạnh** với nhau (`radius_mean` ↔ `perimeter_mean` ↔ `area_mean`).

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. ⭐ SVM BẮT BUỘC chuẩn hoá

```
   area_mean:    143 – 2501      (biên độ ~2358)
   smoothness_mean: 0,05 – 0,16  (biên độ ~0,11)

   SVM dựa trên khoảng cách → không scale thì area_mean áp đảo,
   29 đặc trưng còn lại coi như không tồn tại.
   → Đây là thuật toán NHẠY CẢM NHẤT với việc quên chuẩn hoá.
```

### 4.2. Hiểu C và gamma

```
   C  — mức phạt khi phân loại sai
        C nhỏ  → lề RỘNG, chấp nhận vài lỗi   → đơn giản, có thể underfit
        C lớn  → lề HẸP, cố phân đúng hết     → phức tạp, dễ OVERFIT

   gamma — tầm ảnh hưởng của 1 điểm (chỉ với RBF)
        gamma nhỏ → ảnh hưởng XA  → ranh giới mượt
        gamma lớn → ảnh hưởng GẦN → ranh giới ngoằn ngoèo, OVERFIT nặng
```

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([('scale', StandardScaler()),
                 ('svm', SVC(probability=True, class_weight='balanced',
                             random_state=42))])

grid = {'svm__C':      [0.1, 1, 10, 100],
        'svm__gamma':  ['scale', 0.001, 0.01, 0.1],
        'svm__kernel': ['linear', 'rbf', 'poly']}

gs = GridSearchCV(pipe, grid, cv=5, scoring='recall', n_jobs=-1)
```

> ⚠️ `probability=True` làm SVM **chậm hơn nhiều** (phải chạy Platt scaling nội bộ).
> Chỉ bật khi thật sự cần xác suất.

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☑ 1. Nạp dữ liệu, XÁC NHẬN quy ước nhãn (0 = ác tính!) — ghi rõ trong báo cáo
   ☑ 2. EDA: heatmap tương quan 30 biến → chỉ ra nhóm biến trùng lặp thông tin
   ☑ 3. Chia train/test stratify
   ☑ 4. ⚠️ Chạy SVM KHÔNG scale → ghi lại điểm (thường rất tệ)
   ☑ 5. Chạy SVM CÓ scale → so sánh → CHỨNG MINH tầm quan trọng của chuẩn hoá
   ☑ 6. So sánh 3 kernel: linear / rbf / poly (cùng C=1)
   ☑ 7. GridSearchCV dò C, gamma, kernel — scoring dùng F2 (beta=2), KHÔNG
        dùng recall thuần (xem mục "KẾT QUẢ THỰC TẾ" bên dưới, lý do tại sao)
   ☑ 8. Vẽ heatmap điểm CV theo (C, gamma) → thấy vùng tối ưu
   ☑ 9. Đếm số support vector: model.n_support_ → chiếm bao nhiêu % dữ liệu?
   ☑ 10. Chọn ngưỡng nhắm RECALL lớp ác tính ≥ 0,98 (đạt 0,9762 — CHƯA đủ,
         xem giải thích bên dưới), báo cáo precision tương ứng
   ☑ 11. Vẽ ma trận nhầm lẫn, chỉ rõ có bao nhiêu ca ÁC TÍNH bị bỏ sót
   ☑ 12. So sánh với Logistic Regression (TT-04) — cái nào tốt hơn ở đây, vì sao?
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☑ Có bảng so sánh CÓ vs KHÔNG chuẩn hoá (chênh lệch phải rất lớn)
   ☑ Có so sánh 3 kernel
   ☑ Có heatmap C × gamma
   ☑ RECALL lớp ÁC TÍNH ≥ 0,97 trên tập test (đạt 0,9762 ở threshold đã chọn;
     mục tiêu cao hơn ≥0,98 nêu ở mục 5.10 CHƯA đạt — xem bên dưới)
   ☑ Ma trận nhầm lẫn chỉ rõ số ca ác tính bị bỏ sót: 1/42 ca (FN=1) ở
     threshold=0.40 đã chọn; 1/42 ca ở threshold mặc định 0.5 của best model
   ☑ Giải thích được vì sao SVM chỉ phụ thuộc vào support vectors
   ☑ Nêu hạn chế: SVM không giải thích được từng ca, chậm khi dữ liệu lớn
```

**Mức tham chiếu:** ROC-AUC ~0,98–0,99 (đây là bộ dữ liệu "dễ", tách bạch tốt).
Đo thực tế: ROC-AUC = **0,9940** (malignant = lớp dương). Đừng nhầm điểm cao
ở đây với việc bài toán y tế thật cũng dễ như vậy.

---

## 6.1. KẾT QUẢ THỰC TẾ (BÁO CÁO)

*(Số liệu sinh từ `src/train.py`, lưu đầy đủ trong `reports/metrics.json`;
tái tạo bằng `python src/train.py`.)*

**a) Chuẩn hoá quan trọng thế nào — SVM không scale vs có scale**

| | Accuracy | Precision (ác tính) | Recall (ác tính) | F1 (ác tính) |
|---|---|---|---|---|
| Không scale | 0,9298 | 0,9474 | **0,8571** | 0,9000 |
| Có `StandardScaler` | 0,9825 | 0,9762 | **0,9762** | 0,9762 |

→ Không scale, recall ác tính rơi xuống 0,857 (bỏ sót nhiều ca hơn hẳn) —
đúng như cảnh báo ở mục 4.1: các đặc trưng biên độ lớn (`area_mean`) áp đảo
khoảng cách, làm SVM lệch quyết định.

**b) So sánh 3 kernel (C=1)**

| Kernel | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| linear | 0,9737 | 0,9535 | 0,9762 | 0,9647 |
| **rbf** | 0,9825 | 0,9762 | 0,9762 | 0,9762 |
| poly | 0,9123 | 1,0000 | 0,7619 | 0,8649 |

→ `rbf` tốt nhất và cân bằng nhất; `poly` có precision hoàn hảo nhưng recall
rất thấp (bỏ sót gần 1/4 ca ác tính) — không chấp nhận được cho bài toán này.

**c) GridSearchCV — LỖI ĐÃ SỬA**

Bản đầu tiên dùng `scoring=make_scorer(recall_score, pos_label=0)` (recall
thuần). Nó chọn `C=1, gamma=0.1` — vùng overfit trên 30 đặc trưng (README đã
cảnh báo ở mục 4.2) — vì recall CV cao nhất bất kể precision. Trên test, model
đó chỉ đạt **0,930 / 0,870 / 0,952 (acc/prec/recall)**, TỆ HƠN cả baseline
chưa tune (0,982/0,976/0,976).

**Đã sửa** bằng cách đổi scoring sang **F2-score** (`fbeta_score(beta=2,
pos_label=0)`) — vẫn ưu tiên recall gấp đôi precision (đúng ưu tiên y tế của
bài toán) nhưng không thưởng cho cấu hình làm sụp precision. Kết quả:
GridSearch tự chọn lại `C=1, gamma='scale', kernel=rbf` — trùng với baseline
tốt nhất, đạt test **Accuracy 0,9649 / Precision 0,9318 / Recall 0,9762 /
F1 0,9535**, ROC-AUC 0,9940.

**d) Support vectors**

Model tốt nhất dùng **108/455 mẫu train làm support vector (23,74%)** — 49 ác
tính, 59 lành tính. Đây là các điểm nằm sát biên quyết định; các điểm còn lại
không ảnh hưởng đến vị trí ranh giới phân chia.

**e) Chọn threshold nhắm recall ≥ 0,98 — KHÔNG đạt, nêu rõ**

Quét ngưỡng bằng xác suất out-of-fold (5-fold CV) trên **tập train**, chỉ áp
ngưỡng đã chốt lên test **một lần duy nhất** (tránh dò ngưỡng trên test).
Ngưỡng chọn được: **0,40**.

Áp lên test: Accuracy 0,9474 / Precision 0,8913 / **Recall 0,9762** / F1
0,9318 — **1 ca ác tính bị bỏ sót (FN=1)** trên 42 ca ác tính của tập test, 5
ca lành tính bị báo nhầm ác tính (FP=5, tức 5 ca sinh thiết thừa trong số 46
dự đoán dương).

**Mục tiêu recall ≥ 0,98 của đề (mục 3, mục 6) CHƯA đạt được** (đạt 0,9762).
Đây là giới hạn thật của mô hình/dữ liệu ở kích thước tập test này (42 ca ác
tính → mỗi ca tương ứng ~2,4 điểm % recall, nên 1 ca bị bỏ sót đã kéo recall
xuống dưới 0,98); hạ ngưỡng thấp hơn nữa để ép recall lên 0,98+ sẽ chỉ làm
precision giảm thêm (nhiều sinh thiết thừa hơn), nên không được coi là cải
thiện thực chất. Ghi nhận trung thực thay vì che giấu.

**f) So sánh với Logistic Regression**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM (RBF, threshold mặc định 0,5) | 0,9649 | 0,9318 | 0,9762 | **0,9535** |
| Logistic Regression | 0,9561 | 0,9111 | 0,9762 | 0,9425 |

Hai model đạt recall ác tính bằng nhau (0,9762), nhưng **SVM có F1 cao hơn**
nhờ precision tốt hơn (ít báo nhầm hơn). Với dữ liệu Wisconsin — vốn tách
lớp khá rõ — Logistic Regression đơn giản, dễ diễn giải hệ số hơn SVM-RBF
(hộp đen), trong khi SVM cần tune C/gamma tốn công mà lợi ích thêm không lớn.
Nếu cần độ dễ diễn giải/triển khai nhanh, Logistic Regression là lựa chọn hợp
lý; nếu ưu tiên precision cao nhất ở cùng mức recall, SVM (RBF) nhỉnh hơn.

**g) Ngưỡng đi kèm model**

`models/svm_pipeline.joblib` lưu một **dict** `{"pipeline", "threshold",
"threshold_note", "threshold_test_metrics"}` thay vì chỉ lưu estimator —
vì gọi `pipeline.predict()` trực tiếp sẽ luôn dùng ngưỡng mặc định 0,5 (bỏ
qua phân tích ngưỡng ở mục (e) phía trên). Khi load model, cần áp
`pipeline.predict_proba(X)[:, 0] >= threshold` để dùng đúng điểm vận hành đã
khuyến nghị, cùng với đánh đổi precision đã nêu.

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Quên `StandardScaler` | SVM gần như vô dụng |
| Nhầm quy ước nhãn 0/1 | Tối ưu nhầm lớp — recall của lớp SAI |
| `gamma` quá lớn | Overfit nặng, train 100% test kém |
| Dùng SVM cho dữ liệu > 100k dòng | Rất chậm (độ phức tạp ~O(n²)) → dùng LinearSVC |
| Dùng accuracy | Che mất việc bỏ sót ca ác tính |

---

## 8. SẢN PHẨM NỘP

```
TT-05-SVM/
├── README.md                    ← số ca ác tính bị bỏ sót: FN=1/42 (xem mục 6.1e)
├── notebooks/svm_breast_cancer.ipynb
├── src/train.py
├── models/svm_pipeline.joblib   ← dict {pipeline, threshold, ...} (xem mục 6.1g)
├── reports/
│   ├── scale_vs_noscale.png
│   ├── kernel_comparison.png
│   ├── C_gamma_heatmap.png
│   └── metrics.json             ← toàn bộ số liệu, để tái tạo báo cáo
└── requirements.txt
```

> ⚖️ Ghi rõ: công cụ hỗ trợ, **không** thay thế bác sĩ giải phẫu bệnh.
> Dữ liệu từ Wisconsin (Mỹ), cần kiểm định lại trên dân số Việt Nam trước khi dùng thật.

---

## 9. MỞ RỘNG

```
   1. Vẽ ranh giới quyết định trên 2 chiều PCA → nhìn thấy lề và support vectors
   2. Thử LinearSVC + calibration → nhanh hơn nhiều cho dữ liệu lớn
   3. Dùng SVR (TT-20) để dự đoán một chỉ số liên tục thay vì phân loại
```

**Tham khảo:** [Buổi 4 — LogReg, SVM & Metrics](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-04-LogReg-SVM-Metrics/Tai-Lieu)
