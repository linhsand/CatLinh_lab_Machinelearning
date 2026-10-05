# TT-24 — PCA (PHÂN TÍCH THÀNH PHẦN CHÍNH)
## Nén 561 tín hiệu cảm biến xuống còn vài chục chiều để chạy trên thiết bị đeo

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 5](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-05-Unsupervised-PCA) |
| 🧠 **Nhóm** | **Học KHÔNG giám sát** — Giảm chiều |
| 🔧 **Thuật toán** | PCA (Principal Component Analysis) |
| 🏭 **Lĩnh vực** | IoT · Thiết bị đeo · Giám sát thiết bị công nghiệp |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   PCA tìm các TRỤC MỚI (thành phần chính) theo thứ tự giữ được
   NHIỀU BIẾN ĐỘNG (variance) nhất:

        x₂                        PC1 = trục giữ nhiều biến động nhất
         │    ●●●  ╱PC1           PC2 = trục vuông góc PC1, giữ nhiều thứ nhì
         │  ●●●●╱                 ...
         │ ●●●╱  ╲PC2
         │●●╱      ╲
         └──────────── x₁

   Chiếu dữ liệu lên K trục đầu → giữ được phần lớn thông tin
   với số chiều nhỏ hơn nhiều.
```

⚠️ **PCA KHÔNG phải chọn đặc trưng.** Nó **TẠO ĐẶC TRƯNG MỚI** là tổ hợp tuyến tính
của tất cả đặc trưng cũ. Hệ quả: PC1 **không còn ý nghĩa vật lý** → mất tính giải thích.

---

## 2. BÀI TOÁN THỰC TẾ

```
   Vòng đeo tay theo dõi sức khoẻ thu 561 chỉ số từ gia tốc kế + con quay hồi chuyển
   để nhận biết người đang: đi bộ · lên cầu thang · đứng · ngồi · nằm.

   RÀNG BUỘC PHẦN CỨNG:
     • Chip chỉ có 64 KB RAM
     • Pin phải trụ 7 ngày
     • Xử lý 561 chiều mỗi giây → tốn pin, không đủ bộ nhớ

   → Nén xuống ~50 chiều mà vẫn giữ được độ chính xác nhận dạng
   → Tiết kiệm ~90% chi phí tính toán và truyền dữ liệu.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Human Activity Recognition Using Smartphones (UCI) |
| **Link** | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones |
| **Kích thước** | 10.299 dòng × **561 đặc trưng** |
| **Nhãn** | 6 hoạt động (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) |

**Vì sao chọn bộ này:** 561 chiều là **số chiều thật sự lớn**, các cột lại tương quan
rất mạnh (đều dẫn xuất từ cùng vài tín hiệu gốc) → PCA phát huy tối đa tác dụng.

### ⚠️ Lưu ý

```
   1. Dữ liệu đã được chuẩn hoá sẵn về [-1, 1] trong bộ gốc.
      → Vẫn NÊN StandardScaler để mọi cột có phương sai 1
        (PCA cực nhạy với thang đo — cột phương sai lớn sẽ chiếm hết PC1).

   2. Dữ liệu chia theo NGƯỜI (21 người train, 9 người test).
      → KHÔNG được trộn rồi chia lại ngẫu nhiên: cùng 1 người ở cả train và test
        là rò rỉ (model học đặc điểm cá nhân thay vì học hoạt động).
```

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. Chọn số thành phần bằng phương sai tích luỹ

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

X_scaled = StandardScaler().fit_transform(X_train)     # ⭐ fit CHỈ trên train

pca_full = PCA().fit(X_scaled)
tich_luy = np.cumsum(pca_full.explained_variance_ratio_)

for nguong in [0.80, 0.90, 0.95, 0.99]:
    k = np.argmax(tich_luy >= nguong) + 1
    print(f"Giữ {nguong:.0%} phương sai → cần {k} thành phần "
          f"(giảm {(1 - k/X.shape[1]):.1%} số chiều)")
```

### 4.2. PCA phải nằm TRONG Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('pca',   PCA(n_components=0.95, random_state=42)),   # giữ 95% phương sai
    ('clf',   LinearSVC(max_iter=5000, random_state=42)),
])
```

> ⚠️ `PCA().fit()` trên toàn bộ dữ liệu rồi mới chia train/test là **rò rỉ**:
> các trục chính đã "nhìn thấy" tập test.

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Nạp dữ liệu, GIỮ NGUYÊN cách chia train/test theo người của bộ gốc
   ☐ 2. Chuẩn hoá (fit trên train)
   ☐ 3. PCA đầy đủ → vẽ SCREE PLOT (phương sai từng thành phần)
   ☐ 4. Vẽ đường phương sai TÍCH LUỸ, đánh dấu mốc 80/90/95/99%
   ☐ 5. Lập bảng: ngưỡng phương sai | số chiều cần | % giảm chiều
   ☐ 6. ⭐ THÍ NGHIỆM CHÍNH — đánh đổi số chiều vs độ chính xác:
        với K ∈ {2, 10, 25, 50, 100, 200, 561}, train LinearSVC
        → vẽ 2 đường: accuracy và thời gian train theo K
        → tìm ĐIỂM NGỌT: giảm chiều tối đa mà accuracy giảm < 1%
   ☐ 7. Vẽ scatter PC1–PC2 tô màu theo 6 hoạt động
        → 6 hoạt động có tách được chỉ với 2 chiều không?
   ☐ 8. Phân tích PC1: xem `components_[0]`, 10 đặc trưng gốc đóng góp nhiều nhất
        → PC1 đại diện cho khái niệm vật lý gì? (thường là "đang chuyển động hay tĩnh")
   ☐ 9. ⭐ Đo dung lượng: X gốc bao nhiêu MB, sau PCA còn bao nhiêu MB?
   ☐ 10. Tái tạo ngược (inverse_transform) → đo sai số tái tạo theo K
   ☐ 11. So sánh với các cách giảm chiều khác: SelectKBest, t-SNE, UMAP
         (lưu ý: t-SNE/UMAP chỉ để TRỰC QUAN HOÁ, không dùng làm tiền xử lý cho model)
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Giữ đúng cách chia theo người (không trộn ngẫu nhiên)
   ☐ Có scree plot + đường phương sai tích luỹ
   ☐ Có bảng ngưỡng phương sai → số chiều
   ☐ ⭐ Có biểu đồ đánh đổi accuracy vs số chiều + chỉ ra ĐIỂM NGỌT
   ☐ Có scatter PC1–PC2 tô màu 6 lớp
   ☐ Có phân tích ý nghĩa PC1
   ☐ ⭐ Có số liệu tiết kiệm dung lượng/thời gian cụ thể
   ☐ Nêu hạn chế: mất tính giải thích, PCA chỉ bắt quan hệ TUYẾN TÍNH
```

**Mức tham chiếu:** khoảng 60–70 thành phần giữ được ~95% phương sai (giảm ~88% số
chiều), accuracy giảm không đáng kể (< 1%).

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Không chuẩn hoá trước PCA | Cột phương sai lớn chiếm trọn PC1 |
| `PCA.fit()` trên toàn bộ dữ liệu | Rò rỉ |
| Trộn rồi chia lại theo dòng | Cùng 1 người ở cả train/test → rò rỉ danh tính |
| Kỳ vọng PC1 có ý nghĩa vật lý | PC là tổ hợp tuyến tính, thường không diễn giải trực tiếp |
| Dùng t-SNE/UMAP làm tiền xử lý | Chúng không có `transform` ổn định cho dữ liệu mới |
| Giảm chiều rồi mới chuẩn hoá | Sai thứ tự |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-24-PCA-<HoTen>/
├── README.md          ← có biểu đồ đánh đổi + số liệu tiết kiệm
├── notebooks/pca_har_sensors.ipynb
├── src/{pca_pipeline.py}
├── models/pca_pipeline.joblib
├── reports/{scree_plot.png, variance_tich_luy.png, danh_doi_chieu_accuracy.png, pc1_pc2_scatter.png}
└── requirements.txt
```

**Mở rộng:**
1. **Kernel PCA** (RBF) — bắt được quan hệ phi tuyến, có tốt hơn PCA thường không?
2. **IncrementalPCA** — xử lý dữ liệu không vừa RAM (mô phỏng luồng cảm biến thật)
3. Dùng PCA để **phát hiện bất thường**: điểm có sai số tái tạo cao = hoạt động lạ
   → nối sang bài VAE (KT-19) cùng ý tưởng

**Tham khảo:** [Buổi 5 — PCA](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-05-Unsupervised-PCA/Tai-Lieu)
