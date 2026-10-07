# TT-26 — CNN (CONVOLUTIONAL NEURAL NETWORK)
## Sàng lọc viêm phổi trên ảnh X-quang ngực

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 10](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-10-CNN-ComputerVision) |
| 🧠 **Nhóm** | Deep Learning · Thị giác máy tính |
| 🔧 **Thuật toán** | CNN + Transfer Learning |
| 🏭 **Lĩnh vực** | Y tế · Chẩn đoán hình ảnh |
| ⏱ **Thời lượng** | 8–12 giờ (**cần GPU**) |
| 📈 **Độ khó** | ⭐⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   [Conv → ReLU → Pool] × N  →  Flatten  →  Dense  →  Output
   ├──── TRÍCH XUẤT ĐẶC TRƯNG ────┤        ├─ PHÂN LOẠI ─┤

   Tầng 1 học:  ── │ ╱ ╲        cạnh, đường
   Tầng 2 học:  ╔╗ ○ △          góc, kết cấu
   Tầng 3 học:  vùng mờ, đường viền phổi bất thường
```

**Hai ý tưởng làm nên CNN:** chia sẻ trọng số (1 bộ lọc quét khắp ảnh) và kết nối
cục bộ (mỗi neuron chỉ nhìn 1 vùng nhỏ) → ít tham số hơn MLP hàng trăm lần mà vẫn
giữ được cấu trúc không gian 2D.

---

## 2. BÀI TOÁN THỰC TẾ

```
   Bệnh viện tuyến huyện không có bác sĩ chẩn đoán hình ảnh trực đêm.
   Ảnh X-quang chụp lúc 2h sáng phải chờ tới sáng mới có người đọc.

   → Công cụ SÀNG LỌC tự động: gắn cờ ca nghi ngờ viêm phổi
     để bác sĩ trực ưu tiên xem trước.

   ⚠️ RECALL LÀ TỐI THƯỢNG (≥ 0,97):
      Bỏ sót viêm phổi ở trẻ em có thể dẫn tới tử vong.
      Báo động giả chỉ tốn thêm 1 lần bác sĩ liếc mắt.

   🩺 KHÔNG BAO GIỜ dùng thay bác sĩ — chỉ sắp thứ tự ưu tiên đọc phim.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Chest X-Ray Images (Pneumonia) |
| **Link** | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |
| **Kích thước** | 5.863 ảnh JPEG (~1,2 GB) |
| **Lớp** | NORMAL · PNEUMONIA |
| **Chia sẵn** | train (5.216) · val (**16** ⚠️) · test (624) |

### 🚨 Ba vấn đề nghiêm trọng của bộ dữ liệu này

```
   1. TẬP VALIDATION CHỈ CÓ 16 ẢNH (8 mỗi lớp!)
      → hoàn toàn vô dụng để chọn model, kết quả nhiễu kinh khủng
      → PHẢI tự tách lại validation từ tập train (vd 15%)

   2. MẤT CÂN BẰNG: train có 1.341 NORMAL vs 3.875 PNEUMONIA (~74%)
      → dùng class_weight

   3. PHÂN PHỐI TRAIN ≠ TEST
      Tập test cân bằng hơn (234 NORMAL / 390 PNEUMONIA)
      → accuracy trên train/val sẽ cao hơn test một cách có hệ thống
      → đây là hiện tượng THẬT, phải nêu trong báo cáo
```

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. Luôn bắt đầu bằng Transfer Learning

```
   Train CNN từ đầu với 5.000 ảnh → gần như chắc chắn overfit.
   → Dùng mạng đã học trên ImageNet (1,2 TRIỆU ảnh) rồi thay tầng cuối.
```

```python
import tensorflow as tf

base = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base.trainable = False                          # GIAI ĐOẠN 1: đóng băng

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.AUC(name='auc')])
```

**Giai đoạn 2 — fine-tuning:**
```python
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # ⭐ lr NHỎ hơn 100 lần
              loss='binary_crossentropy', metrics=[...])
```

> ⚠️ Fine-tune với learning rate lớn sẽ **phá huỷ** toàn bộ tri thức ImageNet ngay
> trong vài bước đầu.

### 4.2. Augmentation phải HỢP LÝ VỚI Y TẾ

```
   ✅ Được: xoay nhẹ ±10°, dịch ±10%, phóng to ±10%, đổi độ sáng/tương phản nhẹ
   ❌ CẤM:  lật NGANG (tim nằm bên trái — lật là sai giải phẫu!)
   ❌ CẤM:  lật dọc, biến dạng mạnh
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Tải dữ liệu, ĐẾM ảnh mỗi lớp mỗi tập → phát hiện 3 vấn đề ở mục 3
   ☐ 2. ⭐ Tự tách lại validation 15% từ train (stratify)
   ☐ 3. Hiển thị 8 ảnh mỗi lớp → quan sát bằng mắt sự khác biệt
   ☐ 4. tf.data pipeline: resize 224×224, chuẩn hoá, augmentation (chỉ cho train)
   ☐ 5. Baseline: CNN nhỏ tự xây (3 khối Conv) train từ đầu → ghi kết quả
   ☐ 6. Transfer Learning giai đoạn 1 (đóng băng base) → so sánh
   ☐ 7. Fine-tuning giai đoạn 2 (lr = 1e-5) → so sánh
   ☐ 8. Vẽ learning curves cả 2 giai đoạn
   ☐ 9. Chọn ngưỡng đạt RECALL ≥ 0,97 trên tập validation
   ☐ 10. Đánh giá trên tập TEST 1 lần: ma trận nhầm lẫn
         → ĐẾM CHÍNH XÁC số ca viêm phổi bị BỎ SÓT
   ☐ 11. ⭐ Grad-CAM: vẽ vùng ảnh model chú ý cho 6 ca
         → model có nhìn vào VÙNG PHỔI không, hay nhìn vào chữ/nhãn góc ảnh?
   ☐ 12. Hiển thị 10 ca dự đoán sai → phân tích nguyên nhân
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Đã tự tách lại validation (nêu rõ lý do bộ gốc chỉ có 16 ảnh)
   ☐ Có so sánh: CNN từ đầu vs Transfer Learning vs Fine-tuning
   ☐ Augmentation KHÔNG có lật ngang (nêu lý do giải phẫu)
   ☐ RECALL trên tập test ≥ 0,96
   ☐ Ma trận nhầm lẫn ghi rõ SỐ CA BỎ SÓT
   ☐ ⭐ Có ảnh Grad-CAM + nhận xét model có nhìn đúng vùng phổi không
   ☐ Có phân tích 10 ca sai
   ☐ Có mục CẢNH BÁO Y TẾ trong README
```

**Mức tham chiếu:** Recall ~0,96–0,99 · Accuracy ~0,88–0,93 trên tập test.
Accuracy thấp hơn recall là bình thường do phân phối test khác train.

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Dùng tập val 16 ảnh của bộ gốc | Chọn model theo kết quả ngẫu nhiên |
| Lật ngang ảnh khi augment | Sai giải phẫu, model học sai |
| Fine-tune với lr = 1e-3 | Phá huỷ trọng số ImageNet |
| Train từ đầu với 5.000 ảnh | Overfit nặng |
| Dùng accuracy làm metric chính | Che giấu ca bỏ sót |
| Không kiểm tra Grad-CAM | Model có thể "gian lận" bằng cách nhìn nhãn/ký hiệu trên phim |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-26-CNN-<HoTen>/
├── README.md          ← ⭐ có CẢNH BÁO Y TẾ + số ca bỏ sót
├── notebooks/cnn_chest_xray.ipynb
├── src/{data_pipeline.py, model.py, train.py}
├── models/best_model.keras
├── reports/{learning_curves.png, confusion_matrix.png, gradcam_examples.png, ca_du_doan_sai.png}
└── requirements.txt
```

> ⚖️ **Bắt buộc ghi trong README:** đây là công cụ **sắp thứ tự ưu tiên đọc phim**,
> KHÔNG phải công cụ chẩn đoán. Dữ liệu từ Quảng Châu (Trung Quốc), trên bệnh nhi
> 1–5 tuổi → **không tổng quát hoá** cho người lớn hay dân số khác. Mọi ca đều phải
> có bác sĩ đọc lại.

**Mở rộng:**
1. So sánh 3 kiến trúc: ResNet50 · EfficientNetB0 · DenseNet121 (xem KT-05, KT-06, KT-07)
2. Hiệu chuẩn xác suất (calibration): xác suất model đưa ra có đáng tin không?
3. Ước lượng độ bất định bằng MC Dropout → ca nào model "không chắc" thì chuyển bác sĩ ngay

**Tham khảo:** [Buổi 10 — CNN & Transfer Learning](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-10-CNN-ComputerVision/Tai-Lieu/ly_thuyet_chi_tiet_buoi_10.md)
