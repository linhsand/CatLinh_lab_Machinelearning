# TT-10 — MLP CLASSIFIER (sklearn)
## Đọc số viết tay trên séc / phiếu chuyển khoản

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 7](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-07-Neural-Network) |
| 🧠 **Nhóm** | Mạng nơ-ron · Phân loại đa lớp |
| 🔧 **Thuật toán** | Multi-Layer Perceptron (MLPClassifier) |
| 🏭 **Lĩnh vực** | Ngân hàng · Số hoá chứng từ |
| ⏱ **Thời lượng** | 6–8 giờ |
| 📈 **Độ khó** | ⭐⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   Input        Hidden 1      Hidden 2      Output
   (64 px)      (128 ReLU)    (64 ReLU)     (10 softmax)
     ○ ─────────── ○ ──────────── ○ ────────── ○  "0"
     ○ ─────────── ○ ──────────── ○ ────────── ○  "1"
     ○ ─────────── ○ ──────────── ○ ────────── ○  ...
     ...           ...            ...            ○  "9"

   Mỗi neuron: z = Σwᵢxᵢ + b  →  a = ReLU(z)
   Học bằng BACKPROPAGATION: lan truyền sai số ngược, cập nhật w.
```

**Điểm mấu chốt:** không có hàm phi tuyến (ReLU) thì 100 tầng cũng chỉ mạnh bằng
1 tầng tuyến tính — vì tích của nhiều ma trận vẫn là một ma trận.

---

## 2. BÀI TOÁN THỰC TẾ

```
   Ngân hàng nhận 8.000 séc/ngày phải nhập tay số tiền.
   1 nhân viên nhập được 300 séc/ngày → cần 27 người.
   Tỉ lệ nhập sai của người: ~0,5%.

   → Tự động đọc dãy số → giảm nhân sự, giảm sai sót.

   ⚠️ Ràng buộc nghiệp vụ: sai 1 chữ số trong số tiền là SAI TIỀN THẬT.
      → Model phải trả về ĐỘ TIN CẬY. Nếu tin cậy < 99% → chuyển cho người kiểm tra.
      → Đây gọi là "human-in-the-loop", bắt buộc trong nghiệp vụ tài chính.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Khởi động** | `sklearn.datasets.load_digits()` — 1.797 ảnh 8×8, có sẵn |
| **Chính** | MNIST — 70.000 ảnh 28×28 |
| **Cách lấy MNIST** | `fetch_openml('mnist_784', version=1, as_frame=False)` |
| **Nhãn** | 10 lớp (chữ số 0–9), khá cân bằng |

```python
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0                      # ⭐ BẮT BUỘC chuẩn hoá về [0,1]
y = y.astype(int)
```

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. Chuẩn hoá là bắt buộc

```
   Pixel 0–255 chưa chuẩn hoá → z = Σwx + b rất lớn
   → hàm kích hoạt BÃO HOÀ → đạo hàm ≈ 0 → mạng ĐỨNG YÊN không học.
   → Chia 255 hoặc dùng StandardScaler.
```

### 4.2. Cấu hình

```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),   # hình PHỄU: to → nhỏ
    activation='relu',              # mặc định tốt cho tầng ẩn
    solver='adam',
    alpha=1e-4,                     # regularization L2
    batch_size=128,
    learning_rate_init=1e-3,
    max_iter=100,
    early_stopping=True,            # ⭐ tự tách 10% validation
    n_iter_no_change=10,
    random_state=42, verbose=True,
)
```

### 4.3. Đếm tham số — biết mạng "nặng" cỡ nào

```
   784 → 128 → 64 → 10:
     784×128 + 128 = 100.480
     128×64  +  64 =   8.256
      64×10  +  10 =     650
                     ─────────
              TỔNG  = 109.386 tham số

   Quy tắc kinh nghiệm: cần ~10 mẫu dữ liệu cho mỗi tham số để tránh overfit nặng.
   60.000 mẫu / 109.386 tham số → vẫn cần regularization.
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Khởi động với load_digits() (nhanh) → hiểu quy trình
   ☐ 2. Chuyển sang MNIST, chuẩn hoá /255
   ☐ 3. Baseline: Logistic Regression → thường đạt ~92%
   ☐ 4. ⚠️ Chạy MLP KHÔNG chuẩn hoá → ghi lại (thường tệ hoặc không hội tụ)
   ☐ 5. MLP có chuẩn hoá → so sánh
   ☐ 6. Thử 4 kiến trúc: (64), (128), (128,64), (256,128,64)
        → bảng: accuracy · số tham số · thời gian train
   ☐ 7. Vẽ đường loss theo epoch (mlp.loss_curve_) → chẩn đoán
   ☐ 8. Thử 3 activation: relu / tanh / logistic → giải thích chênh lệch
        (gợi ý: logistic gây vanishing gradient)
   ☐ 9. Thử 3 learning_rate: 1e-2 / 1e-3 / 1e-4 → vẽ 3 đường loss chồng nhau
   ☐ 10. Ma trận nhầm lẫn 10×10 → cặp số nào hay bị nhầm nhất? (thường 4↔9, 3↔5)
   ☐ 11. Hiển thị 20 ảnh bị dự đoán SAI → người có đọc được không?
   ☐ 12. ⭐ Cơ chế human-in-the-loop: đặt ngưỡng tin cậy 99%
         → bao nhiêu % séc tự động được, bao nhiêu % phải chuyển người?
   ☐ 13. So sánh với CNN (TT-26) → chênh lệch bao nhiêu, vì sao?
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Có bảng so sánh CÓ/KHÔNG chuẩn hoá
   ☐ Có bảng so sánh ≥ 4 kiến trúc kèm SỐ THAM SỐ và thời gian
   ☐ Có biểu đồ loss_curve_ và phân tích
   ☐ Có so sánh 3 activation + giải thích vanishing gradient
   ☐ Ma trận nhầm lẫn 10×10 + phân tích cặp số hay nhầm
   ☐ ⭐ Có bảng human-in-the-loop: % tự động vs % cần người ở ngưỡng 99%
   ☐ Accuracy test ≥ 0,96
   ☐ Nêu được hạn chế: MLP làm MẤT cấu trúc không gian của ảnh
```

**Mức tham chiếu:** MLP đạt ~0,97–0,98 trên MNIST. CNN (TT-26) đạt >0,99.

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Quên chia 255 | Mạng không hội tụ |
| Không `early_stopping` | Overfit, tốn thời gian |
| Mạng quá to với dữ liệu ít | Overfit nặng |
| `learning_rate` quá lớn | Loss dao động hoặc thành NaN |
| Dùng activation `logistic` cho tầng ẩn | Vanishing gradient, học rất chậm |
| Bỏ qua ngưỡng tin cậy | Không dùng được trong nghiệp vụ tài chính |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-10-MLP-<HoTen>/
├── README.md          ← có bảng human-in-the-loop
├── notebooks/mlp_digits.ipynb
├── src/train.py
├── models/mlp_pipeline.joblib
├── reports/{loss_curves.png, kien_truc_comparison.png, confusion_10x10.png, anh_sai.png}
└── requirements.txt
```

**Mở rộng:**
1. Vẽ trọng số tầng đầu dưới dạng ảnh 28×28 → mạng "nhìn" thấy gì?
2. Thử làm nhiễu ảnh (xoay 10°, dịch 3px) → accuracy tụt bao nhiêu?
   (chứng minh MLP KHÔNG bất biến dịch chuyển → lý do cần CNN)
3. Chuyển sang Keras (TT-25) để thêm Dropout và BatchNorm

**Tham khảo:** [Buổi 7 — Neural Network](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-07-Neural-Network/Tai-Lieu/ly_thuyet_chi_tiet_buoi_07.md)
