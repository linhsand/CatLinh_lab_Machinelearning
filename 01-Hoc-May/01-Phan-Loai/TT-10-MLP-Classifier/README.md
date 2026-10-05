# TT-10 — MLP CLASSIFIER (sklearn)
## Đọc số viết tay trên séc/phiếu chuyển khoản

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 7](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-07-Neural-Network) |
| 🧠 **Nhóm** | Mạng nơ-ron · Phân loại đa lớp |
| 🔧 **Thuật toán** | Multi-Layer Perceptron (MLPClassifier) |
| 🏭 **Lĩnh vực** | Ngân hàng · Số hoá chứng từ |
| ⏱ **Thời lượng** | 6–8 giờ |
| 📈 **Độ khó** | ⭐⭐⭐ |

> **Kết quả chính (MNIST đầy đủ, test chấm 1 lần):** MLP `(256,128,64)` tanh — **accuracy test 0,9821**
> (baseline LogReg 0,9241 · CNN nhỏ 0,9887). Ở ngưỡng tin cậy đủ an toàn, hệ thống giảm nhân sự
> nhập séc **27 → 18 người** (MLP) hoặc **27 → 16 người** (CNN) mà số séc sai/ngày **không tăng**.
> Ngưỡng 0,99 của đề **không an toàn** với MLP: 238,6 séc sai/ngày, gấp 6 lần hiện tại.

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   Input        Hidden 1      Hidden 2      Hidden 3     Output
   (784 px)     (256 tanh)    (128 tanh)    (64 tanh)    (10 softmax)
     ○ ─────────── ○ ──────────── ○ ──────────── ○ ──────── ○  "0"
     ...           ...            ...            ...          ○  "9"

   Mỗi neuron: z = Σwᵢxᵢ + b  →  a = f(z)
   Học bằng BACKPROPAGATION: lan truyền sai số ngược, cập nhật w.
```

**Điểm mấu chốt:** không có hàm phi tuyến thì 100 tầng cũng chỉ mạnh bằng 1 tầng tuyến
tính — vì tích của nhiều ma trận vẫn là một ma trận.

---

## 2. BÀI TOÁN THỰC TẾ

```
   Ngân hàng nhận 8.000 séc/ngày phải nhập tay số tiền.
   1 nhân viên nhập được 300 séc/ngày → cần 27 người.
   Tỉ lệ nhập sai của người: ~0,5% số séc → ~40 séc sai/ngày.

   ⚠️ Sai 1 chữ số trong số tiền là SAI TIỀN THẬT.
      → Model phải trả về ĐỘ TIN CẬY; ca không đủ chắc → chuyển người (human-in-the-loop).
```

---

## 3. BỘ DỮ LIỆU & CÁCH CHIA

| | |
|---|---|
| **Khởi động** | `load_digits()` — 1.797 ảnh 8×8, pixel [0,16] → chia **/16** |
| **Chính** | MNIST `fetch_openml('mnist_784')` — 70.000 ảnh 28×28, pixel [0,255] → chia **/255** |
| **Dữ liệu** | Không commit vào git (tự cache ở `~/scikit_learn_data/`) — xem [data/DATA_SOURCE.md](data/DATA_SOURCE.md) |

Dùng **toàn bộ** MNIST, không lấy mẫu con:

| Tập | Số ảnh | Dùng để |
|---|---|---|
| Train | 50.000 (stratify từ 60.000 train chuẩn) | huấn luyện |
| **Validation** | 10.000 (phần còn lại của 60.000 train chuẩn) | **MỌI lựa chọn**: kiến trúc, activation, learning rate, ngưỡng HITL, số epoch CNN |
| **Test** | 10.000 (test chuẩn MNIST) | chấm **1 lần** ở cuối, sau khi mọi thứ đã cố định |

---

## 4. CẤU HÌNH CUỐI (chọn trên validation)

```python
Pipeline([
    ("scale", FunctionTransformer(partial(np.multiply, 1/255))),   # nhận ảnh THÔ 0–255
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),   # ← chọn theo val (mục 5.3), không hard-code
        activation="tanh",                   # ← chọn theo val (mục 5.4)
        learning_rate_init=1e-3,             # ← chọn theo val (mục 5.5)
        solver="adam", alpha=1e-4, batch_size=128, max_iter=100,
        early_stopping=True, n_iter_no_change=10, random_state=42,
    )),
])
# Tự động khi max(predict_proba) >= 0,99999  ← ngưỡng chọn trên val (mục 5.8)
```

* `/255` nằm **trong** Pipeline → `models/mlp_pipeline.joblib` nhận thẳng ảnh 0–255; dạng
  `partial(np.multiply, ...)` nên `joblib.load` được ở bất kỳ đâu mà không cần import `train.py`
  (đã kiểm tra: nạp lại trong tiến trình mới → test 0,9821).
* Số tham số: 784×256+256 + 256×128+128 + 128×64+64 + 64×10+10 = **242.762**.
* Tinh chỉnh **tuần tự** kiến trúc → activation → lr (không phải grid đầy đủ 4×3×3).

---

## 5. KẾT QUẢ THỰC NGHIỆM

Số liệu dưới đây là **kết quả chạy thật** (`notebooks/mlp_digits.ipynb` gọi các hàm trong
`src/train.py`, `random_state=42`), lưu tại `reports/` và `reports/tom_tat.json`.

### 5.1. Khởi động & baseline

| | Accuracy |
|---|---|
| `load_digits()` + MLP(64), chia /16 | 0,9806 |
| Logistic Regression trên MNIST (val / test) | 0,9223 / 0,9241 |

### 5.2. Có / không chuẩn hoá — `reports/so_sanh_chuan_hoa.csv` (val, cấu hình `(128,64)` relu)

| | acc val | epoch | loss cuối | thời gian |
|---|---|---|---|---|
| KHÔNG chuẩn hoá (0–255) | 0,9692 | 71 | 0,0274 | 1.199 s |
| **CÓ chuẩn hoá (/255)** | **0,9756** | **22** | **0,0069** | **235 s** |

Chuẩn hoá thắng mọi cột: +0,64 điểm %, ít epoch hơn 3 lần, nhanh hơn 5 lần, loss cuối thấp hơn
4 lần. Mạng *không* chết hẳn như đề dự đoán — `adam` tự chuẩn hoá độ lớn bước cập nhật và ReLU
không bão hoà phía dương — nhưng học chậm và kém hơn rõ.

### 5.3. 4 kiến trúc — `reports/kien_truc_comparison.csv`, `loss_curves.png` (val)

| Kiến trúc | Số tham số | acc val | Epoch | Thời gian train |
|---|---|---|---|---|
| (64) | 50.890 | 0,9714 | 36 | 223 s |
| (128) | 101.770 | 0,9781 | 53 | 615 s |
| (128,64) | 109.386 | 0,9756 | 22 | 235 s |
| **(256,128,64)** ✅ | 242.762 | **0,9794** | 75 | 770 s |

* Chọn tự động bằng `argmax(acc_val)` → **(256,128,64)**. Nó chỉ hơn (128) 0,13 điểm %
  (≈13 ảnh) nhưng nặng gấp 2,4 lần; với 1 seed, các chênh lệch dưới ~0,2 điểm % có thể chỉ là nhiễu.
* **Đường loss:** training loss mọi kiến trúc xuống ~1e-3, trong khi accuracy val nội bộ đi ngang
  ở 0,97–0,985 từ epoch ~15–20. Khoảng cách này là dấu hiệu **bắt đầu overfit** (50.000 mẫu cho
  50k–243k tham số, rất xa quy tắc 10 mẫu/tham số) → `early_stopping` + L2 là cần thiết.
  Loss có các gai (vd (128) ở epoch 51: 1e-3 → 1,8e-2): khi loss rất nhỏ, bước Adam trở nên
  tương đối lớn và văng khỏi đáy.

### 5.4. 3 activation — `reports/activation_comparison.csv` (val, kiến trúc đã chọn)

| Activation | acc val | Loss sau epoch 1 | Epoch | Loss cuối |
|---|---|---|---|---|
| relu | 0,9794 | 0,317 | 75 | 0,0015 |
| **tanh** ✅ | **0,9808** | 0,330 | 50 | 0,0009 |
| logistic | 0,9725 | **1,115** | 25 | 0,0064 |

**Vanishing gradient:** đạo hàm sigmoid ≤ 0,25 → qua 3 tầng ẩn tín hiệu ngược còn ≤ 0,25³ ≈ 1,6%.
Thực đo: sau epoch đầu, loss của `logistic` vẫn là **1,115**, gấp ~3,4 lần relu/tanh, và
accuracy cuối thấp nhất (kém tanh 0,83 điểm %). `tanh` (đạo hàm tối đa 1, đối xứng quanh 0)
nhỉnh hơn `relu` 0,14 điểm % trên mạng 3 tầng này.

### 5.5. 3 learning rate — `reports/learning_rate_comparison.csv`, `learning_rate_curves.png` (val)

| learning_rate_init | acc val | Epoch | Loss cuối |
|---|---|---|---|
| 1e-2 | 0,9457 | 13 | 0,2186 |
| **1e-3** ✅ | **0,9808** | 50 | 0,0009 |
| 1e-4 | 0,9751 | 60 | 0,0018 |

1e-2 quá lớn: loss kẹt ở 0,22 (gấp ~240 lần), dao động, tụt 3,5 điểm %. 1e-4 quá nhỏ: 60 epoch
vẫn kém 0,57 điểm %. *(Cột thời gian của dòng 1e-4 trong CSV không hợp lệ — máy ngủ giữa lúc train.)*

### 5.6. ⭐ Test — chấm 1 lần — `reports/ket_qua_test.csv`

| Model | Số tham số | **Accuracy test** | Ảnh sai / 10.000 |
|---|---|---|---|
| Logistic Regression | 7.850 | 0,9241 | 759 |
| **MLP (256,128,64) tanh** | 242.762 | **0,9821** ✅ ≥ 0,96 | 179 |
| CNN nhỏ (2 conv + 1 dense, PyTorch) | 225.034 | **0,9887** | 113 |

Test (0,9821) ≈ val (0,9808) → không có dấu hiệu lạc quan do chọn mô hình. MLP giảm lỗi 76% so
với LogReg. CNN nhỏ hơn MLP mà sai ít hơn 37%.

### 5.7. Ma trận nhầm lẫn 10×10 — `confusion_10x10.png`, `cap_nham_lan*.csv`, `anh_sai.png` (test)

| Cặp (gộp 2 chiều) | 4↔9 | 2↔7 | 7↔9 | 2↔8 | 5↔8 | … 3↔5 |
|---|---|---|---|---|---|---|
| Số lần nhầm | **20** | 12 | 11 | 8 | 8 | 6 |

* Đúng gợi ý 4↔9 của đề. Theo 1 chiều nhiều nhất là 9→4 (12 lần), 4→9 (8), 7→2 (8).
* Recall thấp nhất ở chữ số 9 (97,2%), 8 (97,6%) và 5 (97,6%). Cao nhất là 0 (99,5%).
* **20 ảnh sai:** phần lớn người vẫn đọc được (7 có gạch ngang → đoán 2, 9 nét mảnh → đoán 4).
  Vài ảnh mơ hồ thật (6 viết như "l", 8 viết thành khối). **7/20 ảnh sai có độ tin cậy ≥ 0,99**,
  vài ảnh ở mức 1,00, tức model sai mà rất tự tin.

### 5.8. ⭐ Human-in-the-loop — `chon_nguong_val.csv`, `human_in_the_loop.csv`

**Mục tiêu đặt ra:** séc tự động không được sai nhiều hơn người (0,5% số séc). Séc có ~7 chữ số
và chỉ đúng khi cả 7 đều đúng, nên mỗi chữ số tự động phải đúng ≥ 0,995^(1/7) ≈ **99,93%**.
**Ngưỡng được chọn trên VAL:** đó là ngưỡng nhỏ nhất đạt mục tiêu trên.

* MLP: **không ngưỡng nào trên val đạt 99,93%** (tốt nhất 99,83% ở 0,99999), nên dùng ngưỡng
  dự phòng cao nhất là **0,99999**.
* CNN: đạt mục tiêu ở ngưỡng **0,999** (val 99,966%).

Kết quả trên **test** (mức chữ số):

| Model | Ngưỡng | % tự động | % chuyển người | Accuracy phần tự động | Lỗi lọt qua |
|---|---|---|---|---|---|
| MLP | 0,99 (đề) | 96,6% | 3,4% | 99,46% | 52 |
| MLP | **0,99999** (val) | 85,2% | 14,8% | **99,965%** | 3 |
| CNN | 0,99 (đề) | 94,5% | 5,5% | 99,88% | 11 |
| CNN | **0,999** (val) | 87,8% | 12,2% | **99,94%** | 5 |

### 5.9. ⭐ Chi phí nghiệp vụ — `chi_phi_hitl.csv`, `chi_phi_theo_so_chu_so.csv`

Mô phỏng 200.000 séc × 7 chữ số lấy từ test. Giả định 8.000 séc/ngày, 300 séc/người/ngày, người
sai 0,5% số séc. **Chế độ A:** séc chỉ tự động khi cả 7 chữ số đều ≥ ngưỡng, ngược lại cả séc
chuyển sang người.

| Phương án | Séc tự động | Séc sang người/ngày | **Nhân viên** | **Séc sai/ngày** |
|---|---|---|---|---|
| 100% người (hiện tại) | 0% | 8.000 | 27 | 40,0 |
| MLP @0,99 (đề) | 78,6% | 1.715 | 6 | **238,6** ❌ |
| MLP @0,99999 | 32,6% | 5.394 | **18** (−33%) | 33,6 ✅ |
| CNN @0,99 | 67,3% | 2.614 | 9 | 55,3 ❌ |
| CNN @0,999 | 40,3% | 4.774 | **16** (−41%) | 37,5 ✅ |

* **"% chuyển người" tính theo chữ số dễ đánh lừa:** MLP @0,99 chỉ chuyển 3,4% chữ số, nhưng
  phải chuyển 21,4% số séc. Ngược lại, 3,7% séc được tự động lại sai tiền.
* **Chế độ B** (người chỉ gõ lại các chữ số bị gắn cờ) cần ít người hơn hẳn (MLP 5, CNN 4)
  nhưng nhiều séc sai hơn (44,3 / 51,7 séc/ngày), vì lỗi của các chữ số tự động cộng dồn.
* **Độ nhạy theo độ dài số tiền** (MLP @0,99999): 4 chữ số → 52,7% séc tự động, 13 người;
  9 chữ số → 23,7% séc tự động, 21 người. Lợi ích giảm theo hàm mũ ≈ pᵏ.
* Giả định các chữ số độc lập là bi quan: trên séc thật cả dãy do một người viết, nên tỉ lệ
  tự động thực tế có thể cao hơn. Cần hiệu chỉnh lại trên séc thật.

### 5.10. So sánh CNN & độ bền dịch/xoay — `do_ben_dich_xoay.csv/png`, `cnn_epochs.csv` (test)

CNN: `Conv(32)→Pool→Conv(64)→Pool→Dense(128)→10`, Adam lr=1e-3, epoch tốt nhất theo val = 8.

| Biến đổi ảnh test | MLP | CNN |
|---|---|---|
| Gốc | 0,9821 | 0,9887 |
| Dịch phải 2px | 0,8499 | 0,9626 |
| Dịch phải 3px | **0,5773** | 0,8956 |
| Dịch chéo 3px | 0,1905 | 0,4068 |
| Xoay 10° | 0,9708 | 0,9795 |
| Xoay 20° | 0,9191 | 0,9435 |

Dịch 3px làm **MLP mất 40 điểm %**, còn CNN chỉ mất 9. Đây là bằng chứng thực nghiệm cho việc
MLP làm mất cấu trúc không gian. Tích chập dùng chung bộ lọc ở mọi vị trí, và pooling cho
bất biến cục bộ.

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☑ Có bảng so sánh CÓ/KHÔNG chuẩn hoá               → 5.2: 0,9692 vs 0,9756 (val)
   ☑ Có bảng so sánh ≥ 4 kiến trúc kèm SỐ THAM SỐ và thời gian → 5.3
   ☑ Có biểu đồ loss_curve_ và phân tích              → 5.3, reports/loss_curves.png
   ☑ Có so sánh 3 activation + giải thích vanishing gradient → 5.4 (logistic loss ep1 = 1,115)
   ☑ Ma trận nhầm lẫn 10×10 + phân tích cặp số hay nhầm → 5.7 (4↔9: 20 lần)
   ☑ ⭐ Có bảng human-in-the-loop: % tự động vs % cần người ở ngưỡng 99% → 5.8 + chi phí 5.9
   ☑ Accuracy test ≥ 0,96                              → 0,9821
   ☑ Nêu được hạn chế: MLP làm MẤT cấu trúc không gian → 5.10 (dịch 3px: 0,98 → 0,58)
```

**Các bước đề yêu cầu:**
```
   ☑ 1. load_digits() (/16)            ☑ 8. 3 activation
   ☑ 2. MNIST đầy đủ, /255             ☑ 9. 3 learning_rate
   ☑ 3. Baseline LogReg (0,9241)       ☑ 10. Ma trận nhầm lẫn 10×10
   ☑ 4. MLP KHÔNG chuẩn hoá            ☑ 11. 20 ảnh sai
   ☑ 5. MLP có chuẩn hoá               ☑ 12. Human-in-the-loop + chi phí
   ☑ 6. 4 kiến trúc                    ☑ 13. So sánh CNN (huấn luyện thật, 0,9887)
   ☑ 7. Đường loss theo epoch          ☑ Mở rộng 2: dịch/xoay ảnh
```

---

## 7. CẠM BẪY — ĐÃ KIỂM CHỨNG

| Cạm bẫy | Kiểm chứng trong bài |
|---|---|
| Quên chia 255 | acc val 0,9692 so với 0,9756, cần 71 epoch thay vì 22 |
| Không `early_stopping` | Train loss ~1e-3 trong khi val đi ngang ở ~0,98 → sẽ overfit tiếp |
| `learning_rate` quá lớn | lr=1e-2: loss kẹt ở 0,22, acc 0,9457 |
| Activation `logistic` | Loss sau epoch 1 là 1,115 (relu/tanh ~0,32), acc thấp nhất |
| Bỏ qua ngưỡng tin cậy / dùng ngưỡng 0,99 cố định | 238,6 séc sai/ngày, gấp 6 lần người nhập |
| **Chọn siêu tham số / ngưỡng trên TEST** | Mọi lựa chọn làm trên val. Test chỉ dùng 1 lần, test 0,9821 ≈ val 0,9808 |

---

## 8. HẠN CHẾ

1. MLP **mất cấu trúc không gian** và không bất biến dịch chuyển (mục 5.10). Khi triển khai
   phải căn giữa ảnh như MNIST, hoặc dùng CNN.
2. MLP **không đạt** mục tiêu "séc sai ≤ người" trên val ở bất kỳ ngưỡng nào; ngưỡng 0,99999
   là ngưỡng dự phòng. CNN đạt mục tiêu.
3. Tinh chỉnh tuần tự và chỉ 1 seed: các chênh lệch dưới ~0,2 điểm % giữa cấu hình chưa có
   ý nghĩa thống kê.
4. Mô phỏng chi phí giả định chữ số độc lập và 7 chữ số/séc. Phân bố ảnh séc thật (giấy,
   máy quét, nét chữ) khác MNIST, nên cần hiệu chỉnh lại ngưỡng.
5. CNN ở đây chỉ là bản nhỏ (0,9887, chưa đạt >0,99). Augmentation, BatchNorm và Dropout
   thuộc TT-25/TT-26.

---

## 9. SẢN PHẨM & CÁCH CHẠY

```
TT-10-MLP-Classifier/
├── README.md
├── data/DATA_SOURCE.md          ← nguồn & cách chia dữ liệu (MNIST không commit)
├── notebooks/mlp_digits.ipynb   ← giải thích từng bước + output thật
├── src/train.py                 ← toàn bộ pipeline (notebook gọi lại các hàm này)
├── models/mlp_pipeline.joblib   ← Pipeline(/255 → MLP), nhận ảnh thô 0–255
├── reports/                     ← CSV/PNG từng bảng + tom_tat.json
└── requirements.txt             ← có torch (bản CPU) cho phần CNN
```

```bash
pip install -r requirements.txt   # torch CPU: --index-url https://download.pytorch.org/whl/cpu
python src/train.py               # ~1–1,5 giờ trên CPU (LogReg + 9 lần fit MLP trên 50.000 ảnh + CNN)
```

```python
import joblib
model = joblib.load("models/mlp_pipeline.joblib")
proba = model.predict_proba(X_raw_0_255)            # (n, 10)
tu_dong = proba.max(axis=1) >= 0.99999               # còn lại → chuyển người
```

**Tham khảo:** [Buổi 7 — Neural Network](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-07-Neural-Network/Tai-Lieu/ly_thuyet_chi_tiet_buoi_07.md)
