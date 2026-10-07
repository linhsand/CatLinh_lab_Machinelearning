# TT-27 — RNN / LSTM
## Dự báo lưu lượng giao thông theo giờ để điều khiển đèn tín hiệu

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 14](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-14-RNN-LSTM-Transformer) |
| 🧠 **Nhóm** | Deep Learning cho dữ liệu CHUỖI |
| 🔧 **Thuật toán** | RNN · **LSTM** · GRU |
| 🏭 **Lĩnh vực** | Giao thông đô thị · Vận hành IT |
| ⏱ **Thời lượng** | 8–10 giờ |
| 📈 **Độ khó** | ⭐⭐⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   MLP xử lý MỖI MẪU ĐỘC LẬP → không có khái niệm "trước/sau".
   RNN có TRẠNG THÁI ẨN h truyền qua từng bước thời gian:

     x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → ŷ
                    ↑ x₂          ↑ x₃
     h mang "ký ức" về những gì đã thấy trước đó.

   ⚠️ RNN thường bị VANISHING GRADIENT: gradient nhân dồn qua nhiều bước
      → tiêu biến → không học được phụ thuộc xa (quá ~10 bước).

   LSTM chữa bằng 3 CỔNG điều khiển dòng thông tin:
     • Cổng QUÊN   — bỏ thông tin cũ không còn cần
     • Cổng VÀO    — nhận thông tin mới nào
     • Cổng RA     — xuất gì ra ngoài
   → giữ được ký ức dài hàng trăm bước.
```

| | RNN | LSTM | GRU |
|---|---|---|---|
| Số tham số | Ít nhất | Nhiều nhất | Trung bình |
| Ký ức dài | ❌ | ✅ | ✅ |
| Tốc độ | Nhanh | Chậm | Nhanh hơn LSTM ~25% |
| Khuyến nghị | Chỉ để học | ⭐ Mặc định | Thử khi cần nhanh |

---

## 2. BÀI TOÁN THỰC TẾ

```
   Sở Giao thông cần dự báo lưu lượng xe qua nút giao 1–6 giờ tới
   để điều chỉnh chu kỳ đèn tín hiệu và phát cảnh báo ùn tắc.

   Dự báo thiếu → đèn không kịp điều chỉnh → ùn tắc kéo dài
   Dự báo thừa → kéo dài đèn xanh vô ích ở hướng vắng

   Đặc thù dữ liệu: có CHU KỲ NGÀY (2 đỉnh giờ cao điểm),
   CHU KỲ TUẦN (cuối tuần khác hẳn), và bị ảnh hưởng bởi THỜI TIẾT + NGÀY LỄ.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Metro Interstate Traffic Volume (UCI) |
| **Link** | https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume |
| **Kích thước** | 48.204 dòng (theo giờ, 2012–2018) |
| **Nhãn** | `traffic_volume` (số xe/giờ, 0–7.280) |

**Cột:** `holiday`, `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `weather_main`,
`weather_description`, `date_time`, `traffic_volume`

### ⚠️ Bốn vấn đề dữ liệu

```
   1. CÓ LỖ HỔNG THỜI GIAN — thiếu nhiều khoảng (không liên tục theo giờ)
      → phải reindex theo giờ đầy đủ rồi xử lý NaN, KHÔNG được coi là liên tục

   2. GIÁ TRỊ NGOẠI LAI PHI LÝ:
      temp = 0 Kelvin (độ không tuyệt đối!) · rain_1h = 9831 mm/giờ
      → phải lọc

   3. DÒNG TRÙNG date_time (do 2 mô tả thời tiết cùng 1 giờ) → khử trùng

   4. `holiday` chỉ đánh dấu ở dòng đầu ngày lễ, không phải cả ngày
      → phải lan ra toàn bộ ngày đó
```

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. Tạo chuỗi cửa sổ trượt

```python
import numpy as np

def tao_chuoi(X, y, do_dai=24, buoc_du_bao=1):
    """do_dai = số giờ quá khứ dùng làm đầu vào"""
    Xs, ys = [], []
    for i in range(len(X) - do_dai - buoc_du_bao + 1):
        Xs.append(X[i:i+do_dai])
        ys.append(y[i+do_dai+buoc_du_bao-1])
    return np.array(Xs), np.array(ys)
# Kết quả: X có shape (số mẫu, 24, số đặc trưng) — đúng định dạng LSTM cần
```

### 4.2. Mô hình

```python
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Input(shape=(24, n_features)),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32),                     # return_sequences=False ở tầng cuối
    layers.Dropout(0.2),
    layers.Dense(1),                     # ⭐ KHÔNG activation (hồi quy)
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
```

### 4.3. ⚠️ Ba nguyên tắc chống rò rỉ cho chuỗi thời gian

```
   ① CHIA THEO THỜI GIAN, tuyệt đối không shuffle khi chia
      (nhưng ĐƯỢC shuffle các CỬA SỔ đã tạo trong lúc train)
   ② Scaler fit CHỈ trên train, transform cho val/test
   ③ Cửa sổ đầu vào chỉ chứa quá khứ — kiểm tra kỹ chỉ số khi cắt
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Làm sạch: khử trùng date_time, lọc temp/rain phi lý, reindex theo giờ
   ☐ 2. Xử lý lỗ hổng: đánh dấu bằng cột cờ, nội suy hoặc cắt thành nhiều đoạn liên tục
   ☐ 3. EDA: lưu lượng TB theo giờ trong ngày · theo thứ · theo tháng
   ☐ 4. Đặc trưng: sin/cos cho giờ và thứ, cờ ngày lễ, one-hot weather_main
   ☐ 5. Chia theo thời gian: 70% train · 15% val · 15% test
   ☐ 6. ⭐ BASELINE BẮT BUỘC (2 cái):
        • Naive: "giờ này = giờ này hôm qua"
        • Seasonal naive: "giờ này = giờ này TUẦN TRƯỚC"
        ⚠️ Baseline thứ 2 rất mạnh — LSTM phải thắng nó mới có giá trị
   ☐ 7. Baseline cây: XGBoost + lag features (TT-31) — thường rất mạnh
   ☐ 8. SimpleRNN → LSTM → GRU: bảng so sánh MAE · thời gian train · số tham số
   ☐ 9. Khảo sát độ dài cửa sổ: 6 / 12 / 24 / 48 giờ → MAE thay đổi thế nào?
   ☐ 10. Vẽ dự báo vs thực tế trên 1 tuần cuối
   ☐ 11. Dự báo NHIỀU BƯỚC: 1h / 3h / 6h tới → MAE tăng bao nhiêu?
   ☐ 12. Phân tích lỗi: sai nhiều nhất vào giờ nào, ngày nào?
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Đã xử lý đủ 4 vấn đề dữ liệu ở mục 3
   ☐ Chia theo thời gian, không rò rỉ
   ☐ ⭐ Có CẢ 2 baseline naive + baseline XGBoost
   ☐ ⭐ LSTM phải THẮNG seasonal naive (nếu không → kết luận trung thực là
        bài này không cần deep learning)
   ☐ Có bảng so sánh RNN vs LSTM vs GRU
   ☐ Có khảo sát độ dài cửa sổ
   ☐ Có biểu đồ dự báo vs thực tế 1 tuần
   ☐ Có kết quả dự báo nhiều bước
```

**Mức tham chiếu:** MAE ~250–400 xe/giờ cho dự báo 1 giờ tới. Nếu XGBoost + lag
features cho kết quả tương đương mà nhanh hơn 20 lần — hãy ghi kết luận đó.

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Coi dữ liệu liên tục dù có lỗ hổng | Cửa sổ chứa 2 thời điểm cách nhau nhiều ngày |
| Shuffle khi CHIA train/test | Rò rỉ tương lai |
| Scaler fit trên toàn bộ dữ liệu | Rò rỉ |
| Không có baseline naive | Không biết LSTM có đáng dùng không |
| Đặt activation ở tầng Dense cuối | Chặn đầu ra hồi quy |
| Cửa sổ quá dài (168 giờ) | Train rất chậm, thường không tốt hơn |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-27-LSTM-<HoTen>/
├── README.md          ← ⭐ có bảng so sánh với baseline naive
├── notebooks/lstm_traffic.ipynb
├── src/{data.py, sequences.py, train.py}
├── models/lstm_traffic.keras
├── reports/{eda_theo_gio.png, rnn_lstm_gru.png, du_bao_vs_thuc_te.png, do_dai_cua_so.png}
└── requirements.txt
```

**Mở rộng:**
1. **Bidirectional LSTM** — hợp lý không? (gợi ý: KHÔNG, vì dự báo tương lai
   thì không được nhìn dữ liệu sau thời điểm hiện tại)
2. **Seq2Seq** dự báo 24 giờ tới cùng lúc thay vì từng giờ
3. So sánh với **Transformer** (TT-28) trên cùng bài toán chuỗi

**Tham khảo:** [Buổi 14 — RNN, LSTM, Transformer](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-14-RNN-LSTM-Transformer/Tai-Lieu) · [Buổi 9 — Time Series](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-09-TimeSeries/Tai-Lieu/ly_thuyet_chi_tiet_buoi_09.md)
