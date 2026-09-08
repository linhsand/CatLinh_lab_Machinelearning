# TT-22 — MLP REGRESSOR
## Dự đoán mức tiêu hao nhiên liệu để tư vấn khách chọn xe

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao) |
| 🧠 **Nhóm** | Mạng nơ-ron cho hồi quy |
| 🔧 **Thuật toán** | MLPRegressor (sklearn) |
| 🏭 **Lĩnh vực** | Ô tô · Tư vấn tiêu dùng |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

Giống MLP phân loại (TT-10) nhưng tầng output **chỉ 1 neuron và KHÔNG có activation**
(để xuất được số thực bất kỳ), hàm loss là MSE thay vì cross-entropy.

```
   Input (7)      Hidden (64 ReLU)   Hidden (32 ReLU)   Output (1, LINEAR)
      ○ ───────────── ○ ───────────────── ○ ─────────────── ○  →  MPG

   ⚠️ Nếu đặt sigmoid ở output → đầu ra bị chặn trong [0,1] → không bao giờ
      xuất ra được 35 MPG. Đây là lỗi kinh điển khi mới học.
```

**Khi nào MLP thắng cây?** Khi quan hệ **trơn và liên tục**. Cây tạo hàm bậc thang;
MLP tạo hàm mượt — hợp với các quan hệ vật lý như tiêu hao nhiên liệu.

---

## 2. BÀI TOÁN THỰC TẾ

```
   Trang tư vấn mua xe muốn hiển thị: "Với thông số này, xe tiêu hao khoảng
   8,5 L/100km — tương đương 21 triệu tiền xăng/năm nếu chạy 15.000 km."

   Giá trị: khách so sánh được CHI PHÍ SỬ DỤNG THẬT giữa các mẫu xe,
   không chỉ nhìn giá bán ban đầu.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Auto MPG (UCI) |
| **Link** | https://archive.ics.uci.edu/dataset/9/auto+mpg |
| **Kích thước** | 398 dòng × 8 cột |
| **Nhãn** | `mpg` (miles per gallon) |

**Đặc trưng:** `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`,
`model_year`, `origin`

### ⚠️ Ba lưu ý

```
   1. `horsepower` có 6 giá trị '?' → đọc vào thành kiểu TEXT
      → pd.to_numeric(errors='coerce') rồi điền median

   2. Bộ chỉ 398 dòng — RẤT ÍT cho mạng nơ-ron
      → mạng to sẽ overfit ngay
      → bắt buộc: mạng nhỏ + early_stopping + regularization alpha

   3. `origin` là mã vùng (1=Mỹ, 2=Châu Âu, 3=Nhật) → biến PHÂN LOẠI, phải one-hot

   4. Đổi đơn vị cho người Việt: L/100km = 235,215 / mpg
```

---

## 4. HƯỚNG ĐI ĐÚNG

```python
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

net = Pipeline([
    ('scale', StandardScaler()),                 # ⭐ BẮT BUỘC cho X
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(64, 32),             # nhỏ vì chỉ có 398 mẫu
        activation='relu', solver='adam',
        alpha=1e-2,                              # regularization MẠNH
        learning_rate_init=1e-3, max_iter=2000,
        early_stopping=True, n_iter_no_change=30,
        random_state=42)),
])

# ⭐ Scale luôn cả y — mạng nơ-ron hội tụ tốt hơn nhiều khi y quanh 0
model = TransformedTargetRegressor(regressor=net, transformer=StandardScaler())
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Đọc dữ liệu, xử lý 6 giá trị '?' ở horsepower
   ☐ 2. One-hot `origin`; bỏ cột `car_name` (định danh)
   ☐ 3. EDA: scatter weight vs mpg (quan hệ nghịch, hơi cong) · mpg theo model_year
   ☐ 4. Baseline: DummyRegressor + Linear Regression + Random Forest
   ☐ 5. ⚠️ MLP KHÔNG scale → thường không hội tụ hoặc rất tệ → ghi lại
   ☐ 6. MLP scale X · MLP scale cả X và y → BẢNG so sánh 3 trường hợp
   ☐ 7. Thử 4 kiến trúc: (16) · (64) · (64,32) · (256,128,64)
        → bảng: RMSE · số tham số · dấu hiệu overfit
        ⚠️ với 398 mẫu, mạng (256,128,64) sẽ overfit rõ rệt — đó là bài học
   ☐ 8. Vẽ `loss_curve_` và `validation_scores_` → chẩn đoán
   ☐ 9. Dò `alpha` ∈ {1e-4, 1e-3, 1e-2, 1e-1} → vẽ RMSE train/test theo alpha
   ☐ 10. So sánh 3 activation: relu / tanh / logistic
   ☐ 11. ⭐ BẢNG SO SÁNH CUỐI với TT-11 (Linear), TT-17 (RF), TT-20 (SVR)
         trên tiêu chí: RMSE · thời gian train · giải thích được không
   ☐ 12. Đổi kết quả sang L/100km và ước tính tiền xăng/năm
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Xử lý đúng 6 giá trị '?' ở horsepower
   ☐ Có bảng so sánh 3 trường hợp scale (không / chỉ X / cả X và y)
   ☐ Có bảng 4 kiến trúc + CHỨNG MINH mạng to overfit với dữ liệu nhỏ
   ☐ Có biểu đồ loss_curve_
   ☐ Có khảo sát alpha
   ☐ ⭐ Có bảng so sánh với ít nhất 3 thuật toán hồi quy khác
   ☐ R² test > 0,85
   ☐ Kết luận trung thực: với 398 dòng dữ liệu bảng, MLP có đáng dùng không?
```

**Mức tham chiếu:** R² ~0,85–0,89. Random Forest thường **ngang hoặc hơn** MLP trên
bộ nhỏ này — và đó chính là kết luận quan trọng nhất của bài.

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Đặt activation ở tầng output | Đầu ra bị chặn, không bao giờ đúng |
| Không scale y | Hội tụ chậm hoặc không hội tụ |
| Mạng quá to với 398 mẫu | Overfit nặng |
| Không `early_stopping` | Train thừa, overfit |
| Để `origin` dạng số | Model hiểu "Nhật > Châu Âu > Mỹ" |
| Kết luận "MLP luôn tốt hơn" | Với dữ liệu bảng nhỏ, cây thường thắng |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-22-MLPRegressor-<HoTen>/
├── README.md          ← có bảng so sánh 4 thuật toán + kết luận trung thực
├── notebooks/mlp_regressor_mpg.ipynb
├── src/train.py
├── models/mlp_reg.joblib
├── reports/{scale_comparison.png, kien_truc_overfit.png, loss_curve.png, alpha_sweep.png}
└── requirements.txt
```

**Mở rộng:**
1. Chuyển sang PyTorch (KT-01) để thêm Dropout, BatchNorm → có cải thiện không?
2. Thử với bộ dữ liệu LỚN hơn (vd Bike Sharing 17k dòng ở TT-19)
   → lúc nào MLP mới bắt đầu thắng cây?
3. Dự báo khoảng: train 10 MLP với seed khác nhau → dùng độ lệch chuẩn làm khoảng tin cậy

**Tham khảo:** [Buổi 7 — Neural Network](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-07-Neural-Network/Tai-Lieu/ly_thuyet_chi_tiet_buoi_07.md) · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
