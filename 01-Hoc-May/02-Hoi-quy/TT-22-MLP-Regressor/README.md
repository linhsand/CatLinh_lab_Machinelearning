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

### ✅ Kết quả thực đo (`notebooks/mlp_regressor_mpg.ipynb`, xem `reports/tom_tat.json`)

Chia 318 train / 80 test (`random_state=42`). Vì tập test chỉ 80 xe, mọi lựa chọn cấu hình
dùng **RMSE cross-validation 5-fold trên tập train**; test chỉ để báo cáo.

**So sánh 3 trường hợp scale** (MLP (64,32), `alpha=1e-2`, early stopping):

| | RMSE CV | RMSE test | R² test | Số vòng |
|---|---|---|---|---|
| Không scale | 3,72 | 3,11 | 0,820 (⚠️ thua cả Linear) | 235 (dừng vì kẹt) |
| Chỉ scale X | 3,00 | 2,36 | 0,896 | 342 |
| Scale X và y | 3,00 | 2,19 | 0,911 | 247 |

**4 kiến trúc** — không regularization (`alpha=1e-5`, không early stopping), mạng to overfit rõ:

| Kiến trúc | Số tham số | RMSE train | RMSE test |
|---|---|---|---|
| (16) | 177 | 2,85 | 2,58 |
| (64) | 705 | 2,36 | 2,13 |
| (64,32) | 2.753 | 1,41 | 2,50 |
| (256,128,64) | 43.777 | **0,72** | **2,62** |

Có regularization thì khoảng cách train–test thu hẹp. Kiến trúc cuối chọn theo **quy tắc 1-SE**
(mạng nhỏ nhất có RMSE CV trong khoảng tốt nhất + 1 std) → **(64)**. `loss_curve_` của mạng to:
R² validation đạt đỉnh 0,94 ở vòng ~65 rồi tụt về 0,85 trong khi loss train vẫn giảm.

**Khảo sát alpha:** với mạng to, đường RMSE CV hình chữ U (3,17 ở 1e-4 → 2,79 ở alpha=1 → 3,48 ở 10);
với mạng (64) gần như phẳng trong khoảng 1e-4..1e-1 → alpha quan trọng khi mạng quá to.
**Activation:** relu (CV 2,87) ≳ tanh (2,97) >> logistic (3,54).

**Model cuối** `models/mlp_reg.joblib`: (64), relu, `alpha=1`, scale X và y →
**RMSE test 2,25 mpg · R² 0,906 · MAE 1,67 mpg ≈ 0,88 L/100km** — đạt tiêu chí R² > 0,85.
Ổn định qua 10 seed: 2,20 ± 0,03 (mạng to + early stopping: 2,35 ± 0,39).

**⭐ Bảng so sánh với các thuật toán hồi quy khác** (train lại trên cùng Auto MPG, cùng tiền xử lý;
RMSE CV = 5 fold × 3 lần lặp trên tập train):

| Thuật toán | RMSE CV (± std) | RMSE test | R² test | Train (s) | Giải thích được |
|---|---|---|---|---|---|
| SVR RBF (TT-20) | **2,70 ± 0,39** | **2,00** | **0,926** | 0,01 | Khó |
| **MLP (64) (TT-22)** | 2,86 ± 0,29 | 2,25 | 0,906 | 0,18 | Khó (hộp đen) |
| Random Forest (TT-17) | 2,97 ± 0,35 | 2,16 | 0,913 | 0,29 | Một phần |
| KNN K=5 (TT-21) | 3,06 ± 0,39 | 2,21 | 0,909 | 0,01 | Có (xe tương tự) |
| Linear Regression (TT-11) | 3,47 ± 0,27 | 2,89 | 0,845 | 0,01 | Có (hệ số) |

**Kết luận trung thực:** MLP đạt yêu cầu và đứng nhóm đầu, nhưng chênh lệch với SVR/Random Forest
nhỏ hơn độ nhiễu của CV — **không thắng rõ**. Hai model tạo hàm mượt (SVR, MLP) nhỉnh hơn cây, khớp
nhận định "quan hệ vật lý trơn", nhưng SVR đạt được điều đó với 2 siêu tham số, còn MLP cần scale X+y,
chọn kiến trúc, alpha, activation và dễ hỏng nếu cấu hình sai. Với 398 dòng dữ liệu bảng, MLP
**không đáng là lựa chọn mặc định**.

**Đổi đơn vị cho người Việt** (15.000 km/năm, xăng giả định 24.000 đ/L): ví dụ Ford F250 (1970)
ước tính 22,8 L/100km ≈ 82 triệu đ/năm; VW Pickup (1982) 5,7 L/100km ≈ 20 triệu đ/năm.

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
TT-22-MLP-Regressor/
├── README.md          ← có bảng so sánh 5 thuật toán + kết luận trung thực
├── data/{auto-mpg.data, auto-mpg.names}   ← file gốc UCI
├── notebooks/mlp_regressor_mpg.ipynb      ← toàn bộ pipeline; mỗi bước có "Logic" + "Giải thích code" + "Đọc kết quả"
├── src/{features.py, train.py}            ← đọc/làm sạch + tiền xử lý dùng chung; script train độc lập
├── models/mlp_reg.joblib                  ← TransformedTargetRegressor(tiền xử lý + MLP, scale y)
├── reports/{eda.png, scale_comparison.png, kien_truc_overfit.png, loss_curve.png, alpha_sweep.png,
│            so_sanh_models.png, so_sanh_scale.csv, so_sanh_kien_truc.csv, so_sanh_models.csv,
│            tom_tat.json, train_metrics.json}
└── requirements.txt
```

**Mở rộng:**
1. Chuyển sang PyTorch (KT-01) để thêm Dropout, BatchNorm → có cải thiện không?
2. Thử với bộ dữ liệu LỚN hơn (vd Bike Sharing 17k dòng ở TT-19)
   → lúc nào MLP mới bắt đầu thắng cây?
3. Dự báo khoảng: train 10 MLP với seed khác nhau → dùng độ lệch chuẩn làm khoảng tin cậy

**Tham khảo:** [Buổi 7 — Neural Network](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-07-Neural-Network/Tai-Lieu/ly_thuyet_chi_tiet_buoi_07.md) · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
