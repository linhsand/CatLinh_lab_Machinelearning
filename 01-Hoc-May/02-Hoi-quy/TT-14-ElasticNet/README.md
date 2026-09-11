# TT-14 — ELASTICNET
## Dự báo tiêu thụ năng lượng toà nhà khi các biến thiết kế dính chặt nhau

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 13](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao) |
| 🧠 **Nhóm** | Hồi quy có regularization kết hợp |
| 🔧 **Thuật toán** | ElasticNet (L1 + L2) |
| 🏭 **Lĩnh vực** | Năng lượng · Thiết kế xây dựng |
| ⏱ **Thời lượng** | 5–7 giờ |
| 📈 **Độ khó** | ⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   ElasticNet = trộn CẢ HAI hình phạt:

        Loss = MSE + λ·[ ρ·Σ|wᵢ|  +  (1−ρ)/2·Σwᵢ² ]
                        └─ L1 ─┘      └─── L2 ───┘

        ρ (l1_ratio) = 1   → giống hệt LASSO
        ρ = 0              → giống hệt RIDGE
        ρ = 0,5            → cân bằng cả hai
```

**Nó sinh ra để vá đúng điểm yếu của Lasso:** khi 2 biến tương quan 0,95, Lasso chọn
ngẫu nhiên 1 biến và bỏ hẳn biến kia. ElasticNet **giữ cả nhóm** biến tương quan
(hiệu ứng gom nhóm — grouping effect) mà vẫn loại được biến vô dụng.

| | Ridge | Lasso | **ElasticNet** |
|---|---|---|---|
| Đưa hệ số về 0 | ❌ | ✅ | ✅ |
| Giữ nhóm biến tương quan | ✅ | ❌ | ✅ |
| Khi số biến > số mẫu | Kém | Tối đa n biến | ✅ Tốt nhất |

---

## 2. BÀI TOÁN THỰC TẾ

```
   Công ty thiết kế cần ước tính TẢI SƯỞI và TẢI LÀM MÁT của toà nhà
   NGAY TỪ BẢN VẼ, trước khi xây, để chọn công suất điều hoà.

   Chọn thừa công suất → lãng phí đầu tư + tốn điện vận hành
   Chọn thiếu công suất → toà nhà không đủ mát → phải cải tạo, cực đắt

   ⚠️ Đặc thù dữ liệu: các biến thiết kế DÍNH CHẶT nhau về mặt hình học
      Diện tích tường ↔ Diện tích mái ↔ Chiều cao ↔ Diện tích sàn
      (đổi 1 cái là các cái kia đổi theo — ràng buộc vật lý)
   → Lasso sẽ bỏ oan biến quan trọng. ElasticNet là lựa chọn đúng.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Energy Efficiency (UCI) |
| **Link** | https://archive.ics.uci.edu/dataset/242/energy+efficiency |
| **Kích thước** | 768 dòng × 8 đặc trưng |
| **Nhãn** | `Y1` = tải sưởi · `Y2` = tải làm mát (**2 bài hồi quy**) |

**Đặc trưng:** `X1` độ gọn tương đối, `X2` diện tích bề mặt, `X3` diện tích tường,
`X4` diện tích mái, `X5` chiều cao tổng, `X6` hướng nhà, `X7` diện tích kính,
`X8` phân bố kính

### ⚠️ Lưu ý dữ liệu

```
   1. X1, X2, X4, X5 tương quan gần như HOÀN HẢO (|r| > 0,95)
      → chính là lý do chọn bộ này cho ElasticNet.

   2. X6 (hướng nhà) và X8 (phân bố kính) là biến PHÂN LOẠI mã hoá bằng số
      → phải one-hot, không để dạng số có thứ tự.

   3. Bộ chỉ có 768 dòng → dùng cross-validation, đừng tin 1 lần chia train/test.
```

---

## 4. HƯỚNG ĐI ĐÚNG

```python
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

pipe = Pipeline([
    ('scale', StandardScaler()),                       # ⭐ BẮT BUỘC
    ('en', ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0], # dò cả tỉ lệ trộn
        alphas=np.logspace(-4, 1, 100),
        cv=5, max_iter=50000, random_state=42)),
])
pipe.fit(X_train, y_train)
print("alpha:", pipe['en'].alpha_, "| l1_ratio:", pipe['en'].l1_ratio_)
```

> 💡 `l1_ratio` gần 1 → dữ liệu ưa Lasso (ít biến thật sự quan trọng).
> `l1_ratio` gần 0 → dữ liệu ưa Ridge (nhiều biến cùng đóng góp).
> Con số máy chọn ra chính là **câu trả lời về bản chất dữ liệu**.

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Nạp dữ liệu, tính ma trận tương quan + VIF → xác nhận đa cộng tuyến nặng
   ☐ 2. One-hot X6, X8; chuẩn hoá các biến số
   ☐ 3. Baseline: DummyRegressor + Linear Regression
   ☐ 4. Chạy 3 model trên CÙNG dữ liệu: Ridge · Lasso · ElasticNet
   ☐ 5. ⭐ BẢNG SO SÁNH: mỗi model giữ bao nhiêu biến? RMSE bao nhiêu?
   ☐ 6. ⭐ Kiểm chứng HIỆU ỨNG GOM NHÓM:
        • Lasso giữ X1 hay X4 hay X5? (nó sẽ chọn 1, bỏ phần còn lại)
        • ElasticNet giữ mấy biến trong nhóm đó?
   ☐ 7. Vẽ heatmap RMSE theo lưới (alpha × l1_ratio)
   ☐ 8. Làm CẢ HAI nhãn Y1 và Y2 → so sánh: biến nào quan trọng cho sưởi,
        biến nào cho làm mát? (kết quả thường KHÁC NHAU — có ý nghĩa kỹ thuật)
   ☐ 9. Kiểm tra ổn định: bootstrap 100 lần → hệ số ElasticNet dao động bao nhiêu?
   ☐ 10. ✍️ Đề xuất 3 thay đổi thiết kế giúp giảm tải năng lượng
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Có bảng VIF chứng minh đa cộng tuyến
   ☐ Có bảng so sánh Ridge / Lasso / ElasticNet (số biến giữ + RMSE)
   ☐ ⭐ Chỉ rõ được hiệu ứng gom nhóm: Lasso bỏ biến nào mà ElasticNet giữ
   ☐ Có heatmap alpha × l1_ratio
   ☐ Làm đủ cả 2 nhãn Y1 và Y2, có so sánh
   ☐ Giải thích được ý nghĩa của l1_ratio mà máy chọn
   ☐ RMSE tốt hơn baseline rõ rệt
```

**Mức tham chiếu:** R² ~0,90–0,92 cho Y1 (tải sưởi). Đây là bộ dữ liệu mô phỏng
nên quan hệ rất sạch — đừng kỳ vọng dữ liệu thật cũng đẹp như vậy.

---

## 6.1. KẾT QUẢ CHẠY THỰC TẾ

**Toàn bộ số liệu dưới đây là kết quả chạy thật** (`notebooks/elasticnet_energy.ipynb`,
tái lập bằng `src/train.py`), dữ liệu tải trực tiếp từ UCI (768 dòng), `random_state=42`.

**Bước 1 — Ma trận tương quan + VIF** (`reports/vif_table.csv`), tính trên 6 biến số liên tục:

| Biến | Mô tả | VIF |
|---|---|---|
| X2 | Diện tích bề mặt | **∞** (vô cực) |
| X3 | Diện tích tường | **∞** (vô cực) |
| X4 | Diện tích mái | **∞** (vô cực) |
| X1 | Độ gọn tương đối | 105,52 |
| X5 | Chiều cao tổng | 31,21 |
| X7 | Diện tích kính | 1,00 |

VIF = ∞ nghĩa là đa cộng tuyến **hoàn hảo** (X2, X3, X4 là tổ hợp tuyến tính chính xác của các
biến còn lại) — dữ liệu Energy Efficiency chỉ có **12 hình khối nhà** khác nhau (lặp lại theo
hướng × kính), và X1–X5 được tính từ đúng các công thức hình học của nhau. Đây là mức đa cộng
tuyến nặng hơn cả README cảnh báo, xác nhận rõ ràng lý do phải dùng ElasticNet thay vì OLS thường.
Các cặp |r| > 0,8: `(X1,X2)=-0,992`, `(X4,X5)=-0,973`, `(X2,X4)=0,881`, `(X2,X5)=-0,858`,
`(X1,X4)=-0,869`, `(X1,X5)=0,828`.

**Sau one-hot X6, X8:** 14 cột (`X1,X2,X3,X4,X5,X7` + `X6_3,X6_4,X6_5` + `X8_1..X8_5`).

**Bảng so sánh 3 model + baseline** (test set, 154 dòng):

| Model | Nhãn Y1 (tải sưởi) RMSE / R² | Nhãn Y2 (tải làm mát) RMSE / R² |
|---|---|---|
| Dummy (trung bình) | 10,238 / -0,006 | 9,666 / -0,008 |
| Linear Regression | 2,872 / 0,9209 | 3,120 / 0,8950 |
| Ridge (alpha≈0,272 / 0,192) | 2,876 / 0,9207 | 3,121 / 0,8949 |
| Lasso (alpha≈0,00129 / 0,00072) | 2,875 / 0,9207 | 3,120 / 0,8949 |
| **ElasticNet** (alpha≈0,00045 / 0,00032, l1_ratio=0,10) | **2,876 / 0,9207** | **3,121 / 0,8949** |

Cả 3 model regularization đều giữ **14/14 biến** ở alpha tối ưu theo CV — với chỉ 14 biến và 614
dòng train (p << n), CV chọn ra alpha rất nhỏ vì mô hình chưa hề overfit, nên gần như không cần
phạt mạnh. RMSE 4 model gần như giống hệt nhau (đều vượt trội hẳn baseline Dummy), khớp đúng mức
tham chiếu R² ~0,90–0,92 cho Y1.

**⭐ Bước 6 — Hiệu ứng gom nhóm.** Ở alpha tối ưu riêng của từng model (rất nhỏ), *cả Lasso lẫn
ElasticNet đều giữ đủ 4/4 biến* trong nhóm tương quan `{X1, X2, X4, X5}` — chưa thấy khác biệt, vì
alpha chưa đủ lớn để ép loại biến nào. Để **thực sự bộc lộ** hiệu ứng gom nhóm, cần ép alpha tăng
dần vượt qua alpha tối ưu (`reports/hieu_ung_gom_nhom_theo_alpha_*.png`):

| Nhãn | alpha ép (Lasso vừa bắt đầu loại biến trong nhóm) | Lasso giữ trong nhóm | ElasticNet (l1_ratio=0,5) giữ trong nhóm | Biến Lasso loại |
|---|---|---|---|---|
| Y1 | 0,0104 | 3/4 | **4/4** | `X2` (diện tích bề mặt) |
| Y2 | 0,0227 | 3/4 | **4/4** | `X2` (diện tích bề mặt) |

Tại đúng alpha đó, hệ số của `X2` là `lasso=0,0000` nhưng `elasticnet=-1,59` (Y1) / `-1,09` (Y2) —
ElasticNet vẫn giữ tín hiệu của X2 thay vì loại bỏ hoàn toàn như Lasso. **Xác nhận đúng lý thuyết:**
Lasso chọn ngẫu nhiên 1 biến trong nhóm tương quan và bỏ hẳn biến còn lại (ở đây luôn là X2, vì X2
tương quan gần như tuyệt đối với X1: r=-0,992), còn ElasticNet phân bổ hệ số cho cả nhóm nhờ thành
phần phạt L2.

**Heatmap alpha × l1_ratio** (`reports/heatmap_alpha_l1ratio_*.png`): RMSE thấp và gần như phẳng
khi alpha ≲ 0,1 (không phụ thuộc l1_ratio), tăng mạnh khi alpha lớn — và tăng **nhanh nhất** ở
l1_ratio=1,0 (Lasso thuần) vì lúc đó Lasso loại sạch cả 4 biến trong nhóm tương quan trước (xem
bảng trên: tại alpha≈10, Lasso còn 0/14 biến), gây underfit nặng hơn hẳn so với l1_ratio nhỏ (gần
Ridge) vẫn giữ được tín hiệu dù hệ số bị co nhỏ.

**Bước 8 — So sánh Y1 (tải sưởi) vs Y2 (tải làm mát)** (`reports/so_sanh_Y1_Y2_tam_quan_trong.csv`):

| Biến | Hệ số ElasticNet Y1 | Hệ số ElasticNet Y2 | Nhận xét |
|---|---|---|---|
| X5 (chiều cao) | **+7,32** | **+7,21** | Quan trọng nhất cho cả hai, gần như bằng nhau |
| X1 (độ gọn) | -6,22 | -7,18 | Quan trọng hơn ~15% cho làm mát |
| X3 (diện tích tường) | **+0,87** | +0,22 | Quan trọng hơn **~4 lần** cho tải sưởi (tường truyền nhiệt mùa lạnh) |
| X8 (phân bố kính, mọi hạng mục) | ~1,4–1,6 | ~0,55–0,71 | Ảnh hưởng gấp ~2 lần lên tải sưởi so với làm mát |
| X7 (diện tích kính) | +2,31 | +1,81 | Quan trọng hơn ~25% cho tải sưởi |

→ Diện tích tường và cách bố trí kính ảnh hưởng đến **tải sưởi** rõ rệt hơn tải làm mát — hợp lý về
mặt vật lý xây dựng: tường và kính là đường thất thoát nhiệt chính vào mùa lạnh, trong khi tải làm
mát mùa nóng chịu ảnh hưởng cân bằng hơn giữa hình khối (X1, X5) và bức xạ mặt trời qua kính.

**Bước 9 — Bootstrap 100 lần (80% dữ liệu mỗi lần), hệ số ElasticNet:**

| Biến | Hệ số trung bình (Y1) | Độ lệch chuẩn | Hệ số biến thiên |
|---|---|---|---|
| X5 | 7,37 | 0,28 | **3,8%** (rất ổn định) |
| X1 | -6,54 | 0,30 | 4,6% |
| X4 | -3,76 | 0,20 | 5,4% |
| X2 | -3,65 | 0,21 | 5,7% |
| X6 (hướng nhà, cả 3 hạng mục) | ±0,01 – 0,04 | ~0,07 | **175% – 593%** (hoàn toàn không ổn định) |

→ Các biến hình học chính (X1, X2, X4, X5, X7) **ổn định cao** (biến thiên < 10%) — kết luận về
tầm quan trọng của chúng đáng tin cậy. Ngược lại, hệ số của X6 (hướng nhà) dao động cực mạnh quanh
0 và đổi dấu giữa các lần lấy mẫu → **hướng nhà không có ảnh hưởng thống kê đáng tin cậy** trong bộ
dữ liệu này, dù mô hình đôi khi gán cho nó hệ số khác 0.

**✍️ Đề xuất 3 thay đổi thiết kế để giảm tải năng lượng** (dựa trên dấu và độ lớn hệ số ElasticNet
đã chuẩn hoá, `reports/tam_quan_trong_bien_*.csv`):

1. **Giảm chiều cao tầng (X5)** — biến quan trọng nhất cho cả hai nhãn (hệ số +7,3 cho Y1, +7,2
   cho Y2, đều dương và lớn nhất). Mỗi đơn vị chiều cao tăng thêm kéo theo mức tăng tải lớn nhất
   trong toàn bộ 14 biến — ưu tiên hàng đầu khi có thể điều chỉnh thiết kế.
2. **Tăng độ gọn tương đối (X1) bằng hình khối nhà vuông vắn hơn** — hệ số âm lớn (-6,2 cho Y1,
   -7,2 cho Y2): nhà càng "gọn" (tỉ lệ diện tích bề mặt/thể tích càng nhỏ) thì tải càng giảm, và
   tác dụng còn rõ hơn cho tải làm mát. Đây là đòn bẩy rẻ nhất vì chỉ cần thay đổi tỉ lệ hình khối,
   không cần vật liệu đắt tiền.
3. **Ưu tiên xử lý tường và kính cho công trình có mùa lạnh kéo dài** — X3 (tường) và X8 (phân bố
   kính) có ảnh hưởng lên tải sưởi (Y1) gấp 2–4 lần so với tải làm mát (Y2). Với công trình cần tối
   ưu chi phí sưởi, nên đầu tư cách nhiệt tường và chọn phân bố kính hợp lý trước; với công trình
   ưu tiên làm mát, tác động của 2 biến này thấp hơn nên có thể hạ ưu tiên đầu tư ở đây.

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Quên chuẩn hoá | Phạt bất công giữa các biến khác đơn vị |
| Chỉ dò `alpha`, cố định `l1_ratio` | Bỏ lỡ điểm tối ưu thật |
| Để X6, X8 dạng số | Model hiểu nhầm "hướng 4 > hướng 2" |
| Tin 1 lần chia train/test với 768 dòng | Phương sai lớn → dùng cross-validation |
| `max_iter` mặc định | Cảnh báo không hội tụ |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-14-ElasticNet/
├── README.md          ← có mục "HIỆU ỨNG GOM NHÓM" (mục 6.1)
├── notebooks/elasticnet_energy.ipynb
├── src/train.py
├── data/ENB2012_data.xlsx             ← cache tải từ UCI, sinh tự động khi chạy train.py
├── models/{elasticnet_Y1.joblib, elasticnet_Y2.joblib}
├── reports/
│   ├── correlation_matrix.csv, vif_table.csv          ← bước 1: đa cộng tuyến
│   ├── so_sanh_3_model_Y1/Y2.csv, .png                ← bước 5: so sánh Ridge/Lasso/ElasticNet
│   ├── hieu_ung_gom_nhom_theo_alpha_Y1/Y2.csv, .png   ← bước 6: hiệu ứng gom nhóm (ép alpha)
│   ├── heatmap_alpha_l1ratio_Y1/Y2.png                ← bước 7: heatmap alpha x l1_ratio
│   ├── so_sanh_Y1_Y2_tam_quan_trong.csv, .png         ← bước 8: so sánh Y1 vs Y2
│   ├── bootstrap_elasticnet_Y1/Y2.csv, .png           ← bước 9: ổn định hệ số
│   ├── tam_quan_trong_bien_Y1/Y2.csv                  ← bước 10: xếp hạng biến (cơ sở đề xuất)
│   └── tom_tat.json                                   ← tổng hợp toàn bộ số liệu
└── requirements.txt
```

**Mở rộng:**
1. Thêm đặc trưng tương tác (`X3 × X7`) → ElasticNet có tự loại bớt không?
2. So sánh với Gradient Boosting Regressor (TT-18) — mất tính giải thích để đổi lấy bao nhiêu % RMSE?
3. Dự đoán đồng thời Y1 và Y2 bằng `MultiTaskElasticNet` — có tốt hơn 2 model riêng không?

**Tham khảo:** [Buổi 13 — Regularization](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-13-Math-Regression-NangCao/Tai-Lieu)
