# TT-30 — ARIMA / SARIMA
## Dự báo lượng khách hàng tháng để lập kế hoạch nhân sự

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 9](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-09-TimeSeries) |
| 🧠 **Nhóm** | Chuỗi thời gian · Mô hình thống kê |
| 🔧 **Thuật toán** | ARIMA · **SARIMA** |
| 🏭 **Lĩnh vực** | Dịch vụ · Hàng không · Bán lẻ |
| ⏱ **Thời lượng** | 6–8 giờ |
| 📈 **Độ khó** | ⭐⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   ARIMA(p, d, q)

   AR(p) — AutoRegressive : dự đoán từ p GIÁ TRỊ quá khứ của chính nó
            yₜ = φ₁yₜ₋₁ + ... + φₚyₜ₋ₚ + ε

   I(d)  — Integrated     : số lần SAI PHÂN để chuỗi thành DỪNG

   MA(q) — Moving Average : dự đoán từ q SAI SỐ dự báo trong quá khứ
            yₜ = θ₁εₜ₋₁ + ... + θqεₜ₋q + ε

   SARIMA(p,d,q)(P,D,Q,s) — thêm phần MÙA VỤ với chu kỳ s
                             s = 12 (tháng) · 7 (ngày) · 24 (giờ)
```

**Ưu thế riêng của ARIMA so với cây/ML:** nó **NGOẠI SUY được xu hướng** (nhờ thành
phần I) và cho **khoảng tin cậy** sẵn — hai thứ mà Random Forest/XGBoost không làm được.

---

## 2. BÀI TOÁN THỰC TẾ

```
   Chuỗi nhà hàng cần dự báo lượng khách 3 tháng tới để:
     • Lên lịch tuyển nhân viên thời vụ (tuyển trước 6 tuần)
     • Đặt hàng nguyên liệu dài hạn
     • Lập kế hoạch dòng tiền

   Dự báo THIẾU → không đủ người phục vụ → khách bỏ đi, mất uy tín
   Dự báo THỪA → trả lương nhân viên ngồi chơi

   ⚠️ Cần KHOẢNG dự báo, không chỉ 1 con số:
      "tháng 12 dự kiến 12.000–14.500 khách" → giúp lập kế hoạch theo kịch bản.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Khởi động** | AirPassengers — 144 tháng (1949–1960), kinh điển |
| **Cách lấy** | `statsmodels.datasets.get_rdataset("AirPassengers")` |
| **Nâng cao** | Rossmann Store Sales (Kaggle) hoặc dữ liệu POS của công ty |

```python
import statsmodels.api as sm
du_lieu = sm.datasets.get_rdataset("AirPassengers").data
# 144 điểm, có TREND tăng rõ và MÙA VỤ chu kỳ 12 tháng, biên độ PHÌNH DẦN
```

### ⚠️ Đặc điểm quyết định cách xử lý

```
   Biên độ dao động mùa vụ TĂNG DẦN theo trend
   → đây là mô hình NHÂN TÍNH (multiplicative), không phải cộng tính

        ∿∿∿∿
    ∿∿∿    ∿∿∿∿
   ∿∿        ∿∿∿∿∿      ← biên độ phình to dần

   → PHẢI lấy log(y) trước để biến thành cộng tính, mô hình xong rồi exp() ngược lại.
   → Bỏ qua bước này là lỗi phổ biến nhất với bộ dữ liệu này.
```

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. Quy trình Box-Jenkins chuẩn

```
   ① NHÌN     — vẽ chuỗi, phân rã trend/seasonal/residual
   ② LÀM DỪNG — kiểm định ADF; nếu p ≥ 0,05 thì sai phân
   ③ ĐỊNH BẬC — nhìn ACF (chọn q) và PACF (chọn p)
   ④ ƯỚC LƯỢNG— fit model
   ⑤ KIỂM TRA — phần dư phải là NHIỄU TRẮNG (Ljung-Box p > 0,05)
   ⑥ DỰ BÁO   — kèm khoảng tin cậy
```

### 4.2. Kiểm định tính dừng

```python
from statsmodels.tsa.stattools import adfuller
import numpy as np

y_log = np.log(y)
stat, p, *_ = adfuller(y_log.dropna())
print(f"ADF p-value = {p:.4f} → {'DỪNG' if p < 0.05 else 'KHÔNG DỪNG'}")
# Không dừng → thử y_log.diff(1) rồi kiểm lại; còn mùa vụ thì thêm .diff(12)
```

### 4.3. Mô hình

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(y_log_train, order=(1,1,1), seasonal_order=(1,1,1,12),
                enforce_stationarity=False, enforce_invertibility=False)
fit = model.fit(disp=False)
print(fit.summary())

kq = fit.get_forecast(steps=24)
du_bao = np.exp(kq.predicted_mean)                 # ⭐ đổi ngược từ log
khoang = np.exp(kq.conf_int(alpha=0.05))           # khoảng tin cậy 95%
```

### 4.4. ⭐ Kiểm tra phần dư — bước hay bị bỏ qua nhất

```python
from statsmodels.stats.diagnostic import acorr_ljungbox
lb = acorr_ljungbox(fit.resid, lags=[12, 24], return_df=True)
# p-value > 0,05 → phần dư là nhiễu trắng → model đã bắt hết cấu trúc ✅
# p-value < 0,05 → CÒN mẫu chưa bắt được → phải chỉnh lại (p,d,q)
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Vẽ chuỗi gốc, nhận diện trend + mùa vụ + biên độ phình dần
   ☐ 2. seasonal_decompose với model='multiplicative' và 'additive' → cái nào hợp?
   ☐ 3. Lấy log → vẽ lại → xác nhận biên độ đã ổn định
   ☐ 4. ADF trên: chuỗi gốc · log · log+diff(1) · log+diff(1)+diff(12)
        → lập bảng p-value → xác định d và D
   ☐ 5. Vẽ ACF và PACF của chuỗi đã dừng → đề xuất (p,q) và (P,Q)
   ☐ 6. Chia theo thời gian: 120 tháng train · 24 tháng test
   ☐ 7. ⭐ BASELINE BẮT BUỘC:
        • Naive: "tháng này = tháng trước"
        • Seasonal naive: "tháng này = tháng này NĂM NGOÁI"  ← rất mạnh
   ☐ 8. SARIMA thủ công theo ACF/PACF
   ☐ 9. Dò tự động: quét lưới (p,d,q)(P,D,Q,12) chọn theo AIC thấp nhất
        → so sánh với bậc chọn thủ công
   ☐ 10. ⭐ Ljung-Box + vẽ biểu đồ chẩn đoán phần dư (fit.plot_diagnostics())
   ☐ 11. Dự báo 24 tháng + khoảng tin cậy, vẽ chồng lên thực tế
   ☐ 12. Tính MAPE trên tập test, so với 2 baseline
   ☐ 13. ⭐ Đo TỈ LỆ PHỦ: bao nhiêu % giá trị thật rơi trong khoảng 95%?
         (lý tưởng ≈ 95%; nếu chỉ 60% thì khoảng tin cậy quá hẹp, không dùng được)
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Xác định đúng mô hình nhân tính → có dùng log
   ☐ Có bảng ADF p-value cho ít nhất 4 phiên bản chuỗi
   ☐ Có ACF/PACF + lý giải cách chọn (p,q)
   ☐ ⭐ Có kiểm định Ljung-Box + biểu đồ chẩn đoán phần dư
   ☐ ⭐ Có CẢ 2 baseline và SARIMA phải thắng seasonal naive
   ☐ Có biểu đồ dự báo + khoảng tin cậy
   ☐ ⭐ Có tỉ lệ phủ của khoảng dự báo
   ☐ Nêu hạn chế: ARIMA chỉ dùng quá khứ của chính chuỗi, không dùng biến ngoài
```

**Mức tham chiếu:** MAPE ~3–5% cho AirPassengers với SARIMA(1,1,1)(1,1,1,12) trên log.

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Không lấy log với chuỗi nhân tính | Dự báo sai lệch tăng dần theo thời gian |
| Quên `exp()` khi báo cáo | Báo số ở thang log |
| Sai phân quá tay | Tăng nhiễu, model tệ đi |
| Bỏ qua kiểm tra phần dư | Model còn sót cấu trúc mà không biết |
| Không có baseline | Không biết model có đáng dùng không |
| Dùng ARIMA cho chuỗi có nhiều biến ngoài | Nên dùng SARIMAX hoặc ML + lag (TT-31) |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-30-ARIMA-<HoTen>/
├── README.md          ← ⭐ có bảng ADF + kiểm định phần dư + tỉ lệ phủ
├── notebooks/arima_forecasting.ipynb
├── src/{stationarity.py, train.py}
├── models/sarima_fit.pkl
├── reports/{decompose.png, acf_pacf.png, residual_diagnostics.png, du_bao_khoang.png}
└── requirements.txt
```

**Mở rộng:**
1. **SARIMAX** — thêm biến ngoại sinh (ngày lễ, khuyến mãi, thời tiết)
2. So sánh với **Prophet** (Meta) — dễ dùng hơn, xử lý ngày lễ tự động
3. So sánh với **XGBoost + lag features** (TT-31): ARIMA thắng ở ngoại suy trend
   và khoảng tin cậy, ML thắng khi có nhiều biến ngoài

**Tham khảo:** [Buổi 9 — Time Series & ARIMA](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-09-TimeSeries/Tai-Lieu/ly_thuyet_chi_tiet_buoi_09.md)
