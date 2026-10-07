# Nguồn dữ liệu

**Flight Price Prediction** (Kaggle, Shubham Bathwal, 2022).
<https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction>

Vé máy bay nội địa Ấn Độ thu thập từ trang EaseMyTrip, 11/02–31/03/2022, 6 hãng, 6 thành phố.
File dùng ở đây là `Clean_Dataset.csv`: **300.153 dòng × 12 cột** (cột đầu `Unnamed: 0` là index thừa).

| Cột | Ý nghĩa |
|---|---|
| `airline` | 6 hãng |
| `flight` | mã chuyến bay (bị **bỏ**: 1.561 giá trị, cây sẽ học thuộc mã chuyến) |
| `source_city`, `destination_city` | 6 thành phố |
| `departure_time`, `arrival_time` | 6 khung giờ (Early_Morning … Late_Night) |
| `stops` | zero / one / two_or_more |
| `class` | Economy / Business |
| `duration` | thời gian bay (giờ) |
| `days_left` | số ngày từ lúc đặt tới ngày bay, **1–49** |
| `price` | giá vé (Rupee), nhãn |

## Cách lấy

File đã có sẵn trong `data/` (~25 MB). Nếu cần tải lại: đăng nhập Kaggle, tải `Clean_Dataset.csv` từ link
trên và đặt vào `data/`. Bản trong repo được lấy từ một mirror công khai trên GitHub
(`Amanrathi-Git/Flight-Pricing-Analytics`), đã kiểm tra đúng 300.153 dòng và đúng 12 cột như bản Kaggle.
