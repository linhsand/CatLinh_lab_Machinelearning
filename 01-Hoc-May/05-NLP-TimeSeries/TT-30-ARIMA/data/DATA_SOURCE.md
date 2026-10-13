# Nguồn dữ liệu

**AirPassengers**: số khách quốc tế hàng tháng của các hãng hàng không Mỹ, 01/1949 – 12/1960 (144 tháng), đơn vị
**nghìn khách**. Nguồn gốc: Box, G. E. P., Jenkins, G. M. & Reinsel, G. C. *Time Series Analysis, Forecasting and
Control* (Series G). Có sẵn trong R (`datasets::AirPassengers`).

## Cách lấy

Không cần làm gì. Lần chạy đầu, `src/stationarity.py` gọi `statsmodels.datasets.get_rdataset("AirPassengers")` (cần
mạng, tải từ kho Rdatasets) rồi lưu `data/AirPassengers.csv` (2 cột: `time` là năm thập phân, `value` là số khách).
Các lần sau đọc file này. Dữ liệu **không** được commit (xem `.gitignore`).

Cột `time` dạng năm thập phân (`1949.083` = tháng 2/1949) được đổi thành chỉ mục tháng `1949-01-01 … 1960-12-01`.

## Đặc điểm

| | |
|---|---|
| Số điểm | 144, không thiếu, không ngoại lai |
| Nhỏ nhất / lớn nhất | 104 (11/1949) / 622 (7/1960) |
| Trend | tăng gần tuyến tính, TB +12,9%/năm (train 1949–1958) |
| Mùa vụ | chu kỳ 12 tháng, đỉnh tháng 7–8, đáy tháng 11 |
| Kiểu | **nhân tính**: biên độ mùa vụ tăng từ 44 (1949) lên 232 (1960) |

## Chia dữ liệu

| Tập | Tháng | Dùng để |
|---|---|---|
| train | 01/1949 – 12/1958 (120) | ADF trên toàn chuỗi để chọn d, D; ước lượng; chọn bậc theo AIC |
| test | 01/1959 – 12/1960 (24) | **chỉ báo cáo**: dự báo 24 tháng một lần từ 12/1958 |
