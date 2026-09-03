# Nguồn dữ liệu

Bộ dữ liệu gốc: **Credit Card Fraud Detection**, do Machine Learning Group -
ULB (Université Libre de Bruxelles) công bố trên Kaggle
(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Môi trường chạy notebook này không có API key Kaggle để gọi `kaggle datasets
download` trực tiếp, nên `creditcard.csv` được tải qua bản mirror công khai
trên GitHub (`nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection`) — xác nhận
**cùng 284.807 dòng, cùng 492 giao dịch gian lận (0,1727%)** như bản gốc
Kaggle, không phải dữ liệu giả lập/tổng hợp.

**Đặc điểm dữ liệu:**
* `Time`: số giây kể từ giao dịch đầu tiên trong tập (trải dài ~48 giờ / 2 ngày).
* `V1`–`V28`: đã qua biến đổi PCA để ẩn thông tin gốc (bảo mật khách hàng),
  không thể diễn giải ý nghĩa nghiệp vụ của từng cột.
* `Amount`: số tiền giao dịch (đơn vị gốc trong bộ dữ liệu là EUR), **chưa
  được scale** trong khi `V1`–`V28` đã scale sẵn qua PCA.
* `Class`: nhãn nhị phân, 1 = gian lận (492 / 284.807 ≈ 0,172%).
