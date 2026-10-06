# Nguồn dữ liệu

**Human Activity Recognition Using Smartphones** (UCI, Anguita và cộng sự, 2013).
<https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>

30 người đeo điện thoại Samsung Galaxy S II ở thắt lưng, làm 6 hoạt động. Gia tốc kế và con quay hồi
chuyển lấy mẫu 50 Hz, cắt thành cửa sổ 2,56 s chồng lấn 50%. Mỗi cửa sổ có **561 đặc trưng** dẫn
xuất (miền thời gian và miền tần số), đã được chuẩn hoá về [−1, 1].

| Tập | Số cửa sổ | Số người |
|---|---|---|
| Train | 7.352 | 21 |
| Test | 2.947 | 9 (không trùng với train) |

## Cách lấy

Không cần làm gì: lần chạy đầu, `src/data.py` tự tải zip (~61 MB, bên trong là một zip lồng nhau),
giải nén vào `data/UCI HAR Dataset/` rồi cache thành `data/har.parquet` (~30 MB). Các lần sau chỉ
đọc file parquet.

Dữ liệu **không** được commit (xem `.gitignore` trong thư mục này) vì sau khi giải nén nặng ~270 MB
và có thể tải lại bất cứ lúc nào.

## Lưu ý

- `features.txt` có **42 tên đặc trưng bị trùng**, ví dụ các cột `fBodyAcc-bandsEnergy()-1,8` lặp
  lại cho các trục X/Y/Z. `src/data.py` thêm hậu tố `__<số thứ tự cột>` cho các tên trùng.
- Giữ **nguyên** cách chia train/test theo người của bộ gốc. `split_by_subject()` có `assert` kiểm tra
  không người nào xuất hiện ở cả hai tập.
