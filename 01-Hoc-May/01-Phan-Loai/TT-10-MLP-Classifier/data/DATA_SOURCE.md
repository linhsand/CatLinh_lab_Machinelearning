# Nguồn dữ liệu

## 1. Khởi động — `load_digits()`
Bộ 1.797 ảnh 8×8 đi kèm sẵn trong `scikit-learn`, không cần tải:

```python
from sklearn.datasets import load_digits
digits = load_digits()
```

## 2. Chính — MNIST (`mnist_784`)
70.000 ảnh chữ số viết tay 28×28, tải qua OpenML:

```python
from sklearn.datasets import fetch_openml
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
```

Lần chạy đầu tiên cần internet; `fetch_openml` tự cache vào
`~/scikit_learn_data/` nên các lần chạy sau (script lẫn notebook) không tải
lại. Không commit dữ liệu MNIST vào git (xem `.gitignore` trong thư mục này)
vì file khá lớn (~55 MB nén) và có thể tải lại bất cứ lúc nào.

## Cỡ mẫu dùng trong `src/train.py` / notebook

Để giữ thời gian chạy toàn bộ 10+ cấu hình MLP (4 kiến trúc × 3 activation ×
3 learning rate + model cuối) ở mức vài phút thay vì hàng giờ, các thí
nghiệm so sánh trong bài này lấy mẫu **stratify** từ MNIST gốc:
`TRAIN_SAMPLE = 12.000`, `TEST_SAMPLE = 3.000` (đặt trong `src/train.py`).
Đây là lựa chọn thực dụng cho môi trường lab/CI, không phải giới hạn của
thuật toán — đổi `TRAIN_SAMPLE = None` trong `src/train.py` để chạy trên
toàn bộ 60.000/10.000 mẫu chuẩn của MNIST (sẽ lâu hơn nhiều, đặc biệt kiến
trúc `(256,128,64)`).
