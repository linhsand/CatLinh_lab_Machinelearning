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

## Cách chia dữ liệu trong `src/train.py` / notebook

Dùng **toàn bộ** MNIST, không lấy mẫu con:

| Tập | Số ảnh | Nguồn | Dùng để |
|---|---|---|---|
| Train | 50.000 | 60.000 ảnh train chuẩn, tách stratify (`random_state=42`) | huấn luyện |
| Validation | 10.000 | phần còn lại của 60.000 ảnh train chuẩn | **mọi lựa chọn**: kiến trúc, activation, learning rate, ngưỡng human-in-the-loop, số epoch CNN |
| Test | 10.000 | 10.000 ảnh test chuẩn của MNIST | chấm **1 lần** ở cuối, sau khi mọi lựa chọn đã cố định |

`early_stopping=True` của `MLPClassifier` còn tự cắt thêm 10% *bên trong*
tập train (5.000 ảnh) để quyết định dừng — tách biệt với tập validation ở trên.
