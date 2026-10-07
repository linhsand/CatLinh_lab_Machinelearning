# Nguồn dữ liệu

**Ames Housing** (Dean De Cock, 2011), bản dùng trong cuộc thi Kaggle
[House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).

`train.csv`: **1.460 căn nhà × 81 cột** (`Id`, 79 đặc trưng, nhãn `SalePrice` bằng USD), bán tại Ames, Iowa,
2006–2010. Ý nghĩa từng cột và từng mã chữ xem trong `data_description.txt` của cuộc thi.

## Cách lấy

File đã có sẵn trong `data/`. Nếu bị xoá, `src/train.py` tự tải lại qua **OpenML**
(`fetch_openml(data_id=42165)`), một mirror công khai giữ nguyên nội dung của bản Kaggle (đã kiểm tra: đúng
1.460 dòng × 81 cột, đúng tên cột). Có tài khoản Kaggle thì có thể tải `train.csv` trực tiếp và ghi đè.

## Lưu ý

- 19 cột có giá trị thiếu. **17 cột** thiếu nghĩa là "không có tiện ích" (vd `PoolQC` = NaN là nhà không có bể bơi),
  chỉ **2 cột** thiếu thật (`LotFrontage`, `Electrical`). Xem `reports/phan_loai_gia_tri_thieu.csv`.
- `SalePrice` lệch phải mạnh (skew ≈ 1,88), nên huấn luyện trên `log1p`.
