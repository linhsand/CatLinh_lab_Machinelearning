# Nguồn dữ liệu

**Health Insurance Cross Sell Prediction** (Kaggle, Anmol Kumar).
<https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction>

Khách hàng đang mua bảo hiểm sức khoẻ, nhãn `Response` = 1 nếu quan tâm đến bảo hiểm ô tô.

| | |
|---|---|
| `train.csv` | 381.109 dòng × 12 cột (`id` + 10 đặc trưng + `Response`), **12,26%** quan tâm, không có giá trị thiếu |
| `test.csv` | 127.037 dòng, **không có nhãn** (dùng cho bảng xếp hạng Kaggle) → bài này **không dùng** |

## Cách lấy

Không cần làm gì. Lần chạy đầu, `src/data.py` gọi `kagglehub.dataset_download(...)` (bộ dữ liệu công khai, không cần
API key), rồi copy `train.csv` vào thư mục này. Các lần sau đọc thẳng `data/train.csv`.

Dữ liệu **không** được commit (xem `.gitignore`) vì tải lại được bất cứ lúc nào.

## Chia dữ liệu

`train.csv` được chia **70/15/15 có stratify** theo `Response`, `random_state=42` (`data.split`):

| Tập | Số dòng | Dùng để |
|---|---:|---|
| train | 266.775 | fit model, fit scaler / one-hot / bảng mã embedding |
| val | 57.167 | EarlyStopping, chọn kiến trúc, class_weight, cách mã hoá, ngưỡng gọi |
| test | 57.167 | **chỉ báo cáo** |

## Lưu ý

- `Region_Code` và `Policy_Sales_Channel` lưu dạng số thực (`28.0`) nhưng là **mã phân loại**. `load_raw()` đổi
  thành số nguyên. Mạng nơ-ron nhận chúng dưới dạng one-hot hoặc Embedding, LightGBM nhận dưới dạng `categorical_feature`.
- `Gender` và `Vehicle_Damage` là chuỗi, được đổi thành 0/1 (`Male`=1, `Yes`=1). `Vehicle_Age` có 3 mức
  (`< 1 Year`, `1-2 Year`, `> 2 Years`).
