# Nguồn dữ liệu

**Metro Interstate Traffic Volume** (UCI Machine Learning Repository, John Hogue, 2019).
<https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume>

Lưu lượng xe theo giờ trên đường cao tốc I-94 hướng tây (trạm 301 của MnDOT, giữa Minneapolis và St Paul, Minnesota, Mỹ),
kèm thời tiết (OpenWeatherMap) và ngày lễ. 48.204 dòng, từ 2012-10-02 09:00 đến 2018-09-30 23:00.

| Cột | Ý nghĩa |
|---|---|
| `holiday` | Tên ngày lễ (chỉ ghi ở dòng 0h), `None` nếu không phải lễ |
| `temp` | Nhiệt độ (Kelvin) |
| `rain_1h`, `snow_1h` | Lượng mưa / tuyết trong giờ (mm) |
| `clouds_all` | Mây che phủ (%) |
| `weather_main`, `weather_description` | Thời tiết (11 nhóm / 38 mô tả) |
| `date_time` | Giờ địa phương |
| `traffic_volume` | **Nhãn:** số xe / giờ (0–7.280) |

## Cách lấy

Không cần làm gì. Lần chạy đầu, `src/data.py` tải file zip từ UCI và lưu `Metro_Interstate_Traffic_Volume.csv.gz`
(~400 KB) vào thư mục này. Dữ liệu **không** được commit (xem `.gitignore`).

Đọc với `keep_default_na=False`: chuỗi `"None"` ở cột `holiday` nghĩa là "không phải ngày lễ". Pandas mặc định sẽ biến nó
thành NaN.

## Bốn vấn đề và cách xử lý (`data.clean`)

| Vấn đề | Số liệu | Xử lý |
|---|---|---|
| 1. Lỗ hổng thời gian | 52.551 giờ trong khoảng, chỉ **40.575** giờ có dữ liệu → thiếu **11.976 giờ**. Lỗ dài nhất **308 ngày** (2014-08-08 → 2015-06-11) | Reindex theo giờ. Lỗ ≤ 3 giờ (2.771 giờ) nội suy tuyến tính + cột cờ `noi_suy`. Lỗ > 3 giờ (136 lỗ, 9.205 giờ) để NaN, chuỗi bị cắt thành nhiều đoạn |
| 2. Ngoại lai phi lý | `temp = 0 K` (10 dòng, 31/01–02/02/2014), `rain_1h = 9.831 mm` (1 dòng) | Đặt NaN rồi nội suy |
| 3. Trùng `date_time` | **7.629** dòng trùng. Cùng `traffic_volume`, chỉ khác mô tả thời tiết (vd "Rain" + "Drizzle") | Gộp 1 dòng/giờ. One-hot `weather_main` lấy max, giữ cả 2 điều kiện |
| 4. `holiday` chỉ ở dòng đầu ngày | 53 dòng ghi tên lễ | Lan ra cả 24 giờ của ngày đó → **1.272 giờ** ngày lễ |

Không cửa sổ nào được chứa giờ NaN (`sequences.valid_anchors` kiểm tra), và nhãn dùng để train/đánh giá chỉ lấy giờ có
**dữ liệu thật** (không lấy giờ nội suy).

## Chia theo thời gian

70 / 15 / 15 trên các giờ có giá trị (thật + nội suy), **không xáo trộn**:

| Tập | Từ | Đến |
|---|---|---|
| train | 2012-10-02 09:00 | 2017-04-06 10:00 |
| val | 2017-04-06 11:00 | 2018-01-02 19:00 |
| test | 2018-01-02 20:00 | 2018-09-30 23:00 |

Train chứa cả lỗ 308 ngày. Val và test gần như liên tục.
