# Nguồn dữ liệu

**AG News Classification** (Zhang, Zhao & LeCun, 2015, *Character-level Convolutional Networks for Text Classification*),
trích từ kho tin AG's corpus (hơn 1 triệu bài từ hơn 2.000 nguồn tin, 2004–2005).
Hugging Face: <https://huggingface.co/datasets/fancyzhx/ag_news>

| | |
|---|---|
| Train | 120.000 tin, 30.000 / lớp |
| Test | 7.600 tin, 1.900 / lớp |
| Lớp | 0 World · 1 Sports · 2 Business · 3 Sci/Tech |
| Văn bản | tiêu đề + mô tả ngắn, trung bình 38 từ (p99 = 70, dài nhất 177) |

Không có tin trùng lặp trong train, và không có tin test nào xuất hiện trong train.

## Cách lấy

Không cần làm gì. Lần chạy đầu, `src/data.py` gọi `datasets.load_dataset("fancyzhx/ag_news")` rồi lưu
`data/ag_news.parquet`. Các lần sau đọc file này. Dữ liệu **không** được commit (xem `.gitignore`).

⚠️ Tên cũ `load_dataset("ag_news")` trong đề **không còn chạy** với `datasets` ≥ 4 (lỗi
`Repository id must be 'namespace/name'`). Bộ dữ liệu đã chuyển sang repo `fancyzhx/ag_news`.

## Rác HTML và cách làm sạch (`data.clean_text`)

| Loại rác | Ví dụ | Số tin (127.600) | Xử lý |
|---|---|---:|---|
| Mã ký tự mất dấu `&` | `AUSTRALIA #39;S` (= `'`), `#36;` (= `$`), `#151;` (= `—`) | 33.348 (26,1%) | Thêm `&` rồi `html.unescape` |
| Thẻ HTML đã mã hoá | `activist &lt;b&gt;...&lt;/b&gt;` | 5.542 (4,3%) | `unescape` 2 lần rồi xoá thẻ |
| `\` thay cho xuống dòng | `Wall Street's dwindling\band of` | 13.954 (10,9%) | Đổi thành khoảng trắng |
| `quot;` / `&amp;` | `quot;Hello quot;` | 5.056 (4,0%) | Như dòng 1 |

Sau khi làm sạch, cả 4 loại đều còn **0** tin.

## Chia dữ liệu

| Tập | Số tin | Dùng để |
|---|---:|---|
| train | 108.000 | fit (90% train gốc, stratify, seed 42) |
| val | 12.000 | EarlyStopping, chọn C / số head / epoch |
| test | 7.600 | **chỉ báo cáo** (test gốc) |
| train 20k | 20.000 | tập con stratify (5.000 / lớp) để fine-tune DistilBERT trên CPU, và TF-IDF đối chứng |

## Lưu ý

- Nhãn nguồn tin `(Reuters)` / `(AP)` / `(AFP)` có trong 20% tin và phân bố lệch: 29% tin World so với 14–20% các lớp khác.
  Đây là thông tin có sẵn khi phân loại thật, nên giữ lại.
- Tin năm 2004–2005: các mô hình sẽ yếu hơn với chủ đề mới (tên người, công ty, công nghệ sau 2005).
