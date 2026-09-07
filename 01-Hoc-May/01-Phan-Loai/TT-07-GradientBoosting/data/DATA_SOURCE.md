# Nguồn dữ liệu

Bộ dữ liệu gốc: **Adult Census Income**, UCI Machine Learning Repository
(https://archive.ics.uci.edu/dataset/2/adult).

Môi trường chạy notebook này không có quyền truy cập trực tiếp
`archive.ics.uci.edu`, nên `adult.csv` được tải qua bản mirror công khai
trên GitHub (`jbrownlee/Datasets/adult-all.csv`) — cùng 48.842 dòng, cùng
phân phối nhãn (~23.9% `>50K`) như bản gốc UCI.

**Lưu ý:** bản mirror này đã được strip khoảng trắng thừa quanh các giá trị
chuỗi sẵn, trong khi bản UCI gốc có định dạng `' ?'` (khoảng trắng + dấu hỏi)
cho giá trị thiếu. Notebook và `src/train.py` vẫn viết code làm sạch **phòng
thủ** (`.str.strip()` trước khi so khớp `"?"`) để chạy đúng với cả hai định
dạng — đúng tinh thần bẫy #1/#2 mà đề bài (README.md) mô tả.

**Nhãn cũng được làm sạch phòng thủ tương tự:** file phân phối chính thức của
UCI (`adult.test`) ghi nhãn kèm dấu chấm cuối (`">50K."` thay vì `">50K"`) vì
đây vốn là dòng comment cuối file gốc. `adult.csv` (bản mirror dùng ở đây)
không có dấu chấm này, nhưng code vẫn `.str.rstrip(".")` trước khi so khớp
`">50K"` — nếu không, việc gộp thêm `adult.test` hoặc đổi sang nguồn UCI gốc
sẽ khiến toàn bộ nhãn bị đọc thành `0` một cách âm thầm (không lỗi, không
cảnh báo), làm sai lệch mọi kết quả phía sau mà không ai nhận ra.
