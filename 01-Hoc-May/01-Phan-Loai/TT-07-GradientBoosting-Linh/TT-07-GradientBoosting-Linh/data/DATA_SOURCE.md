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
