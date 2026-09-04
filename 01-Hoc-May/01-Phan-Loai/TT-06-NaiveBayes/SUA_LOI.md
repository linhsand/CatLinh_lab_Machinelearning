# NHẬT KÝ SỬA LỖI — TT-06

Ánh xạ từng điểm trong nhận xét sang thay đổi cụ thể trong mã nguồn và tài liệu.

---

## ❶ NGHIÊM TRỌNG — `data/spam.csv` là file giả lập, không phải bộ UCI

**Nhận xét:** repo chứa file 1.144 dòng do `data/make_demo_data.py` tự sinh, trong khi README §3/§7
khẳng định đó là bộ UCI 5.572 dòng. Notebook chạy trên dữ liệu thật nhưng file thật không có trong
repo, nên `python src/train.py` không tái tạo được README.

**Đã sửa:**

| Việc | Chi tiết |
| :--- | :--- |
| Xoá `data/make_demo_data.py` | Không còn đường nào sinh ra dữ liệu giả trong dự án. |
| Thêm `data/download_data.py` | Tải bộ thật rồi **xác thực 3 lớp**: SHA-256 `440e6ea9…`, số dòng thô 5.572 + phân phối 4.825/747, số dòng sau lọc trùng 5.169 + phân phối 4.516/653. Sai bất kỳ điều kiện nào → thoát mã lỗi 1, **không ghi file**. |
| Thêm `data/.gitignore` | `spam.csv` không bao giờ vào git nữa, nên không thể commit nhầm file giả lần thứ hai. |
| `train.py` kiểm tra lại khi nạp | Sai số dòng hoặc sai phân phối nhãn → in thông báo chỉ rõ cách khắc phục rồi thoát mã lỗi 1. Muốn dùng bộ khác phải nói rõ bằng `--skip-data-check`. |
| README §3.1 | Ghi thẳng sự cố cũ và lý do dữ liệu không được commit. |

**Kiểm chứng:** clone mới → `pip install -r requirements.txt` → `python data/download_data.py` →
`python src/train.py` sinh lại toàn bộ `reports/` khớp với các bảng trong README.

---

## ❷ NGHIÊM TRỌNG — `reports/` đã commit phủ định chính README

**Nhận xét:** `bang_so_sanh_3_to_hop.csv` cho Precision = Recall = F1 = 1.0 ở cả 3 tổ hợp; lưới
alpha × ngram cho 1.0 ở cả 8 cấu hình (vô nghĩa); `ca_du_doan_sai.csv` chỉ có dòng tiêu đề trong
khi README trình bày bảng 10 ca sai.

**Đã sửa:**

* Toàn bộ `reports/` được sinh lại từ **một lần chạy duy nhất trên dữ liệu thật**. Không còn giá
  trị 1.0 nào; `ca_du_doan_sai.csv` chứa đủ **16 ca sai** (3 FP + 13 FN), không cắt bớt.
* `train.py` sinh thêm **`reports/ket_qua.md`** chứa mọi bảng số dưới dạng Markdown. README tuyên
  bố ngay đầu file rằng mọi con số của nó trích từ đây. Đây là biện pháp **cấu trúc**: README và
  `reports/` không còn hai nguồn sự thật độc lập để lệch nhau.
* `train.py` sinh **`reports/run_metadata.json`** ghi seed, phiên bản scikit-learn/pandas, dấu vân
  tay dữ liệu và thời điểm chạy.
* Thêm cảnh báo tự động: nếu số ca sai bằng 0, script in ra rằng đây là dấu hiệu điển hình của dữ
  liệu giả lập hoặc rò rỉ. Sự cố cũ sẽ tự tố cáo nó ngay trên màn hình.
* README §9.1 hướng dẫn cách đối chiếu README với `ket_qua.md` sau mỗi lần chạy.

---

## ❸ Ngưỡng và lưới siêu tham số được chọn trên tập TEST

**Nhận xét:** `precision_recall_curve` chạy trên `y_test` → T = 0.0009, rồi báo cáo Precision/Recall
trên chính tập test đó; lưới alpha × ngram cũng chấm bằng `y_test`. Đúng lỗi đã tự sửa ở TT-04/TT-05.

**Đã sửa** — tập test bị khoá và chỉ mở đúng một lần:

| Quyết định | Trước | Sau |
| :--- | :--- | :--- |
| Chọn tổ hợp vector hoá × mô hình | `y_test` | 5-fold CV trên train |
| Dò `alpha` × `ngram_range` | `y_test` | 5-fold CV trên train |
| Chọn ngưỡng quyết định | `precision_recall_curve(y_test, …)` | `precision_recall_curve` trên xác suất **out-of-fold** của train |
| Đánh giá cuối | cùng tập test đã dùng để chọn | Test mở một lần ở bước 12, ngưỡng đã đóng băng từ trước |

Bổ sung thêm:

* Đo **Precision tại ngưỡng đã chọn trên từng fold** để định lượng sai số: 0.9811 ± 0.0136,
  khoảng ±1 s.e. là [0.9750, 0.9872].
* EDA chuyển sang tính **chỉ trên train** — nhìn phân bố tập test rồi mới quyết định đặc trưng
  cũng là rò rỉ, dù nhẹ.
* Hình mới `chon_nguong_pr_curve.png` vẽ đường PR trên validation kèm điểm vận hành, nói rõ ngay
  trên tiêu đề rằng ngưỡng được chọn ở đâu.

**Kết quả thay đổi thế nào:** ngưỡng đi từ T = 0.0009 lên **T = 0.4170**, và Precision công bố đi
từ 0.9836 (rò rỉ) xuống **0.9752 (thật)** — tức **không đạt** ràng buộc 0.98. README §5.7 ghi nhận
thẳng điều này kèm phân tích: độ lệch nằm trong khoảng ±1 s.e. đã đo, chỉ cần bớt một False
Positive là ràng buộc đạt, và cách xử lý đúng là chốt ngưỡng với biên an toàn chứ không phải chỉnh
ngưỡng cho tới khi số liệu đẹp.

---

## ❹ `train.py` thoái bộ so với chuẩn của TT-02..05

**Nhận xét:** không có `main()` / `if __name__ == "__main__"`, `DATA_PATH` và `REPORT_DIR` theo cwd,
không `makedirs` → crash trên clone mới.

**Đã sửa:**

| Vấn đề | Cách sửa |
| :--- | :--- |
| Không có `main()` | Toàn bộ logic nằm trong các hàm; `main(argv=None)` điều phối; `if __name__ == "__main__": raise SystemExit(main())`. Import file này không còn chạy 300 dòng tác dụng phụ. |
| Đường dẫn theo cwd | `ROOT = Path(__file__).resolve().parents[1]`. Chạy được từ bất kỳ thư mục nào. |
| Không `makedirs` | `args.reports.mkdir(parents=True, exist_ok=True)` và tương tự cho `models`, ngay đầu `main()` trước mọi thao tác ghi. Đã kiểm chứng bằng cách xoá sạch cả hai thư mục rồi chạy lại. |
| Không có tham số dòng lệnh | `argparse`: `--data --reports --models --seed --test-size --folds --target-precision --skip-data-check`. |
| Thiếu thoát lỗi tử tế | Thiếu dữ liệu hoặc dữ liệu sai → thông báo chỉ rõ lệnh khắc phục, thoát mã 1, không để traceback trần. |
| Notebook | Cùng phương pháp, cùng seed với `train.py`; đường dẫn cũng neo theo repo root. |

---

## ❺ Các điểm sửa thêm (không có trong nhận xét)

Rà lại toàn bộ dự án thì thấy thêm bốn chỗ sai về nội dung chuyên môn:

**a) Diễn giải sai bảng top-20 từ.** Bảng cũ xếp theo $\log P(w \mid \text{spam})$ nên `to`, `the`,
`is`, `and` đứng đầu, rồi README gán cho chúng ý nghĩa nghiệp vụ ("`the` — mạo từ xác định"). Đại
lượng đó đo tần suất trong lớp spam chứ không đo khả năng phân biệt. Đã giữ bảng gốc (đúng yêu cầu
đề bài) nhưng nói rõ vì sao nó gây hiểu lầm, và thêm §5.8.1 xếp theo **log-odds**
$\log P(w|\text{spam}) - \log P(w|\text{ham})$ — cho ra `claim`, `prize`, `150p`, `have won`, tức
chữ ký thật của spam đầu số trả phí.

**b) Mô tả sai cơ chế zero-probability.** README cũ viết "chỉ cần một từ lạ là tích xác suất về 0".
Thực tế từ hoàn toàn lạ **vô hại** vì `CountVectorizer` loại nó khỏi từ vựng ngay từ khâu vector
hoá. Thủ phạm là từ **có** trong từ vựng nhưng đếm được 0 lần ở một lớp. Và hậu quả không phải xác
suất bằng 0 mà là **`NaN`**: $\log 0 = -\infty$ ở cả hai lớp, chuẩn hoá thành `NaN`, nhãn trả về
tuỳ ý. §5.4.1 chứng minh bằng bảng 5 tình huống chạy thật, trong đó có ca một tin rác lọt lưới chỉ
vì chứa chữ `'gt'`.

**c) Đo độ trễ sai chế độ.** Bản cũ đo `predict()` trên cả lô 1.034 tin rồi chia đều, gọi kết quả
là "0.019 ms/tin" và kết luận "nhanh gấp 250 lần SLA". Gateway xử lý từng tin khi nó đến, không xử
lý theo lô. Đo lại từng tin một: **p95 = 0.629 ms**, dư khoảng 8 lần so với SLA 5 ms. Kết luận
(Naive Bayes thừa nhanh) không đổi, nhưng biên an toàn thật là 8 lần chứ không phải 250 lần. §7.3
tính lại ngân sách độ trễ hai tầng từ con số đúng này.

**d) Kết luận vượt quá dữ liệu.** Bản cũ viết "Naive Bayes vượt trội hoàn toàn" dựa trên chênh lệch
F1 = 0.0045 giữa các tổ hợp — nhỏ hơn độ lệch chuẩn giữa các fold (0.0136). Đã sửa thành "cả ba
tương đương, chọn theo F1 out-of-fold một cách nhất quán", và ghi chú rằng so sánh với Logistic
Regression chưa hoàn toàn công bằng vì LogReg để nguyên `class_weight="balanced"` (vốn đẩy về phía
Recall) mà không được dò ngưỡng như Naive Bayes.

---

## Tóm tắt thay đổi số liệu

| Chỉ số | Bản trước (công bố) | Bản này (thật) | Vì sao khác |
| :--- | :---: | :---: | :--- |
| Nguồn dữ liệu của `reports/` | giả lập 1.144 dòng | UCI 5.169 dòng sau lọc trùng | ❶ |
| Tổ hợp được chọn | TF-IDF + BernoulliNB | TF-IDF + MultinomialNB | ❸ — chọn bằng CV thay vì test |
| Ngưỡng | T = 0.0009 | **T = 0.4170** | ❸ — chọn trên validation |
| Precision (spam) | 0.9836 | **0.9752** | ❸ — 0.9836 là số bị rò rỉ |
| Recall (spam) | 0.9160 | **0.9008** | ❸ |
| F1 (spam) | 0.9306 | **0.9365** | ❸ |
| Ma trận nhầm lẫn | 901 / 2 / 11 / 120 | **900 / 3 / 13 / 118** | ❸ |
| Độ trễ mỗi tin | 0.019 ms | **p95 = 0.629 ms** | ❺c — đo đúng chế độ |
| Số ca sai trong CSV | 0 (chỉ có header) | **16** | ❷ |
| Đạt ràng buộc Precision ≥ 0.98 | "đạt" | **không đạt trên test** | ❸ — ghi nhận trung thực |
