# Nguồn dữ liệu

Bộ dữ liệu gốc: **NSL-KDD** (bản cải tiến của KDD Cup 99), công bố bởi
Canadian Institute for Cybersecurity (CIC) — Đại học New Brunswick
(https://www.unb.ca/cic/datasets/nsl.html).

Môi trường chạy notebook này không có sẵn bản tải chính thức từ trang CIC
(yêu cầu form/đăng ký), nên `KDDTrain+.txt` và `KDDTest+.txt` được tải qua
bản mirror công khai trên GitHub
(`jmnwong/NSL-KDD-Dataset`) — xác nhận **đúng 125.973 dòng train / 22.544
dòng test**, đúng số dòng công bố chính thức của NSL-KDD, không phải dữ liệu
giả lập/tổng hợp.

**Cấu trúc cột (43 cột, không có header trong file gốc):**

41 đặc trưng theo đúng thứ tự chuẩn NSL-KDD (`duration`, `protocol_type`,
`service`, `flag`, `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`,
`urgent`, `hot`, `num_failed_logins`, `logged_in`, `num_compromised`,
`root_shell`, `su_attempted`, `num_root`, `num_file_creations`,
`num_shells`, `num_access_files`, `num_outbound_cmds`, `is_host_login`,
`is_guest_login`, `count`, `srv_count`, `serror_rate`, `srv_serror_rate`,
`rerror_rate`, `srv_rerror_rate`, `same_srv_rate`, `diff_srv_rate`,
`srv_diff_host_rate`, `dst_host_count`, `dst_host_srv_count`,
`dst_host_same_srv_rate`, `dst_host_diff_srv_rate`,
`dst_host_same_src_port_rate`, `dst_host_srv_diff_host_rate`,
`dst_host_serror_rate`, `dst_host_srv_serror_rate`, `dst_host_rerror_rate`,
`dst_host_srv_rerror_rate`) + cột nhãn `label` (loại tấn công cụ thể, vd.
`normal`, `neptune`, `satan`, ...) + cột `difficulty_level` (độ khó gán bởi
bộ tạo dữ liệu, không dùng để huấn luyện).

**Đặc điểm quan trọng đã xác nhận bằng code (`src/train.py`):**
* `protocol_type` (3 mức), `service` (~70 mức), `flag` (11 mức) là dạng
  phân loại → one-hot encode (`handle_unknown='ignore'` vì tập test có vài
  mức `service` không xuất hiện trong tập train).
* Tập **test chứa 17 loại tấn công KHÔNG có trong tập train**
  (`apache2, httptunnel, mailbomb, mscan, named, processtable, ps, saint,
  sendmail, snmpgetattack, snmpguess, sqlattack, udpstorm, worm, xlock,
  xsnoop, xterm`) — mô phỏng tấn công zero-day, đây là **thiết kế cố ý** của
  bộ dữ liệu, không phải lỗi tải dữ liệu.
* Lớp U2R cực hiếm: 52/125.973 ≈ 0,041% ở tập train → gộp nhãn nhị phân
  `normal` (0) vs `attack` (1) trước khi huấn luyện chính, giữ `attack_category`
  (DoS/Probe/R2L/U2R) riêng để phục vụ EDA và phần mở rộng đa lớp.
