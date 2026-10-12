# Nguồn dữ liệu

**Chest X-Ray Images (Pneumonia)** (Kaggle, Paul Mooney), gốc từ Kermany và cộng sự (2018), *Cell* 172(5).
<https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>

Ảnh X-quang ngực thẳng (trước–sau) của **bệnh nhi 1–5 tuổi** tại Trung tâm Y tế Phụ nữ và Trẻ em Quảng Châu (Trung Quốc).
Nhãn do 2 bác sĩ đọc, bác sĩ thứ ba kiểm tra lại tập test.

| Tập gốc | NORMAL | PNEUMONIA | Tổng | % PNEUMONIA |
|---|---:|---:|---:|---:|
| train | 1.341 | 3.875 | 5.216 | 74,3% |
| val | 8 | 8 | **16** | 50,0% |
| test | 234 | 390 | 624 | 62,5% |

Tên file ghi rõ tác nhân: `person1_bacteria_1.jpeg`, `person1_virus_6.jpeg` (PNEUMONIA), `IM-0115-0001.jpeg`,
`NORMAL2-IM-1427-0001.jpeg` (NORMAL). Phần `personN` / `IM-NNNN` là **mã bệnh nhân**. Một bệnh nhân có thể có tới 31 phim.

## Cách lấy

Không cần làm gì. Lần chạy đầu, `src/data_pipeline.py` gọi `kagglehub.dataset_download(...)` (bộ công khai, không cần
API key, file nén ~2,3 GB, có 2 bản sao `chest_xray/` và `chest_xray/chest_xray/` + rác `__MACOSX`). Mỗi ảnh được đọc 1 lần,
chuyển xám, resize 224×224 rồi lưu thành **`data/cache_224.npz`** (uint8, 294 MB, ~1 phút). Các lần sau chỉ đọc file này.

Ảnh gốc và cache **không** được commit (xem `.gitignore`) vì tải lại được bất cứ lúc nào.

## Chia lại dữ liệu

Tập val gốc chỉ có 16 ảnh → **gộp vào train** (5.232 ảnh) rồi tự tách 1/7 ≈ 14,3% làm val bằng
`StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)`: giữ tỉ lệ lớp **và** không để một bệnh nhân nằm ở
cả train lẫn val (nếu không, model "nhận ra" bệnh nhân thay vì bệnh). Tập test gốc giữ nguyên.

| Tập | Số ảnh | NORMAL | PNEUMONIA | Bệnh nhân | % PNEUMONIA |
|---|---:|---:|---:|---:|---:|
| train | 4.484 | 1.156 | 3.328 | 2.452 | 74,2% |
| val | 748 | 193 | 555 | 409 | 74,2% |
| test | 624 | 234 | 390 | 427 | 62,5% |

## Lưu ý

- **Mã bệnh nhân ở test được đánh số riêng.** 170 mã `personN` xuất hiện ở cả train gốc và test gốc, nhưng khác tác nhân
  (vd `person1_bacteria` ở train, `person1_virus` ở test) và độ khác biệt điểm ảnh giữa các cặp này (23–106) bằng các cặp
  ngẫu nhiên (28–53). Đây là trùng số, không phải cùng người, nên test gốc được dùng nguyên.
- Kích thước ảnh gốc rất khác nhau: rộng 384–2.916 px (trung vị 1.281), cao 127–2.713 px (trung vị 888), **98,9% nằm
  ngang**. 283 ảnh lưu dạng RGB, còn lại ảnh xám → tất cả đổi về xám. Resize thẳng về 224×224 (không giữ tỉ lệ),
  giống cách làm phổ biến với bộ này.
- Nhiều phim PNEUMONIA có chữ/ký hiệu thêm (giờ chụp, mã máy `KV70`, điện cực) mà phim NORMAL ít có → nguy cơ model học
  "đường tắt". Xem mục Grad-CAM trong README.
