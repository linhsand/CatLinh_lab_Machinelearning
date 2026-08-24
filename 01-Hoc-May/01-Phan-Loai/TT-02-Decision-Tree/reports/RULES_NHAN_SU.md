# 5 QUY TẮC CHO PHÒNG NHÂN SỰ

(File này được sinh tự động bởi `src/train.py` từ dữ liệu thực tế — tỉ lệ nghỉ việc trung bình toàn bộ tập dữ liệu là **16.1%**, dùng làm mốc so sánh cho từng nhóm bên dưới. Các nhóm này bám theo các nút chia quan trọng nhất của cây quyết định — xem `cay_quyet_dinh.png` và `feature_importance.png`.)

## 1. Nhân viên MỚI (TotalWorkingYears <= 2 năm kinh nghiệm)
- Số nhân viên trong nhóm: 123 (8.4% tổng số)
- Tỉ lệ nghỉ việc của nhóm: 43.9% (chênh lệch so với trung bình chung: +27.8%)

## 2. Nhân viên có làm THÊM GIỜ (OverTime = Yes)
- Số nhân viên trong nhóm: 416 (28.3% tổng số)
- Tỉ lệ nghỉ việc của nhóm: 30.5% (chênh lệch so với trung bình chung: +14.4%)

## 3. Cấp bậc thấp (JobLevel = 1) VÀ có làm thêm giờ
- Số nhân viên trong nhóm: 156 (10.6% tổng số)
- Tỉ lệ nghỉ việc của nhóm: 52.6% (chênh lệch so với trung bình chung: +36.4%)

## 4. KHÔNG có cổ phần thưởng (StockOptionLevel = 0)
- Số nhân viên trong nhóm: 631 (42.9% tổng số)
- Tỉ lệ nghỉ việc của nhóm: 24.4% (chênh lệch so với trung bình chung: +8.3%)

## 5. Từng làm > 4 công ty trước đó VÀ tuổi <= 37
- Số nhân viên trong nhóm: 136 (9.3% tổng số)
- Tỉ lệ nghỉ việc của nhóm: 32.4% (chênh lệch so với trung bình chung: +16.2%)
