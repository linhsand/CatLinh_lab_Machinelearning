# Báo cáo kết quả — TT-01 KNN Classifier: Sàng lọc nguy cơ tiểu đường

## 1. Dữ liệu và xử lý thiếu
Bộ dữ liệu Pima Indians Diabetes có 5 cột không thể mang giá trị 0 về mặt y sinh
(`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`). Các giá trị 0 này
được coi là dữ liệu thiếu, thay bằng `NaN` rồi impute bằng median trong Pipeline
(để tránh rò rỉ thông tin từ tập test vào tập train). Tỷ lệ thiếu đáng chú ý:

| Cột | % thiếu |
|---|---|
| Insulin | 48,7% |
| SkinThickness | 29,6% |

Đây là mức thiếu lớn, vì vậy lựa chọn median-impute (ổn định hơn mean trước outlier)
là hợp lý, nhưng cần lưu ý sai số impute ở Insulin có thể ảnh hưởng đáng kể tới mô hình.

## 2. Vai trò của chuẩn hoá dữ liệu
So sánh KNN K=5 có và không chuẩn hoá (`reports/comparison_scale_vs_noscale.csv`):

| | Recall | Accuracy |
|---|---|---|
| Baseline (Dummy) | 0,000 | 0,649 |
| KNN K=5 — không chuẩn hoá | 0,500 | 0,675 |
| KNN K=5 — có chuẩn hoá | 0,611 | 0,753 |

Recall tăng từ 0,5 lên 0,611 khi chuẩn hoá, vì KNN dựa trên khoảng cách Euclid/Manhattan:
các đặc trưng có thang đo lớn (ví dụ Insulin, Glucose) sẽ lấn át các đặc trưng thang đo nhỏ
nếu không scale, làm méo khoảng cách và chọn sai láng giềng.

## 3. Chọn tham số bằng GridSearchCV
`GridSearchCV` dò K ∈ {1,3,...,31}, `weights` ∈ {uniform, distance},
`metric` ∈ {euclidean, manhattan}, với `scoring="recall"` (không dùng accuracy vì bài
toán y tế cần ưu tiên phát hiện ca dương tính, chi phí bỏ sót bệnh nhân cao hơn chi phí
báo động giả). Pipeline (impute + scale + KNN) được đặt **bên trong** `GridSearchCV` với
`StratifiedKFold(5, shuffle=True, random_state=42)` để impute/scale được fit lại đúng
trên từng fold train, chống rò rỉ dữ liệu.

## 4. Sanity check K=1
Với K=1, accuracy trên chính tập train đạt 1,0 — dự đoán như kỳ vọng vì mỗi điểm train
là láng giềng gần nhất của chính nó. Đây không phải là dấu hiệu mô hình tốt, mà là minh
chứng cho overfitting; kết quả này không phản ánh khả năng tổng quát hoá và không được
dùng để chọn mô hình cuối cùng (mô hình cuối chọn qua CV trên tập train, không dùng train
accuracy).

## 5. Đánh giá trên tập test (chạm 1 lần duy nhất)
Ngưỡng phân loại (threshold) được dò bằng `StratifiedKFold` 5-fold **trên train**
(hàm `find_best_threshold`, mỗi fold `clone()` lại pipeline nên không rò rỉ dữ liệu). Vì K=13,
`weights="uniform"`, `predict_proba` chỉ nhận 14 giá trị rời rạc (bội số của 1/13), nên các
threshold ứng viên được lấy từ chính các giá trị xác suất thực sự xuất hiện trên các fold
validation (điểm giữa hai mức liền kề) — cho ra đúng 15 điểm vận hành thực sự khác nhau, thay
vì quét một lưới liên tục bước 0.01 (51 giá trị) mà phần lớn là ảo vì rơi vào cùng một mức
lượng tử. Trong 15 điểm đó, điểm được chọn là điểm có recall cao nhất trong số các điểm đạt
precision ≥ 0,50 trên CV — **ràng buộc 0,50 là một giả định làm việc** (mỗi 2 ca cảnh báo có
tối đa 1 ca âm tính thật), dựa trên lập luận bỏ sót ca bệnh (FN) tốn kém hơn báo động giả (FP,
chỉ tốn thêm một lượt xét nghiệm khẳng định), **chứ không dựa trên số liệu vận hành thực tế
của phòng khám nào** — cần được xác nhận lại trước khi triển khai.

Một vài điểm vận hành khác trong cùng 15 điểm đó (CV trên train), để thấy rõ đánh đổi:

| Threshold | Recall (CV) | Precision (CV) | Ghi chú |
|---|---|---|---|
| 0,04 | 0,981 | 0,404 | gần như báo dương tính tất cả, quá nhiều báo động giả |
| **0,19** | **0,921** | **0,505** | **điểm được chọn** — recall cao nhất đạt precision ≥ 0,50 |
| 0,35 | 0,795 | 0,595 | cân bằng hơn, đổi recall lấy ít báo động giả hơn |
| 0,42 | 0,668 | 0,657 | precision/recall xấp xỉ nhau |
| 0,50 | 0,603 | 0,721 | ưu tiên precision, gần với điểm GridSearchCV mặc định (K=13, ngưỡng 0,5) |

Model tốt nhất (GridSearchCV, K=13) chỉ được áp lên tập test **một lần duy nhất** với threshold
0,19 đã chọn. Ma trận nhầm lẫn trên tập test (`reports/confusion_matrix.png`,
`reports/results_log.json`):

```
            Dự đoán 0   Dự đoán 1
Thực 0         57           43
Thực 1         8            46
```

- Recall = 0,852 → mô hình phát hiện được khoảng 85% các ca dương tính thực sự (8/54 ca bị bỏ sót).
- Precision = 0,517 → trong các ca được cảnh báo dương tính, khoảng 52% là dương tính thực sự (43/100 ca thực âm tính bị báo động giả).
- F1 = 0,643
- Accuracy = 0,669 — thấp, gần ngang baseline Dummy (0,649), vì threshold thấp nên model
  nghiêng mạnh về việc dự đoán dương tính, đánh đổi accuracy để lấy recall.
- PR-AUC = 0,616
- Threshold = 0,19

**Góc nhìn nghiệp vụ:** với ngưỡng này, cứ 100 ca thực sự âm tính thì có 43 ca bị gọi xét
nghiệm lại oan. Nếu phòng khám không đủ năng lực xử lý khối lượng tái xét nghiệm này, nên
chuyển sang một điểm vận hành precision cao hơn trong bảng trên (ví dụ threshold 0,35 hoặc
0,42) — bảng ở trên cho phép chọn theo năng lực thực tế thay vì cố định một con số.

### Hạn chế và hướng khắc phục chưa thực hiện
Recall test (0,852) đã vượt khoảng tham chiếu (0,65–0,75) nhưng đổi lại 43 cảnh báo giả trên
100 ca thực sự âm tính — precision và accuracy đều thấp. Các hướng cải thiện có thể thử thêm
nhưng chưa được đánh giá trong lần chạy này:
- Đặt lại ràng buộc dò threshold theo năng lực thực tế của phòng khám (ví dụ giới hạn tỷ lệ FP
  chấp nhận được) thay vì ràng buộc precision ≥ 0,50 không giải thích.
- Feature engineering (ví dụ tạo biến tương tác Glucose×BMI) hoặc thử impute bằng KNNImputer
  thay vì median, đặc biệt với Insulin (48,7% thiếu).
- Thử oversampling (SMOTE) trên tập train.
- KNN không hỗ trợ `class_weight`; có thể chuyển hẳn sang mô hình hỗ trợ trọng số lớp
  (Logistic Regression, xem mục 6) nếu cần kiểm soát đánh đổi recall/precision mượt hơn.

## 6. So sánh KNN tối ưu với Logistic Regression
So sánh bằng 5-fold CV trên tập train (không chạm test), kết quả trong
`reports/comparison_knn_vs_logreg.csv`. Logistic Regression dùng `class_weight="balanced"`
nên thường cho recall cao hơn KNN nhưng đổi lại precision thấp hơn — đây là một baseline
tham khảo hợp lý nếu recall vẫn là ưu tiên hàng đầu và cần một mô hình dễ diễn giải hơn.

## 7. Kết luận
Pipeline chống rò rỉ dữ liệu đúng chuẩn (impute + scale nằm trong CV/GridSearchCV), chọn
đúng metric tối ưu (recall) cho bài toán y tế, và có thực nghiệm định lượng chứng minh vai
trò của chuẩn hoá. Recall trên tập test (0,852) vượt khoảng tham chiếu của đề bài (0,65–0,75),
nhưng đây không phải là điểm dừng lý tưởng: precision (0,517) và accuracy (0,669) đều thấp,
tức cứ khoảng 100 ca thực sự âm tính thì có 43 ca bị cảnh báo nhầm là dương tính. Với một hệ
thống sàng lọc, đây có thể là đánh đổi chấp nhận được (bỏ sót bệnh đắt hơn báo động giả), nhưng
mức FP cụ thể cần được phòng khám xác nhận là khả thi về mặt vận hành (đủ nguồn lực xét nghiệm
lại) trước khi triển khai — đây là hạn chế chính cần làm rõ ở vòng lặp tiếp theo, cùng các hướng
đã nêu ở mục 5.

## 8. Cách chạy
```bash
cd <thư mục dự án TT-01>
pip install -r requirements.txt
python train.py
```
Script tự tạo `reports/` và `models/` nếu chưa có, và đọc dữ liệu từ đường dẫn tuyệt đối
`data/pima-indians-diabetes.csv` tính theo vị trí file `train.py` (không phụ thuộc thư mục
đang đứng khi chạy lệnh).