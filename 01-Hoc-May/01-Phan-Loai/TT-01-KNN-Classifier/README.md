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
| KNN K=5 — không chuẩn hoá | 0,500 | 0.675 |
| KNN K=5 — có chuẩn hoá | 0,611 | 0.753 |

Recall tăng từ 0.5 lên 0.611 khi chuẩn hoá, vì KNN dựa trên khoảng cách Euclid/Manhattan: các đặc trưng có thang đo lớn (ví dụ Insulin, Glucose) sẽ lấn át các đặc trưng thang đo nhỏ nếu không scale, làm méo khoảng cách và chọn sai láng giềng.

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
Ngưỡng phân lại (threshold) được dò bằng StratifiedKFold5-fold trên train (hàm find_best_threshold, mỗi fold clone()lại pipeline nên không rò rỉ dữ liệu), ưu tiên recall với ràng buộc precision ≥ 0.50 trên CV. Kết quả: threshold = 0.16. Sau đó model tốt nhất (GridSearchCV, K=13) chỉ được áp lên tập test một lần duy nhất với threshold này. My trận nhầm lẫn trên tập test ( reports/confusion_matrix.png, reports/results_log.json):

```
            Dự đoán 0   Dự đoán 1
Thực 0         57           43
Thực 1         8            46
```

- Recall = 0.852 → mô hình phát hiện được khoảng 85% các ca dương tính thực sự (8/54 ca bị bỏ sót).
- Precision = 0.517 → trong các ca được cảnh báo dương tính, khoảng 52% là dương tính thực sự (43/89 là báo động giả).
- F1 = 0.643
- Accuracy = 0.669 — thấp, gần ngang baseline Dummy (0.649), vì threshold 0.16 rất thấp nên model nghiêng mạnh về việc dự đoán dương tính, đánh đổi accuracy để lấy recall.
- PR-AUC = 0.616
- Threshold = 0.16

Đánh đổi ở điểm vận hành này chưa được thẩm định kỹ: ràng buộc "precision ≥ 0.50" khi dò threshold there tự đặt, chưa có căn cứ nghiệp vụ rõ ràng (ví dụ năng lực xét nghiệm lại của phòng khám, chi phí một ca dương tính giả so với một ca bỏ sót). Ngoài ra với K=13 và weights="uniform", predict_probachỉ nhận các giá trị rời rạc dạng bội số của 1/13, nên lưới threshold bước 0.01 phần lớn không đổi kết quả — độ mịn của việc dò threshold thấp hơn vẻ ngoài của nó.

### Hạn chế và hướng khắc phục chưa thực hiện đầy đủ
Recall test chưa đạt mức tham chiếu. Các hướng cải thiện có thể thử thêm nhưng chưa được
đánh giá trong lần chạy này:
- Feature engineering (ví dụ tạo biến tương tác Glucose×BMI) hoặc thử impute bằng KNNImputer
  thay vì median, đặc biệt với Insulin (48,7% thiếu).
- Thử oversampling (SMOTE) trên tập train thay vì chỉ điều chỉnh ngưỡng.
- KNN không hỗ trợ `class_weight`; có thể thử `weights="distance"` kết hợp ngưỡng thấp hơn,
  hoặc chuyển hẳn sang mô hình hỗ trợ trọng số lớp (Logistic Regression, xem mục 6).

## 6. So sánh KNN tối ưu với Logistic Regression
So sánh bằng 5-fold CV trên tập train (không chạm test), kết quả trong
`reports/comparison_knn_vs_logreg.csv`. Logistic Regression dùng `class_weight="balanced"`
nên thường cho recall cao hơn KNN nhưng đổi lại precision thấp hơn — đây là một baseline
tham khảo hợp lý nếu recall vẫn là ưu tiên hàng đầu và cần một mô hình dễ diễn giải hơn.

## 7. Kết luận
Pipeline chống rò rỉ dữ liệu đúng chuẩn (impute + scale nằm trong CV/GridSearchCV), chọn đúng metric tối ưu (recall) cho bài toán y tế, và có thực nghiệm định lượng chứng minh vai trò của chuẩn hoá. Recall trên tập test (0.852) vượt khoảng tham chiếu của đề bài (0.65–0.75), nhưng đây không phải là điểm dừng lý tưởng: precision (0.517) và accuracy (0.669) đều thấp, tức cứ khoảng 100 ca thực sự âm tính thì có 43 ca bị cảnh báo nhầm là dương tính. Với một hệ thống sàng lọc, đây có thể là đánh đổi chấp nhận được (bỏ sót bệnh đắt hơn báo động giả), nhưng mức FP cụ thể cần được phòng khám xác nhận là khả thi về mặt vận hành (đủ nguồn lực xét nghiệm lại) trước chi three.

## 8. Cách chạy
```bash
cd <thư mục dự án TT-01>
pip install -r requirements.txt
python train.py
```
Script tự tạo `reports/` và `models/` nếu chưa có, và đọc dữ liệu từ đường dẫn tuyệt đối
`data/pima-indians-diabetes.csv` tính theo vị trí file `train.py` (không phụ thuộc thư mục
đang đứng khi chạy lệnh).