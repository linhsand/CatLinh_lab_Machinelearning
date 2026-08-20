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
| KNN K=5 — không chuẩn hoá | 0,500 | — |
| KNN K=5 — có chuẩn hoá | 0,611 | — |

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
Ma trận nhầm lẫn trên tập test với mô hình tốt nhất và ngưỡng phân loại được tối ưu qua
CV trên train (`reports/confusion_matrix.png`, `reports/results_log.json`):

```
            Dự đoán 0   Dự đoán 1
Thực 0         82           18
Thực 1         26           28
```

- Recall = 28 / (28+26) ≈ 0,519 → mô hình bỏ sót khoảng 48% bệnh nhân thực sự mắc tiểu đường.
- Recall test (0,519) **thấp hơn** khoảng tham chiếu 0,65–0,75 yêu cầu trong đề bài.

### Hạn chế và hướng khắc phục chưa thực hiện đầy đủ
Recall test chưa đạt mức tham chiếu. Các hướng cải thiện có thể thử thêm nhưng chưa được
đánh giá trong lần chạy này:
- Hạ ngưỡng phân loại (`threshold`) sâu hơn nữa, đánh đổi lấy precision thấp hơn.
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
Pipeline chống rò rỉ dữ liệu đúng chuẩn (impute + scale nằm trong CV/GridSearchCV), chọn
đúng metric tối ưu (recall) cho bài toán y tế, và có thực nghiệm định lượng chứng minh vai
trò của chuẩn hoá. Tuy nhiên recall trên tập test (0,519) chưa đạt khoảng tham chiếu của đề
bài (0,65–0,75); đây là hạn chế chính cần cải thiện ở vòng lặp tiếp theo, theo các hướng đã
nêu ở mục 5.
