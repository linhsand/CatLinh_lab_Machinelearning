# TT-03 — RANDOM FOREST
## Ai sẽ mở sổ tiết kiệm? — Tối ưu danh sách gọi telesales

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 3](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-03-Feature-Eng-Tree) |
| 🧠 **Nhóm** | Phân loại · Ensemble (Bagging) |
| 🔧 **Thuật toán** | Random Forest |
| 🏭 **Lĩnh vực** | Ngân hàng · Telesales · Marketing |
| ⏱ **Thời lượng** | 6–8 giờ |
| 📈 **Độ khó** | ⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   Một cây (TT-02) rất KHÔNG ỔN ĐỊNH: đổi chút dữ liệu → cây khác hẳn.
   Random Forest chữa bằng cách trồng NHIỀU cây rồi cho bỏ phiếu.

     Cây 1 ──┐   mỗi cây học trên:
     Cây 2 ──┤     • một mẫu BOOTSTRAP khác nhau (lấy có hoàn lại)
     Cây 3 ──┼──▶  • một TẬP CON đặc trưng ngẫu nhiên tại mỗi nút
      ...    │
     Cây 500─┘   → BỎ PHIẾU ĐA SỐ

   Hai nguồn ngẫu nhiên này làm các cây KHÁC NHAU → sai số triệt tiêu lẫn nhau
   → giảm VARIANCE mà không tăng bias.
```

---

## 2. BÀI TOÁN THỰC TẾ

```
   Ngân hàng chạy chiến dịch telesales mời mở sổ tiết kiệm có kỳ hạn.
   45.000 khách hàng · chỉ ~11% đồng ý.

   Đội telesales có 20 người, mỗi người gọi 50 cuộc/ngày = 1.000 cuộc/ngày.
   Gọi hết 45.000 khách mất 45 ngày và tốn ~450 triệu chi phí nhân sự.

   → Cần XẾP HẠNG khách theo khả năng đồng ý, gọi 5.000 người
     có xác suất cao nhất trước.
   → Metric: PRECISION@5000 và LIFT so với gọi ngẫu nhiên.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | Bank Marketing (UCI) |
| **Link** | https://archive.ics.uci.edu/dataset/222/bank+marketing |
| **File** | `bank-additional-full.csv` (phân tách bằng dấu `;`) |
| **Kích thước** | 41.188 dòng × 21 cột |
| **Nhãn** | `y` (yes/no) — khoảng **11,3% yes** |

**Nhóm cột:** khách hàng (`age`, `job`, `marital`, `education`, `housing`, `loan`),
chiến dịch (`contact`, `month`, `day_of_week`, `campaign`, `pdays`, `previous`),
kinh tế vĩ mô (`emp.var.rate`, `cons.price.idx`, `euribor3m`, `nr.employed`)

### 🚨 BẪY LỚN NHẤT — cột `duration` gây RÒ RỈ

```
   duration = thời lượng cuộc gọi tính bằng giây.

   Vấn đề: chỉ biết được SAU KHI đã gọi xong.
   Mà nếu khách nói chuyện 20 phút → gần như chắc chắn họ đồng ý.

   → Để cột này lại: AUC vọt lên ~0,94 (đẹp giả tạo)
   → Bỏ cột này    : AUC ~0,79 (con số THẬT, dùng được để chọn ai gọi)

   ⚠️ Chính tài liệu UCI ghi rõ: "should be discarded for a realistic
      predictive model". PHẢI BỎ cột này.
```

> Đây là ví dụ **rò rỉ dữ liệu** hoàn hảo để luyện tập. Bài yêu cầu học viên chạy
> **cả hai** phiên bản để tận mắt thấy chênh lệch.

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. Siêu tham số quan trọng

| Tham số | Ý nghĩa | Khuyến nghị |
|---------|---------|-------------|
| `n_estimators` | Số cây | 300–500 (càng nhiều càng ổn định, chỉ tốn thời gian) |
| `max_features` | Số đặc trưng xét mỗi nút | `'sqrt'` — ⭐ nguồn ngẫu nhiên chính |
| `max_depth` | Độ sâu | `None` thường ổn, giới hạn nếu overfit |
| `min_samples_leaf` | Mẫu tối thiểu mỗi lá | 1–5 |
| `class_weight` | Xử lý lệch | `'balanced_subsample'` |
| `n_jobs` | Số luồng | `-1` (dùng hết CPU) |

> 💡 `max_features='sqrt'` là thứ phân biệt Random Forest với Bagging thường.
> Đặt `max_features=None` → các cây giống nhau → mất hết lợi ích.

### 4.2. OOB Score — validation miễn phí

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=400, max_features='sqrt',
                            class_weight='balanced_subsample',
                            oob_score=True, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
print("OOB score:", rf.oob_score_)     # ước lượng trên mẫu không được bootstrap
```

Mỗi cây chỉ học trên ~63% dữ liệu; 37% còn lại (out-of-bag) dùng để chấm điểm
→ có ước lượng ngoài mẫu **mà không cần tách tập validation**.

### 4.3. ⚠️ Feature importance mặc định bị THIÊN LỆCH

```
   feature_importances_ của sklearn (dựa trên giảm Gini) ưu ái:
     • biến LIÊN TỤC (nhiều điểm cắt khả dĩ)
     • biến phân loại NHIỀU MỨC

   → Với dữ liệu này (job 12 mức, month 10 mức) kết quả dễ gây hiểu sai.
   → Dùng PERMUTATION IMPORTANCE mới đáng tin:
```

```python
from sklearn.inspection import permutation_importance
r = permutation_importance(rf, X_test, y_test, n_repeats=10,
                           scoring='average_precision', random_state=42)
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Đọc CSV với sep=';'
   ☐ 2. ⭐ Chạy 2 phiên bản: CÓ duration và KHÔNG có duration
        → lập bảng so sánh AUC → viết nhận xét về rò rỉ
        → từ bước 3 trở đi CHỈ dùng phiên bản KHÔNG có duration
   ☐ 3. Xử lý pdays = 999 (nghĩa là "chưa từng liên hệ") → tạo cột cờ riêng
   ☐ 4. Xử lý giá trị 'unknown' ở job/education/housing → coi là 1 mức riêng
   ☐ 5. Pipeline: OneHotEncoder(handle_unknown='ignore') cho biến phân loại
   ☐ 6. Baseline: DummyClassifier + Decision Tree đơn (TT-02) để so sánh
   ☐ 7. Random Forest, in oob_score_
   ☐ 8. Vẽ đường AUC theo n_estimators = 10..500 → tìm điểm bão hoà
   ☐ 9. So sánh feature_importances_ vs permutation_importance
   ☐ 10. Tính PRECISION@5000 và LIFT so với gọi ngẫu nhiên
   ☐ 11. Vẽ đường LIFT / Cumulative gain
```

```python
def lift_at_k(y_true, y_proba, k):
    idx = np.argsort(y_proba)[-k:]
    tl_top = y_true.iloc[idx].mean()
    return tl_top / y_true.mean()        # gọi nhóm này hiệu quả gấp mấy lần ngẫu nhiên?
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ Có bảng so sánh CÓ/KHÔNG cột duration + giải thích rò rỉ
   ☐ Xử lý đúng pdays = 999
   ☐ Báo cáo OOB score
   ☐ Có biểu đồ AUC theo số cây → chỉ ra điểm bão hoà
   ☐ Có so sánh 2 loại feature importance
   ☐ Báo cáo PRECISION@5000 và LIFT bằng con số cụ thể
   ☐ So sánh RF với 1 cây đơn → chứng minh ensemble tốt hơn
```

**Mức tham chiếu (không có `duration`):** ROC-AUC ~0,78–0,80, PR-AUC ~0,42–0,48.
Nếu AUC > 0,93 → bạn quên bỏ `duration`.

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Giữ cột `duration` | Rò rỉ — model vô dụng khi triển khai thật |
| Để `pdays = 999` như số | Model coi là "999 ngày trước" — sai bản chất |
| Tin `feature_importances_` mặc định | Kết luận sai về biến nào quan trọng |
| `n_estimators` quá ít (10–50) | Chưa tận dụng được sức mạnh ensemble |
| Dùng accuracy | 88,7% chỉ bằng cách đoán "no" cho tất cả |

---

## 8. SẢN PHẨM NỘP

```
TT-03-RandomForest-<HoTen>/
├── README.md                          ← có mục "BÀI HỌC VỀ RÒ RỈ DỮ LIỆU"
├── notebooks/{01_leakage_demo.ipynb, 02_random_forest.ipynb}
├── src/train.py
├── models/rf_pipeline.joblib
├── outputs/danh_sach_goi_top5000.csv  ← ⭐ sản phẩm bàn giao cho telesales
├── reports/{auc_theo_so_cay.png, permutation_importance.png, lift_curve.png}
└── requirements.txt
```

---

## 9. MỞ RỘNG

```
   1. So sánh Bagging (max_features=None) vs Random Forest (max_features='sqrt')
      → chứng minh việc lấy ngẫu nhiên đặc trưng thật sự có tác dụng
   2. Tính chi phí–lợi ích: 1 cuộc gọi tốn 10.000đ, 1 khách mở sổ lãi 500.000đ
      → gọi bao nhiêu người thì LỢI NHUẬN cao nhất?
   3. So sánh với XGBoost (TT-08) trên cùng bộ dữ liệu
```

**Tham khảo:** [Buổi 3 — Tree & Random Forest](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-03-Feature-Eng-Tree/Tai-Lieu) · [Buổi 6 — Ensemble](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-06-Ensemble-EndToEnd/Tai-Lieu)
