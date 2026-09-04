# KẾT QUẢ THỰC NGHIỆM — SINH TỰ ĐỘNG

> File này do `src/train.py` sinh ra, **không sửa tay**. Mọi bảng số trong
> `README.md` đều trích từ đây, nên README và `reports/` không thể mâu thuẫn nhau.

* Thời điểm chạy: `2026-09-04T14:28:18`
* Lệnh: `python src/train.py --seed 42 --folds 5 --target-precision 0.98`
* Dữ liệu: `data\spam.csv` — 5,572 dòng thô → 5,169 dòng sau khi lọc trùng
* Chia tập: train 4,135 / test 1,034 (80/20, stratify, seed 42)
* Phiên bản: scikit-learn 1.8.0, pandas 3.0.1

## 1. Thống kê độ dài tin nhắn (tập train)

| label   |   count |     mean |     std |   50% |   min |   max |
|:--------|--------:|---------:|--------:|------:|------:|------:|
| ham     |    3613 |  69.9029 | 54.6336 |    52 |     2 |   632 |
| spam    |     522 | 137.971  | 30.1626 |   148 |    13 |   224 |

## 2. So sánh tổ hợp — chấm bằng 5-fold CV trên TRAIN

Đây là bảng dùng để **chọn** mô hình, nên bắt buộc tính trên out-of-fold của train.

| Tổ hợp                          |   Precision (CV) |   Recall (CV) |   F1 (CV) |   Accuracy (CV) |   Train 1 lần (s) |   5-fold CV (s) |
|:--------------------------------|-----------------:|--------------:|----------:|----------------:|------------------:|----------------:|
| TfidfVectorizer + MultinomialNB |           0.9936 |        0.887  |    0.9372 |          0.985  |            0.2404 |           1.22  |
| TfidfVectorizer + BernoulliNB   |           0.9957 |        0.8812 |    0.935  |          0.9845 |            0.2819 |           1.199 |
| CountVectorizer + MultinomialNB |           0.9503 |        0.9157 |    0.9327 |          0.9833 |            0.2291 |           1.285 |
| Logistic Regression + TFIDF     |           0.9156 |        0.9349 |    0.9251 |          0.9809 |            0.3012 |           1.509 |
| Baseline: DummyClassifier       |           0      |        0      |    0      |          0.8738 |            0.1128 |           0.671 |

Tổ hợp Naive Bayes được chọn: **TfidfVectorizer + MultinomialNB**

## 3. Lưới alpha × ngram_range — chấm bằng 5-fold CV trên TRAIN

|   alpha | ngram_range   |   Precision (CV) |   Recall (CV) |   F1 (CV) |   Accuracy (CV) |
|--------:|:--------------|-----------------:|--------------:|----------:|----------------:|
|    0.1  | (1, 2)        |           0.9936 |        0.887  |    0.9372 |          0.985  |
|    0.01 | (1, 2)        |           0.9769 |        0.8927 |    0.9329 |          0.9838 |
|    0.1  | (1, 1)        |           0.9808 |        0.8793 |    0.9273 |          0.9826 |
|    0.01 | (1, 1)        |           0.9726 |        0.8831 |    0.9257 |          0.9821 |
|    0.5  | (1, 1)        |           0.9977 |        0.8333 |    0.9081 |          0.9787 |
|    0.5  | (1, 2)        |           0.9976 |        0.7931 |    0.8837 |          0.9736 |
|    1    | (1, 1)        |           1      |        0.751  |    0.8578 |          0.9686 |
|    1    | (1, 2)        |           1      |        0.6992 |    0.823  |          0.962  |

Cấu hình được chọn: **alpha = 0.1, ngram_range = (1, 2)**

## 4. Ngưỡng quyết định

| Hạng mục | Giá trị |
| :--- | :--- |
| Ràng buộc nghiệp vụ | Precision (spam) ≥ 0.98 |
| Ngưỡng chọn trên OOF của train | **T = 0.417011** |
| Precision ước lượng (OOF gộp) | 0.9812 |
| Recall ước lượng (OOF gộp) | 0.8985 |
| Precision theo từng fold | [0.9681, 0.9894, 0.9792, 1.0, 0.9691] |
| Precision trung bình ± độ lệch chuẩn | 0.9811 ± 0.0136 |
| Đạt ràng buộc trên validation | Có |

Độ lệch chuẩn giữa các fold cho thấy ràng buộc Precision ≥ 0.98 chỉ đúng
**theo kỳ vọng**. Trên một tập test hữu hạn (chỉ 131 tin spam), sai lệch một
vài False Positive đã đủ kéo Precision xuống dưới mục tiêu — đây là hạn chế
thống kê, không phải lỗi lập trình.

## 5. Top 20 từ đặc trưng của SPAM

### 5a. Theo log P(từ | spam)

|   Hạng | Từ/Cụm từ   |   log P(từ | spam) |
|-------:|:------------|-------------------:|
|      1 | to          |            -5.0681 |
|      2 | call        |            -5.1412 |
|      3 | your        |            -5.4771 |
|      4 | free        |            -5.4927 |
|      5 | for         |            -5.6416 |
|      6 | you         |            -5.7036 |
|      7 | or          |            -5.7501 |
|      8 | txt         |            -5.789  |
|      9 | now         |            -5.8088 |
|     10 | text        |            -5.8301 |
|     11 | mobile      |            -5.8935 |
|     12 | reply       |            -5.9092 |
|     13 | stop        |            -5.9095 |
|     14 | from        |            -5.9308 |
|     15 | the         |            -5.9322 |
|     16 | claim       |            -5.9957 |
|     17 | ur          |            -5.9977 |
|     18 | is          |            -6.0166 |
|     19 | www         |            -6.0393 |
|     20 | have        |            -6.0569 |

### 5b. Theo log-odds log P(w|spam) − log P(w|ham)

|   Hạng | Từ/Cụm từ   |   log P(w|spam) - log P(w|ham) |
|-------:|:------------|-------------------------------:|
|      1 | claim       |                         5.8611 |
|      2 | prize       |                         5.7755 |
|      3 | 150p        |                         5.37   |
|      4 | your mobile |                         5.3561 |
|      5 | have won    |                         5.3423 |
|      6 | co          |                         5.3327 |
|      7 | co uk       |                         5.2951 |
|      8 | 18          |                         5.2787 |
|      9 | nokia       |                         5.2579 |
|     10 | to claim    |                         5.2298 |
|     11 | guaranteed  |                         5.1728 |
|     12 | 1000        |                         5.16   |
|     13 | 16          |                         5.1511 |
|     14 | 500         |                         5.1463 |
|     15 | tone        |                         5.1261 |
|     16 | www         |                         5.0353 |
|     17 | ringtone    |                         5.0125 |
|     18 | cs          |                         5.0107 |
|     19 | 000         |                         4.9751 |
|     20 | stop to     |                         4.9719 |

## 6. Độ trễ suy luận

| Chế độ                              |   Trung vị p50 (ms) |   p95 (ms) |   p99 (ms) |   Trung bình (ms) |
|:------------------------------------|--------------------:|-----------:|-----------:|------------------:|
| Từng tin một (giống gateway thật)   |              1.1984 |     1.8745 |     2.6777 |            1.2671 |
| Theo lô 1034 tin (vector hoá 1 lần) |            nan      |   nan      |   nan      |            0.051  |

## 7. KẾT QUẢ CUỐI CÙNG TRÊN TEST (lần chạm duy nhất)

| Mô hình                                            |   Precision |   Recall |     F1 |   Accuracy |
|:---------------------------------------------------|------------:|---------:|-------:|-----------:|
| TfidfVectorizer + MultinomialNB @ T=0.417          |      0.9752 |   0.9008 | 0.9365 |     0.9845 |
| TfidfVectorizer + MultinomialNB @ T=0.5 (mặc định) |      0.9833 |   0.9008 | 0.9402 |     0.9855 |
| Logistic Regression (balanced) + TFIDF             |      0.9516 |   0.9008 | 0.9255 |     0.9816 |
| Baseline: đoán toàn 'ham'                          |      0      |   0      | 0      |     0.8733 |

Ma trận nhầm lẫn tại T = 0.417011:

| | Dự đoán HAM | Dự đoán SPAM |
| :--- | ---: | ---: |
| **Thực tế HAM** | 900 (TN) | 3 (FP) |
| **Thực tế SPAM** | 13 (FN) | 118 (TP) |

## 8. Phân tích ca sai

Tổng 16 ca sai trên test: 3 False Positive,
13 False Negative.
Danh sách đầy đủ: `reports/ca_du_doan_sai.csv`.

| loai_loi       | thuc_te   | du_doan   |   xac_suat_spam | noi_dung                                                                                                       |
|:---------------|:----------|:----------|----------------:|:---------------------------------------------------------------------------------------------------------------|
| False Positive | ham       | spam      |       0.467167  | Waiting for your call.                                                                                         |
| False Positive | ham       | spam      |       0.547256  | K:)eng rocking in ashes:)                                                                                      |
| False Positive | ham       | spam      |       0.90818   | Nokia phone is lovly..                                                                                         |
| False Negative | spam      | ham       |       0.340476  | Burger King - Wanna play footy at a top stadium? Get 2 Burger King before 1st Sept and go Large or Super with  |
| False Negative | spam      | ham       |       0.325979  | ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE MINS. INDIA CUST SERVs SED YES. L8ER GOT MEGA BILL. 3 DONT GIV A |
| False Negative | spam      | ham       |       0.186488  | 88066 FROM 88066 LOST 3POUND HELP                                                                              |
| False Negative | spam      | ham       |       0.176256  | Check Out Choose Your Babe Videos @ sms.shsex.netUN fgkslpoPW fgkslpo                                          |
| False Negative | spam      | ham       |       0.14351   | Xmas & New Years Eve tickets are now on sale from the club, during the day from 10am till 8pm, and on Thurs, F |
| False Negative | spam      | ham       |       0.126239  | ringtoneking 84484                                                                                             |
| False Negative | spam      | ham       |       0.0887567 | Would you like to see my XXX pics they are so hot they were nearly banned in the uk!                           |

## 9. Minh hoạ alpha = 0

```
[cảnh báo sklearn] RuntimeWarning: divide by zero encountered in log
[cảnh báo sklearn] RuntimeWarning: invalid value encountered in subtract
Từ chỉ xuất hiện trong SPAM ở tập train: 'claim' (72 lần spam / 0 lần ham)
Từ chỉ xuất hiện trong HAM ở tập train : 'gt' (221 lần ham / 0 lần spam)

Tình huống                                          P(spam) α=0   nhãn  P(spam) α=0.1   nhãn
--------------------------------------------------------------------------------------------
Tin HAM bình thường                                    0.00e+00    ham       1.15e-13    ham
Tin HAM + 1 từ chỉ có ở spam ('claim')                      NaN    ham       3.02e-10    ham
Tin SPAM bình thường                                   1.00e+00   spam       1.00e+00   spam
Tin SPAM + 1 từ chỉ có ở ham ('gt')                         NaN    ham       1.00e+00   spam
Tin SPAM + 1 từ HOÀN TOÀN lạ ('xyzzyqwerty7788')       1.00e+00   spam       1.00e+00   spam

Đọc bảng — hai dòng có từ 'độc' cho ra NaN, không phải một xác suất sai. Với alpha = 0 thì P(w|c) = 0 nên log P(w|c) = −∞. Tin nhắn đó chứa cả từ vắng mặt ở ham lẫn từ vắng mặt ở spam, nên log-likelihood của CẢ HAI lớp đều bằng −∞; khi chuẩn hoá, −∞ − (−∞) = NaN. Bộ phân loại không trả về phán quyết sai — nó không trả về gì cả, và nhãn dự đoán trở thành tuỳ ý.
Dòng cuối cho thấy từ HOÀN TOÀN lạ lại vô hại: 'xyzzyqwerty7788' không nằm trong từ vựng nên bị CountVectorizer loại thẳng từ khâu vector hoá. Thủ phạm thật là từ CÓ trong từ vựng nhưng đếm được 0 lần ở một lớp — đúng tình huống mà Laplace smoothing sinh ra để xử lý.
Với alpha = 0.1, cùng những tin đó vẫn cho xác suất hữu hạn và nhãn đúng hướng. Lưu ý Naive Bayes vốn quá tự tin (xác suất bão hoà về ~1e-16 hoặc ~1.0) do giả định độc lập nhân dồn hàng chục thừa số — đó cũng chính là lý do phải chọn ngưỡng bằng thực nghiệm ở bước 10 thay vì mặc định 0.5.
```
