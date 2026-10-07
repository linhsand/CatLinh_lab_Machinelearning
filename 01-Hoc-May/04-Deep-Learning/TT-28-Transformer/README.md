# TT-28 — TRANSFORMER
## Tự động phân loại chủ đề tin tức cho toà soạn

| | |
|---|---|
| 🎓 **Khoá** | HỌC MÁY · [Buổi 14](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-14-RNN-LSTM-Transformer) |
| 🧠 **Nhóm** | Deep Learning · NLP · Kiến trúc Attention |
| 🔧 **Thuật toán** | Transformer Encoder (tự xây) + fine-tune DistilBERT |
| 🏭 **Lĩnh vực** | Truyền thông · Toà soạn · Tổng hợp tin |
| ⏱ **Thời lượng** | 9–12 giờ (**cần GPU**) |
| 📈 **Độ khó** | ⭐⭐⭐⭐ |

---

## 1. THUẬT TOÁN NÀY LÀ GÌ

```
   LSTM (TT-27) đọc TUẦN TỰ  →  chậm, khó nhớ xa
   Transformer nhìn TOÀN BỘ câu CÙNG LÚC bằng SELF-ATTENTION

        Attention(Q,K,V) = softmax( Q·Kᵀ / √d ) · V

   Trực giác: mỗi từ "hỏi" mọi từ khác trong câu
              "ai liên quan tới tôi nhất?" rồi lấy thông tin từ đó.

   Câu: "Ngân hàng cho vay với lãi suất thấp"
        Từ "lãi suất" chú ý mạnh vào "ngân hàng", "vay"
        → hiểu đúng ngữ cảnh tài chính (không phải "bờ sông")
```

**Ba thành phần bắt buộc:**
```
   ① Positional Encoding — vì attention không có khái niệm thứ tự,
                            phải CỘNG thêm thông tin vị trí
   ② Multi-Head Attention — nhiều "góc nhìn" song song (ngữ pháp, ngữ nghĩa…)
   ③ Feed-Forward + Residual + LayerNorm — ổn định huấn luyện mạng sâu
```

---

## 2. BÀI TOÁN THỰC TẾ

```
   Toà soạn nhận 5.000 tin/ngày từ các nguồn RSS.
   Biên tập viên phải phân loại thủ công vào chuyên mục:
   Thế giới · Thể thao · Kinh doanh · Khoa học-Công nghệ

   1 người phân loại được ~400 tin/ngày → cần 12 người.
   → Tự động hoá: model phân loại, người chỉ kiểm tra ca model không chắc.
```

---

## 3. BỘ DỮ LIỆU

| | |
|---|---|
| **Tên** | AG News Classification |
| **Cách lấy** | `from datasets import load_dataset; load_dataset("ag_news")` |
| **Kích thước** | 120.000 train + 7.600 test |
| **Lớp** | 4 chuyên mục, **cân bằng hoàn toàn** (30.000 mỗi lớp) |

**Phương án tiếng Việt (nâng cao):** dùng `UIT-VSFC` hoặc tự thu thập tin từ RSS
báo Việt Nam. Khi đó cần **tách từ** bằng `underthesea` trước khi tokenize.

---

## 4. HƯỚNG ĐI ĐÚNG

### 4.1. Làm CẢ HAI nhánh để hiểu bản chất

```
   NHÁNH A — TỰ XÂY Transformer Encoder (Keras)
     → hiểu positional encoding, multi-head attention hoạt động ra sao
     → accuracy ~88–90%

   NHÁNH B — FINE-TUNE DistilBERT có sẵn (Hugging Face)
     → cách làm THỰC TẾ trong công việc
     → accuracy ~94–95%

   → Báo cáo phải so sánh: công sức bỏ ra vs kết quả thu về.
```

### 4.2. Nhánh A — khối Transformer tối giản

```python
import tensorflow as tf
from tensorflow.keras import layers

class KhoiTransformer(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.att  = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)
        self.ffn  = tf.keras.Sequential([layers.Dense(d_ff, activation='relu'),
                                         layers.Dense(d_model)])
        self.ln1  = layers.LayerNormalization(epsilon=1e-6)
        self.ln2  = layers.LayerNormalization(epsilon=1e-6)
        self.do1  = layers.Dropout(dropout)
        self.do2  = layers.Dropout(dropout)

    def call(self, x, training=False):
        a = self.do1(self.att(x, x), training=training)
        x = self.ln1(x + a)                       # ⭐ residual + norm
        f = self.do2(self.ffn(x), training=training)
        return self.ln2(x + f)
```

> ⚠️ Thiếu **positional encoding** thì Transformer coi câu như một cái túi từ
> (giống Bag-of-Words) — mất hoàn toàn thông tin thứ tự. Đây là lỗi hay gặp nhất
> khi tự xây.

### 4.3. Nhánh B — fine-tune DistilBERT

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ten = "distilbert-base-uncased"          # tiếng Việt: "vinai/phobert-base"
tok = AutoTokenizer.from_pretrained(ten)
model = AutoModelForSequenceClassification.from_pretrained(ten, num_labels=4)
# learning rate cho fine-tune: 2e-5 đến 5e-5 (RẤT NHỎ), 2–3 epoch là đủ
```

---

## 5. CÁC BƯỚC THỰC HIỆN

```
   ☐ 1. Nạp AG News, kiểm tra cân bằng lớp, xem 5 mẫu mỗi lớp
   ☐ 2. ⭐ BASELINE BẮT BUỘC: TF-IDF + LinearSVC (xem TT-29)
        → thường đạt ~91–92% chỉ trong 30 giây!
        → đây là mốc mà Transformer phải vượt qua
   ☐ 3. Tokenize + padding, xây từ điển (nhánh A)
   ☐ 4. Nhánh A: Embedding + Positional Encoding + 2 khối Transformer + GlobalAvgPool + Dense
   ☐ 5. ⚠️ THÍ NGHIỆM: bỏ positional encoding → accuracy tụt bao nhiêu?
   ☐ 6. Khảo sát số head: 1 / 2 / 4 / 8 → accuracy và thời gian
   ☐ 7. Nhánh B: fine-tune DistilBERT 2–3 epoch
   ☐ 8. ⭐ BẢNG SO SÁNH 4 CÁCH:
        TF-IDF+SVC · LSTM · Transformer tự xây · DistilBERT
        theo: accuracy · thời gian train · số tham số · thời gian dự đoán 1 tin
   ☐ 9. Ma trận nhầm lẫn 4×4 → cặp chuyên mục nào hay nhầm?
        (thường Business ↔ Sci/Tech)
   ☐ 10. ⭐ Trực quan hoá attention: chọn 3 tin, vẽ heatmap từ nào chú ý từ nào
   ☐ 11. Cơ chế human-in-the-loop: ngưỡng tin cậy 95%
         → bao nhiêu % tin tự động, bao nhiêu % cần biên tập viên?
```

---

## 6. TIÊU CHÍ HOÀN THÀNH

```
   ☐ ⭐ Có baseline TF-IDF + LinearSVC
   ☐ Nhánh A tự xây có positional encoding + thí nghiệm bỏ nó đi
   ☐ Nhánh B fine-tune thành công với lr nhỏ (2e-5 – 5e-5)
   ☐ ⭐ Có BẢNG SO SÁNH 4 cách kèm thời gian và số tham số
   ☐ Có ma trận nhầm lẫn + phân tích cặp lớp hay nhầm
   ☐ ⭐ Có heatmap attention cho ít nhất 3 mẫu
   ☐ Có bảng human-in-the-loop
   ☐ Kết luận trung thực: chênh lệch 3% accuracy có đáng với chi phí GPU không?
```

**Mức tham chiếu:** TF-IDF+SVC ~91% · Transformer tự xây ~88–90% · DistilBERT ~94–95%.
⚠️ Transformer tự xây **thua** TF-IDF là chuyện bình thường với 120k mẫu — đó là
bài học quan trọng: kiến trúc hiện đại cần **rất nhiều dữ liệu** mới phát huy.

---

## 7. CẠM BẪY

| Cạm bẫy | Hậu quả |
|---------|---------|
| Quên positional encoding | Mất thứ tự từ, thành Bag-of-Words |
| Fine-tune với lr = 1e-3 | Phá huỷ trọng số đã học sẵn |
| Không có baseline TF-IDF | Không biết Transformer có đáng dùng không |
| `max_length` quá lớn (512) | Chậm gấp nhiều lần mà không cải thiện |
| Fine-tune 10 epoch | Overfit — BERT chỉ cần 2–3 epoch |
| Quên tách từ với tiếng Việt | Tokenizer cắt sai từ ghép |

---

## 8. SẢN PHẨM NỘP & MỞ RỘNG

```
TT-28-Transformer-<HoTen>/
├── README.md          ← ⭐ có bảng so sánh 4 cách + kết luận chi phí/lợi ích
├── notebooks/{01_baseline_tfidf.ipynb, 02_transformer_tu_xay.ipynb, 03_finetune_bert.ipynb}
├── src/{transformer_block.py, train.py}
├── models/
├── reports/{so_sanh_4_cach.png, attention_heatmap.png, confusion_4x4.png}
└── requirements.txt
```

**Mở rộng:**
1. Chuyển sang tiếng Việt với **PhoBERT** (xem KT-14)
2. Chưng cất tri thức (knowledge distillation): dùng DistilBERT dạy lại model nhỏ hơn
3. Phân loại đa nhãn: 1 tin có thể thuộc nhiều chuyên mục cùng lúc

**Tham khảo:** [Buổi 14 — Transformer](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-14-RNN-LSTM-Transformer/Tai-Lieu) · [Buổi 8 — NLP](https://github.com/TruongTanNghia/Training-Machine-learning/tree/main/Buoi-08-NLP/Tai-Lieu/ly_thuyet_chi_tiet_buoi_08.md)
