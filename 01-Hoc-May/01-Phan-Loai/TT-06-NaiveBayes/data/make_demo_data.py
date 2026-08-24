"""
make_demo_data.py
------------------
File này CHỈ để demo — nó tự sinh ra một bộ SMS ham/spam giả lập có cùng
"tính cách thống kê" với bộ SMS Spam Collection thật (UCI):
  - Tỉ lệ ~87% ham / ~13% spam
  - Spam thường DÀI hơn ham, hay chứa từ khoá quảng cáo/khuyến mãi
  - Có một số dòng trùng lặp (giống lỗi thật của bộ UCI)
  - Lưu bằng encoding latin-1 để bạn tập luyện xử lý đúng encoding

>>> KHI LÀM BÀI THẬT: tải file spam.csv gốc từ
    https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
    bỏ vào data/spam.csv rồi chạy thẳng src/train.py — không cần sửa code.
"""
import random
import csv

random.seed(42)

HAM_TEMPLATES = [
    "Hey, are you free tonight? Let's grab dinner",
    "Ok lar... I'll call you later",
    "I'm on my way, be there in 10 mins",
    "Can you pick up milk on your way home?",
    "Happy birthday! Hope you have a great day",
    "Did you finish the report yet?",
    "Sorry I missed your call, was in a meeting",
    "See you at the usual place tomorrow",
    "Thanks for helping me move last weekend",
    "What time does the movie start?",
    "I'll be there in five minutes",
    "Can we reschedule our meeting to Friday?",
    "Just landed, will call you soon",
    "Don't forget to bring your charger",
    "Mom said dinner is ready, come home",
    "Lol that's so funny, send me the video",
    "I'm running late, traffic is bad",
    "Let's meet at the coffee shop at 9",
    "How was your exam today?",
    "I left my keys at your place, can you check?",
    "Good morning, hope you slept well",
    "The kids are asking when you'll be back",
    "Can you send me the address again?",
    "I'll pay you back on Friday, promise",
    "Nah I don't think I can make it tonight",
    "U dun say so early hor, u c already then say",
    "Cool, see you then",
    "What are you doing this weekend?",
    "I'm at the gym, call you after",
    "Please remember to feed the cat",
]

SPAM_TEMPLATES = [
    "FREE entry into our weekly draw! Text WIN to 80086 now to claim your prize before it expires",
    "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot! Call 09061701444 now",
    "Congratulations! You've been selected to receive a FREE £1000 cash reward. Text CLAIM to 88888 now",
    "WINNER!! As a valued customer you have been selected to receive a £900 prize reward! Call now",
    "You have won a guaranteed cash prize! Reply CASH to claim your reward of £2000 now before offer ends",
    "FreeMsg: Txt CALL to 86888 and claim your reward of 3 hours talk time to use from your phone now",
    "URGENT! Your mobile number has won a prize of £5000! Call 09061743806 from a landline to claim",
    "SIX chances to win CASH! From 100 to 20,000 pounds. Text CSH11 and send to 87575",
    "Free ringtone waiting for you to be collected. Simply text PASSWORD to 68866 now",
    "Congratulations! You have been selected to receive a free iPhone. Claim now by texting CLAIM to 84484",
    "Win a brand new car by entering our free draw. Text CAR to 89545 before offer closes tonight",
    "Your account has been credited with £500. Reply YES to claim your bonus now, offer ends today",
    "FREE MSG: You are eligible for a discount on your next mobile bill. Call 08000930705 now to claim",
    "Get a free holiday voucher worth £2000, text HOLIDAY to 80488 to claim before this offer expires",
    "You have been randomly selected to win a shopping voucher. Text SHOP to 85023 now for free entry",
    "Last chance! Claim your free prize now by calling 09050002311, offer closes at midnight tonight",
    "Bonus prize alert! You have won a £1500 cash reward, text WON to 87066 now to claim your reward",
    "FreeMsg: Hey there darling, txt back to claim your free gift, hurry offer closes soon",
    "Reminder: your reward of £750 is waiting, claim now by texting CLAIM to 82277 before it expires",
    "Text STOP to opt out or reply YES to claim your free entry into this week's £1000 prize draw",
]


HAM_PREFIX = ["", "Hey, ", "Btw, ", "Ok, ", "Actually, ", "Well, ", "So ", "Also, "]
HAM_SUFFIX = ["", " thanks", " ok?", " :)", " see ya", " let me know", " no rush", " haha"]
SPAM_PREFIX = ["", "ALERT: ", "NOTICE: ", "Dear customer, ", "Hi, "]
SPAM_SUFFIX = ["", " Reply STOP to opt out", " T&Cs apply", " Offer ends soon", " Limited time only"]


def jitter(s: str, is_spam: bool) -> str:
    """Ghép prefix/suffix + nhiễu nhẹ để sinh nhiều biến thể câu khác nhau,
    mô phỏng việc dữ liệu SMS thật rất đa dạng dù cùng chủ đề."""
    prefix = random.choice(SPAM_PREFIX if is_spam else HAM_PREFIX)
    suffix = random.choice(SPAM_SUFFIX if is_spam else HAM_SUFFIX)
    s = prefix + s + suffix
    if random.random() < 0.15:
        s = s.upper()
    if random.random() < 0.25:
        s = s + f" {random.randint(1000, 9999)}"
    return s


rows = []
N_HAM = 950
N_SPAM = 140

for _ in range(N_HAM):
    rows.append(("ham", jitter(random.choice(HAM_TEMPLATES), is_spam=False)))
for _ in range(N_SPAM):
    rows.append(("spam", jitter(random.choice(SPAM_TEMPLATES), is_spam=True)))

random.shuffle(rows)

# Cố tình chèn thêm ~5% dòng trùng lặp y hệt, giống lỗi có thật trong bộ UCI
dup_count = int(0.05 * len(rows))
rows.extend(random.sample(rows, dup_count))
random.shuffle(rows)

with open("/home/claude/TT-06-NaiveBayes/data/spam.csv", "w", newline="", encoding="latin-1") as f:
    writer = csv.writer(f)
    writer.writerow(["v1", "v2", "Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
    for label, text in rows:
        writer.writerow([label, text, "", "", ""])

print(f"Đã tạo {len(rows)} dòng ({N_HAM} ham gốc, {N_SPAM} spam gốc, +{dup_count} dòng trùng lặp cố ý)")
