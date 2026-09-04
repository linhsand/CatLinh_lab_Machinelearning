"""
Tải bộ dữ liệu SMS Spam Collection (UCI #228) và XÁC THỰC tính toàn vẹn.
===========================================================================
Chạy: python data/download_data.py

Script này thay thế hoàn toàn `make_demo_data.py` (đã xoá). Lý do:
`make_demo_data.py` sinh ra dữ liệu GIẢ LẬP, khiến mọi số liệu trong
reports/ trở nên vô nghĩa (Precision = Recall = F1 = 1.0) và mâu thuẫn
với README. Dự án chỉ được phép chạy trên dữ liệu thật.

Sau khi tải, script kiểm tra 3 lớp:
  1. SHA-256 của file gốc.
  2. Số dòng = 5.572 và phân phối nhãn = 4.825 ham / 747 spam.
  3. Sau drop_duplicates = 5.169 dòng (4.516 ham / 653 spam).
Nếu bất kỳ kiểm tra nào thất bại -> thoát với mã lỗi 1, KHÔNG ghi file.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd

# Thư mục gốc dự án = thư mục cha của thư mục chứa file này (không phụ thuộc cwd)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "spam.csv"

# Nguồn chính: UCI. Nguồn dự phòng: bản mirror giữ nguyên định dạng Kaggle
# (cột v1/v2, encoding latin-1) để `pd.read_csv(..., encoding='latin-1')` chạy được.
SOURCES = [
    "https://raw.githubusercontent.com/mohitgupta-omg/"
    "Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv",
]

EXPECTED_SHA256 = "440e6ea9fa825578abfdd7b7932ef8393d72ef86c0c33f64676705ce40b1dfc2"
EXPECTED_ROWS = 5572
EXPECTED_LABELS = {"ham": 4825, "spam": 747}
EXPECTED_ROWS_DEDUP = 5169
EXPECTED_LABELS_DEDUP = {"ham": 4516, "spam": 653}


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(urls: list[str], timeout: int = 60) -> bytes:
    last_err: Exception | None = None
    for url in urls:
        print(f"[tải]  {url}")
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = resp.read()
            print(f"       -> {len(data):,} byte")
            return data
        except Exception as exc:  # noqa: BLE001
            print(f"       !! thất bại: {exc}")
            last_err = exc
    raise RuntimeError(f"Không tải được dữ liệu từ bất kỳ nguồn nào ({last_err})")


def verify(data: bytes, strict_hash: bool = True) -> pd.DataFrame:
    """Xác thực nội dung. Ném AssertionError nếu không khớp kỳ vọng."""
    digest = sha256_of(data)
    print(f"[hash] sha256 = {digest}")
    if digest != EXPECTED_SHA256:
        msg = (
            f"SHA-256 KHÔNG khớp.\n"
            f"  kỳ vọng: {EXPECTED_SHA256}\n"
            f"  nhận được: {digest}"
        )
        if strict_hash:
            raise AssertionError(msg)
        print(f"[cảnh báo] {msg}\n  -> bỏ qua do --no-strict-hash, vẫn kiểm tra nội dung.")

    import io

    df = pd.read_csv(io.BytesIO(data), encoding="latin-1")
    assert {"v1", "v2"}.issubset(df.columns), f"Thiếu cột v1/v2, có: {list(df.columns)}"
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})

    counts = df["label"].value_counts().to_dict()
    print(f"[kiểm] số dòng thô        : {len(df):,} (kỳ vọng {EXPECTED_ROWS:,})")
    print(f"[kiểm] phân phối nhãn thô : {counts} (kỳ vọng {EXPECTED_LABELS})")
    assert len(df) == EXPECTED_ROWS, f"Số dòng {len(df)} != {EXPECTED_ROWS}"
    assert counts == EXPECTED_LABELS, f"Phân phối nhãn {counts} != {EXPECTED_LABELS}"

    dedup = df.drop_duplicates(subset=["text"])
    counts_d = dedup["label"].value_counts().to_dict()
    print(f"[kiểm] sau drop_duplicates: {len(dedup):,} (kỳ vọng {EXPECTED_ROWS_DEDUP:,})")
    print(f"[kiểm] phân phối sau lọc  : {counts_d} (kỳ vọng {EXPECTED_LABELS_DEDUP})")
    assert len(dedup) == EXPECTED_ROWS_DEDUP, f"Sau lọc trùng {len(dedup)} != {EXPECTED_ROWS_DEDUP}"
    assert counts_d == EXPECTED_LABELS_DEDUP, f"Phân phối sau lọc {counts_d} != {EXPECTED_LABELS_DEDUP}"

    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Tải & xác thực bộ SMS Spam Collection (UCI).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Đường dẫn file đích.")
    ap.add_argument("--force", action="store_true", help="Ghi đè nếu file đã tồn tại.")
    ap.add_argument("--no-strict-hash", action="store_true",
                    help="Chỉ cảnh báo khi sai SHA-256 (vẫn bắt buộc đúng số dòng/nhãn).")
    args = ap.parse_args()

    out: Path = args.out
    if out.exists() and not args.force:
        print(f"[bỏ qua] {out} đã tồn tại. Dùng --force để tải lại.")
        try:
            verify(out.read_bytes(), strict_hash=not args.no_strict_hash)
        except AssertionError as exc:
            print(f"\n[LỖI] File hiện có KHÔNG phải bộ dữ liệu thật:\n  {exc}", file=sys.stderr)
            print("  -> Chạy lại với --force để tải bản đúng.", file=sys.stderr)
            return 1
        print("[OK] File hiện có là bộ dữ liệu UCI thật.")
        return 0

    try:
        data = download(SOURCES)
        verify(data, strict_hash=not args.no_strict_hash)
    except (AssertionError, RuntimeError) as exc:
        print(f"\n[LỖI] {exc}", file=sys.stderr)
        print("KHÔNG ghi file. Vui lòng tải thủ công từ:", file=sys.stderr)
        print("  https://archive.ics.uci.edu/dataset/228/sms+spam+collection", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"\n[OK] Đã lưu bộ dữ liệu THẬT vào: {out}")
    print("     Bước tiếp theo: python src/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
