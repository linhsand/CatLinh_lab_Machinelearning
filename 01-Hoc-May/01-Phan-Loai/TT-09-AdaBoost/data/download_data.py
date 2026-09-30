"""
Tải NSL-KDD (KDDTrain+.txt, KDDTest+.txt) và XÁC THỰC tính toàn vẹn.
=====================================================================
Chạy: python data/download_data.py

Dữ liệu (~22MB) KHÔNG được commit vào git (xem .gitignore) — script này tải lại
từ bản mirror công khai và kiểm tra:
  1. SHA-256 khớp đúng bản đã dùng để sinh toàn bộ số liệu trong reports/.
  2. Số dòng = 125.973 (train) / 22.544 (test) — đúng số công bố của NSL-KDD.
Nếu kiểm tra thất bại -> thoát với mã lỗi 1, KHÔNG ghi file.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
MIRROR = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/"

FILES = {
    "KDDTrain+.txt": {
        "sha256": "1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95",
        "rows": 125_973,
    },
    "KDDTest+.txt": {
        "sha256": "fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84",
        "rows": 22_544,
    },
}


def main() -> int:
    for name, spec in FILES.items():
        out = DATA_DIR / name
        if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() == spec["sha256"]:
            print(f"[ok]   {name} đã có sẵn, SHA-256 khớp")
            continue

        url = MIRROR + name.replace("+", "%2B")
        print(f"[tải]  {url}")
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()

        digest = hashlib.sha256(data).hexdigest()
        n_rows = data.count(b"\n")
        if digest != spec["sha256"] or n_rows != spec["rows"]:
            print(f"[lỗi]  {name}: sha256={digest}, rows={n_rows:,} "
                  f"(kỳ vọng {spec['sha256']}, {spec['rows']:,})")
            return 1
        out.write_bytes(data)
        print(f"[ok]   {name}: {len(data):,} byte, {n_rows:,} dòng, SHA-256 khớp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
