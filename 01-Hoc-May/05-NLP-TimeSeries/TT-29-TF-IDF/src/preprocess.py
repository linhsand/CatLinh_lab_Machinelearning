"""Nap va tien xu ly 20 Newsgroups cho TT-29 (luyen cho bai phan loai ticket CSKH).

- fetch_20newsgroups (sklearn tu tai ~14 MB lan dau, cache o ~/scikit_learn_data). Train 11.314 / test 7.532,
  chia SAN theo thoi gian dang bai (test la cac bai dang SAU) -> giu nguyen, khong tron lai.
- Cac kieu bo phan gay ro ri (tham so `remove` cua sklearn):
    headers : moi dong truoc dong trong dau tien: From (email + ten nguoi), Subject, Organization, Lines,
              Nntp-Posting-Host... Dong "Newsgroups:" (dap an truc tiep) chi co o 45 / 11.314 bai.
    footers : khoi cuoi bai sau dong "--" (chu ky: ten, email, truong/cong ty).
    quotes  : dong trich bai truoc ("> ...", "In article <...> X writes:").
  + them bien the "giu Subject": bo het nhung giu lai tieu de, vi ticket that LUON co tieu de.
- an_danh(): thay email / URL / so dien thoai bang the chung. Ticket that BAT BUOC an danh truoc khi dung; o day
  con giup model khong hoc thuoc dia chi email cua tung nguoi gui.
- Tach 20% train lam VAL (stratify) de chon tham so. TEST chi de bao cao.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
REMOVE_ALL = ("headers", "footers", "quotes")

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
_URL = re.compile(r"(https?://|ftp://|www\.)\S+", re.I)
_PHONE = re.compile(r"(\+?1[-. ])?\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b")   # dang My: (412) 268-3000
_WS = re.compile(r"\s+")


def an_danh(s: str) -> str:
    s = _EMAIL.sub(" EMAIL ", s)
    s = _URL.sub(" URL ", s)
    s = _PHONE.sub(" PHONE ", s)
    return _WS.sub(" ", s).strip()


def _subjects(subset: str) -> list[str]:
    """Dong Subject cua tung bai (cung thu tu voi fetch_20newsgroups(subset, random_state mac dinh))."""
    raw = fetch_20newsgroups(subset=subset).data
    out = []
    for d in raw:
        head = d.split("\n\n", 1)[0]
        m = re.search(r"^Subject:\s*(.*)$", head, re.M)
        out.append(re.sub(r"^(re:\s*)+", "", m.group(1).strip(), flags=re.I) if m else "")
    return out


def load(remove=REMOVE_ALL, keep_subject: bool = False, anonymize: bool = True) -> dict:
    """{'train', 'test'}: DataFrame (text, label) + 'names' (20 ten nhom)."""
    out = {}
    for subset in ["train", "test"]:
        b = fetch_20newsgroups(subset=subset, remove=remove)
        text = list(b.data)
        if keep_subject:
            subj = _subjects(subset)
            assert len(subj) == len(text)
            text = [f"{s}\n{t}" for s, t in zip(subj, text)]
        if anonymize:
            text = [an_danh(t) for t in text]
        else:
            text = [_WS.sub(" ", t).strip() for t in text]
        out[subset] = pd.DataFrame({"text": text, "label": b.target})
        out["names"] = list(b.target_names)
    return out


def split_val(train: pd.DataFrame, frac: float = 0.2):
    tr, va = train_test_split(train, test_size=frac, stratify=train["label"], random_state=RANDOM_STATE)
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def short_name(name: str) -> str:
    """'comp.sys.ibm.pc.hardware' -> 'ibm.pc.hardware' (de ve hinh)."""
    p = name.split(".")
    return ".".join(p[-2:]) if len(p) > 2 else name
