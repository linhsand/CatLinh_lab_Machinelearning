"""Tai, kiem tra, chia lai va tao tf.data cho bo Chest X-Ray Images (Pneumonia) (Kaggle) cho TT-26.

- Lan dau: tai bang kagglehub (bo cong khai, khong can API key, ~2,3 GB nen KHONG copy vao data/).
  Moi anh JPEG (rong 384-2.916 px, 99% nam ngang) duoc doc 1 lan, chuyen xam, resize 224x224 roi luu
  thanh data/cache_224.npz (uint8, ~300 MB). Cac lan sau chi doc file cache nay.
- 3 van de cua bo du lieu (xem README): val goc chi 16 anh, mat can bang 1:2,9, phan phoi train != test.
  -> GOP train goc + val goc (5.232 anh) roi TU TACH 15% lam val, stratify theo nhan va
     THEO BENH NHAN (1 benh nhan co the co nhieu phim; de chung o train va val la ro ri).
  -> Tap test goc (624 anh) giu nguyen, chi dung 1 lan de bao cao.
- tf.data: anh xam -> 3 kenh (mang ImageNet can 3 kenh), gia tri 0..255. Chuan hoa nam TRONG model
  (EfficientNet co san lop Rescaling + Normalization; CNN tu xay co Rescaling(1/255)) nen file .keras tu du.
- Augmentation (chi cho train): xoay +-10 do, dich +-10%, phong to +-10%, do sang/tuong phan nhe.
  KHONG lat ngang (tim nam ben trai -> lat la sai giai phau), KHONG lat doc.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE = DATA_DIR / "cache_224.npz"
KAGGLE_ID = "paultimothymooney/chest-xray-pneumonia"

IMG_SIZE = 224
CLASSES = ["NORMAL", "PNEUMONIA"]          # nhan 0 / 1 (1 = viem phoi = lop duong tinh)
RANDOM_STATE = 42
VAL_FRACTION = 0.15


def _find_root() -> Path:
    """Thu muc chua train/ val/ test/ (ban zip Kaggle co 2 ban sao chest_xray/ va chest_xray/chest_xray/)."""
    import kagglehub  # chi can khi chua co cache
    src = Path(kagglehub.dataset_download(KAGGLE_ID))
    for cand in sorted(src.rglob("train"), key=lambda p: len(p.parts)):
        if (cand / "NORMAL").is_dir() and "__MACOSX" not in cand.parts:
            return cand.parent
    raise FileNotFoundError(f"Khong tim thay train/NORMAL trong {src}")


def patient_id(fname: str, label: str) -> str:
    """Ma benh nhan tu ten file.

    PNEUMONIA: person1_bacteria_1.jpeg, person1_virus_6.jpeg  -> 'P-person1'
    NORMAL:    IM-0115-0001.jpeg, NORMAL2-IM-1427-0001.jpeg    -> 'N-IM-0115', 'N-NORMAL2-IM-1427'
    """
    stem = Path(fname).stem
    if label == "PNEUMONIA":
        m = re.match(r"(person\d+)_", stem)
        return "P-" + (m.group(1) if m else stem)
    m = re.match(r"((?:NORMAL2-)?IM-\d+)-", stem)
    return "N-" + (m.group(1) if m else stem)


def subtype(fname: str, label: str) -> str:
    """NORMAL / bacteria / virus (ten file PNEUMONIA ghi ro tac nhan)."""
    if label == "NORMAL":
        return "normal"
    return "virus" if "virus" in fname.lower() else "bacteria"


def _load_image(path: Path, size: int = IMG_SIZE) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        im.draft("L", (size * 2, size * 2))   # JPEG: giai ma o do phan giai thap hon -> nhanh hon nhieu
        im = im.convert("L").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def build_cache() -> None:
    """Doc toan bo anh 1 lan -> data/cache_224.npz (anh uint8) + bang chi muc (file, tap goc, nhan, benh nhan)."""
    root = _find_root()
    rows = []
    for split in ["train", "val", "test"]:
        for label in CLASSES:
            for p in sorted((root / split / label).glob("*.jpeg")):
                rows.append({"file": p.name, "path": str(p), "tap_goc": split, "lop": label})
    print(f"[data] Doc va resize {len(rows):,} anh tu {root} (1 lan) ...", flush=True)
    imgs = np.stack([_load_image(Path(r["path"])) for r in rows])
    idx = pd.DataFrame(rows).drop(columns="path")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, images=imgs, ten_file=idx["file"].to_numpy(str), tap_goc=idx["tap_goc"].to_numpy(str),
             lop=idx["lop"].to_numpy(str))
    print(f"[data] Da luu {CACHE} ({imgs.nbytes / 1e6:.0f} MB)", flush=True)


def load_index() -> tuple[pd.DataFrame, np.ndarray]:
    """Bang chi muc (1 dong / anh) + mang anh (N, 224, 224) uint8 cung thu tu."""
    if not CACHE.exists():
        build_cache()
    z = np.load(CACHE)
    df = pd.DataFrame({"file": z["ten_file"], "tap_goc": z["tap_goc"], "lop": z["lop"]})
    df["y"] = (df["lop"] == "PNEUMONIA").astype(int)
    df["benh_nhan"] = [patient_id(f, l) for f, l in zip(df["file"], df["lop"])]
    df["tac_nhan"] = [subtype(f, l) for f, l in zip(df["file"], df["lop"])]
    return df, z["images"]


def count_table(df: pd.DataFrame) -> pd.DataFrame:
    """Buoc 1: so anh moi lop moi tap goc + ti le PNEUMONIA."""
    t = df.pivot_table(index="tap_goc", columns="lop", values="y", aggfunc="size", fill_value=0)
    t = t.reindex(["train", "val", "test"])[CLASSES]
    t["tong"] = t.sum(axis=1)
    t["ti_le_PNEUMONIA"] = t["PNEUMONIA"] / t["tong"]
    t["so_benh_nhan"] = df.groupby("tap_goc")["benh_nhan"].nunique()
    return t


def resplit(df: pd.DataFrame, val_fraction: float = VAL_FRACTION) -> pd.DataFrame:
    """Buoc 2: gop train goc + val goc (16 anh) roi tach ~15% lam val moi.

    StratifiedGroupKFold(n_splits=round(1/0.15)=7): giu ti le lop VA khong de 1 benh nhan nam o ca 2 tap.
    Lay 1 fold lam val. Tra ve df co them cot 'tap' in {train, val, test}.
    """
    df = df.copy()
    df["tap"] = np.where(df["tap_goc"] == "test", "test", "train")
    pool = df[df["tap"] == "train"]
    sgkf = StratifiedGroupKFold(n_splits=round(1 / val_fraction), shuffle=True, random_state=RANDOM_STATE)
    _, val_idx = next(sgkf.split(pool, pool["y"], groups=pool["benh_nhan"]))
    df.loc[pool.index[val_idx], "tap"] = "val"
    return df


def split_table(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby("tap").agg(so_anh=("y", "size"), NORMAL=("y", lambda s: int((s == 0).sum())),
                              PNEUMONIA=("y", "sum"), so_benh_nhan=("benh_nhan", "nunique"))
    t["ti_le_PNEUMONIA"] = t["PNEUMONIA"] / t["so_anh"]
    return t.reindex(["train", "val", "test"])


def augmenter(seed: int = RANDOM_STATE) -> tf.keras.Sequential:
    """Augmentation hop ly voi y te. RandomRotation nhan ti le cua 1 vong: 10 do = 10/360."""
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.RandomRotation(10 / 360, fill_mode="constant", seed=seed),
        L.RandomTranslation(0.1, 0.1, fill_mode="constant", seed=seed + 1),
        L.RandomZoom(0.1, fill_mode="constant", seed=seed + 2),
        L.RandomBrightness(0.1, value_range=(0, 255), seed=seed + 3),
        L.RandomContrast(0.1, seed=seed + 4),
        # Co y KHONG co RandomFlip: lat ngang dua tim sang phai (sai giai phau), lat doc vo nghia.
    ], name="augment")


def make_dataset(images: np.ndarray, y: np.ndarray, training: bool, batch: int = 32,
                 seed: int = RANDOM_STATE) -> tf.data.Dataset:
    """(N,224,224) uint8 -> batch (B,224,224,3) float32 0..255 + nhan float32. Train: xao tron + augmentation."""
    ds = tf.data.Dataset.from_tensor_slices((images, y.astype(np.float32)))
    if training:
        ds = ds.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch)
    to_rgb = lambda x: tf.repeat(tf.cast(x[..., None], tf.float32), 3, axis=-1)  # xam -> 3 kenh giong nhau
    if training:
        aug = augmenter(seed)
        ds = ds.map(lambda x, t: (tf.clip_by_value(aug(to_rgb(x), training=True), 0, 255), t),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda x, t: (to_rgb(x), t), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def to_batch(images: np.ndarray) -> np.ndarray:
    """Anh xam uint8 (N,224,224) -> (N,224,224,3) float32 de dua thang vao model (Grad-CAM, ve hinh)."""
    return np.repeat(images[..., None].astype(np.float32), 3, axis=-1)
