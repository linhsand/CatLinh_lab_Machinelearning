"""TT-24 - PCA: nen 561 dac trung cam bien (UCI HAR) de chay tren thiet bi deo.

Nguyen tac danh gia:
  * Giu NGUYEN cach chia theo NGUOI cua bo goc (21 nguoi train / 9 nguoi test).
  * StandardScaler + PCA nam TRONG Pipeline -> chi fit tren train (hoac tren fold train cua CV).
  * MOI lua chon (diem ngot K) lam bang GroupKFold(5) theo NGUOI tren train.
    Tap test chi dung de BAO CAO.

Cac buoc (theo README de bai):
  1-2. Nap du lieu, chuan hoa                       -> load()
  3-5. Scree plot, phuong sai tich luy, bang nguong   -> run_variance_analysis()
  6.   Danh doi so chieu vs accuracy + thoi gian      -> run_dimension_tradeoff(), choose_sweet_spot()
  7.   Scatter PC1-PC2 theo 6 hoat dong              -> run_pc_scatter()
  8.   Y nghia PC1                                   -> run_pc1_analysis()
  9.   Dung luong / truyen du lieu / bo nho / phep tinh -> run_cost_analysis()
  10.  Sai so tai tao theo K                         -> run_reconstruction()
  11.  So sanh SelectKBest, LDA, t-SNE, UMAP          -> run_supervised_comparison(), run_embedding_comparison()
  Mo rong: Kernel PCA, IncrementalPCA, phat hien bat thuong -> run_kernel_pca(), run_incremental_pca(),
           run_anomaly_detection()
  Model cuoi                                         -> build_final_models()

Chay: python src/pca_pipeline.py
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

import data as D

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42
N_FOLDS = 5
K_GRID = [2, 5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 561]   # 561 = khong giam chieu
VARIANCE_THRESHOLDS = [0.80, 0.90, 0.95, 0.99]
MAX_ACC_DROP = 0.01            # diem ngot: mat toi da 1 diem % accuracy CV so voi 561 chieu
N_CLASSES = 6
# Thiet bi deo (de bai): RAM 64 KB. Cua so tin hieu 2,56 s chong lan 50% -> 1 cua so moi 1,28 s
DEVICE_RAM_BYTES = 64 * 1024
WINDOWS_PER_DAY = int(24 * 3600 / 1.28)          # 67.500 cua so/ngay
BYTES_PER_FLOAT = 4                              # float32 tren vi dieu khien
EMBED_SAMPLE = 3000                              # t-SNE/UMAP chi de truc quan -> lay mau 3.000 diem train

ACTIVITY_ORDER = list(D.ACTIVITIES.values())
COLORS = dict(zip(ACTIVITY_ORDER, ["#1f77b4", "#2ca02c", "#17becf", "#ff7f0e", "#d62728", "#9467bd"]))


def log(msg: str) -> None:
    print(f"[TT-24] {msg}", flush=True)


def save_fig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / name, dpi=130)
    plt.close(fig)


def make_pipe(k: int | None, clf=None) -> Pipeline:
    """scale -> (PCA k chieu) -> classifier. k=None hoac 561 -> khong PCA."""
    steps = [("scale", StandardScaler())]
    if k is not None and k < 561:
        steps.append(("pca", PCA(n_components=k, random_state=RANDOM_STATE)))
    steps.append(("clf", clf if clf is not None else LinearSVC(max_iter=5000, random_state=RANDOM_STATE)))
    return Pipeline(steps)


def cv_score(pipe, X, y, groups) -> np.ndarray:
    return cross_val_score(pipe, X, y, groups=groups, cv=GroupKFold(N_FOLDS), n_jobs=N_FOLDS)


# ----------------------------------------------------------------------------
# 1-2. Nap du lieu (chia theo nguoi) + chuan hoa
# ----------------------------------------------------------------------------
def load():
    df = D.load_har()
    Xtr, Xte, ytr, yte, groups, feats = D.split_by_subject(df)
    log(f"  Train {Xtr.shape} ({len(set(groups))} nguoi) | Test {Xte.shape} "
        f"({df.loc[df.split == 'test', 'subject'].nunique()} nguoi) | khong nguoi nao o ca 2 tap")
    var = Xtr.var(axis=0)
    log(f"  Gia tri trong [{Xtr.min():.0f}, {Xtr.max():.0f}] nhung phuong sai cot tu {var.min():.4f} "
        f"den {var.max():.3f} (gap {var.max() / var.min():.0f} lan) -> van phai StandardScaler")
    return df, Xtr, Xte, ytr, yte, groups, feats


# ----------------------------------------------------------------------------
# 3-5. Scree plot, phuong sai tich luy, bang nguong -> so chieu
# ----------------------------------------------------------------------------
def run_variance_analysis(Xtr):
    scaler = StandardScaler().fit(Xtr)
    pca_full = PCA(random_state=RANDOM_STATE).fit(scaler.transform(Xtr))
    pca_raw = PCA(random_state=RANDOM_STATE).fit(Xtr)            # de so sanh: KHONG chuan hoa
    evr, evr_raw = pca_full.explained_variance_ratio_, pca_raw.explained_variance_ratio_
    cum, cum_raw = np.cumsum(evr), np.cumsum(evr_raw)

    rows = []
    for t in VARIANCE_THRESHOLDS:
        k, k_raw = int(np.argmax(cum >= t)) + 1, int(np.argmax(cum_raw >= t)) + 1
        rows.append({"nguong_phuong_sai": t, "so_chieu_co_chuan_hoa": k, "giam_chieu_%": (1 - k / 561) * 100,
                     "so_chieu_khong_chuan_hoa": k_raw, "giam_chieu_khong_chuan_hoa_%": (1 - k_raw / 561) * 100})
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / "bang_nguong_phuong_sai.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(range(1, 51), evr[:50] * 100, color="#2980b9")
    axes[0].set_xlabel("Thanh phan chinh"); axes[0].set_ylabel("% phuong sai")
    axes[0].set_title("Scree plot - 50 thanh phan dau (co chuan hoa)")
    axes[1].semilogy(range(1, 562), evr * 100, color="#2980b9")
    axes[1].set_xlabel("Thanh phan chinh"); axes[1].set_ylabel("% phuong sai (log)")
    axes[1].set_title("Scree plot - ca 561 thanh phan (thang log)")
    save_fig(fig, "scree_plot.png")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(range(1, 562), cum * 100, label="Co StandardScaler", color="#2980b9", lw=2)
    ax.plot(range(1, 562), cum_raw * 100, label="Khong chuan hoa", color="#7f8c8d", ls="--")
    for t, k in zip(VARIANCE_THRESHOLDS, table["so_chieu_co_chuan_hoa"]):
        ax.axhline(t * 100, color="#c0392b", lw=0.6, ls=":")
        ax.plot(k, t * 100, "o", color="#c0392b")
        ax.annotate(f"{t:.0%}: K={k}", (k, t * 100), textcoords="offset points", xytext=(8, -12), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("So thanh phan K (log)"); ax.set_ylabel("% phuong sai tich luy")
    ax.set_title("Phuong sai tich luy theo so thanh phan"); ax.legend()
    save_fig(fig, "variance_tich_luy.png")

    log(f"  PC1 giu {evr[0]:.1%}, PC2 {evr[1]:.1%}, PC3 {evr[2]:.1%} phuong sai (co chuan hoa); "
        f"khong chuan hoa: PC1 {evr_raw[0]:.1%}")
    for r in table.itertuples():
        log(f"    {r.nguong_phuong_sai:.0%}: can {r.so_chieu_co_chuan_hoa:3d} chieu (giam {r._3:.1f}%) | "
            f"khong chuan hoa: {r.so_chieu_khong_chuan_hoa:3d}")
    return scaler, pca_full, pca_raw, table


# ----------------------------------------------------------------------------
# 6. Danh doi so chieu vs accuracy + thoi gian (LinearSVC), chon diem ngot bang CV
# ----------------------------------------------------------------------------
def run_dimension_tradeoff(Xtr, ytr, Xte, yte, groups) -> pd.DataFrame:
    rows = []
    for k in K_GRID:
        pipe = make_pipe(k)
        cv = cv_score(pipe, Xtr, ytr, groups)
        t0 = time.perf_counter()
        pipe.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = pipe.predict(Xte)
        pred_ms = (time.perf_counter() - t0) / len(Xte) * 1e3
        rows.append({"K": k, "giam_chieu_%": (1 - k / 561) * 100,
                     "phuong_sai_giu_%": float(pipe.named_steps["pca"].explained_variance_ratio_.sum() * 100)
                     if "pca" in pipe.named_steps else 100.0,
                     "acc_cv": cv.mean(), "acc_cv_std": cv.std(), "acc_test": accuracy_score(yte, pred),
                     "thoi_gian_fit_s": fit_s, "thoi_gian_du_doan_ms_moi_mau": pred_ms})
        r = rows[-1]
        log(f"    K={k:3d}: phuong sai {r['phuong_sai_giu_%']:5.1f}%  acc CV={r['acc_cv']:.4f}+-{r['acc_cv_std']:.3f}  "
            f"test={r['acc_test']:.4f}  fit={fit_s:5.2f}s")
    df = pd.DataFrame(rows)
    df["mat_acc_cv_vs_561_diem"] = (df.loc[df.K == 561, "acc_cv"].iloc[0] - df["acc_cv"]) * 100
    df.to_csv(REPORTS_DIR / "danh_doi_chieu_accuracy.csv", index=False)
    return df


def choose_sweet_spot(tradeoff: pd.DataFrame) -> int:
    """K NHO NHAT ma accuracy CV giam <= 1 diem % so voi 561 chieu (chi dung CV, khong nhin test)."""
    full = tradeoff.loc[tradeoff.K == 561, "acc_cv"].iloc[0]
    ok = tradeoff[tradeoff["acc_cv"] >= full - MAX_ACC_DROP]
    k = int(ok["K"].min())
    log(f"  => DIEM NGOT (CV): K={k} (acc CV {ok.loc[ok.K == k, 'acc_cv'].iloc[0]:.4f} vs 561 chieu {full:.4f})")
    return k


def plot_tradeoff(tradeoff: pd.DataFrame, k_sweet: int, k95: int) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.errorbar(tradeoff["K"], tradeoff["acc_cv"] * 100, yerr=tradeoff["acc_cv_std"] * 100, fmt="o-",
                 color="#2980b9", capsize=3, label="Accuracy CV theo nguoi (dung de CHON)")
    ax1.plot(tradeoff["K"], tradeoff["acc_test"] * 100, "s--", color="#7f8c8d", label="Accuracy test (chi bao cao)")
    full = tradeoff.loc[tradeoff.K == 561, "acc_cv"].iloc[0] * 100
    ax1.axhline(full - MAX_ACC_DROP * 100, color="#c0392b", ls=":", lw=1, label="Nguong: 561 chieu - 1 diem")
    ax1.axvline(k_sweet, color="#27ae60", ls="--", label=f"Diem ngot K={k_sweet}")
    ax1.axvline(k95, color="#8e44ad", ls=":", label=f"95% phuong sai K={k95}")
    ax1.set_xscale("log"); ax1.set_xlabel("So chieu K (log)"); ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(45, 100)
    ax2 = ax1.twinx()
    ax2.plot(tradeoff["K"], tradeoff["thoi_gian_fit_s"], "^-", color="#e67e22", alpha=0.7, label="Thoi gian train (s)")
    ax2.set_ylabel("Thoi gian train LinearSVC + PCA (s)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    ax1.set_title("Danh doi so chieu vs accuracy va thoi gian (Scaler -> PCA(K) -> LinearSVC)")
    save_fig(fig, "danh_doi_chieu_accuracy.png")


# ----------------------------------------------------------------------------
# 7. Scatter PC1-PC2 theo 6 hoat dong
# ----------------------------------------------------------------------------
def run_pc_scatter(scaler, pca_full, Xtr, ytr, Xte, yte, groups) -> dict:
    Z = pca_full.transform(scaler.transform(Xtr))[:, :2]
    fig, ax = plt.subplots(figsize=(9, 7))
    for act in ACTIVITY_ORDER:
        m = ytr == act
        ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.4, color=COLORS[act], label=act)
    ax.set_xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0]:.1%} phuong sai)")
    ax.set_ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1]:.1%} phuong sai)")
    ax.set_title("Train chieu len PC1-PC2: dong vs tinh tach ro, trong moi nhom thi chong lan")
    ax.legend(markerscale=3)
    save_fig(fig, "pc1_pc2_scatter.png")

    # Dinh luong: (a) 6 lop voi 2 chieu, (b) "dong vs tinh" chi voi PC1
    moving_tr = np.isin(ytr, list(D.MOVING)).astype(int)
    moving_te = np.isin(yte, list(D.MOVING)).astype(int)
    pc1_pipe = Pipeline([("scale", StandardScaler()), ("pca", PCA(1, random_state=RANDOM_STATE)),
                         ("clf", LogisticRegression())])
    cv_bin = cross_val_score(pc1_pipe, Xtr, moving_tr, groups=groups, cv=GroupKFold(N_FOLDS))
    pc1_pipe.fit(Xtr, moving_tr)
    res = {"acc_6_lop_2_chieu_cv": float(cv_score(make_pipe(2), Xtr, ytr, groups).mean()),
           "acc_dong_vs_tinh_chi_PC1_cv": float(cv_bin.mean()),
           "acc_dong_vs_tinh_chi_PC1_test": float(pc1_pipe.score(Xte, moving_te))}
    pc_means = pd.DataFrame(Z, columns=["PC1", "PC2"]).assign(activity=ytr).groupby("activity").mean()
    pc_means = pc_means.reindex(ACTIVITY_ORDER)
    pc_means.to_csv(REPORTS_DIR / "pc1_pc2_trung_binh_theo_hoat_dong.csv")
    log(f"  6 lop chi voi 2 chieu: acc CV={res['acc_6_lop_2_chieu_cv']:.3f}; dong vs tinh chi voi PC1: "
        f"CV={res['acc_dong_vs_tinh_chi_PC1_cv']:.4f}, test={res['acc_dong_vs_tinh_chi_PC1_test']:.4f}")
    log("  PC1/PC2 trung binh theo hoat dong: " + ", ".join(
        f"{a}=({r.PC1:+.1f},{r.PC2:+.1f})" for a, r in pc_means.iterrows()))
    return {**res, "pc_trung_binh": pc_means.round(3).to_dict(orient="index")}


# ----------------------------------------------------------------------------
# 8. Y nghia PC1
# ----------------------------------------------------------------------------
def signal_group(name: str) -> str:
    """'tBodyAcc-mean()-X' -> 'tBodyAcc'; 'angle(X,gravityMean)' -> 'angle'."""
    return "angle" if name.startswith("angle") else name.split("-")[0]


def run_pc1_analysis(pca_full, pca_raw, feats) -> dict:
    load1 = pd.Series(pca_full.components_[0], index=feats)
    top = load1.reindex(load1.abs().sort_values(ascending=False).index).head(10)
    top_df = top.rename("he_so_PC1").to_frame()
    top_df["nhom_tin_hieu"] = [signal_group(n) for n in top.index]
    top_df.to_csv(REPORTS_DIR / "pc1_top10_dac_trung.csv")

    sq = load1 ** 2
    groups = sq.groupby([signal_group(n) for n in feats]).sum().sort_values(ascending=False)
    domain = sq.groupby(["time" if n.startswith("t") else "freq" if n.startswith("f") else "angle"
                         for n in feats]).sum()
    stat = sq.groupby([("mean/std/mad/max/min/energy/iqr" if any(s in n for s in
                        ["mean()", "std()", "mad()", "max()", "min()", "energy()", "iqr()", "sma()"])
                        else "khac") for n in feats]).sum()
    n_needed = int(np.argmax(np.cumsum(np.sort(sq.values)[::-1]) >= 0.5)) + 1
    groups.to_csv(REPORTS_DIR / "pc1_dong_gop_theo_nhom_tin_hieu.csv", header=["ty_trong_binh_phuong_he_so"])

    load_raw = pd.Series(pca_raw.components_[0], index=feats)
    top_raw = load_raw.abs().sort_values(ascending=False).head(5)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].barh(top.index[::-1], top.values[::-1], color="#2980b9")
    axes[0].set_title("PC1: 10 dac trung goc dong gop nhieu nhat")
    axes[0].tick_params(axis="y", labelsize=8)
    axes[1].barh(groups.index[::-1], groups.values[::-1] * 100, color="#27ae60")
    axes[1].set_xlabel("% tong binh phuong he so PC1")
    axes[1].set_title("PC1 theo nhom tin hieu")
    save_fig(fig, "pc1_phan_tich.png")

    log(f"  PC1: can {n_needed}/561 dac trung de gom 50% trong so -> PC1 la tong hop RAT NHIEU cot, khong 1 cot nao chu dao")
    log("  Top 10 he so PC1: " + ", ".join(f"{n}({v:+.3f})" for n, v in top.items()))
    log("  Nhom tin hieu (% trong so PC1): " + ", ".join(f"{g}={v:.1%}" for g, v in groups.head(6).items()))
    log("  Khong chuan hoa, PC1 bi chi phoi boi: " + ", ".join(f"{n}({v:.3f})" for n, v in top_raw.items()))
    return {"top10": top_df.round(4).reset_index().rename(columns={"index": "dac_trung"}).to_dict(orient="records"),
            "nhom_tin_hieu_%": (groups * 100).round(2).to_dict(),
            "mien_thoi_gian_vs_tan_so_%": (domain * 100).round(2).to_dict(),
            "so_dac_trung_gom_50%_trong_so": n_needed,
            "top5_khong_chuan_hoa": top_raw.round(4).to_dict()}


# ----------------------------------------------------------------------------
# 9. Chi phi: dung luong, truyen du lieu, bo nho model, phep tinh moi lan suy luan
# ----------------------------------------------------------------------------
def linear_cost(k: int | None) -> dict:
    """So tham so / phep nhan-cong (MAC) cho pipeline scale -> PCA(k) -> linear classifier 6 lop."""
    d = 561
    scale = 2 * d                                    # mean + std
    if k is None or k >= d:
        params, macs = scale + N_CLASSES * d + N_CLASSES, d + N_CLASSES * d
    else:
        params = scale + k * d + d + N_CLASSES * k + N_CLASSES       # components + mean_ + W + b
        macs = d + k * d + N_CLASSES * k
    return {"tham_so": params, "bo_nho_model_KB": params * BYTES_PER_FLOAT / 1024, "MAC_moi_lan": macs}


def run_cost_analysis(Xtr, Xte, ytr, k_sweet: int, k95: int, lda_k: int = N_CLASSES - 1) -> pd.DataFrame:
    n_all = len(Xtr) + len(Xte)
    rows = []
    configs = [("Goc 561 chieu", None), (f"PCA K={k95} (95% phuong sai)", k95),
               (f"PCA K={k_sweet} (diem ngot)", k_sweet), ("PCA K=50", 50), (f"LDA {lda_k} chieu", "lda")]
    for name, k in configs:
        dim = 561 if k is None else (lda_k if k == "lda" else k)
        c = linear_cost(None if k is None else (lda_k if k == "lda" else k))
        rows.append({
            "cau_hinh": name, "so_chieu": dim,
            "dung_luong_du_lieu_MB_float64": n_all * dim * 8 / 1e6,
            "truyen_moi_cua_so_bytes": dim * BYTES_PER_FLOAT,
            "truyen_moi_ngay_MB": WINDOWS_PER_DAY * dim * BYTES_PER_FLOAT / 1e6,
            "tham_so_model": c["tham_so"], "bo_nho_model_KB": c["bo_nho_model_KB"],
            "vua_64KB_RAM": c["bo_nho_model_KB"] * 1024 <= DEVICE_RAM_BYTES,
            "MAC_moi_lan_suy_luan": c["MAC_moi_lan"],
            # Phan loai TUYEN TINH sau PCA tuong duong 1 ma tran W.P (6x561) -> gop lai thi chi phi = goc
            "MAC_neu_gop_W_P": 561 + N_CLASSES * 561 if k is not None else c["MAC_moi_lan"],
        })
    df = pd.DataFrame(rows)

    # Model phi tuyen (RBF-SVM): chi phi du doan ~ so vector ho tro x so chieu -> PCA giam that
    rbf_rows = []
    for name, k in [("RBF-SVM 561 chieu", None), (f"RBF-SVM PCA K={k_sweet}", k_sweet), ("RBF-SVM PCA K=50", 50)]:
        pipe = make_pipe(k, SVC(kernel="rbf"))
        t0 = time.perf_counter(); pipe.fit(Xtr, ytr); fit_s = time.perf_counter() - t0
        t0 = time.perf_counter(); pipe.predict(Xte); pred_ms = (time.perf_counter() - t0) / len(Xte) * 1e3
        n_sv = int(pipe.named_steps["clf"].n_support_.sum())
        dim = 561 if k is None else k
        rbf_rows.append({"cau_hinh": name, "so_chieu": dim, "so_vector_ho_tro": n_sv,
                         "MAC_moi_lan_suy_luan": n_sv * dim + (0 if k is None else 561 * k),
                         "bo_nho_model_KB": (n_sv * dim + (0 if k is None else 561 * k)) * BYTES_PER_FLOAT / 1024,
                         "thoi_gian_fit_s": fit_s, "thoi_gian_du_doan_ms_moi_mau": pred_ms})
    rbf = pd.DataFrame(rbf_rows)
    df.to_csv(REPORTS_DIR / "chi_phi_tuyen_tinh.csv", index=False)
    rbf.to_csv(REPORTS_DIR / "chi_phi_rbf_svm.csv", index=False)
    for r in df.itertuples():
        log(f"    {r.cau_hinh:28s} du lieu {r.dung_luong_du_lieu_MB_float64:6.1f} MB | truyen {r.truyen_moi_ngay_MB:6.1f} MB/ngay | "
            f"model {r.bo_nho_model_KB:7.1f} KB (64KB: {r.vua_64KB_RAM}) | MAC {r.MAC_moi_lan_suy_luan:,} (gop: {r.MAC_neu_gop_W_P:,})")
    for r in rbf.itertuples():
        log(f"    {r.cau_hinh:28s} SV={r.so_vector_ho_tro} MAC={r.MAC_moi_lan_suy_luan:,} model={r.bo_nho_model_KB:,.0f} KB "
            f"fit={r.thoi_gian_fit_s:.2f}s du doan={r.thoi_gian_du_doan_ms_moi_mau:.3f} ms/mau")
    return df, rbf


# ----------------------------------------------------------------------------
# 10. Sai so tai tao theo K
# ----------------------------------------------------------------------------
def run_reconstruction(scaler, Xtr, Xte) -> pd.DataFrame:
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
    rows = []
    for k in [k for k in K_GRID if k < 561]:
        p = PCA(k, random_state=RANDOM_STATE).fit(Ztr)
        err_tr = np.mean((Ztr - p.inverse_transform(p.transform(Ztr))) ** 2)
        err_te = np.mean((Zte - p.inverse_transform(p.transform(Zte))) ** 2)
        rows.append({"K": k, "MSE_tai_tao_train": err_tr, "MSE_tai_tao_test": err_te,
                     "phuong_sai_mat_train_%": err_tr / Ztr.var(axis=0).mean() * 100,
                     "phuong_sai_mat_test_%": err_te / Zte.var(axis=0).mean() * 100})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "sai_so_tai_tao.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(df["K"], df["MSE_tai_tao_train"], "o-", label="train")
    ax.plot(df["K"], df["MSE_tai_tao_test"], "s--", label="test (nguoi chua gap)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("So thanh phan K"); ax.set_ylabel("MSE tai tao (khong gian da chuan hoa)")
    ax.set_title("Sai so tai tao (inverse_transform) theo K")
    ax.legend()
    save_fig(fig, "sai_so_tai_tao.png")
    for r in df.itertuples():
        log(f"    K={r.K:3d}: MSE train={r.MSE_tai_tao_train:.4f} test={r.MSE_tai_tao_test:.4f} "
            f"(mat {r._4:.1f}% / {r._5:.1f}% phuong sai)")
    return df


# ----------------------------------------------------------------------------
# 11a. So sanh cach giam chieu dung cho MODEL: PCA vs SelectKBest vs LDA (co giam sat)
# ----------------------------------------------------------------------------
def run_supervised_comparison(Xtr, ytr, Xte, yte, groups) -> pd.DataFrame:
    rows = []
    for k in [10, 25, 50, 100, 200]:
        for name, red in [("PCA", PCA(k, random_state=RANDOM_STATE)), ("SelectKBest(f_classif)", SelectKBest(f_classif, k=k))]:
            pipe = Pipeline([("scale", StandardScaler()), ("red", red),
                             ("clf", LinearSVC(max_iter=5000, random_state=RANDOM_STATE))])
            cv = cv_score(pipe, Xtr, ytr, groups)
            rows.append({"phuong_phap": name, "K": k, "acc_cv": cv.mean(), "acc_cv_std": cv.std(),
                         "acc_test": pipe.fit(Xtr, ytr).score(Xte, yte)})
    for k in [2, 5]:
        pipe = Pipeline([("scale", StandardScaler()), ("red", LinearDiscriminantAnalysis(n_components=k)),
                         ("clf", LinearSVC(max_iter=5000, random_state=RANDOM_STATE))])
        cv = cv_score(pipe, Xtr, ytr, groups)
        rows.append({"phuong_phap": "LDA (co giam sat)", "K": k, "acc_cv": cv.mean(), "acc_cv_std": cv.std(),
                     "acc_test": pipe.fit(Xtr, ytr).score(Xte, yte)})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "so_sanh_giam_chieu.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, g in df.groupby("phuong_phap"):
        ax.errorbar(g["K"], g["acc_cv"] * 100, yerr=g["acc_cv_std"] * 100, fmt="o-", capsize=3, label=name)
    ax.set_xscale("log"); ax.set_xlabel("So chieu giu lai"); ax.set_ylabel("Accuracy CV theo nguoi (%)")
    ax.set_title("PCA (khong giam sat) vs SelectKBest vs LDA (co giam sat) - cung LinearSVC")
    ax.legend()
    save_fig(fig, "so_sanh_giam_chieu.png")
    for r in df.itertuples():
        log(f"    {r.phuong_phap:24s} K={r.K:3d}: acc CV={r.acc_cv:.4f}+-{r.acc_cv_std:.3f} test={r.acc_test:.4f}")
    return df


# ----------------------------------------------------------------------------
# 11b. t-SNE / UMAP - CHI de truc quan hoa (khong co transform on dinh cho du lieu moi)
# ----------------------------------------------------------------------------
def run_embedding_comparison(Xtr, ytr, scaler) -> pd.DataFrame:
    import umap  # noqa: import cham (numba) -> chi import khi can

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(Xtr), EMBED_SAMPLE, replace=False)
    Z, y = scaler.transform(Xtr[idx]), ytr[idx]
    embeds = {}
    t0 = time.perf_counter(); embeds["PCA"] = PCA(2, random_state=RANDOM_STATE).fit_transform(Z); t_pca = time.perf_counter() - t0
    t0 = time.perf_counter()
    embeds["t-SNE"] = TSNE(2, perplexity=30, init="pca", random_state=RANDOM_STATE).fit_transform(Z)
    t_tsne = time.perf_counter() - t0
    t0 = time.perf_counter()
    embeds["UMAP"] = umap.UMAP(n_components=2, random_state=RANDOM_STATE).fit_transform(Z)
    t_umap = time.perf_counter() - t0

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, (name, E), secs in zip(axes, embeds.items(), [t_pca, t_tsne, t_umap]):
        # Do tach lop trong 2D: 5-NN cross-val tren CHINH embedding (chi so mo ta, khong phai model)
        knn = cross_val_score(KNeighborsClassifier(5), E, y, cv=5).mean()
        rows.append({"phuong_phap": name, "thoi_gian_s": secs, "acc_5NN_trong_2D": knn,
                     "co_transform_cho_du_lieu_moi": name == "PCA"})
        for act in ACTIVITY_ORDER:
            m = y == act
            ax.scatter(E[m, 0], E[m, 1], s=4, alpha=0.5, color=COLORS[act], label=act)
        ax.set_title(f"{name} ({secs:.1f}s) - 5-NN trong 2D: {knn:.1%}")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(markerscale=3, fontsize=7)
    save_fig(fig, "pca_tsne_umap.png")
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "pca_tsne_umap.csv", index=False)
    for r in df.itertuples():
        log(f"    {r.phuong_phap:6s}: {r.thoi_gian_s:6.1f}s  5-NN trong 2D = {r.acc_5NN_trong_2D:.3f}")
    return df


# ----------------------------------------------------------------------------
# Mo rong 1. Kernel PCA (RBF) vs PCA tuyen tinh
# ----------------------------------------------------------------------------
def run_kernel_pca(Xtr, ytr, Xte, yte, groups, k: int = 50) -> pd.DataFrame:
    rows = []
    for name, red in [("PCA tuyen tinh", PCA(k, random_state=RANDOM_STATE)),
                      ("Kernel PCA (RBF, gamma=1/561)", KernelPCA(k, kernel="rbf", eigen_solver="randomized",
                                                                   random_state=RANDOM_STATE))]:
        pipe = Pipeline([("scale", StandardScaler()), ("red", red),
                         ("clf", LinearSVC(max_iter=5000, random_state=RANDOM_STATE))])
        res = cross_validate(pipe, Xtr, ytr, groups=groups, cv=GroupKFold(N_FOLDS), n_jobs=N_FOLDS)
        t0 = time.perf_counter(); pipe.fit(Xtr, ytr); fit_s = time.perf_counter() - t0
        t0 = time.perf_counter(); acc = pipe.score(Xte, yte); pred_ms = (time.perf_counter() - t0) / len(Xte) * 1e3
        rows.append({"phuong_phap": name, "K": k, "acc_cv": res["test_score"].mean(), "acc_test": acc,
                     "thoi_gian_fit_s": fit_s, "thoi_gian_du_doan_ms_moi_mau": pred_ms,
                     "phai_luu_toan_bo_train_de_transform": name.startswith("Kernel")})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "kernel_pca.csv", index=False)
    for r in df.itertuples():
        log(f"    {r.phuong_phap:30s} K={k}: acc CV={r.acc_cv:.4f} test={r.acc_test:.4f} fit={r.thoi_gian_fit_s:.1f}s "
            f"du doan={r.thoi_gian_du_doan_ms_moi_mau:.3f} ms/mau")
    return df


# ----------------------------------------------------------------------------
# Mo rong 2. IncrementalPCA - mo phong luong du lieu khong vua RAM
# ----------------------------------------------------------------------------
def run_incremental_pca(scaler, Xtr, ytr, Xte, yte, k: int, batch_size: int = 500) -> dict:
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
    pca = PCA(k, random_state=RANDOM_STATE).fit(Ztr)
    ipca = IncrementalPCA(k, batch_size=batch_size)
    t0 = time.perf_counter()
    for start in range(0, len(Ztr), batch_size):
        ipca.partial_fit(Ztr[start:start + batch_size])
    ipca_s = time.perf_counter() - t0
    # Do giong nhau giua 2 khong gian con: trung binh cos goc chinh (1 = trung khit)
    cos = np.linalg.svd(pca.components_ @ ipca.components_.T, compute_uv=False)
    clf_pca = LinearSVC(max_iter=5000, random_state=RANDOM_STATE).fit(pca.transform(Ztr), ytr)
    clf_ipca = LinearSVC(max_iter=5000, random_state=RANDOM_STATE).fit(ipca.transform(Ztr), ytr)
    res = {"K": k, "batch_size": batch_size, "so_batch": int(np.ceil(len(Ztr) / batch_size)),
           "phuong_sai_PCA_%": float(pca.explained_variance_ratio_.sum() * 100),
           "phuong_sai_IncrementalPCA_%": float(ipca.explained_variance_ratio_.sum() * 100),
           "cos_goc_chinh_trung_binh": float(cos.mean()), "cos_goc_chinh_nho_nhat": float(cos.min()),
           "acc_test_PCA": float(clf_pca.score(pca.transform(Zte), yte)),
           "acc_test_IncrementalPCA": float(clf_ipca.score(ipca.transform(Zte), yte)),
           "bo_nho_moi_batch_MB": batch_size * 561 * 8 / 1e6, "bo_nho_ca_ma_tran_MB": Ztr.nbytes / 1e6,
           "thoi_gian_IncrementalPCA_s": ipca_s}
    pd.DataFrame([res]).to_csv(REPORTS_DIR / "incremental_pca.csv", index=False)
    log(f"    IncrementalPCA K={k}, {res['so_batch']} batch x {batch_size}: phuong sai {res['phuong_sai_IncrementalPCA_%']:.2f}% "
        f"(PCA {res['phuong_sai_PCA_%']:.2f}%), cos goc chinh TB={res['cos_goc_chinh_trung_binh']:.4f} "
        f"min={res['cos_goc_chinh_nho_nhat']:.4f}, acc test {res['acc_test_IncrementalPCA']:.4f} vs {res['acc_test_PCA']:.4f}, "
        f"RAM/batch {res['bo_nho_moi_batch_MB']:.1f} MB vs {res['bo_nho_ca_ma_tran_MB']:.1f} MB")
    return res


# ----------------------------------------------------------------------------
# Mo rong 3. Phat hien bat thuong bang sai so tai tao: hoat dong CHUA TUNG GAP co bi phat hien?
# ----------------------------------------------------------------------------
def run_anomaly_detection(Xtr, ytr, Xte, yte, k: int) -> pd.DataFrame:
    rows = []
    for unseen in ACTIVITY_ORDER:
        keep = ytr != unseen
        scaler = StandardScaler().fit(Xtr[keep])
        pca = PCA(k, random_state=RANDOM_STATE).fit(scaler.transform(Xtr[keep]))
        Z = scaler.transform(Xte)
        err = np.mean((Z - pca.inverse_transform(pca.transform(Z))) ** 2, axis=1)
        is_new = (yte == unseen).astype(int)
        thr = np.percentile(np.mean((scaler.transform(Xtr[keep]) - pca.inverse_transform(
            pca.transform(scaler.transform(Xtr[keep])))) ** 2, axis=1), 95)   # nguong = p95 sai so tren train
        flagged = err > thr
        rows.append({"hoat_dong_chua_gap": unseen, "AUC_sai_so_tai_tao": roc_auc_score(is_new, err),
                     "sai_so_TB_hoat_dong_moi": err[is_new == 1].mean(), "sai_so_TB_hoat_dong_da_biet": err[is_new == 0].mean(),
                     "ty_le_bat_duoc_o_nguong_p95_train_%": flagged[is_new == 1].mean() * 100,
                     "bao_dong_gia_o_nguong_p95_train_%": flagged[is_new == 0].mean() * 100})
    df = pd.DataFrame(rows).sort_values("AUC_sai_so_tai_tao", ascending=False)
    df.to_csv(REPORTS_DIR / "phat_hien_bat_thuong.csv", index=False)
    for r in df.itertuples():
        log(f"    Bo {r.hoat_dong_chua_gap:18s} khoi train: AUC={r.AUC_sai_so_tai_tao:.3f}  bat duoc {r._5:.1f}% "
            f"bao dong gia {r._6:.1f}% (nguong p95)")
    return df


# ----------------------------------------------------------------------------
# Model cuoi: PCA (diem ngot) + LinearSVC, va LDA 5 chieu de doi chieu
# ----------------------------------------------------------------------------
def build_final_models(Xtr, ytr, Xte, yte, k_sweet: int) -> dict:
    out = {}
    for name, pipe, path in [
        (f"PCA K={k_sweet} + LinearSVC", make_pipe(k_sweet), "pca_pipeline.joblib"),
        ("Goc 561 chieu + LinearSVC", make_pipe(None), None),
        ("LDA 5 chieu + LinearSVC", Pipeline([("scale", StandardScaler()),
                                               ("lda", LinearDiscriminantAnalysis(n_components=5)),
                                               ("clf", LinearSVC(max_iter=5000, random_state=RANDOM_STATE))]),
         "lda_pipeline.joblib"),
    ]:
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        rep = classification_report(yte, pred, labels=ACTIVITY_ORDER, output_dict=True, zero_division=0)
        out[name] = {"acc_test": accuracy_score(yte, pred),
                     "f1_theo_lop": {a: round(rep[a]["f1-score"], 4) for a in ACTIVITY_ORDER}}
        if path:
            joblib.dump(pipe, MODELS_DIR / path)
        if name.startswith("PCA"):
            cm = confusion_matrix(yte, pred, labels=ACTIVITY_ORDER)
            pd.DataFrame(cm, index=ACTIVITY_ORDER, columns=ACTIVITY_ORDER).to_csv(REPORTS_DIR / "confusion_matrix_pca.csv")
            fig, ax = plt.subplots(figsize=(7.5, 6.5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(6)); ax.set_xticklabels(ACTIVITY_ORDER, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(6)); ax.set_yticklabels(ACTIVITY_ORDER, fontsize=8)
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_xlabel("Du doan"); ax.set_ylabel("Thuc te")
            ax.set_title(f"Ma tran nham lan tren test - {name}")
            fig.colorbar(im, ax=ax, fraction=0.046)
            save_fig(fig, "confusion_matrix_pca.png")
            out[name]["cap_nham_nhieu_nhat"] = [
                {"thuc_te": ACTIVITY_ORDER[i], "du_doan": ACTIVITY_ORDER[j], "so_lan": int(cm[i, j])}
                for i, j in sorted(((i, j) for i in range(6) for j in range(6) if i != j),
                                   key=lambda t: -cm[t])[:3]]
        log(f"    {name:28s} acc test={out[name]['acc_test']:.4f}  F1: " +
            ", ".join(f"{a[:8]}={v:.3f}" for a, v in out[name]["f1_theo_lop"].items()))
    log("  Da luu models/pca_pipeline.joblib (model de bai yeu cau) va models/lda_pipeline.joblib")
    return out


def main() -> None:
    log("1-2. Nap du lieu (giu cach chia theo nguoi) ...")
    df, Xtr, Xte, ytr, yte, groups, feats = load()

    log("3-5. Scree plot, phuong sai tich luy, bang nguong ...")
    scaler, pca_full, pca_raw, var_table = run_variance_analysis(Xtr)
    k95 = int(var_table.loc[var_table.nguong_phuong_sai == 0.95, "so_chieu_co_chuan_hoa"].iloc[0])

    log("6. Danh doi so chieu vs accuracy (GroupKFold theo nguoi tren train) ...")
    tradeoff = run_dimension_tradeoff(Xtr, ytr, Xte, yte, groups)
    k_sweet = choose_sweet_spot(tradeoff)
    plot_tradeoff(tradeoff, k_sweet, k95)

    log("7. Scatter PC1-PC2 ...")
    scatter = run_pc_scatter(scaler, pca_full, Xtr, ytr, Xte, yte, groups)

    log("8. Y nghia PC1 ...")
    pc1 = run_pc1_analysis(pca_full, pca_raw, feats)

    log("9. Chi phi dung luong / truyen / bo nho / phep tinh ...")
    cost_lin, cost_rbf = run_cost_analysis(Xtr, Xte, ytr, k_sweet, k95)

    log("10. Sai so tai tao theo K ...")
    recon = run_reconstruction(scaler, Xtr, Xte)

    log("11a. PCA vs SelectKBest vs LDA ...")
    comp = run_supervised_comparison(Xtr, ytr, Xte, yte, groups)

    log("11b. PCA vs t-SNE vs UMAP (truc quan hoa) ...")
    emb = run_embedding_comparison(Xtr, ytr, scaler)

    log("Mo rong 1. Kernel PCA ...")
    kpca = run_kernel_pca(Xtr, ytr, Xte, yte, groups)

    log("Mo rong 2. IncrementalPCA ...")
    ipca = run_incremental_pca(scaler, Xtr, ytr, Xte, yte, k_sweet)

    log("Mo rong 3. Phat hien bat thuong bang sai so tai tao ...")
    anomaly = run_anomaly_detection(Xtr, ytr, Xte, yte, k95)

    log("Model cuoi ...")
    final = build_final_models(Xtr, ytr, Xte, yte, k_sweet)

    save_summary(var_table=var_table, k95=k95, tradeoff=tradeoff, k_sweet=k_sweet, scatter=scatter, pc1=pc1,
                 cost_lin=cost_lin, cost_rbf=cost_rbf, recon=recon, comp=comp, emb=emb, kpca=kpca, ipca=ipca,
                 anomaly=anomaly, final=final, pca_full=pca_full)
    log("HOAN THANH. Xem reports/ va models/.")


def save_summary(var_table, k95, tradeoff, k_sweet, scatter, pc1, cost_lin, cost_rbf, recon, comp, emb,
                 kpca, ipca, anomaly, final, pca_full) -> None:
    summary = {
        "du_lieu": {"train": 7352, "test": 2947, "nguoi_train": 21, "nguoi_test": 9, "so_dac_trung": 561},
        "phuong_sai_PC1_PC2_PC3_%": [round(float(v) * 100, 2) for v in pca_full.explained_variance_ratio_[:3]],
        "bang_nguong_phuong_sai": var_table.to_dict(orient="records"),
        "K_95_phan_tram_phuong_sai": k95,
        "danh_doi_chieu_accuracy": tradeoff.to_dict(orient="records"),
        "diem_ngot_K_chon_bang_CV": k_sweet,
        "pc1_pc2": scatter, "pc1": pc1,
        "chi_phi_tuyen_tinh": cost_lin.to_dict(orient="records"),
        "chi_phi_rbf_svm": cost_rbf.to_dict(orient="records"),
        "sai_so_tai_tao": recon.to_dict(orient="records"),
        "so_sanh_giam_chieu": comp.to_dict(orient="records"),
        "pca_tsne_umap": emb.to_dict(orient="records"),
        "kernel_pca": kpca.to_dict(orient="records"),
        "incremental_pca": ipca,
        "phat_hien_bat_thuong": anomaly.to_dict(orient="records"),
        "model_cuoi_test": final,
    }
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log("  Da luu reports/tom_tat.json")


if __name__ == "__main__":
    main()
