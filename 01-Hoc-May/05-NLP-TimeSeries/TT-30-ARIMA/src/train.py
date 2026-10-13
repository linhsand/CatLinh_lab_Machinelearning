"""TT-30 - ARIMA / SARIMA: du bao luong khach thang (AirPassengers) de lap ke hoach nhan su.

Nguyen tac danh gia:
  * Chia THEO THOI GIAN: 120 thang train (1949-1958) - 24 thang test (1959-1960). Du bao 24 thang MOT LAN tu goc
    cuoi train (khong nhin gia tri test nao) -> moi mo hinh, ke ca baseline, deu du bao "mu" 24 buoc.
  * Chon bac mo hinh: ACF/PACF (thu cong) va AIC (luoi) - deu CHI dung train. Test chi de bao cao.
  * Them: backtest 4 goc du bao (thang 84 / 96 / 108 / 120) vi 24 diem test la qua it de ket luan.

Cac buoc (theo README de bai):
  1-3.  Chuoi goc, phan ra, log                       -> run_look()
  4-5.  ADF 6 phien ban, ACF/PACF                     -> run_stationarity()
  6-7.  Chia 120/24, 2 baseline                       -> split(), baselines()
  8.    SARIMA thu cong (+ ban cua de, + khong log)   -> run_manual()
  9.    Quet luoi AIC (+ bay enforce_*=False)         -> run_grid(), run_aic_trap()
  10.   Ljung-Box + bieu do chan doan phan du         -> run_residuals()
  11-13. Du bao 24 thang + khoang, MAPE, ti le phu    -> run_forecast()
  Them: du bao cuon chieu 1 thang, backtest nhieu goc -> one_step(), run_backtest()

Chay: python src/train.py   (~2 phut tren CPU)
"""
from __future__ import annotations

import itertools
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.statespace.sarimax import SARIMAX

import stationarity as S

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
for d in (REPORTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

N_TRAIN, H = 120, 24
S_ = 12
BURN_IN = 13          # 1 + 12 phan du dau tien cua mo hinh sai phan (1)(12) la gia tri khoi tao, bo khi kiem tra


def log(msg: str) -> None:
    print(msg, flush=True)


def save_fig(fig, name: str) -> None:
    S.save(fig, REPORTS_DIR / name)


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAPE_%": float(np.mean(np.abs(p - y) / y) * 100), "MAE": float(np.mean(np.abs(p - y))),
            "RMSE": float(np.sqrt(np.mean((p - y) ** 2))), "sai_lech_TB_%": float(np.mean((p - y) / y) * 100)}


# ------------------------------------------------------------------ 1-3. Nhin
def run_look(y: pd.Series) -> dict:
    amp = S.amplitude_table(y)
    amp.to_csv(REPORTS_DIR / "bien_do_mua_vu.csv")
    dec = S.decompose_compare(y, REPORTS_DIR / "decompose.png")
    dec.to_csv(REPORTS_DIR / "decompose_so_sanh.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(17, 3.8))
    axes[0].plot(y.index, y, c="#4C72B0"); axes[0].set_title("Chuoi goc (nghin khach / thang)")
    axes[1].plot(y.index, np.log(y), c="#55A868"); axes[1].set_title("log(y): bien do gan nhu khong doi")
    axes[2].plot(amp.index, amp["bien_do_goc"] / amp["bien_do_goc"].iloc[0], "o-", label="bien do goc (max - min)")
    axes[2].plot(amp.index, amp["bien_do_log"] / amp["bien_do_log"].iloc[0], "s-", label="bien do sau log")
    axes[2].axhline(1, c="gray", lw=0.8); axes[2].set_title("Bien do mua vu moi nam (chia cho nam 1949)")
    axes[2].legend(fontsize=8)
    save_fig(fig, "chuoi_goc_va_log.png")
    return {"bien_do": amp, "decompose": dec}


def run_stationarity(y: pd.Series) -> dict:
    adf = S.adf_table(y)
    adf.to_csv(REPORTS_DIR / "adf.csv", index=False)
    acfp = S.plot_acf_pacf(np.log(y).diff().diff(12), REPORTS_DIR / "acf_pacf.png")
    acfp.to_csv(REPORTS_DIR / "acf_pacf.csv", index=False)
    return {"adf": adf, "acf_pacf": acfp}


# ------------------------------------------------------------------ 6-7. Chia + baseline
def split(y: pd.Series, n_train: int = N_TRAIN, h: int = H):
    return y.iloc[:n_train], y.iloc[n_train:n_train + h]


def baselines(train: pd.Series, h: int) -> dict[str, np.ndarray]:
    """Du bao h buoc tu cuoi train, khong nhin tuong lai.
    naive: lap lai thang cuoi. seasonal naive: lap lai 12 thang cuoi.
    (them) seasonal naive + drift: 12 thang cuoi x toc do tang TB moi nam cua train (tinh tren log)."""
    last_year = train.iloc[-S_:].to_numpy()
    reps = int(np.ceil(h / S_))
    snaive = np.tile(last_year, reps)[:h]
    growth = np.exp(np.log(train).diff(S_).dropna().mean())      # he so tang TB nam qua nam
    years_ahead = np.repeat(np.arange(1, reps + 1), S_)[:h]
    return {"Naive (thang truoc)": np.repeat(train.iloc[-1], h), "Seasonal naive (cung thang nam ngoai)": snaive,
            "Seasonal naive + drift (them)": snaive * growth ** (years_ahead - 1) * growth}


# ------------------------------------------------------------------ SARIMA chung
def fit_sarima(train: pd.Series, order, seasonal, use_log: bool = True, enforce: bool = True):
    """enforce=True (mac dinh cua statsmodels), KHONG dung enforce_*=False nhu code mau cua de: khi tat, statsmodels
    khoi tao "diffuse" va bo so quan sat dau = so trang thai cua mo hinh (14-27 tuy bac) -> AIC moi mo hinh tinh tren
    so diem KHAC NHAU, khong so sanh duoc (xem run_aic_trap)."""
    z = np.log(train) if use_log else train
    m = SARIMAX(z, order=order, seasonal_order=(*seasonal, S_), enforce_stationarity=enforce,
                enforce_invertibility=enforce)
    return m.fit(disp=False)


def run_aic_trap(train, test) -> pd.DataFrame:
    """Cung 5 mo hinh, fit voi enforce_*=False (code cua de) va True (mac dinh)."""
    rows = []
    for o, so in [((1, 1, 0), (0, 1, 0)), ((2, 1, 2), (0, 1, 0)), ((0, 1, 1), (0, 1, 1)), ((1, 1, 1), (1, 1, 1)),
                  ((0, 1, 1), (1, 1, 1))]:
        for enf in (False, True):
            f = fit_sarima(train, o, so, enforce=enf)
            rows.append({"mo_hinh": f"{o}{(*so, S_)}", "enforce": enf, "so_diem_bo_dau": f.loglikelihood_burn,
                         "so_diem_tinh_likelihood": f.nobs_effective, "loglik": f.llf, "AIC": f.aic,
                         "MAPE_test_%": metrics(test, forecast(f, len(test))["du_bao"])["MAPE_%"]})
    t = pd.DataFrame(rows)
    t["hang_AIC_trong_nhom"] = t.groupby("enforce")["AIC"].rank().astype(int)
    t.to_csv(REPORTS_DIR / "bay_aic_enforce.csv", index=False)
    return t


def one_step(y, train, test, order, seasonal) -> pd.DataFrame:
    """Du bao CUON CHIEU 1 thang toi tren test: tham so co dinh tu train, moi thang dung du lieu that den thang truoc.
    Baseline tuong ung: naive = thang truoc that, seasonal naive = cung thang nam ngoai that."""
    fit = fit_sarima(train, order, seasonal)
    full = fit.apply(np.log(y.iloc[:len(train) + len(test)]))        # giu nguyen tham so, chi loc lai
    pred = np.exp(full.get_prediction(start=len(train), dynamic=False).predicted_mean)
    yy = y.iloc[:len(train) + len(test)]
    rows = [{"mo_hinh": "SARIMA 1 buoc (tham so tu train)", **metrics(test, pred)},
            {"mo_hinh": "Naive 1 buoc (thang truoc that)", **metrics(test, yy.shift(1).iloc[len(train):])},
            {"mo_hinh": "Seasonal naive 1 buoc (nam ngoai that)", **metrics(test, yy.shift(12).iloc[len(train):])}]
    t = pd.DataFrame(rows)
    t.to_csv(REPORTS_DIR / "du_bao_1_buoc.csv", index=False)
    return t


def forecast(fit, h: int, use_log: bool = True) -> pd.DataFrame:
    f = fit.get_forecast(steps=h)
    out = pd.DataFrame({"du_bao": f.predicted_mean})
    for a, tag in [(0.05, "95"), (0.20, "80")]:
        ci = f.conf_int(alpha=a)
        out[f"duoi_{tag}"], out[f"tren_{tag}"] = ci.iloc[:, 0], ci.iloc[:, 1]
    return np.exp(out) if use_log else out      # exp() moi cot -> ve thang so khach that


def coverage(y_true, fc: pd.DataFrame, tag: str) -> float:
    y = np.asarray(y_true)
    return float(np.mean((y >= fc[f"duoi_{tag}"].to_numpy()) & (y <= fc[f"tren_{tag}"].to_numpy())))


def evaluate(name: str, train, test, order, seasonal, use_log=True) -> dict:
    t0 = time.time(); fit = fit_sarima(train, order, seasonal, use_log); secs = time.time() - t0
    fc = forecast(fit, len(test), use_log)
    lb = acorr_ljungbox(fit.resid.iloc[BURN_IN:], lags=[12, 24], return_df=True)
    return {"mo_hinh": name, "order": str(order), "seasonal_order": str((*seasonal, S_)), "log": use_log,
            "AIC": fit.aic if use_log else np.nan, "BIC": fit.bic if use_log else np.nan,
            **metrics(test, fc["du_bao"]), "phu_95_%": coverage(test, fc, "95") * 100,
            "phu_80_%": coverage(test, fc, "80") * 100, "LjungBox_p_lag12": float(lb["lb_pvalue"].iloc[0]),
            "LjungBox_p_lag24": float(lb["lb_pvalue"].iloc[1]), "thoi_gian_fit_s": secs, "_fit": fit, "_fc": fc}


def _table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])


# ------------------------------------------------------------------ 8. Thu cong
def run_manual(train, test) -> tuple[pd.DataFrame, dict]:
    """(0,1,1)(0,1,1,12): ACF cua log+d1+d12 chi vuot nguong o lag 1 va lag 12 roi tat -> MA(1) + SMA(1).
    (1,1,1)(1,1,1,12): bac goi y cua de bai. + ban (0,1,1)(0,1,1,12) KHONG log de thay cam bay."""
    runs = {"thu_cong": evaluate("SARIMA thu cong (0,1,1)(0,1,1,12) - log", train, test, (0, 1, 1), (0, 1, 1)),
            "de_bai": evaluate("SARIMA cua de (1,1,1)(1,1,1,12) - log", train, test, (1, 1, 1), (1, 1, 1)),
            "khong_log": evaluate("SARIMA (0,1,1)(0,1,1,12) - KHONG log", train, test, (0, 1, 1), (0, 1, 1), use_log=False)}
    t = _table(list(runs.values()))
    t.to_csv(REPORTS_DIR / "sarima_thu_cong.csv", index=False)
    return t, runs


# ------------------------------------------------------------------ 9. Luoi AIC
def run_grid(train, test, pq=(0, 1, 2), PQ=(0, 1, 2)) -> tuple[pd.DataFrame, dict]:
    """d = D = 1 co dinh theo ADF. AIC chi so sanh duoc giua cac mo hinh CUNG muc sai phan (cung du lieu dau vao)."""
    rows = []
    t0 = time.time()
    for p, q, P, Q in itertools.product(pq, pq, PQ, PQ):
        try:
            fit = fit_sarima(train, (p, 1, q), (P, 1, Q))
            fc = forecast(fit, len(test))
            rows.append({"order": (p, 1, q), "seasonal_order": (P, 1, Q, S_), "AIC": fit.aic, "BIC": fit.bic,
                         "MAPE_test_%": metrics(test, fc["du_bao"])["MAPE_%"]})
        except Exception as e:                       # mot so to hop khong hoi tu
            rows.append({"order": (p, 1, q), "seasonal_order": (P, 1, Q, S_), "AIC": np.nan, "BIC": np.nan,
                         "MAPE_test_%": np.nan, "loi": str(e)[:60]})
    secs = time.time() - t0
    t = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    t["hang_AIC"] = np.arange(1, len(t) + 1)
    t["hang_MAPE_test"] = t["MAPE_test_%"].rank().astype("Int64")
    t.to_csv(REPORTS_DIR / "luoi_aic.csv", index=False)
    best = t.iloc[0]
    log(f"[grid] {len(t)} mo hinh trong {secs:.0f}s | AIC tot nhat {best['order']}{best['seasonal_order']} "
        f"AIC={best['AIC']:.1f} MAPE test={best['MAPE_test_%']:.2f}%")
    run = evaluate(f"SARIMA theo AIC {best['order']}{best['seasonal_order']} - log", train, test, tuple(best["order"]),
                   tuple(best["seasonal_order"][:3]))
    return t, run


# ------------------------------------------------------------------ 10. Phan du
def run_residuals(run: dict) -> pd.DataFrame:
    fit = run["_fit"]
    r = fit.resid.iloc[BURN_IN:]
    lb = acorr_ljungbox(r, lags=[6, 12, 18, 24], return_df=True)
    jb, jb_p, skew, kurt = jarque_bera(r)
    fig = fit.plot_diagnostics(figsize=(13, 8), lags=24)
    fig.suptitle(f"Chan doan phan du - {run['mo_hinh']} (bo {BURN_IN} phan du dau)", y=1.01)
    # plot_diagnostics tu bo giai doan khoi tao (loglikelihood_burn) nen khong can cat them
    save_fig(fig, "residual_diagnostics.png")
    t = lb.reset_index().rename(columns={"index": "lag"})
    t["ket_luan"] = np.where(t["lb_pvalue"] > 0.05, "nhieu trang (khong bac bo H0)", "CON cau truc")
    t.attrs.update({"JarqueBera_p": float(jb_p), "skew": float(skew), "kurtosis": float(kurt),
                    "do_lech_chuan_phan_du_log": float(r.std())})
    t.to_csv(REPORTS_DIR / "ljung_box.csv", index=False)
    return t


# ------------------------------------------------------------------ 11-13. Du bao
def run_forecast(y, train, test, final: dict, base: dict) -> pd.DataFrame:
    fc = final["_fc"]
    rows = [{"mo_hinh": k, **metrics(test, v)} for k, v in base.items()]
    rows.append({"mo_hinh": final["mo_hinh"], **metrics(test, fc["du_bao"]), "phu_95_%": final["phu_95_%"],
                 "phu_80_%": final["phu_80_%"]})
    t = pd.DataFrame(rows)
    t.to_csv(REPORTS_DIR / "so_sanh_test.csv", index=False)
    detail = fc.assign(thuc_te=test.to_numpy(), seasonal_naive=base["Seasonal naive (cung thang nam ngoai)"])
    detail["trong_95"] = (detail["thuc_te"] >= detail["duoi_95"]) & (detail["thuc_te"] <= detail["tren_95"])
    detail.round(1).to_csv(REPORTS_DIR / "du_bao_24_thang.csv")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(train.index[-48:], train.iloc[-48:], c="k", lw=1.2, label="train (4 nam cuoi)")
    ax.plot(test.index, test, c="k", lw=2, marker="o", ms=3, label="thuc te (test)")
    ax.fill_between(fc.index, fc["duoi_95"], fc["tren_95"], color="#4C72B0", alpha=0.15, label="khoang 95%")
    ax.fill_between(fc.index, fc["duoi_80"], fc["tren_80"], color="#4C72B0", alpha=0.25, label="khoang 80%")
    ax.plot(fc.index, fc["du_bao"], c="#4C72B0", lw=2, label=f"SARIMA (MAPE {final['MAPE_%']:.1f}%)")
    sn = base["Seasonal naive (cung thang nam ngoai)"]
    ax.plot(test.index, sn, c="#C44E52", ls="--", lw=1.2, label=f"seasonal naive (MAPE {metrics(test, sn)['MAPE_%']:.1f}%)")
    ax.set_ylabel("nghin khach / thang"); ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"Du bao 24 thang (1959-1960) tu goc 12/1958 - {final['mo_hinh']}")
    ax = axes[1]
    err = (fc["du_bao"] - test) / test * 100
    ax.bar(test.index, err, width=20, color=np.where(err > 0, "#DD8452", "#4C72B0"))
    ax.axhline(0, c="k", lw=0.8); ax.set_ylabel("sai so %"); ax.set_title("Sai so SARIMA theo thang (cam = du bao thua)")
    save_fig(fig, "du_bao_khoang.png")
    return t


# ------------------------------------------------------------------ Them: backtest
def run_backtest(y, order, seasonal, origins=(84, 96, 108, 120), h=H) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for o in origins:
        tr, te = split(y, o, h)
        b = baselines(tr, len(te))
        fit = fit_sarima(tr, order, seasonal)
        fc = forecast(fit, len(te))
        rows.append({"goc": tr.index[-1].strftime("%m/%Y"), "so_thang_train": o, "mo_hinh": "SARIMA",
                     **metrics(te, fc["du_bao"]), "phu_95_%": coverage(te, fc, "95") * 100,
                     "phu_80_%": coverage(te, fc, "80") * 100})
        for k, v in b.items():
            rows.append({"goc": tr.index[-1].strftime("%m/%Y"), "so_thang_train": o, "mo_hinh": k, **metrics(te, v)})
    t = pd.DataFrame(rows)
    t.to_csv(REPORTS_DIR / "backtest.csv", index=False)
    summary = t.groupby("mo_hinh", sort=False)[["MAPE_%", "phu_95_%", "phu_80_%"]].mean()
    summary.to_csv(REPORTS_DIR / "backtest_tom_tat.csv")
    return t, summary


def save_model(fit, train: pd.Series, order, seasonal, path: Path = MODELS_DIR / "sarima_fit.pkl") -> None:
    """Luu gon: bac + tham so + chuoi train (vai KB). fit.save() mac dinh nang ~14 MB (luu ca mang trung gian cua bo
    loc Kalman), con fit.save(remove_data=True) thi khong du bao duoc nua."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump({"order": tuple(order), "seasonal_order": (*seasonal, S_), "params": fit.params.to_dict(),
                     "train": train, "log": True}, f)


def load_model(path: Path = MODELS_DIR / "sarima_fit.pkl"):
    """Dung lai ket qua fit tu tham so da luu (khong uoc luong lai): load_model().get_forecast(24)."""
    import pickle
    with open(path, "rb") as f:
        m = pickle.load(f)
    mod = SARIMAX(np.log(m["train"]), order=m["order"], seasonal_order=m["seasonal_order"])
    return mod.filter(pd.Series(m["params"]))


def save_summary(**kw) -> None:
    def conv(o):
        if isinstance(o, pd.DataFrame):
            return o.reset_index().to_dict(orient="records")
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        return str(o)
    with open(REPORTS_DIR / "tom_tat.json", "w", encoding="utf-8") as f:
        json.dump(kw, f, ensure_ascii=False, indent=2, default=conv)
    log("[save] reports/tom_tat.json")


def main() -> None:
    y = S.load()
    look = run_look(y); log(look["decompose"].to_string())
    st = run_stationarity(y); log(st["adf"].to_string())
    train, test = split(y)
    base = baselines(train, H)
    for k, v in base.items():
        log(f"[baseline] {k}: {metrics(test, v)}")
    trap = run_aic_trap(train, test); log(trap.to_string())
    man_t, man = run_manual(train, test); log(man_t.to_string())
    grid_t, grid_run = run_grid(train, test); log(grid_t.head(10).to_string())
    # Mo hinh cuoi: AIC thap nhat giua ban thu cong va ban luoi (chi dung train)
    cands = [man["thu_cong"], man["de_bai"], grid_run]
    final = min(cands, key=lambda r: r["AIC"])
    log(f"[chon] {final['mo_hinh']} (AIC {final['AIC']:.1f})")
    save_model(final["_fit"], train, eval(final["order"]), eval(final["seasonal_order"])[:3])
    lb = run_residuals(final); log(lb.to_string()); log(str(lb.attrs))
    comp = run_forecast(y, train, test, final, base); log(comp.to_string())
    order, seasonal = eval(final["order"]), eval(final["seasonal_order"])[:3]
    os_ = one_step(y, train, test, order, seasonal); log(os_.to_string())
    bt, bts = run_backtest(y, order, seasonal); log(bts.to_string())
    save_summary(adf=st["adf"], decompose=look["decompose"], bay_aic=trap, thu_cong=man_t, luoi_top10=grid_t.head(10),
                 chon=final["mo_hinh"], ljung_box=lb, ljung_box_attrs=lb.attrs, so_sanh_test=comp, mot_buoc=os_,
                 backtest=bts)


if __name__ == "__main__":
    main()
