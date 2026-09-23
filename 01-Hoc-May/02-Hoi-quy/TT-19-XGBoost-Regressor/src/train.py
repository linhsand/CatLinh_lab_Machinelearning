"""Huan luyen mo hinh XGBoost chinh thuc cho du bao nhu cau thue xe dap (TT-19).

Chay doc lap (khong can mo Jupyter): python src/train.py
Ghi model vao models/xgb_bike.json va chi so vao reports/train_metrics.json

Chien luoc (xem giai thich day du trong notebooks/xgboost_bike_demand.ipynb):
  1. Early-stop tren (train=nam 1)/(val=9 thang dau nam 2) chi de TIM so cay,
     vi tap train don le khong "thay" duoc muc cau tang manh cua nam 2.
  2. Refit tren toan bo train+val (voi so cay da chon) truoc khi danh gia
     tren test (3 thang cuoi nam 2) -> giai quyet van de ngoai suy cua cay.
  3. Tinh chinh sieu tham so bang RandomizedSearchCV (TimeSeriesSplit) tren
     chinh train+val, roi lap lai buoc early-stop/refit voi tham so tot nhat.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

sys.path.append(str(Path(__file__).resolve().parent))
from features import build_features, time_based_split  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
TARGET = "cnt"
RANDOM_STATE = 42

PARAM_DIST = {
    "max_depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "reg_alpha": [0, 0.1, 0.5, 1.0],
}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pick_n_estimators(params: dict, X: pd.DataFrame, y: pd.Series, holdout_frac=0.15) -> int:
    """Early stopping tren mot lat cuoi (theo thoi gian) cua X/y de chon so cay."""
    cut = int(len(X) * (1 - holdout_frac))
    p = dict(params, n_estimators=3000)
    m = xgb.XGBRegressor(**p, early_stopping_rounds=100, eval_metric="rmse")
    m.fit(X.iloc[:cut], y.iloc[:cut], eval_set=[(X.iloc[cut:], y.iloc[cut:])], verbose=False)
    return max(int(m.best_iteration), 50)


def main() -> None:
    df_raw = pd.read_csv(DATA_DIR / "hour.csv", parse_dates=["dteday"])
    df = build_features(df_raw, drop_leak=True)
    feature_cols = [c for c in df.columns if c != TARGET]

    train, val, test = time_based_split(df)
    X_trval = pd.concat([train[feature_cols], val[feature_cols]])
    y_trval = pd.concat([train[TARGET], val[TARGET]])
    X_test, y_test = test[feature_cols], test[TARGET]

    base = xgb.XGBRegressor(n_estimators=400, tree_method="hist", n_jobs=-1,
                             random_state=RANDOM_STATE, objective="reg:squarederror")
    search = RandomizedSearchCV(base, PARAM_DIST, n_iter=25, scoring="neg_root_mean_squared_error",
                                 cv=TimeSeriesSplit(n_splits=3), random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X_trval, y_trval)
    best_params = dict(search.best_params_, tree_method="hist", n_jobs=-1,
                        random_state=RANDOM_STATE, objective="reg:squarederror")

    n_estimators = pick_n_estimators(best_params, X_trval, y_trval)

    t0 = time.time()
    model = xgb.XGBRegressor(**best_params, n_estimators=n_estimators)
    model.fit(X_trval, y_trval)
    fit_time_s = time.time() - t0

    pred_test = np.clip(model.predict(X_test), 0, None)
    metrics = {
        "best_params": search.best_params_,
        "n_estimators": n_estimators,
        "rmse_test": rmse(y_test, pred_test),
        "r2_test": float(r2_score(y_test, pred_test)),
        "mae_test": float(mean_absolute_error(y_test, pred_test)),
        "fit_time_s": fit_time_s,
        "n_features": len(feature_cols),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    model.save_model(MODELS_DIR / "xgb_bike.json")
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
