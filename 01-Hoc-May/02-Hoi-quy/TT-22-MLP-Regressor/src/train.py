"""Huan luyen MLP Regressor chinh thuc du doan mpg (TT-22).

Chay doc lap: python src/train.py
Cau hinh lay tu notebook (chon bang cross-validation tren tap train):
  kien truc (64,) theo quy tac 1-SE, alpha=1, relu, khong early stopping,
  scale ca X (trong pipeline) va y (TransformedTargetRegressor).
Ghi model vao models/mlp_reg.joblib va chi so vao reports/train_metrics.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from features import load_clean, make_preprocessor, mpg_to_l100km, split_xy  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
RANDOM_STATE = 42
HIDDEN = (64,)
ALPHA = 1.0
ACTIVATION = "relu"


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_model() -> TransformedTargetRegressor:
    net = Pipeline([
        ("pre", make_preprocessor(scale=True)),
        ("mlp", MLPRegressor(hidden_layer_sizes=HIDDEN, activation=ACTIVATION, solver="adam",
                             alpha=ALPHA, learning_rate_init=1e-3, max_iter=2000,
                             early_stopping=False, n_iter_no_change=30, random_state=RANDOM_STATE)),
    ])
    return TransformedTargetRegressor(regressor=net, transformer=StandardScaler())


def main() -> None:
    X, y = split_xy(load_clean())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    model = build_model()
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time_s = time.time() - t0

    pred = model.predict(X_test)
    metrics = {
        "hidden_layer_sizes": list(HIDDEN),
        "alpha": ALPHA,
        "activation": ACTIVATION,
        "rmse_test_mpg": rmse(y_test, pred),
        "r2_test": float(r2_score(y_test, pred)),
        "mae_test_mpg": float(mean_absolute_error(y_test, pred)),
        "mae_test_l100km": float(mean_absolute_error(mpg_to_l100km(y_test), mpg_to_l100km(pred))),
        "n_iter": int(model.regressor_.named_steps["mlp"].n_iter_),
        "fit_time_s": fit_time_s,
        "n_train": len(X_train),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "mlp_reg.joblib")
    with open(REPORTS_DIR / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
