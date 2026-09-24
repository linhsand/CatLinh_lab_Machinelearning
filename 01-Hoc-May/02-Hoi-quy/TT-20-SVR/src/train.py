"""Huan luyen SVR chinh thuc cho du doan cuong do chiu nen be tong (TT-20).

Chay doc lap: python src/train.py
Ghi pipeline vao models/svr_pipeline.joblib va chi so vao reports/train_metrics.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import joblib

sys.path.append(str(Path(__file__).resolve().parent))
from features import build_features, split_xy  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
RANDOM_STATE = 42

PARAM_GRID = {
    "regressor__svr__C": [1, 10, 100, 1000],
    "regressor__svr__gamma": ["scale", 0.01, 0.1, 1],
    "regressor__svr__epsilon": [0.01, 0.1, 0.5],
}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    df_raw = pd.read_csv(DATA_DIR / "concrete.csv")
    df = build_features(df_raw)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    pipe = Pipeline([("scale", StandardScaler()), ("svr", SVR(kernel="rbf"))])
    model = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())

    grid = GridSearchCV(model, PARAM_GRID, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
    t0 = time.time()
    grid.fit(X_train, y_train)
    fit_time_s = time.time() - t0

    best_model = grid.best_estimator_
    pred_test = best_model.predict(X_test)
    svr_final = best_model.regressor_.named_steps["svr"]
    n_sv = int(svr_final.support_.shape[0])

    metrics = {
        "best_params": grid.best_params_,
        "cv_rmse": float(-grid.best_score_),
        "rmse_test": rmse(y_test, pred_test),
        "r2_test": float(r2_score(y_test, pred_test)),
        "mae_test": float(mean_absolute_error(y_test, pred_test)),
        "n_support_vectors": n_sv,
        "n_train": len(X_train),
        "support_vector_ratio_%": round(n_sv / len(X_train) * 100, 1),
        "fit_time_s": fit_time_s,
        "n_features": X.shape[1],
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "svr_pipeline.joblib")
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
