"""Huan luyen KNN Regressor chinh thuc cho dinh gia nha California (TT-21).

Chay doc lap: python src/train.py
Ghi pipeline vao models/knn_pipeline.joblib va chi so vao reports/train_metrics.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from features import load_clean, split_xy  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
RANDOM_STATE = 42
BEST_K = 10


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    df = load_clean()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=BEST_K, weights="distance", n_jobs=-1)),
    ])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    fit_time_s = time.time() - t0

    pred_test = pipe.predict(X_test)
    metrics = {
        "n_neighbors": BEST_K,
        "rmse_test": rmse(y_test, pred_test),
        "r2_test": float(r2_score(y_test, pred_test)),
        "mae_test": float(mean_absolute_error(y_test, pred_test)),
        "fit_time_s": fit_time_s,
        "n_train": len(X_train),
        "n_features": X.shape[1],
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "knn_pipeline.joblib")
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
