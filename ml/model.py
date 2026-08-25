"""Train / evaluate / persist the Linear Regression models.

Persistence contract
--------------------
Every trained model is written to ``models/<key>_linreg.joblib`` together with
the SHA-256 fingerprint of the CSV it was trained on and a schema version.
``get_model()`` reuses that artifact and only retrains when

  * no artifact exists,
  * the CSV fingerprint no longer matches (the data changed),
  * the schema version changed (the feature/training code changed), or
  * the caller explicitly asks with ``force=True`` (the Retrain button).
"""
from __future__ import annotations

import datetime as _dt
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    MARKETS,
    MIN_TEST_ROWS,
    MODEL_DIR,
    SCHEMA_VERSION,
    SIGNAL_THRESHOLD_K,
    TEST_SIZE,
    model_path,
)
from .data import file_fingerprint, load_market, market_path
from .features import TARGET, build_features, split_supervised


def _make_pipeline() -> Pipeline:
    # The scaler lives *inside* the pipeline, so when it is fitted on the
    # training slice only, the test slice cannot leak into the scaling stats.
    return Pipeline(
        [("scaler", StandardScaler()), ("model", LinearRegression())]
    )


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min()) if len(equity) else 0.0


def _backtest(actual: np.ndarray, pred: np.ndarray, threshold: float) -> dict:
    """Long/flat/short on the model signal, one bar held, no costs."""
    position = np.where(pred > threshold, 1.0, np.where(pred < -threshold, -1.0, 0.0))
    strat = position * actual
    equity = np.cumprod(1.0 + strat)
    bh = np.cumprod(1.0 + actual)
    ann = 252.0
    sharpe = (
        float(strat.mean() / strat.std() * np.sqrt(ann)) if strat.std() > 0 else 0.0
    )
    trades = int((np.diff(np.concatenate([[0.0], position])) != 0).sum())
    return {
        "equity": equity,
        "buy_hold_equity": bh,
        "strategy_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "buy_hold_return": float(bh[-1] - 1.0) if len(bh) else 0.0,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(equity),
        "buy_hold_max_drawdown": _max_drawdown(bh),
        "n_long": int((position > 0).sum()),
        "n_short": int((position < 0).sum()),
        "n_flat": int((position == 0).sum()),
        "n_trades": trades,
        "win_rate": float((strat[position != 0] > 0).mean())
        if (position != 0).any()
        else 0.0,
    }


def _grade(dir_acc: float, baseline_dir: float, r2: float, ic: float) -> tuple[str, str]:
    """Honest label. Daily price prediction is close to a coin flip and the
    grading here says so rather than dressing up noise."""
    edge = dir_acc - baseline_dir
    if r2 > 0 and dir_acc >= 0.55 and ic > 0.05:
        return ("Promising", "Beats the naive baseline on direction, error and correlation on unseen data.")
    if dir_acc >= 0.52 and ic > 0.02:
        return ("Slight edge", "A small directional edge on unseen data - well within what luck can produce over one test window.")
    if dir_acc >= 0.50 or r2 > -0.01:
        return ("Coin-flip", "No reliable edge on unseen data. Treat every signal as low confidence.")
    return ("No edge", "The model performs at or below a naive baseline on unseen data. Do not trade this signal.")


def train_market(key: str) -> dict:
    """Fit, evaluate on a strictly-later holdout, and return the bundle."""
    df, report = load_market(key)
    frame, feature_names = build_features(df)
    supervised, live_row = split_supervised(frame, feature_names)

    X = supervised[feature_names].to_numpy(dtype=float)
    y = supervised[TARGET].to_numpy(dtype=float)
    dates = supervised["Date"].to_numpy()
    closes = supervised["Close"].to_numpy(dtype=float)

    n = len(supervised)
    n_test = max(MIN_TEST_ROWS, int(round(n * TEST_SIZE)))
    if n_test >= n - MIN_TEST_ROWS:
        raise ValueError(f"{key}: not enough rows ({n}) for an honest split")
    split = n - n_test

    # ---- chronological split: train = older, test = newer & unseen ----------
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    eval_pipe = _make_pipeline().fit(X_tr, y_tr)
    pred_te = eval_pipe.predict(X_te)
    pred_tr = eval_pipe.predict(X_tr)

    # ---- honest metrics, all computed on the unseen slice -------------------
    naive = float(y_tr.mean())          # best constant guess known at train time
    # A usable baseline must be chosen without looking at the test outcomes.
    # Pick the majority direction from the training period, then apply that
    # fixed guess to the later test period.
    train_majority_sign = 1.0 if (y_tr > 0).mean() >= 0.5 else -1.0
    baseline_dir = float((np.sign(y_te) == train_majority_sign).mean())
    test_majority_rate = float(max((y_te > 0).mean(), (y_te <= 0).mean()))
    ic = float(np.corrcoef(pred_te, y_te)[0, 1]) if np.std(pred_te) > 0 else 0.0
    dir_acc = float((np.sign(pred_te) == np.sign(y_te)).mean())

    thr_bt = SIGNAL_THRESHOLD_K * float(np.std(pred_tr))  # from train only
    backtest = _backtest(y_te, pred_te, thr_bt)

    # ---- robustness: expanding-window walk-forward over the whole history ---
    cv_scores = []
    for tr_idx, te_idx in TimeSeriesSplit(n_splits=5).split(X):
        p = _make_pipeline().fit(X[tr_idx], y[tr_idx]).predict(X[te_idx])
        cv_scores.append(float((np.sign(p) == np.sign(y[te_idx])).mean()))

    grade, grade_note = _grade(dir_acc, baseline_dir, r2_score(y_te, pred_te), ic)

    metrics = {
        "n_rows": n,
        "n_train": int(split),
        "n_test": int(n_test),
        "train_start": str(pd.Timestamp(dates[0]).date()),
        "train_end": str(pd.Timestamp(dates[split - 1]).date()),
        "test_start": str(pd.Timestamp(dates[split]).date()),
        "test_end": str(pd.Timestamp(dates[-1]).date()),
        "test_r2": float(r2_score(y_te, pred_te)),
        "train_r2": float(r2_score(y_tr, pred_tr)),
        "test_mae": float(mean_absolute_error(y_te, pred_te)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_te, pred_te))),
        "baseline_mae": float(mean_absolute_error(y_te, np.full_like(y_te, naive))),
        "baseline_rmse": float(
            np.sqrt(mean_squared_error(y_te, np.full_like(y_te, naive)))
        ),
        "directional_accuracy": dir_acc,
        "baseline_directional_accuracy": baseline_dir,
        "test_majority_direction_rate": test_majority_rate,
        "baseline_direction": "up" if train_majority_sign > 0 else "down",
        "information_coefficient": ic,
        "cv_directional_accuracy_mean": float(np.mean(cv_scores)),
        "cv_directional_accuracy_std": float(np.std(cv_scores)),
        "cv_folds": cv_scores,
        "grade": grade,
        "grade_note": grade_note,
        "backtest": backtest,
    }

    # ---- deployment model: same pipeline refit on every supervised row ------
    # Metrics above stay untouched -- they come from data this refit has not
    # been scored on. Refitting only lets tomorrow's prediction use the most
    # recent bars as well.
    final_pipe = _make_pipeline().fit(X, y)
    pred_all = final_pipe.predict(X)
    threshold = SIGNAL_THRESHOLD_K * float(np.std(pred_all))

    lr: LinearRegression = final_pipe.named_steps["model"]
    scaler: StandardScaler = final_pipe.named_steps["scaler"]

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "key": key,
        "market": MARKETS[key]["name"],
        "symbol": MARKETS[key]["symbol"],
        "fingerprint": report["fingerprint"],
        "trained_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "pipeline": final_pipe,
        "feature_names": feature_names,
        "coefficients": lr.coef_.astype(float),
        "intercept": float(lr.intercept_),
        "feature_mean": scaler.mean_.astype(float),
        "feature_scale": scaler.scale_.astype(float),
        "metrics": metrics,
        "threshold": threshold,
        "backtest_threshold": thr_bt,
        "train_return_std": float(np.std(y_tr)),
        "full_return_std": float(np.std(y)),
        "data_report": report,
        "live_row": live_row.reset_index(drop=True),
        "test_frame": pd.DataFrame(
            {
                "Date": pd.to_datetime(dates[split:]),
                "Close": closes[split:],
                "actual": y_te,
                "predicted": pred_te,
            }
        ),
        "history": pd.DataFrame(
            {"Date": frame["Date"], "Close": frame["Close"]}
        ).dropna(),
    }
    return bundle


def save_model(bundle: dict) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = model_path(bundle["key"])
    joblib.dump(bundle, path, compress=3)
    return path


def artifact_status(key: str) -> dict:
    """Describe the saved artifact without training anything."""
    path = model_path(key)
    csv = market_path(key)
    info = {"key": key, "path": path, "exists": os.path.exists(path)}
    if not info["exists"]:
        info["state"] = "missing"
        return info
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # corrupt / version mismatch
        info["state"] = "unreadable"
        info["error"] = str(exc)
        return info
    info["trained_at"] = bundle.get("trained_at")
    info["schema_version"] = bundle.get("schema_version")
    current_fp = file_fingerprint(csv)
    if bundle.get("schema_version") != SCHEMA_VERSION:
        info["state"] = "stale-schema"
    elif bundle.get("fingerprint") != current_fp:
        info["state"] = "stale-data"
    else:
        info["state"] = "current"
    return info


def get_model(key: str, force: bool = False) -> tuple[dict, str]:
    """Return (bundle, source) where source is 'cache' or 'trained'.

    This is the function that makes refreshing the website cheap: it only
    reaches the expensive training path when something actually changed.
    """
    path = model_path(key)
    if not force and os.path.exists(path):
        try:
            bundle = joblib.load(path)
            if bundle.get("schema_version") == SCHEMA_VERSION and bundle.get(
                "fingerprint"
            ) == file_fingerprint(market_path(key)):
                return bundle, "cache"
        except Exception:
            pass  # fall through and rebuild
    bundle = train_market(key)
    try:
        save_model(bundle)
    except OSError:
        # Read-only hosting can still serve a freshly trained, in-memory model.
        # Streamlit's resource cache prevents repeated training in the session.
        return bundle, "trained-memory"
    return bundle, "trained"
