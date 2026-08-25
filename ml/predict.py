"""Ties the model, the signal logic and the explanation into one result."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import explain, trading


def predict(bundle: dict, *, budget: float, risk_pct: float, atr_mult: float,
            rr: float, allow_short: bool = True) -> dict:
    live = bundle["live_row"].iloc[0]
    names = bundle["feature_names"]
    X = live[names].to_numpy(dtype=float).reshape(1, -1)

    pred_return = float(bundle["pipeline"].predict(X)[0])
    price = float(live["Close"])
    as_of = pd.Timestamp(live["Date"])

    threshold = float(bundle["threshold"])
    signal = trading.classify(pred_return, threshold)
    strength, strength_label = trading.confidence(
        pred_return, threshold, bundle["metrics"]["grade"]
    )

    plan = trading.build_plan(
        signal=signal,
        price=price,
        pred_return=pred_return,
        live=live,
        budget=budget,
        risk_pct=risk_pct,
        atr_mult=atr_mult,
        rr=rr,
        allow_short=allow_short,
    )

    # Honest uncertainty: the typical size of this model's error on data it was
    # never scored on. The point forecast is far smaller than this band.
    rmse = float(bundle["metrics"]["test_rmse"])
    predicted_price = price * (1.0 + pred_return)

    story = explain.narrative(bundle, live, pred_return, signal)

    return {
        "signal": signal,
        "predicted_return": pred_return,
        "predicted_price": predicted_price,
        "price": price,
        "as_of": as_of,
        "threshold": threshold,
        "strength": strength,
        "strength_label": strength_label,
        "price_low": price * (1.0 + pred_return - rmse),
        "price_high": price * (1.0 + pred_return + rmse),
        "typical_error_pct": rmse * 100.0,
        "plan": plan,
        "explanation": story,
        "live": live,
    }
