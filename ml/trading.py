"""Turning a predicted return into a signal, a position size and risk levels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MAX_STOP_PCT, MIN_STOP_PCT

BUY, HOLD, SELL = "BUY", "HOLD", "SELL"


def classify(pred_return: float, threshold: float) -> str:
    if pred_return > threshold:
        return BUY
    if pred_return < -threshold:
        return SELL
    return HOLD


def volatility_stop_pct(live: pd.Series, atr_mult: float) -> tuple[float, str]:
    """Stop distance as a fraction of price, plus the basis it came from."""
    if "atr_pct" in live.index and np.isfinite(live["atr_pct"]) and live["atr_pct"] > 0:
        raw, basis = float(live["atr_pct"]), "14-day ATR"
    else:
        # Close-only series (WTI): approximate ATR with the 20-day return
        # standard deviation. ATR runs ~1.5x daily sigma for most markets.
        vol = float(live.get("vol_20d", 0.0) or 0.0)
        raw, basis = max(vol, 1e-4) * 1.5, "20-day volatility (no OHLC available)"
    stop = float(np.clip(raw * atr_mult, MIN_STOP_PCT, MAX_STOP_PCT))
    return stop, basis


def build_plan(
    *,
    signal: str,
    price: float,
    pred_return: float,
    live: pd.Series,
    budget: float,
    risk_pct: float,
    atr_mult: float,
    rr: float,
    allow_short: bool = True,
) -> dict:
    """Risk-first position sizing.

    The stop distance is set by volatility, then the position is sized so that
    being stopped out costs exactly the amount of capital the user is willing
    to risk -- never the other way round.
    """
    stop_pct, basis = volatility_stop_pct(live, atr_mult)
    risk_amount = budget * risk_pct / 100.0

    # Direction the plan is drawn for: the signal when there is one, otherwise
    # the raw sign of the prediction so the user still sees the levels.
    direction = 1 if signal == BUY else (-1 if signal == SELL else (1 if pred_return >= 0 else -1))

    uncapped_value = risk_amount / stop_pct if stop_pct > 0 else 0.0
    position_value = min(uncapped_value, budget)   # no leverage
    capped = uncapped_value > budget + 1e-9

    units = position_value / price if price > 0 else 0.0
    actual_risk = position_value * stop_pct

    if direction > 0:
        stop_loss = price * (1.0 - stop_pct)
        take_profit = price * (1.0 + stop_pct * rr)
    else:
        stop_loss = price * (1.0 + stop_pct)
        take_profit = price * (1.0 - stop_pct * rr)

    tradeable = signal in (BUY, SELL) and (allow_short or signal == BUY)

    return {
        "signal": signal,
        "direction": "Long" if direction > 0 else "Short",
        "tradeable": tradeable,
        "stop_pct": stop_pct,
        "stop_basis": basis,
        "risk_amount_requested": risk_amount,
        "risk_amount_actual": actual_risk if tradeable else 0.0,
        "position_value": position_value if tradeable else 0.0,
        "position_value_if_taken": position_value,
        "units": units if tradeable else 0.0,
        "units_if_taken": units,
        "capital_used_pct": (position_value / budget * 100.0) if budget > 0 else 0.0,
        "capped_by_budget": capped,
        "entry": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reward_risk": rr,
        "max_loss": actual_risk,
        "max_gain": actual_risk * rr,
    }


def confidence(pred_return: float, threshold: float, grade: str) -> tuple[float, str]:
    """Signal strength (0-100), explicitly NOT a probability of being right."""
    if threshold <= 0:
        strength = 0.0
    else:
        strength = float(np.clip(abs(pred_return) / (threshold * 3.0), 0.0, 1.0) * 100.0)
    # A strong-looking signal from a model with no measured edge is not a
    # confident signal, so the grade caps it.
    cap = {"Promising": 100.0, "Slight edge": 70.0, "Coin-flip": 45.0, "No edge": 25.0}
    strength = min(strength, cap.get(grade, 45.0))
    if strength >= 66:
        label = "High"
    elif strength >= 33:
        label = "Moderate"
    else:
        label = "Low"
    return strength, label
