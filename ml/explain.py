"""Plain-English explanation of a single prediction.

A linear model is fully transparent: the prediction is the intercept plus
`coefficient x standardised feature` for every feature. Those per-feature
products are exactly "how much each indicator moved today's forecast", which
is what the UI shows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_LABELS, HIGHER_IS_BULLISH

PERCENT_FEATURES = {
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "mom_accel",
    "px_vs_sma5", "px_vs_sma10", "px_vs_sma20", "px_vs_sma50",
    "sma10_vs_sma50", "macd_norm", "vol_10d", "vol_20d", "vol_regime",
    "atr_pct", "range_pct", "gap_pct", "dist_high20", "dist_low20",
    "vol_ratio", "vol_chg_1d", "vol_trend",
}


def format_value(name: str, value: float) -> str:
    if name in ("rsi_7", "rsi_14"):
        return f"{value * 100:.0f}"
    if name == "close_loc":
        return f"{value * 100:.0f}% of range"
    if name == "zscore_20":
        return f"{value:+.2f} sd"
    if name in PERCENT_FEATURES:
        return f"{value * 100:+.2f}%"
    return f"{value:.3f}"


def contributions(bundle: dict, live: pd.Series) -> pd.DataFrame:
    """Per-feature push on today's forecast, in return units."""
    names = bundle["feature_names"]
    x = live[names].to_numpy(dtype=float)
    z = (x - bundle["feature_mean"]) / bundle["feature_scale"]
    contrib = bundle["coefficients"] * z
    df = pd.DataFrame(
        {
            "feature": names,
            "label": [FEATURE_LABELS.get(n, n) for n in names],
            "value": x,
            "display": [format_value(n, v) for n, v in zip(names, x)],
            "zscore": z,
            "contribution": contrib,
        }
    )
    df["abs"] = df["contribution"].abs()
    return df.sort_values("abs", ascending=False).reset_index(drop=True)


def _unusual(z: float) -> str:
    a = abs(z)
    if a >= 2.0:
        return "an extreme reading"
    if a >= 1.0:
        return "an unusual reading"
    if a >= 0.4:
        return "a mildly elevated reading" if z > 0 else "a mildly depressed reading"
    return "a fairly normal reading"


def narrative(bundle: dict, live: pd.Series, pred_return: float, signal: str,
              top_n: int = 5) -> dict:
    """Return the pieces the UI assembles into the explanation panel."""
    contrib = contributions(bundle, live)
    top = contrib.head(top_n)

    bullets = []
    for _, row in top.iterrows():
        push_up = row["contribution"] > 0
        bullets.append(
            {
                "label": row["label"],
                "display": row["display"],
                "direction": "up" if push_up else "down",
                "contribution_pct": row["contribution"] * 100.0,
                "note": _unusual(row["zscore"]),
                "bullish_high": row["feature"] in HIGHER_IS_BULLISH,
            }
        )

    up = contrib[contrib["contribution"] > 0]["contribution"].sum()
    down = contrib[contrib["contribution"] < 0]["contribution"].sum()

    # --- market context, independent of the model ---
    context = []
    rsi = float(live.get("rsi_14", np.nan)) * 100.0
    if np.isfinite(rsi):
        if rsi >= 70:
            context.append(f"RSI(14) is {rsi:.0f} - overbought territory, which often precedes cooling off.")
        elif rsi <= 30:
            context.append(f"RSI(14) is {rsi:.0f} - oversold territory, where bounces are more common.")
        else:
            context.append(f"RSI(14) is {rsi:.0f} - neither overbought nor oversold.")

    s20 = float(live.get("px_vs_sma20", np.nan))
    s50 = float(live.get("px_vs_sma50", np.nan))
    if np.isfinite(s20) and np.isfinite(s50):
        if s20 > 0 and s50 > 0:
            context.append(f"Price sits {s20*100:+.1f}% above its 20-day and {s50*100:+.1f}% above its 50-day average - an uptrend.")
        elif s20 < 0 and s50 < 0:
            context.append(f"Price sits {s20*100:+.1f}% vs its 20-day and {s50*100:+.1f}% vs its 50-day average - a downtrend.")
        else:
            context.append(f"Price is {s20*100:+.1f}% vs its 20-day and {s50*100:+.1f}% vs its 50-day average - a mixed, range-bound trend.")

    v10, v20 = float(live.get("vol_10d", np.nan)), float(live.get("vol_20d", np.nan))
    if np.isfinite(v10) and np.isfinite(v20) and v20 > 0:
        ratio = v10 / v20
        state = "rising" if ratio > 1.15 else ("falling" if ratio < 0.85 else "steady")
        context.append(f"Short-term volatility is {state} ({v10*100:.2f}% daily vs {v20*100:.2f}% over 20 days), which sets how wide the stop needs to be.")

    if "vol_ratio" in live.index and np.isfinite(live["vol_ratio"]):
        vr = float(live["vol_ratio"])
        # the feature is clipped at +10.0, so do not print the bound as a
        # measurement -- say what it actually means
        amount = ("more than 10x" if vr >= 9.99
                  else f"{vr*100:+.0f}% versus")
        if vr > 0.25:
            context.append(f"Volume is running {amount} its 20-day average - conviction behind the move.")
        elif vr < -0.25:
            context.append(f"Volume is {vr*100:+.0f}% versus its 20-day average - thin participation, so the move is less reliable.")

    grade = bundle["metrics"]["grade"]
    verdict = {
        "BUY": "the indicators lean net-positive",
        "SELL": "the indicators lean net-negative",
        "HOLD": "the positive and negative indicators roughly cancel out",
    }[signal]

    summary = (
        f"Adding up every indicator, the model forecasts {pred_return*100:+.2f}% for the next session "
        f"because {verdict}. Bullish indicators contribute {up*100:+.2f}% and bearish ones {down*100:+.2f}%."
    )

    return {
        "summary": summary,
        "bullets": bullets,
        "context": context,
        "bullish_total": up,
        "bearish_total": down,
        "grade": grade,
        "table": contrib,
    }
