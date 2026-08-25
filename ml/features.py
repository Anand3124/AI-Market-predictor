"""Technical feature engineering.

Leakage rules obeyed everywhere in this module:
  * every feature at row t is built only from bars at t and earlier
    (rolling windows are backward looking, `shift(-1)` is never used here);
  * the target is the *next* day's return, created with a single forward shift;
  * the last row has no target -- it is kept separately as the live prediction
    input and is never part of training or testing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import WARMUP_BARS

TARGET = "target_next_return"

# Human readable labels used by the explanation engine in the UI.
FEATURE_LABELS = {
    "ret_1d": "1-day price change",
    "ret_3d": "3-day price change",
    "ret_5d": "5-day momentum",
    "ret_10d": "10-day momentum",
    "ret_20d": "20-day momentum",
    "mom_accel": "momentum acceleration (5d vs 20d)",
    "px_vs_sma5": "price vs 5-day moving average",
    "px_vs_sma10": "price vs 10-day moving average",
    "px_vs_sma20": "price vs 20-day moving average",
    "px_vs_sma50": "price vs 50-day moving average",
    "sma10_vs_sma50": "short vs long trend (10d vs 50d MA)",
    "macd_norm": "MACD trend strength",
    "rsi_7": "7-day RSI (short-term overbought / oversold)",
    "rsi_14": "14-day RSI (overbought / oversold)",
    "vol_10d": "10-day volatility",
    "vol_20d": "20-day volatility",
    "vol_regime": "volatility regime (10d vs 20d)",
    "atr_pct": "ATR (average true range, % of price)",
    "range_pct": "today's high-low range",
    "close_loc": "close position inside today's range",
    "gap_pct": "overnight gap",
    "dist_high20": "distance below the 20-day high",
    "dist_low20": "distance above the 20-day low",
    "zscore_20": "price z-score vs 20-day mean",
    "vol_ratio": "volume vs its 20-day average",
    "vol_chg_1d": "1-day volume change",
    "vol_trend": "5-day vs 20-day volume trend",
}

# Features where a *higher* value normally means "more bullish". Used only to
# phrase the explanation, never in the maths.
HIGHER_IS_BULLISH = {
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "mom_accel",
    "px_vs_sma5", "px_vs_sma10", "px_vs_sma20", "px_vs_sma50",
    "sma10_vs_sma50", "macd_norm", "close_loc", "dist_low20", "zscore_20",
}


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder's smoothing, backward looking only.
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 means an unbroken run of gains -> RSI 100.
    rsi = rsi.where(avg_loss.ne(0.0), 100.0)
    return rsi.where(avg_gain.ne(0.0) | avg_loss.ne(0.0), 50.0)


def _atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr / close


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a frame with features + target, and the list of feature names.

    The frame keeps every row (including the final, target-less one) so the
    caller can slice off the live row.
    """
    out = pd.DataFrame(index=df.index)
    out["Date"] = df["Date"]
    out["Close"] = df["Close"]

    close = df["Close"]
    ret1 = close.pct_change(fill_method=None)

    # --- price change / momentum ---
    out["ret_1d"] = ret1
    out["ret_3d"] = close.pct_change(3, fill_method=None)
    out["ret_5d"] = close.pct_change(5, fill_method=None)
    out["ret_10d"] = close.pct_change(10, fill_method=None)
    out["ret_20d"] = close.pct_change(20, fill_method=None)
    out["mom_accel"] = out["ret_5d"] - out["ret_20d"] * (5.0 / 20.0)

    # --- moving averages (expressed as a relative distance, so they are
    #     comparable across price levels and across markets) ---
    sma5 = close.rolling(5).mean()
    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    out["px_vs_sma5"] = close / sma5 - 1.0
    out["px_vs_sma10"] = close / sma10 - 1.0
    out["px_vs_sma20"] = close / sma20 - 1.0
    out["px_vs_sma50"] = close / sma50 - 1.0
    out["sma10_vs_sma50"] = sma10 / sma50 - 1.0

    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    out["macd_norm"] = (ema12 - ema26) / close

    # --- RSI ---
    out["rsi_7"] = _rsi(close, 7) / 100.0
    out["rsi_14"] = _rsi(close, 14) / 100.0

    # --- volatility ---
    vol10 = ret1.rolling(10).std()
    vol20 = ret1.rolling(20).std()
    out["vol_10d"] = vol10
    out["vol_20d"] = vol20
    out["vol_regime"] = vol10 / vol20.replace(0.0, np.nan) - 1.0

    # --- position within the recent range ---
    high20 = close.rolling(20).max()
    low20 = close.rolling(20).min()
    out["dist_high20"] = close / high20 - 1.0
    out["dist_low20"] = close / low20.replace(0.0, np.nan) - 1.0
    out["zscore_20"] = (close - sma20) / close.rolling(20).std().replace(0.0, np.nan)

    # --- intraday shape, only when real OHLC exists ---
    if {"High", "Low", "Open"}.issubset(df.columns):
        out["atr_pct"] = _atr_pct(df)
        rng = (df["High"] - df["Low"])
        out["range_pct"] = rng / close
        out["close_loc"] = ((close - df["Low"]) / rng.replace(0.0, np.nan)).fillna(0.5)
        out["gap_pct"] = df["Open"] / close.shift(1) - 1.0

    # --- volume, only when it exists ---
    if "Volume" in df.columns:
        vol = df["Volume"].astype(float)
        vsma20 = vol.rolling(20).mean()
        vsma5 = vol.rolling(5).mean()
        out["vol_ratio"] = vol / vsma20.replace(0.0, np.nan) - 1.0
        out["vol_chg_1d"] = vol.pct_change(fill_method=None)
        out["vol_trend"] = vsma5 / vsma20.replace(0.0, np.nan) - 1.0

    # --- target: tomorrow's close-to-close return (the only forward shift) ---
    out[TARGET] = close.shift(-1) / close - 1.0

    feature_names = [c for c in out.columns if c not in ("Date", "Close", TARGET)]

    out = out.replace([np.inf, -np.inf], np.nan)
    # Volume ratios can explode on a near-zero base; cap them so a handful of
    # rows cannot dominate a least-squares fit.
    for col in ("vol_ratio", "vol_chg_1d", "vol_trend"):
        if col in out.columns:
            out[col] = out[col].clip(-5.0, 10.0)

    return out, feature_names


def split_supervised(frame: pd.DataFrame, feature_names: list[str]):
    """Split into (supervised rows, live row).

    `live` is the most recent bar: it has complete features but no known
    target yet, which is exactly what tomorrow's prediction is made from.
    """
    usable = frame.iloc[WARMUP_BARS:].copy()
    usable = usable.dropna(subset=feature_names)

    live = usable[usable[TARGET].isna()]
    supervised = usable.dropna(subset=[TARGET]).reset_index(drop=True)
    live_row = live.iloc[[-1]] if len(live) else supervised.iloc[[-1]]
    return supervised, live_row
