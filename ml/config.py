"""Central configuration: markets, paths and model hyper-parameters."""
from __future__ import annotations

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Bump this when features / training logic change so that stale artifacts
# on disk are detected and rebuilt automatically.
SCHEMA_VERSION = 5

# --- market registry -------------------------------------------------------
# key -> metadata. `file` is resolved relative to BASE_DIR.
MARKETS = {
    "spy": {
        "name": "S&P 500 ETF",
        "symbol": "SPY",
        "file": "spy_daily.csv",
        "icon": "📈",
        "unit": "USD",
        "blurb": "Broad US large-cap equity exposure.",
    },
    "nasdaq100": {
        "name": "Nasdaq-100 Index",
        "symbol": "NAS100",
        "file": "nasdaq100_daily.csv",
        "icon": "💻",
        "unit": "pts",
        "blurb": "US mega-cap technology benchmark.",
    },
    "gold": {
        "name": "Gold",
        "symbol": "GC=F",
        "file": "gold_daily.csv",
        "icon": "🥇",
        "unit": "USD/oz",
        "blurb": "Precious metal / defensive store of value.",
    },
    "btc": {
        "name": "Bitcoin / USD",
        "symbol": "BTC-USD",
        "file": "btc_daily.csv",
        "icon": "₿",
        "unit": "USD",
        "blurb": "High-volatility digital asset, trades 24/7.",
    },
    "wti": {
        "name": "WTI Crude Oil Spot",
        "symbol": "WTI-SPOT",
        "file": "wti_close_daily.csv",
        "icon": "🛢️",
        "unit": "USD/bbl",
        "blurb": "Energy benchmark. Close-only series (no OHLC / volume).",
    },
}

# --- training -------------------------------------------------------------
TEST_SIZE = 0.20          # newest 20% of rows are held out, never trained on
MIN_TEST_ROWS = 120       # refuse to trust a test set smaller than this
WARMUP_BARS = 60          # longest look-back window used by the features

# --- trading / signal -----------------------------------------------------
# A prediction must exceed this multiple of the model's own in-sample
# prediction spread before it is called BUY or SELL instead of HOLD.
SIGNAL_THRESHOLD_K = 0.50
DEFAULT_STOP_ATR_MULT = 1.5   # stop distance = mult x recent ATR (or vol proxy)
DEFAULT_RR = 2.0              # take-profit = RR x stop distance
MIN_STOP_PCT = 0.005          # never place a stop tighter than 0.5%
MAX_STOP_PCT = 0.30           # ... or wider than 30%


def market_path(key: str) -> str:
    return os.path.join(BASE_DIR, MARKETS[key]["file"])


def model_path(key: str) -> str:
    return os.path.join(MODEL_DIR, f"{key}_linreg.joblib")
