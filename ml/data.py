"""CSV loading, cleaning and fingerprinting.

The fingerprint is what lets the app skip retraining: a saved model records the
hash of the CSV it was trained on, so a model is only rebuilt when the
underlying dataset actually changes.
"""
from __future__ import annotations

import hashlib
import os

import pandas as pd

from .config import MARKETS, market_path

REQUIRED = ["Date", "Close"]
OPTIONAL = ["Open", "High", "Low", "Volume"]


def file_fingerprint(path: str) -> str:
    """SHA-256 of the file's bytes -- changes if and only if the data changes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_market(key: str) -> tuple[pd.DataFrame, dict]:
    """Load one market CSV, clean it, and report what was cleaned.

    Returns (dataframe, report). The dataframe is sorted by date, has a unique
    monotonic Date column and a strictly positive Close.
    """
    if key not in MARKETS:
        raise KeyError(f"unknown market '{key}'")
    path = market_path(key)
    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset missing: {path}")

    raw = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} is missing column(s): {missing}")

    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    report = {"rows_raw": len(df)}

    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    dupes = int(df["Date"].duplicated().sum())
    df = df.drop_duplicates(subset="Date", keep="last")

    for col in ["Close"] + OPTIONAL:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["Close"])
    report["dropped_missing_close"] = n_before - len(df)

    # A price series cannot be zero or negative. WTI's 2020-04-20 settlement of
    # -36.98 is a real historical print but it is meaningless for a return-based
    # model (returns become undefined), so those bars are removed.
    n_before = len(df)
    df = df[df["Close"] > 0]
    report["dropped_nonpositive_close"] = n_before - len(df)

    # OHLC gaps on thin holidays -> fall back to Close so the bar stays usable.
    for col in ["Open", "High", "Low"]:
        if col in df.columns:
            df[col] = df[col].fillna(df["Close"])
            df.loc[df[col] <= 0, col] = df["Close"]

    if "Volume" in df.columns:
        report["filled_volume"] = int(df["Volume"].isna().sum())
        # Missing volume on a holiday is not "zero traded" -- carry the last
        # observation forward, then fall back to the median.
        df["Volume"] = df["Volume"].ffill()
        df["Volume"] = df["Volume"].fillna(df["Volume"].median())
        df.loc[df["Volume"] <= 0, "Volume"] = pd.NA
        df["Volume"] = df["Volume"].ffill().bfill()

    df = df.reset_index(drop=True)

    report["duplicate_dates"] = dupes
    report["rows_clean"] = len(df)
    report["start"] = df["Date"].iloc[0].date().isoformat() if len(df) else None
    report["end"] = df["Date"].iloc[-1].date().isoformat() if len(df) else None
    report["has_ohlc"] = all(c in df.columns for c in ["Open", "High", "Low"])
    report["has_volume"] = "Volume" in df.columns
    report["fingerprint"] = file_fingerprint(path)
    report["file"] = os.path.basename(path)

    if len(df) < 250:
        raise ValueError(f"{report['file']}: only {len(df)} usable rows, need >= 250")

    return df, report
