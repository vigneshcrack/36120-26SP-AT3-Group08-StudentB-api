"""Market data and Student B's own features (MA5, MA20, TrendStrength), with a time-based cache."""

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
from mykrml_25740287.data.crypto import (
    MarketDataError,
    UnsupportedTokenError,
    fetch_kraken_daily_ohlcv,
    normalise_token,
)
from mykrml_25740287.features.crypto import (
    compute_features,
    compute_market_risk_components,
    compute_market_risk_index,
    features_owned_by,
)

from .errors import market_data_unavailable, unsupported_token

OWN_FEATURES: list[str] = features_owned_by("B")  # ["MA5", "MA20", "TrendStrength"]

_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_lock = threading.Lock()


def validate_token(token: str) -> str:
    """Return the normalised token or raise the brief's 'Unsupported cryptocurrency' error."""
    try:
        return normalise_token(token)
    except UnsupportedTokenError:
        raise unsupported_token(token) from None


def get_candles(token: str, max_age_seconds: int) -> pd.DataFrame:
    """Daily OHLCV for ``token`` (incomplete current day removed), cached for ``max_age_seconds``."""
    now = time.monotonic()
    with _lock:
        cached = _cache.get(token)
        if cached and now - cached[0] < max_age_seconds:
            return cached[1]
    try:
        candles = fetch_kraken_daily_ohlcv(token)
    except MarketDataError:
        with _lock:  # serve the last good download rather than failing, if we have one
            cached = _cache.get(token)
        if cached:
            return cached[1]
        raise market_data_unavailable() from None
    with _lock:
        _cache[token] = (now, candles)
    return candles


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def feature_frame(candles: pd.DataFrame) -> pd.DataFrame:
    """All 13 contract features for every day, computed with the shared package."""
    return compute_features(candles)


def to_json_number(value) -> float | None:
    """Convert numpy/pandas numbers to JSON-safe floats (NaN/inf become null)."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def own_features_payload(candles: pd.DataFrame, history_days: int) -> tuple[dict, list[dict]]:
    """Latest values of Student B's features and the last ``history_days`` rows (with close)."""
    feats = feature_frame(candles)[OWN_FEATURES]
    latest = {name: to_json_number(feats[name].iloc[-1]) for name in OWN_FEATURES}
    recent = feats.tail(history_days).join(candles["close"])
    history = [
        {"date": idx.date().isoformat(), "close": to_json_number(row["close"]),
         **{name: to_json_number(row[name]) for name in OWN_FEATURES}}
        for idx, row in recent.iterrows()
    ]
    return latest, history


def risk_history(candles: pd.DataFrame, days: int) -> list[dict]:
    """Same-day Market Risk Index and its weighted components for the last ``days`` days.

    This is the realised (not forecast) index, computed from Kraken candles with the brief's
    formula, so the Streamlit page can plot how risk has evolved.
    """
    feats = feature_frame(candles)
    close = candles["close"]
    mri = compute_market_risk_index(feats, close)
    comps = compute_market_risk_components(feats, close)
    frame = pd.DataFrame({
        "close": close,
        "MRI": mri,
        "volatility_points": 40 * comps["VRN"],
        "drawdown_points": 30 * comps["DRN"],
        "downside_return_points": 20 * comps["RRN"],
        "gap_risk_points": 10 * comps["GRN"],
    }).dropna().tail(days)
    return [
        {"date": idx.date().isoformat(), **{k: to_json_number(v) for k, v in row.items()}}
        for idx, row in frame.iterrows()
    ]
