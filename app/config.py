"""Runtime configuration, read from environment variables (never hard-coded URLs or secrets)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "mri_lightgbm.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

API_VERSION = "1.0.0"
FORECAST_HORIZON = 7


@dataclass(frozen=True)
class Settings:
    """Service settings.

    Attributes
    ----------
    student_a_url, student_c_url : str or None
        Base URLs of the peer FastAPI services that own Return20 (A) and
        Volatility20 / MaxPrice20 / GapRisk (C). ``None`` means "not configured".
    peer_timeout_seconds : float
        Timeout for each call to a peer ``/features`` endpoint.
    peer_retries : int
        Extra attempts after a failed peer call.
    market_cache_seconds : int
        How long Kraken candles are reused before downloading them again.
    peer_cache_seconds : int
        How long a successful peer response is reused without calling the peer.
    stale_peer_max_age_seconds : int
        Maximum age of the last successful peer response that may be used when the
        peer is down (it is flagged as stale in the response).
    peer_fallback_mode : str
        What to do when a peer feature is unavailable: ``"missing"`` (default, as the
        brief requires: predict with missing values) or ``"local"`` (compute the
        feature locally with the shared package; documented emergency option).
    github_url : str
        Repository URL reported by ``GET /``.
    """

    student_a_url: str | None
    student_c_url: str | None
    peer_timeout_seconds: float
    peer_retries: int
    market_cache_seconds: int
    peer_cache_seconds: int
    stale_peer_max_age_seconds: int
    peer_fallback_mode: str
    github_url: str


def _url(name: str) -> str | None:
    value = os.getenv(name, "").strip().rstrip("/")
    return value or None


@lru_cache
def get_settings() -> Settings:
    """Read settings once per process (tests clear the cache to change them)."""
    mode = os.getenv("PEER_FALLBACK_MODE", "missing").strip().lower()
    if mode not in {"missing", "local"}:
        mode = "missing"
    return Settings(
        student_a_url=_url("STUDENT_A_URL"),
        student_c_url=_url("STUDENT_C_URL"),
        peer_timeout_seconds=float(os.getenv("PEER_TIMEOUT_SECONDS", "10")),
        peer_retries=int(os.getenv("PEER_RETRIES", "1")),
        market_cache_seconds=int(os.getenv("MARKET_CACHE_SECONDS", "900")),
        peer_cache_seconds=int(os.getenv("PEER_CACHE_SECONDS", "900")),
        stale_peer_max_age_seconds=int(os.getenv("STALE_PEER_MAX_AGE_SECONDS", "86400")),
        peer_fallback_mode=mode,
        github_url=os.getenv(
            "GITHUB_URL", "https://github.com/vigneshcrack/36120-26SP-AT3-Group08-StudentB-api"
        ),
    )
