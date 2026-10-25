"""Calls to the other students' /features endpoints, with timeouts, retries and caching.

If a peer cannot be reached, its features are returned as missing (the brief requires the
prediction service to keep operating with missing values). A recent successful response
(up to STALE_PEER_MAX_AGE_SECONDS old) is reused and flagged as stale, which keeps
predictions accurate through short peer outages and Render cold starts.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field

import httpx

from .config import FORECAST_HORIZON, Settings

logger = logging.getLogger("mri_api")

# features this model needs from each peer (exact JSON keys from CONTRACT.md)
PEER_FEATURES: dict[str, list[str]] = {
    "student_a": ["Return20"],
    "student_c": ["Volatility20", "MaxPrice20", "GapRisk"],
}

_peer_cache: dict[tuple[str, str], tuple[float, dict[str, float]]] = {}


@dataclass
class PeerResult:
    values: dict[str, float] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def clear_cache() -> None:
    _peer_cache.clear()


def _peer_url(peer: str, settings: Settings) -> str | None:
    return {"student_a": settings.student_a_url, "student_c": settings.student_c_url}[peer]


def _parse(payload: dict, wanted: list[str]) -> dict[str, float]:
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, dict):
        raise ValueError("response has no 'features' object")  # noqa: TRY004
    parsed = {}
    for name in wanted:
        value = features.get(name)
        if isinstance(value, (int, float)) and math.isfinite(value):
            parsed[name] = float(value)
    if not parsed:
        raise ValueError(f"none of {wanted} present in response")
    return parsed


async def _call_peer(client: httpx.AsyncClient, peer: str, token: str, settings: Settings) -> dict[str, float]:
    url = _peer_url(peer, settings)
    if url is None:
        raise RuntimeError("peer URL not configured")
    last_error: Exception | None = None
    for attempt in range(settings.peer_retries + 1):
        try:
            response = await client.post(
                f"{url}/features", json={"token": token, "forecast_horizon": FORECAST_HORIZON}
            )
            response.raise_for_status()
            return _parse(response.json(), PEER_FEATURES[peer])
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if attempt < settings.peer_retries:
                await asyncio.sleep(0.5 * (attempt + 1))
    raise RuntimeError(str(last_error))


async def _get_peer(client: httpx.AsyncClient, peer: str, token: str, settings: Settings) -> tuple[str, dict, str, str | None]:
    """Return (peer, values, source label, error message)."""
    now = time.monotonic()
    cached = _peer_cache.get((peer, token))
    if cached and now - cached[0] < settings.peer_cache_seconds:
        return peer, cached[1], f"{peer}_api", None
    try:
        values = await _call_peer(client, peer, token, settings)
        _peer_cache[(peer, token)] = (now, values)
        return peer, values, f"{peer}_api", None
    except Exception as exc:  # noqa: BLE001 - any failure means "peer unavailable"
        logger.warning("Peer %s unavailable for %s: %s", peer, token, exc)
        if cached and now - cached[0] < settings.stale_peer_max_age_seconds:
            age_min = int((now - cached[0]) // 60)
            return peer, cached[1], f"{peer}_api (cached, {age_min} min old)", str(exc)
        return peer, {}, "unavailable", str(exc)


async def gather_peer_features(token: str, settings: Settings) -> PeerResult:
    """Fetch all peer features in parallel; failures become missing values."""
    result = PeerResult()
    timeout = httpx.Timeout(settings.peer_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        responses = await asyncio.gather(*(_get_peer(client, p, token, settings) for p in PEER_FEATURES))
    for peer, values, source, error in responses:
        for name in PEER_FEATURES[peer]:
            if name in values:
                result.values[name] = values[name]
                result.sources[name] = source
            else:
                result.sources[name] = "unavailable"
        if error:
            result.errors[peer] = error
    return result
