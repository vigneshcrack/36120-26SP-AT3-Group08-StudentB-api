from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mykrml_25740287.data.crypto import MarketDataError
from mykrml_25740287.features.crypto import compute_features

FIXTURE = Path(__file__).parent / "fixtures" / "kraken_btc_eth_60d.csv"


def load_candles(token: str) -> pd.DataFrame:
    """Real Kraken daily candles (last 60 days of the 23 Oct 2026 snapshot)."""
    df = pd.read_csv(FIXTURE, parse_dates=["date"])
    return df[df.token == token].drop(columns="token").set_index("date").sort_index()


def peer_values(token: str) -> dict[str, dict[str, float]]:
    """What Student A and Student C would return, computed with the same shared formulas."""
    latest = compute_features(load_candles(token)).iloc[-1]
    return {
        "student_a": {"Return20": float(latest["Return20"])},
        "student_c": {k: float(latest[k]) for k in ["Volatility20", "MaxPrice20", "GapRisk"]},
    }


@pytest.fixture
def client(monkeypatch):
    """Test client with Kraken mocked by real fixture data and peers configured but mocked."""
    monkeypatch.setenv("STUDENT_A_URL", "https://student-a.example")
    monkeypatch.setenv("STUDENT_C_URL", "https://student-c.example")
    monkeypatch.setenv("PEER_RETRIES", "0")

    from app import config, market, peers

    config.get_settings.cache_clear()
    market.clear_cache()
    peers.clear_cache()

    def fake_fetch(token, **_):
        if token not in {"BTC", "ETH"}:
            raise MarketDataError("no fixture")
        return load_candles(token)

    monkeypatch.setattr(market, "fetch_kraken_daily_ohlcv", fake_fetch)

    state = {"down": set()}

    async def fake_call_peer(_client, peer, token, _settings):
        if peer in state["down"]:
            raise RuntimeError("connection refused")
        return peer_values(token)[peer]

    monkeypatch.setattr(peers, "_call_peer", fake_call_peer)

    from app.main import app

    with TestClient(app) as test_client:
        test_client.peers_down = state["down"]
        yield test_client
    config.get_settings.cache_clear()
