import json
from datetime import date, timedelta

import joblib
import pandas as pd
import pytest
from mykrml_25740287.features.crypto import compute_features, compute_market_risk_index

from app.config import METRICS_PATH, MODEL_PATH
from tests.conftest import load_candles

SUPPORTED = ["BTC", "ETH", "XRP", "SOL", "DOGE", "XLM", "TRX", "XMR"]


# ---------------------------------------------------------------- simple endpoints
def test_root_lists_required_endpoints(client):
    body = client.get("/").json()
    assert body["project"] == "Market Risk Index API"
    assert body["algorithm"] == "LightGBM Regressor"
    for endpoint in ["/", "/health", "/features", "/predict", "/model-metadata"]:
        assert endpoint in body["endpoints"]
    assert body["github"].startswith("https://github.com/")


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["peers_configured"] == {"student_a": True, "student_c": True}


def test_model_metadata_comes_from_training_artefacts(client):
    body = client.get("/model-metadata").json()
    saved = json.loads(METRICS_PATH.read_text())
    assert body["target"] == "Market Risk Index"
    assert body["forecast_horizon"] == 7
    assert body["features"] == saved["features"]
    assert body["feature_importance"] == saved["feature_importance"]
    assert body["metrics"] == saved["metrics"]
    assert body["classification_labels"] is None
    assert sum(body["feature_importance"].values()) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------- /features
def test_features_returns_owned_features(client):
    response = client.post("/features", json={"token": "btc", "forecast_horizon": 7})
    assert response.status_code == 200
    body = response.json()
    assert body["token"] == "BTC"
    assert set(body["features"]) == {"MA5", "MA20", "TrendStrength"}
    candles = load_candles("BTC")
    expected = compute_features(candles).iloc[-1]
    assert body["features"]["MA20"] == pytest.approx(expected["MA20"])
    assert body["as_of_date"] == candles.index[-1].date().isoformat()
    assert len(body["history"]) == 1


def test_features_history_has_close_and_nulls_for_warmup(client):
    body = client.post("/features", json={"token": "ETH", "history_days": 60}).json()
    assert len(body["history"]) == 60
    assert body["history"][0]["MA20"] is None  # not enough history yet -> null, never NaN
    assert body["history"][-1]["close"] is not None


# ---------------------------------------------------------------- /predict
def _expected_primary(token: str) -> float:
    bundle = joblib.load(MODEL_PATH)
    candles = load_candles(token)
    feats = compute_features(candles).iloc[[-1]].copy()
    close = candles["close"].iloc[[-1]].to_numpy()
    feats["DR"] = (feats["MaxPrice20"] - close) / feats["MaxPrice20"]
    feats["MRI"] = compute_market_risk_index(feats, pd.Series(close, index=feats.index))
    change = bundle["primary_model"].predict(feats[bundle["primary_features"]])[0]
    return float(feats["MRI"].iloc[0] + bundle["shrink"] * change)


def test_predict_matches_brief_contract_and_model(client):
    response = client.post("/predict", json={"token": "BTC", "forecast_horizon": 7})
    assert response.status_code == 200
    body = response.json()
    for key in ["prediction", "forecast_horizon", "prediction_date"]:
        assert key in body
    assert body["forecast_horizon"] == 7
    as_of = date.fromisoformat(body["as_of_date"])
    assert date.fromisoformat(body["prediction_date"]) == as_of + timedelta(days=7)
    assert 0 <= body["prediction"] <= 100
    assert body["model_used"] == "primary"
    assert body["missing_features"] == []
    assert body["prediction"] == pytest.approx(_expected_primary("BTC"), abs=0.01)


def test_predict_explanation_adds_up(client):
    body = client.post("/predict", json={"token": "ETH", "forecast_horizon": 7}).json()
    explanation = body["explanation"]
    total = explanation["baseline"] + sum(explanation["contributions"].values())
    assert total == pytest.approx(body["prediction"], abs=0.02)


def test_predict_uses_fallback_when_student_a_is_down(client):
    client.peers_down.add("student_a")
    body = client.post("/predict", json={"token": "BTC", "forecast_horizon": 7}).json()
    assert body["model_used"] == "fallback"
    assert "Return20" in body["missing_features"]
    assert body["feature_sources"]["Return20"] == "unavailable"
    assert body["current_mri"] is None
    assert 0 <= body["prediction"] <= 100


def test_predict_still_answers_when_all_peers_are_down(client):
    client.peers_down.update({"student_a", "student_c"})
    response = client.post("/predict", json={"token": "BTC", "forecast_horizon": 7})
    assert response.status_code == 200
    assert set(response.json()["missing_features"]) >= {"Return20", "Volatility20", "GapRisk"}


def test_predict_reuses_recent_peer_response_when_peer_goes_down(client, monkeypatch):
    monkeypatch.setenv("PEER_CACHE_SECONDS", "0")  # force a new call each time
    from app import config
    config.get_settings.cache_clear()
    client.post("/predict", json={"token": "BTC", "forecast_horizon": 7})  # fills the cache
    client.peers_down.add("student_c")
    body = client.post("/predict", json={"token": "BTC", "forecast_horizon": 7}).json()
    assert body["model_used"] == "primary"
    assert "cached" in body["feature_sources"]["Volatility20"]


def test_local_fallback_mode_recomputes_missing_peer_features(client, monkeypatch):
    monkeypatch.setenv("PEER_FALLBACK_MODE", "local")
    from app import config
    config.get_settings.cache_clear()
    client.peers_down.update({"student_a", "student_c"})
    body = client.post("/predict", json={"token": "BTC", "forecast_horizon": 7}).json()
    assert body["model_used"] == "primary"
    assert body["feature_sources"]["Return20"].startswith("local_fallback")


# ---------------------------------------------------------------- /risk-history
def test_risk_history_components_sum_to_mri(client):
    body = client.post("/risk-history", json={"token": "BTC", "days": 10}).json()
    assert len(body["history"]) == 10
    row = body["history"][-1]
    parts = row["volatility_points"] + row["drawdown_points"] + row["downside_return_points"] + row["gap_risk_points"]
    assert parts == pytest.approx(row["MRI"])


# ---------------------------------------------------------------- errors
def test_unsupported_token_uses_brief_error_format(client):
    response = client.post("/predict", json={"token": "ADA", "forecast_horizon": 7})
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "Unsupported cryptocurrency"
    assert body["supported_tokens"] == SUPPORTED


@pytest.mark.parametrize("horizon", [1, 6, 8, 30])
def test_unsupported_horizon(client, horizon):
    response = client.post("/predict", json={"token": "BTC", "forecast_horizon": horizon})
    assert response.status_code == 422
    assert response.json()["supported_horizons"] == [7]


@pytest.mark.parametrize("payload", [{}, {"forecast_horizon": 7}, {"token": "BTC", "forecast_horizon": "seven"}])
def test_invalid_request_body(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "Invalid request"
    assert "traceback" not in json.dumps(body).lower()


def test_history_days_out_of_range(client):
    response = client.post("/features", json={"token": "BTC", "history_days": 1000})
    assert response.status_code == 422


def test_market_data_unavailable_returns_503(client):
    response = client.post("/predict", json={"token": "SOL", "forecast_horizon": 7})  # no fixture
    assert response.status_code == 503
    assert response.json()["error"] == "Market data source unavailable"


def test_unknown_route_returns_json_error(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert "error" in response.json()
