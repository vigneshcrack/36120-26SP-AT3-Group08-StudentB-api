"""Market Risk Index API (Student B, Group 08) - 36120 Advanced Machine Learning, AT3.

Predicts the Market Risk Index (MRI) of a cryptocurrency 7 days ahead with LightGBM.
Return20 is requested from Student A's API and Volatility20 / MaxPrice20 / GapRisk from
Student C's API; daily candles come from the Kraken public API. When a peer API is
unavailable the service keeps answering, using missing values and a fallback model.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from . import market
from .config import API_VERSION, FORECAST_HORIZON, get_settings
from .errors import register_error_handlers, unsupported_horizon
from .model import MRIModel, risk_level
from .peers import gather_peer_features
from .schemas import FeaturesRequest, FeaturesResponse, HistoryRequest, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mri_api")

ENDPOINTS = ["/", "/health", "/features", "/predict", "/model-metadata", "/risk-history", "/docs"]
STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE["model"] = MRIModel()  # fail fast at start-up if the artefacts are missing
    logger.info("Model %s loaded", STATE["model"].version)
    yield
    STATE.clear()


app = FastAPI(
    title="Market Risk Index API",
    version=API_VERSION,
    description="Forecasts the Market Risk Index (0-100) of a cryptocurrency 7 days ahead (Group 08, Student B).",
    lifespan=lifespan,
)
register_error_handlers(app)


def _model() -> MRIModel:
    return STATE["model"]


def _check_horizon(value: int) -> None:
    if value != FORECAST_HORIZON:
        raise unsupported_horizon(value)


@app.get("/")
def root():
    """API metadata and self-documentation."""
    return {
        "project": "Market Risk Index API",
        "description": "Predicts the Market Risk Index (MRI) of cryptocurrencies 7 days into the future.",
        "algorithm": "LightGBM Regressor",
        "version": API_VERSION,
        "github": get_settings().github_url,
        "endpoints": ENDPOINTS,
        "group": "08",
        "student": "Student B - Vignesh Selvam (25740287)",
        "supported_tokens": _model().metadata["supported_tokens"],
        "forecast_horizon": FORECAST_HORIZON,
        "owned_features": market.OWN_FEATURES,
    }


@app.get("/health")
def health():
    """Liveness and readiness: confirms the model is loaded and which peers are configured."""
    settings = get_settings()
    return {
        "status": "ok",
        "model_loaded": "model" in STATE,
        "model_version": _model().version,
        "peers_configured": {"student_a": settings.student_a_url is not None,
                             "student_c": settings.student_c_url is not None},
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }


@app.post("/features", response_model=FeaturesResponse)
async def features(request: FeaturesRequest):
    """Engineered features owned by Student B (MA5, MA20, TrendStrength) for the latest day."""
    _check_horizon(request.forecast_horizon)
    token = market.validate_token(request.token)
    candles = await run_in_threadpool(market.get_candles, token, get_settings().market_cache_seconds)
    latest, history = market.own_features_payload(candles, request.history_days)
    return {
        "token": token,
        "as_of_date": candles.index[-1].date().isoformat(),
        "source": "kraken",
        "owner": "Student B",
        "features": latest,
        "history": history,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Forecast the Market Risk Index 7 days after the latest complete daily candle."""
    _check_horizon(request.forecast_horizon)
    token = market.validate_token(request.token)
    settings = get_settings()
    candles = await run_in_threadpool(market.get_candles, token, settings.market_cache_seconds)
    as_of = candles.index[-1]
    close = float(candles["close"].iloc[-1])

    peers = await gather_peer_features(token, settings)
    values, sources = dict(peers.values), dict(peers.sources)
    if settings.peer_fallback_mode == "local":  # documented emergency option, off by default
        local = market.feature_frame(candles).iloc[-1]
        for name, source in list(sources.items()):
            if source == "unavailable" and market.to_json_number(local[name]) is not None:
                values[name], sources[name] = float(local[name]), "local_fallback (shared package)"

    model = _model()
    inputs = model.build_inputs(values, close)
    result = model.predict(inputs)
    sources["DR"] = "derived: MaxPrice20 + Kraken close" if result.inputs["DR"] is not None else "unavailable"
    sources["MRI"] = "derived: brief formula" if result.inputs["MRI"] is not None else "unavailable"

    return {
        "prediction": round(result.value, 2),
        "forecast_horizon": FORECAST_HORIZON,
        "prediction_date": (as_of + timedelta(days=FORECAST_HORIZON)).date().isoformat(),
        "token": token,
        "as_of_date": as_of.date().isoformat(),
        "current_mri": None if result.current_mri is None else round(result.current_mri, 2),
        "risk_level": risk_level(result.value),
        "model_used": result.model_used,
        "model_version": model.version,
        "missing_features": result.missing,
        "feature_sources": sources,
        "inputs": result.inputs,
        "explanation": {
            "baseline": round(result.baseline, 4),
            "contributions": {k: round(v, 4) for k, v in result.contributions.items()},
            "method": result.method,
        },
    }


@app.get("/model-metadata")
def model_metadata():
    """Target, algorithm, horizon, features and importance - read from the training artefacts."""
    meta = _model().metadata
    return {
        "target": meta["target"],
        "target_description": meta["target_description"],
        "algorithm": meta["algorithm"],
        "forecast_horizon": meta["forecast_horizon"],
        "features": meta["features"],
        "feature_importance": meta["feature_importance"],
        "permutation_importance_test_mae_increase": meta.get("permutation_importance_test_mae_increase"),
        "classification_labels": meta.get("classification_labels"),
        "feature_sources": meta.get("feature_sources"),
        "model_version": meta["model_version"],
        "primary_model": meta.get("primary_model"),
        "fallback_model": meta.get("fallback_model"),
        "shrink": meta.get("shrink"),
        "hyperparameters": meta.get("hyperparameters"),
        "search_space": meta.get("search_space"),
        "optuna": meta.get("optuna"),
        "metrics": meta.get("metrics"),
        "degradation_test_mae": meta.get("degradation_test_mae"),
        "training_period": meta.get("training_period"),
        "test_period": meta.get("test_period"),
        "data_source": meta.get("data_source"),
        "created_at": meta.get("created_at"),
    }


@app.post("/risk-history")
async def risk_history(request: HistoryRequest):
    """Realised daily Market Risk Index and its weighted components (for charts)."""
    _check_horizon(request.forecast_horizon)
    token = market.validate_token(request.token)
    candles = await run_in_threadpool(market.get_candles, token, get_settings().market_cache_seconds)
    return {"token": token, "as_of_date": candles.index[-1].date().isoformat(),
            "history": market.risk_history(candles, request.days)}
