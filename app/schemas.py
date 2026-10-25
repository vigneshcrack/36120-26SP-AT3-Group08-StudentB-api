"""Request and response models (they also document the API in /docs)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    token: str = Field(..., description="Cryptocurrency symbol", examples=["BTC"])
    forecast_horizon: int = Field(7, description="Days ahead to forecast; only 7 is supported", examples=[7])


class FeaturesRequest(PredictRequest):
    history_days: int = Field(
        1, ge=1, le=365, description="Number of most recent days to include in 'history'", examples=[30]
    )


class HistoryRequest(PredictRequest):
    days: int = Field(90, ge=1, le=690, description="Number of most recent days of risk history", examples=[90])


class Explanation(BaseModel):
    baseline: float = Field(..., description="Starting point of the forecast before feature contributions")
    contributions: dict[str, float] = Field(..., description="Contribution of each input (MRI points)")
    method: str


class PredictResponse(BaseModel):
    prediction: float = Field(..., description="Forecast Market Risk Index (0-100) at prediction_date")
    forecast_horizon: int
    prediction_date: str
    token: str
    as_of_date: str
    current_mri: float | None = Field(None, description="Market Risk Index on as_of_date (null if inputs missing)")
    risk_level: str
    model_used: str = Field(..., description="'primary' (all inputs) or 'fallback' (missing peer features)")
    model_version: str
    missing_features: list[str]
    feature_sources: dict[str, str]
    inputs: dict[str, float | None]
    explanation: Explanation


class FeaturesResponse(BaseModel):
    token: str
    as_of_date: str
    source: str
    owner: str
    features: dict[str, float | None]
    history: list[dict]
