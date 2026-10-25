"""Loading the trained Market Risk Index models and producing explained predictions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from mykrml_25740287.features.crypto import compute_market_risk_index

from .config import METRICS_PATH, MODEL_PATH

RISK_BANDS = [(20, "Low"), (40, "Moderate"), (60, "High"), (101, "Very high")]


def risk_level(value: float) -> str:
    """Plain-language band for a 0-100 risk score (display aid only)."""
    for upper, label in RISK_BANDS:
        if value < upper:
            return label
    return "Very high"


@dataclass
class Prediction:
    value: float
    model_used: str
    current_mri: float | None
    inputs: dict[str, float | None]
    missing: list[str]
    baseline: float
    contributions: dict[str, float]
    method: str


class MRIModel:
    """Wraps the bundle saved by the experiment notebook (primary residual + fallback level model)."""

    def __init__(self, model_path=MODEL_PATH, metrics_path=METRICS_PATH):
        bundle = joblib.load(model_path)
        self.primary = bundle["primary_model"]
        self.fallback = bundle["fallback_model"]
        self.formulation = bundle["primary_formulation"]
        self.primary_features: list[str] = list(bundle["primary_features"])
        self.fallback_features: list[str] = list(bundle["fallback_features"])
        self.shrink: float = float(bundle["shrink"]) if bundle.get("shrink") is not None else 1.0
        self.version: str = bundle.get("model_version", "1.0.0")
        self.metadata: dict = json.loads(metrics_path.read_text())

    @staticmethod
    def build_inputs(peer_values: dict[str, float], close: float) -> dict[str, float]:
        """Turn raw peer features + today's close into the model inputs (DR and MRI(t)).

        Missing peer values stay NaN, so DR and/or MRI(t) become NaN when they cannot be
        computed, exactly as during training with missing-feature augmentation.
        """
        nan = float("nan")
        row = pd.DataFrame([{
            "Return20": peer_values.get("Return20", nan),
            "Volatility20": peer_values.get("Volatility20", nan),
            "MaxPrice20": peer_values.get("MaxPrice20", nan),
            "GapRisk": peer_values.get("GapRisk", nan),
        }])
        row["DR"] = (row["MaxPrice20"] - close) / row["MaxPrice20"]
        row["MRI"] = compute_market_risk_index(row, pd.Series([close]))
        return {k: float(v) for k, v in row.iloc[0].items()}

    def predict(self, inputs: dict[str, float]) -> Prediction:
        clean = {k: (None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v))
                 for k, v in inputs.items()}
        missing = [f for f in self.primary_features if clean.get(f) is None]
        nan = float("nan")

        if not missing and self.formulation == "residual":
            X = pd.DataFrame([[clean[f] for f in self.primary_features]], columns=self.primary_features)
            change = float(self.primary.predict(X)[0])
            value = clean["MRI"] + self.shrink * change
            contrib = self.primary.predict(X, pred_contrib=True)[0]  # SHAP values + expected value
            baseline = clean["MRI"] + self.shrink * float(contrib[-1])
            contributions = {f: self.shrink * float(c) for f, c in zip(self.primary_features, contrib[:-1])}
            model_used, method = "primary", "LightGBM SHAP values of the predicted 7-day change, scaled by the shrink factor and added to today's MRI"
        else:
            X = pd.DataFrame([[clean.get(f) if clean.get(f) is not None else nan for f in self.fallback_features]],
                             columns=self.fallback_features)
            value = float(self.fallback.predict(X)[0])
            contrib = self.fallback.predict(X, pred_contrib=True)[0]
            baseline = float(contrib[-1])
            contributions = {f: float(c) for f, c in zip(self.fallback_features, contrib[:-1])}
            model_used, method = "fallback", "LightGBM SHAP values of the level model (missing inputs follow learned default branches)"

        return Prediction(
            value=float(np.clip(value, 0, 100)),
            model_used=model_used,
            current_mri=clean.get("MRI"),
            inputs={k: clean.get(k) for k in ["Return20", "Volatility20", "MaxPrice20", "GapRisk", "DR", "MRI"]},
            missing=missing,
            baseline=baseline,
            contributions=contributions,
            method=method,
        )
