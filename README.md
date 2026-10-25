# Market Risk Index API — Group 08, Student B

**Subject:** 36120 Advanced Machine Learning Application, Spring 2026 (UTS) — AT3 Data Product with Machine Learning
**Group:** 08 · **Student B:** Vignesh Selvam (25740287)
**Live service:** _Render URL — add after deployment_ · interactive docs at `<URL>/docs`

| Group 08 member | Role | Student ID |
|---|---|---|
| Ryan Cruikshank | Student A – Asset Performance Index | _TBC_ |
| Vignesh Selvam | **Student B – Market Risk Index (this API)** | 25740287 |
| Vinnie Bui | Student C – Trend Category | _TBC_ |
| Nandeeswari Kumaravelu | Student D – Trading Opportunity Rating | _TBC_ |

## What this service does

It forecasts the **Market Risk Index (MRI)** of a cryptocurrency **7 days ahead**. MRI is a 0–100
score defined by the AT3 brief:

```
MRI = 100 × (0.40·min(Volatility20/0.10, 1) + 0.30·min(DR/0.40, 1)
            + 0.20·min(|min(Return20, 0)|/0.30, 1) + 0.10·min(GapRisk/0.10, 1)),
DR  = (MaxPrice20 − Close) / MaxPrice20
```

Supported tokens: `BTC, ETH, XRP, SOL, DOGE, XLM, TRX, XMR` (one pooled model for all eight).

### How a prediction is made

```
                       ┌──────────────────────────┐
POST /predict ───────▶ │ 1. Kraken daily candles   │  (latest complete day = as_of_date)
                       │ 2. Student A /features    │──▶ Return20
                       │ 3. Student C /features    │──▶ Volatility20, MaxPrice20, GapRisk
                       │    (parallel, timeout,    │
                       │     retry, cache)         │
                       │ 4. DR and MRI(t) computed │  (shared package mykrml_25740287)
                       │ 5. model selection        │
                       └────────────┬─────────────┘
              all inputs present?   │
            yes ────────────────────┴──────────────── no
   primary model (LightGBM, residual):          fallback model (LightGBM, level):
   MRI(t) + 0.678 × predicted 7-day change      predicts MRI(t+7) from available
                                                features; missing values follow
                                                learned default branches
```

- **Primary model** – predicts the 7-day *change* in risk and adds a shrunk version of it to
  today's MRI. On the held-out test period (Jul–Oct 2026) it achieved **MAE 3.70** vs 3.91 for the
  persistence forecast ("risk next week = risk today", 5.3 % better) and 11.29 for the mean.
- **Fallback model** – used automatically when a peer API is down (trained with
  missing-feature augmentation). Measured test MAE: 4.40 with Student A down, 9.26 with
  Student C down.
- If a peer fails, its **last successful response (≤ 24 h old)** is reused and flagged as
  `cached` in `feature_sources`, so short outages and Render cold starts do not degrade accuracy.
- Every prediction includes an **explanation**: LightGBM SHAP contributions of each input
  (`baseline + Σ contributions = prediction`).

Full experiment (EDA, feature ablation, Optuna, evaluation): group experiments repository,
`notebooks/student_b/36120_26SP_AT3_08_25740287_experiment.ipynb`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | API metadata and endpoint list |
| GET | `/health` | Health check (model loaded, peers configured) |
| POST | `/features` | Student B's engineered features: `MA5`, `MA20`, `TrendStrength` (+ optional history) |
| POST | `/predict` | 7-day-ahead MRI forecast with explanation |
| GET | `/model-metadata` | Target, algorithm, horizon, features, feature importance, metrics (read from `models/metrics.json`) |
| POST | `/risk-history` | Realised daily MRI and its weighted components (for charts) |
| GET | `/docs` | Interactive Swagger documentation |

### Examples

```bash
curl -X POST <URL>/predict -H "Content-Type: application/json" \
     -d '{"token": "BTC", "forecast_horizon": 7}'
```
```json
{
  "prediction": 13.73,
  "forecast_horizon": 7,
  "prediction_date": "2026-10-29",
  "token": "BTC",
  "as_of_date": "2026-10-22",
  "current_mri": 13.52,
  "risk_level": "Low",
  "model_used": "primary",
  "model_version": "1.0.0",
  "missing_features": [],
  "feature_sources": {"Return20": "student_a_api", "Volatility20": "student_c_api", "MaxPrice20": "student_c_api",
                      "GapRisk": "student_c_api", "DR": "derived: MaxPrice20 + Kraken close", "MRI": "derived: brief formula"},
  "inputs": {"Return20": -0.04, "Volatility20": 0.018, "MaxPrice20": 113000.0, "GapRisk": 0.00001, "DR": 0.08, "MRI": 13.52},
  "explanation": {"baseline": 13.4, "contributions": {"Return20": 0.2, "Volatility20": 0.05, "DR": 0.03, "GapRisk": 0.01, "MRI": 0.04},
                  "method": "LightGBM SHAP values ..."}
}
```
(values illustrative)

```bash
curl -X POST <URL>/features -H "Content-Type: application/json" \
     -d '{"token": "ETH", "forecast_horizon": 7, "history_days": 30}'
```

### Errors (same JSON shape everywhere)

| Situation | Status | Body |
|---|---|---|
| Unknown token | 400 | `{"error": "Unsupported cryptocurrency", "supported_tokens": [...]}` |
| `forecast_horizon` ≠ 7 | 422 | `{"error": "Unsupported forecast_horizon", "supported_horizons": [7]}` |
| Missing or invalid fields | 422 | `{"error": "Invalid request", "details": [...]}` |
| Kraken unreachable (no cached data) | 503 | `{"error": "Market data source unavailable", ...}` |
| Unknown path | 404 | `{"error": "Not Found"}` |
| Unexpected error | 500 | `{"error": "Internal server error"}` (never a stack trace) |

A peer API being down is **not** an error: the prediction is still returned (HTTP 200) with
`model_used: "fallback"` and the missing inputs listed in `missing_features`.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `STUDENT_A_URL` | – | Base URL of Student A's API (provides `Return20`) |
| `STUDENT_C_URL` | – | Base URL of Student C's API (provides `Volatility20`, `MaxPrice20`, `GapRisk`) |
| `PEER_TIMEOUT_SECONDS` | `10` | Timeout per peer call |
| `PEER_RETRIES` | `1` | Extra attempts after a failed peer call |
| `PEER_CACHE_SECONDS` | `900` | Reuse a successful peer response for this long |
| `STALE_PEER_MAX_AGE_SECONDS` | `86400` | Max age of a cached peer response used during an outage |
| `MARKET_CACHE_SECONDS` | `900` | Reuse Kraken candles for this long |
| `PEER_FALLBACK_MODE` | `missing` | `missing` (brief behaviour) or `local` (recompute unavailable peer features locally with the shared package – emergency option) |
| `GITHUB_URL` | this repo | URL shown by `GET /` |

## Run locally

Requirements: Git and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/vigneshcrack/36120-26SP-AT3-Group08-StudentB-api.git
cd 36120-26SP-AT3-Group08-StudentB-api
uv python install 3.12.13
uv sync                                  # installs pinned dependencies (incl. mykrml-25740287 from TestPyPI)
uv run uvicorn app.main:app --reload     # http://127.0.0.1:8000/docs
```
Optional: set `STUDENT_A_URL` / `STUDENT_C_URL` before starting (PowerShell:
`$env:STUDENT_A_URL="https://..."`). Without them the API still works, using the fallback model.

Without uv: `python -m venv .venv`, activate it, `pip install -r requirements.txt`, then
`uvicorn app.main:app --reload`.

## Tests

```bash
uv run pytest
```
23 tests run offline: Kraken is replaced by real candles stored in `tests/fixtures/` and the
peer APIs by deterministic stand-ins. They cover the brief's request/response contract, the
prediction against the saved model, explanation additivity, each peer being down, cached and
local fallbacks, Kraken being down, and all invalid-input cases.

## Deploy on Render

1. Push this repository to GitHub (private).
2. Render Dashboard → **New → Web Service** → connect the repository.
3. Settings:
   - **Language:** Python 3 · **Instance type:** Free
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Health check path:** `/health`
4. **Environment variables:** `PYTHON_VERSION=3.12.13`, `STUDENT_A_URL=<Student A Render URL>`,
   `STUDENT_C_URL=<Student C Render URL>`.
5. **Create Web Service**; when the build finishes, open `<URL>/docs` and run `/predict`.

(Alternatively use the included `render.yaml`: New → Blueprint.)

## Project structure

```
├── app/
│   ├── main.py      # FastAPI routes
│   ├── model.py     # loads the model bundle, primary/fallback logic, SHAP explanations
│   ├── peers.py     # async calls to Student A/C /features (timeout, retry, cache)
│   ├── market.py    # Kraken candles (cached) and Student B's features
│   ├── schemas.py   # request/response models (shown in /docs)
│   ├── errors.py    # brief-format JSON errors
│   └── config.py    # environment-variable settings and paths
├── models/
│   ├── mri_lightgbm.joblib   # primary + fallback LightGBM models and shrink factor
│   └── metrics.json          # metadata/metrics written by the experiment notebook
├── tests/                    # pytest suite + real-data fixture
├── requirements.txt          # pinned versions for Render
├── pyproject.toml / uv.lock  # uv project definition
├── render.yaml               # optional Render blueprint
└── README.md
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| First request takes 30–60 s | Render free instances sleep; call `/health` first to wake the service |
| `model_used: "fallback"` | A peer URL is not set or the peer is down; see `feature_sources`, check `/health` |
| 503 "Market data source unavailable" | Kraken unreachable or rate-limited; retry after a minute |
| Build fails finding `mykrml-25740287` | `requirements.txt` must keep the `--extra-index-url https://test.pypi.org/simple/` line |
| Start-up error loading the model | `models/mri_lightgbm.joblib` and `models/metrics.json` must be committed |

## AI usage acknowledgement

Generative AI (Claude) was used to support the design, coding, testing and documentation of this
service, in line with the subject's AI usage policy. All code was reviewed, executed and tested by
the student, who can explain and justify every design decision.
