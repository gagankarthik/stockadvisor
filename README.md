# MarketDesk

US market analysis engine exposed as an HTTP **API**, designed to be hosted on
**AWS Lambda**. Adaptive factor scoring, a calibrated ML pattern model,
technical signals, an allocation engine with risk analysis, multi-source quote
validation, and saved-plan performance tracking.

> Formerly a Streamlit app. The UI has been removed; all logic now lives in the
> importable `marketdesk` package and is served by a FastAPI app. The old code
> is preserved under `legacy/` for reference.

⚠ Educational tool, not financial advice. Quotes may be delayed up to 15 min.

## Architecture

Training is **decoupled** from serving — the future-proof pattern for ML on
Lambda:

```
                 (scheduled, heavy)                    (per request, fast)
 EventBridge ─▶ train.py / Lambda  ──writes──▶  S3 artifact store  ──reads──▶  API Lambda ─▶ API Gateway / Function URL
                 • fetch ~500 tickers            • model.joblib                 • FastAPI + Mangum
                 • train + calibrate model       • snapshot.json                • dashboard / screener
                 • score the universe            • closes.pkl                   • allocation / stock detail
                 • write snapshot                • model_history.json           • plans
```

- **`marketdesk/`** — Streamlit-free package: `config`, `store` (local **or** S3),
  `providers/` (yfinance, Finnhub, Alpha Vantage), `features`, `model`,
  `analysis`, `signals`, `service`, `plans`, `universe`.
- **`api/`** — FastAPI app (`main.py`), Pydantic schemas, Mangum Lambda
  `handler.py`.
- **`train.py`** — the refresh job (CLI **and** `lambda_handler`).
- **`Dockerfile` / `Dockerfile.train`** — Lambda container images.

## What's improved in the ML model (`marketdesk/model.py` + `features.py`)

The original was an RF + GradientBoosting ensemble on a single 80/20 time split.
This version is materially stronger and production-shaped:

1. **Cross-sectional rank-normalized features** — every feature is ranked to
   `[-1, 1]` *within each date*, so signals are regime-stationary and comparable
   (the standard empirical-asset-pricing transform). Train and inference share
   one code path, so they can never drift.
2. **Calibrated probabilities** — each learner is isotonically calibrated, so
   "ML %" is an honest probability, not just a ranking.
3. **Purged, embargoed walk-forward CV** — overlapping 21-day labels are handled
   with an embargo gap; the model reports the **rank IC** (ranking skill) next
   to AUC/accuracy.
4. **Recency-weighted training** — recent regimes count more (exponential decay).
5. **Three-learner ensemble** — Random Forest + Extra Trees + HistGradientBoosting.
6. **Richer feature set** — adds market-relative (excess) momentum, risk-adjusted
   momentum, a volatility-regime ratio, and a normalized MACD histogram.
7. **Self-describing artifact** — the model serializes (joblib) with a model card
   (feature contract, horizon, metrics, sklearn version) for safe loading on the
   serving side.

## Run locally

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements-dev.txt
cp .env.example .env                                # add FINNHUB_KEY / ALPHAVANTAGE_KEY

# 1) Refresh job: train the model + build the snapshot (writes to .artifacts/)
python train.py

# 2) Serve the API
uvicorn api.main:app --reload
# Interactive docs at http://127.0.0.1:8000/docs
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Snapshot availability, age, staleness, `data_through`. |
| `GET` | `/dashboard` | Breadth, regime, VIX, sector trend, top movers, ETFs, model card. |
| `GET` | `/screener` | Scored stocks. Query: `profile`, `adaptive`, `use_ml`, `sector`, `signal`, `min_score`, `search`, `limit`. |
| `GET` | `/stocks/{ticker}` | Live quote + 3-source validation, profile, analyst recs, news, price history, snapshot row. |
| `POST` | `/allocation` | Dollar/share plan across core ETFs + top stocks, with risk analysis. |
| `POST` | `/plans` / `GET` `/plans` / `DELETE` `/plans/{i}` | Save and track plans vs an SPY buy-and-hold. |
| `GET` | `/model` / `/model/history` | Current model card and the training history. |
| `POST` | `/admin/refresh` | On-demand rebuild (heavy escape hatch; prefer the scheduled job). |

## Quant research toolkit — `marketdesk/quant/`

A per-symbol OHLCV research pipeline, separate from the cross-sectional serving
path. No look-ahead anywhere (proven in tests), no hardcoded tickers/dates,
type-hinted throughout.

- **`validation`** — completeness, z-score/IQR outliers, OHLC consistency,
  volume checks, duplicates, corporate-action detection, tiered gap filling.
- **`features`** — 40+ causal features (price/volume/technical/statistical/
  microstructure) + a fit/transform `FeatureEngineer` (winsorize, correlation
  prune, scaling) that learns on train only.
- **`targets`** — direction / return / volatility / risk-adjusted signal /
  quantile / forward-regime labels.
- **`splits`** — time-series split + purged, embargoed walk-forward.
- **`models`** — Lambda-safe zoo (quantile GBM, RF, conformal ElasticNet/SVR,
  GBM classifier; XGBoost/LightGBM/CatBoost auto-register if installed). **Every
  model emits prediction intervals.**
- **`hpo`** — Optuna TPE search + median pruning, objective = purged-CV Sharpe,
  coarse→fine.
- **`backtest`** — costs + slippage, full metrics suite. **`drift`** — PSI +
  concept drift.

```bash
# Train the zoo for one symbol with hyperparameter search:
python train_quant.py --ticker AAPL --period 5y --target logret_5d --optimize --trials 40
```

## Deploy to AWS — push to `main` → GitHub Actions → SAM

CI/CD is wired: a push to `main` runs the tests, builds both Lambda **container
images**, and deploys via **AWS SAM** (`template.yaml`). No manual steps, no
SageMaker.

It provisions: an **S3 artifact bucket**, the **API Lambda** (FastAPI via Mangum,
public **Function URL**), and the **refresh Lambda** (more memory/timeout, on an
**EventBridge** weekday schedule) — both sharing the same S3 prefix.

**One-time setup** (repo → Settings → Secrets and variables → Actions):

| Kind | Name | Value |
|---|---|---|
| Variable | `AWS_REGION` | e.g. `us-east-1` |
| Variable | `CORS_ORIGIN` | your UI origin (or `*`) |
| Secret | `AWS_DEPLOY_ROLE_ARN` | IAM role ARN trusted by GitHub OIDC |
| Secret | `FINNHUB_KEY` / `ALPHAVANTAGE_KEY` / `OPENAI_API_KEY` | provider keys |

After the first deploy, grab the API URL from the workflow log (or
`sam list stack-outputs`) and set it as `NEXT_PUBLIC_API_BASE_URL` in the UI.
Deploy locally with `sam build && sam deploy` if you prefer.

## Configuration

All config is environment-driven (12-factor) via `marketdesk/config.py`. Keys:
`FINNHUB_KEY`, `ALPHAVANTAGE_KEY`, `OPENAI_API_KEY`, and `MARKETDESK_*` overrides
(`ARTIFACT_URI`, `INDICES`, `PRICE_PERIOD`, `HORIZON_DAYS`, `ML_BLEND_WEIGHT`,
`CORS_ORIGINS`, …). See `.env.example`.

## Data sources

- **Yahoo Finance** (yfinance): price history, fallback quotes.
- **Finnhub** (free tier 60/min): real-time quotes, profiles, analyst recs,
  earnings calendar, company news.
- **Alpha Vantage** (free tier ≈25/day): independent third quote source for
  cross-validation (✅ VERIFIED / 🟡 MINOR DIVERGENCE / 🔴 MISMATCH).
