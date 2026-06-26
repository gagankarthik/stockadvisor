"""API endpoint smoke tests against a synthetic snapshot (no network)."""

import io
import json

import joblib
import pytest
from fastapi.testclient import TestClient

from marketdesk import analysis, service, signals
from marketdesk.config import get_settings
from marketdesk.model import train_model
from marketdesk.store import build_store


@pytest.fixture(scope="module")
def client(synthetic_closes, stock_tickers, synthetic_info):
    settings = get_settings()
    store = build_store(settings.artifact_uri)

    model = train_model(synthetic_closes, stock_tickers, settings)
    metrics = analysis.compute_metrics(synthetic_closes, synthetic_info)
    stocks = service._component_scores(metrics)
    probs = model.predict_latest(synthetic_closes, synthetic_closes["SPY"])
    stocks["ML %"] = (probs * 100).reindex(stocks.index)
    sig = signals.compute_signals(synthetic_closes)
    stocks = stocks.join(sig[["Signal", "Signal Pts", "Reasons"]], how="left")
    breadth = analysis.market_breadth(synthetic_closes, metrics)

    snapshot = {
        "generated_at": "2025-01-01T00:00:00+00:00",
        "data_through": str(synthetic_closes.index[-1].date()),
        "indices": ["SYN"], "n_stocks": len(stock_tickers),
        "ics": analysis.factor_ic(synthetic_closes, stock_tickers),
        "breadth": breadth, "regime": analysis.market_regime(breadth), "vix": 17.0,
        "sectors": analysis.sector_performance(metrics).reset_index().to_dict("records"),
        "movers": {"gainers": [], "losers": []},
        "etfs": [{"ticker": t, "name": t, "sector": "Broad Market",
                  "price": float(metrics.loc[t, "Price"]),
                  "change_pct": float(metrics.loc[t, "1D %"])}
                 for t in ("SPY", "QQQ", "SCHD", "GLD", "AGG") if t in metrics.index],
        "stocks": stocks.reset_index(names="Ticker").to_dict("records"),
        "model_card": model.card.__dict__,
    }
    store.put_json(service.SNAPSHOT_KEY, snapshot)
    buf = io.BytesIO()
    joblib.dump(synthetic_closes, buf)
    store.put_bytes(service.CLOSES_KEY, buf.getvalue())

    import api.main as apimain
    return TestClient(apimain.app)


@pytest.mark.parametrize("path", [
    "/health", "/dashboard", "/model", "/model/history",
    "/screener", "/screener?profile=Aggressive&use_ml=false&limit=5",
])
def test_get_endpoints_ok_and_json(client, path):
    r = client.get(path)
    assert r.status_code == 200, r.text
    json.dumps(r.json())  # no NaN / Inf leaked into the payload


def test_screener_is_score_sorted(client):
    stocks = client.get("/screener?limit=20").json()["stocks"]
    scores = [s["score"] for s in stocks if s["score"] is not None]
    assert scores == sorted(scores, reverse=True)


def test_allocation_endpoint(client):
    r = client.post("/allocation",
                    json={"amount": 5000, "profile": "Balanced", "n_picks": 6})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["allocation"] and "risk" in body
    assert body["risk"]["risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_invalid_ticker_rejected(client):
    assert client.get("/stocks/!!!").status_code == 400
