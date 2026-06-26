"""FastAPI application factory and routes.

Serves the precomputed daily snapshot from the artifact store for the heavy
endpoints (dashboard, screener, allocation) and hits providers live for
single-ticker detail. Designed to run identically under uvicorn locally and
behind API Gateway + Lambda via `api.handler`.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from marketdesk import __version__
from marketdesk.config import Settings, get_settings
from marketdesk.model import HISTORY_KEY, load_history
from marketdesk.providers import alphavantage, finnhub, prices, quotes
from marketdesk.service import (SNAPSHOT_KEY, allocation_from_snapshot,
                                build_snapshot, load_closes, score_snapshot)
from marketdesk.store import ArtifactStore, build_store
from marketdesk import plans as plans_mod

from . import schemas

settings: Settings = get_settings()
store: ArtifactStore = build_store(settings.artifact_uri)


# ---- helpers ----------------------------------------------------------------

def _clean(obj):
    """Recursively replace NaN/Inf with None so the payload is valid JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _require_snapshot() -> dict:
    snap = store.get_json(SNAPSHOT_KEY)
    if not snap:
        raise HTTPException(503, "No market snapshot available yet. Run the "
                                 "refresh job (train.py) to generate one.")
    return snap


def _row_to_stock(ticker: str, row: dict) -> dict:
    return _clean({
        "ticker": ticker,
        "name": row.get("Name"),
        "sector": row.get("Sector"),
        "price": row.get("Price"),
        "score": row.get("Score"),
        "ml_pct": row.get("ML %"),
        "confidence_pct": row.get("Confidence %"),
        "lean": row.get("Lean"),
        "signal": row.get("Signal"),
        "rsi": row.get("RSI"),
        "volatility_pct": row.get("Volatility %"),
        "ret_1m_pct": row.get("1M %"),
        "ret_12_1m_pct": row.get("12-1M %"),
        "reasons": row.get("Reasons"),
    })


# ---- app --------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.api_title, version=__version__,
        description="US market analysis: adaptive scoring, an ML pattern model, "
                    "technical signals, allocation, and risk. Not financial advice.",
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=list(settings.cors_origins),
        allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/", response_model=schemas.Health, tags=["meta"])
    @app.get("/health", response_model=schemas.Health, tags=["meta"])
    def health():
        age = None
        stale = False
        data_through = None
        mtime = store.last_modified(SNAPSHOT_KEY)
        available = mtime is not None
        if available:
            age = time.time() - mtime
            stale = age > settings.snapshot_max_age_seconds
            snap = store.get_json(SNAPSHOT_KEY) or {}
            data_through = snap.get("data_through")
        return schemas.Health(
            status="ok", version=__version__, snapshot_available=available,
            snapshot_age_seconds=age, snapshot_stale=stale,
            data_through=data_through,
        )

    @app.get("/dashboard", response_model=schemas.DashboardOut, tags=["market"])
    def dashboard():
        snap = _require_snapshot()
        card = snap.get("model_card") or None
        return _clean({
            "generated_at": snap["generated_at"],
            "data_through": snap["data_through"],
            "regime": snap.get("regime", "NEUTRAL"),
            "vix": snap.get("vix"),
            "breadth": snap.get("breadth", {}),
            "sectors": snap.get("sectors", []),
            "movers": snap.get("movers", {"gainers": [], "losers": []}),
            "etfs": snap.get("etfs", []),
            "ics": snap.get("ics", {}),
            "model": card,
        })

    @app.get("/screener", response_model=schemas.ScreenerOut, tags=["market"])
    def screener(
        profile: str = Query("Balanced", pattern="^(Conservative|Balanced|Aggressive)$"),
        adaptive: bool = True,
        use_ml: bool = True,
        sector: str | None = None,
        min_score: float = 0.0,
        signal: str | None = Query(None, pattern="^(BUY|HOLD|SELL)$"),
        limit: int = Query(100, ge=1, le=600),
        search: str | None = None,
    ):
        snap = _require_snapshot()
        scored = score_snapshot(snap, profile, adaptive, use_ml, settings)
        rows = []
        for ticker, row in scored.iterrows():
            d = row.to_dict()
            if sector and str(d.get("Sector")) != sector:
                continue
            if signal and d.get("Signal") != signal:
                continue
            if (d.get("Score") or 0) < min_score:
                continue
            if search:
                hay = f"{ticker} {d.get('Name', '')}".lower()
                if search.lower() not in hay:
                    continue
            rows.append(_row_to_stock(str(ticker), d))
            if len(rows) >= limit:
                break
        return _clean({
            "profile": profile, "adaptive": adaptive, "use_ml": use_ml,
            "count": len(rows), "data_through": snap["data_through"],
            "stocks": rows,
        })

    @app.get("/stocks/{ticker}", tags=["market"])
    def stock_detail(ticker: str, period: str = "1y", interval: str = "1d"):
        ticker = ticker.upper().strip()
        if not alphavantage.valid_symbol(ticker):
            raise HTTPException(400, "Invalid ticker symbol.")

        snap = store.get_json(SNAPSHOT_KEY) or {}
        scored = score_snapshot(snap, settings=settings) if snap.get("stocks") else None
        snap_row = None
        if scored is not None and ticker in scored.index:
            snap_row = _row_to_stock(ticker, scored.loc[ticker].to_dict())

        live = quotes.live_quote(ticker, settings.finnhub_key)
        valid = quotes.triangulate(
            ticker, live[0] if live else None,
            settings.finnhub_key, settings.alphavantage_key)

        today = datetime.now(timezone.utc).date()
        try:
            hist = prices.ticker_history(ticker, period, interval)
            history = [
                {"date": str(idx.date()), "open": float(r["Open"]),
                 "high": float(r["High"]), "low": float(r["Low"]),
                 "close": float(r["Close"]), "volume": int(r.get("Volume", 0) or 0)}
                for idx, r in hist.tail(400).iterrows()
            ]
        except Exception:
            history = []

        return _clean({
            "ticker": ticker,
            "quote": {"price": live[0], "change_pct": live[1]} if live else None,
            "validation": valid,
            "profile": finnhub.profile(ticker, settings.finnhub_key),
            "recommendations": finnhub.recommendations(ticker, settings.finnhub_key),
            "news": finnhub.company_news(
                ticker, str(today - timedelta(days=14)), str(today),
                settings.finnhub_key),
            "snapshot": snap_row,
            "history": history,
        })

    @app.post("/allocation", response_model=schemas.AllocationOut, tags=["advice"])
    def allocation(req: schemas.AllocationRequest):
        snap = _require_snapshot()
        closes = load_closes(store)
        if closes is None:
            raise HTTPException(503, "Price history unavailable; run the refresh job.")
        result = allocation_from_snapshot(
            snap, closes, req.amount, req.profile, req.n_picks,
            req.adaptive, req.use_ml, settings)
        return _clean({"amount": req.amount, "profile": req.profile, **result})

    @app.get("/model", response_model=schemas.ModelCardOut, tags=["model"])
    def model_card():
        snap = store.get_json(SNAPSHOT_KEY) or {}
        card = snap.get("model_card")
        if not card:
            raise HTTPException(404, "No trained model yet.")
        return _clean(card)

    @app.get("/model/history", tags=["model"])
    def model_history():
        return _clean(load_history(store))

    @app.get("/plans", tags=["advice"])
    def list_plans():
        saved = plans_mod.load_plans(store, settings.plans_key)
        return _clean({"plans": plans_mod.evaluate_plans(saved)})

    @app.post("/plans", tags=["advice"])
    def save_plan(req: schemas.SavePlanRequest):
        snap = _require_snapshot()
        closes = load_closes(store)
        if closes is None:
            raise HTTPException(503, "Price history unavailable; run the refresh job.")
        result = allocation_from_snapshot(
            snap, closes, req.amount, req.profile, req.n_picks,
            req.adaptive, req.use_ml, settings)
        holdings = [{"ticker": r["Ticker"], "shares": r["Shares"], "price": r["Price"]}
                    for r in result["allocation"]]
        plan = plans_mod.save_plan(
            store, settings.plans_key, req.amount, req.profile, holdings,
            result["spy_price"] or 0.0)
        return _clean({"saved": plan})

    @app.delete("/plans/{index}", tags=["advice"])
    def delete_plan(index: int):
        ok = plans_mod.delete_plan(store, settings.plans_key, index)
        if not ok:
            raise HTTPException(404, "Plan not found.")
        return {"deleted": index}

    @app.post("/admin/refresh", tags=["meta"])
    def refresh():
        """Rebuild the snapshot on demand. Heavy (downloads the universe); the
        scheduled refresh job is the normal path — this is an escape hatch."""
        snap = build_snapshot(settings, store)
        return {"status": "refreshed", "data_through": snap["data_through"],
                "n_stocks": snap["n_stocks"]}

    return app


app = create_app()
