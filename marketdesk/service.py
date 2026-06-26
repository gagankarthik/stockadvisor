"""Orchestration: build the daily market snapshot, and score it per request.

The expensive work (download ~500 tickers, compute indicators, run the ML
model) happens once in `build_snapshot`, invoked by the decoupled refresh job.
The result — plus the raw price matrix — is written to the artifact store. The
API then serves instantly from that snapshot and only does cheap, profile-
specific scoring (`score_snapshot`) at request time.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import analysis, signals
from .config import Settings
from .model import MarketModel, log_history, train_model
from .providers.prices import fetch_prices
from .store import ArtifactStore
from .universe import ETFS, full_info

SNAPSHOT_KEY = "snapshot.json"
CLOSES_KEY = "closes.pkl"

# Sub-score columns that don't depend on the risk profile (computed once).
_COMPONENT_COLS = ["Momentum", "TrendScore", "LowVol", "RSIScore"]


def _component_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    """Profile-independent 0-100 sub-scores for every stock."""
    df = metrics[metrics["Type"] == "Stock"].copy()
    momentum_raw = 0.45 * df["12-1M %"] + 0.35 * df["6M %"] + 0.20 * df["3M %"]
    df["Momentum"] = momentum_raw.rank(pct=True) * 100
    df["TrendScore"] = df["Trend"] / 3 * 100
    df["LowVol"] = (-df["Volatility %"]).rank(pct=True) * 100
    df["RSIScore"] = (100 - (df["RSI"] - 55).abs() * 2.2).clip(0, 100)
    return df


# ---------------------------------------------------------------------------
# Snapshot construction (refresh job)
# ---------------------------------------------------------------------------

def _vix_level() -> float | None:
    try:
        import yfinance as yf
        h = yf.Ticker("^VIX").history(period="5d")["Close"]
        return float(h.iloc[-1]) if not h.empty else None
    except Exception:
        return None


def build_snapshot(settings: Settings, store: ArtifactStore,
                   model: MarketModel | None = None,
                   persist: bool = True) -> dict:
    """Fetch data, (re)train the model if needed, score the universe, and write
    the snapshot + price matrix to the store. Returns the snapshot dict."""
    info = full_info(settings.indices)
    closes = fetch_prices(list(info.index), settings.price_period)
    metrics = analysis.compute_metrics(closes, info)
    stock_tickers = list(metrics[metrics["Type"] == "Stock"].index)
    ics = analysis.factor_ic(closes, stock_tickers)

    if model is None:
        model = MarketModel.load(store) or train_model(closes, stock_tickers, settings)
        if model is not None and persist:
            model.save(store)
            log_history(store, model.card)

    stocks = _component_scores(metrics)

    # ML probabilities from the (possibly freshly trained) model
    if model is not None:
        spy = closes["SPY"] if "SPY" in closes.columns else None
        probs = model.predict_latest(closes, spy)
        stocks["ML %"] = (probs * 100).reindex(stocks.index)
    else:
        stocks["ML %"] = np.nan

    # technical signals
    sig = signals.compute_signals(closes)
    stocks = stocks.join(sig[["Signal", "Signal Pts", "Reasons"]], how="left")

    breadth = analysis.market_breadth(closes, metrics)
    gainers, losers = analysis.top_movers(metrics)
    sectors = analysis.sector_performance(metrics)

    etf_rows = [
        {"ticker": t, "name": ETFS[t][0], "sector": ETFS[t][1],
         "price": float(metrics.loc[t, "Price"]),
         "change_pct": float(metrics.loc[t, "1D %"])}
        for t in ETFS if t in metrics.index
    ]

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_through": str(closes.index[-1].date()),
        "indices": list(settings.indices),
        "n_stocks": len(stock_tickers),
        "ics": ics,
        "breadth": breadth,
        "regime": analysis.market_regime(breadth),
        "vix": _vix_level(),
        "sectors": sectors.reset_index().to_dict(orient="records"),
        "movers": {
            "gainers": gainers.reset_index().to_dict(orient="records"),
            "losers": losers.reset_index().to_dict(orient="records"),
        },
        "etfs": etf_rows,
        "stocks": stocks.reset_index(names="Ticker").to_dict(orient="records"),
        "model_card": model.card.__dict__ if model is not None else None,
    }

    if persist:
        store.put_json(SNAPSHOT_KEY, snapshot)
        buf = io.BytesIO()
        joblib.dump(closes, buf)
        store.put_bytes(CLOSES_KEY, buf.getvalue())
    return snapshot


# ---------------------------------------------------------------------------
# Request-time scoring (API)
# ---------------------------------------------------------------------------

def stocks_frame(snapshot: dict) -> pd.DataFrame:
    """Reconstruct the per-stock DataFrame (indexed by ticker) from a snapshot."""
    df = pd.DataFrame(snapshot.get("stocks", []))
    if df.empty:
        return df
    return df.set_index("Ticker")


def score_snapshot(snapshot: dict, profile: str = "Balanced",
                   adapt: bool = True, use_ml: bool = True,
                   settings: Settings | None = None) -> pd.DataFrame:
    """Cheap, profile-specific scoring over a prebuilt snapshot.

    Combines the four factor sub-scores with adaptive weights, optionally blends
    in the ML rank, then computes a cross-engine confidence (do the factor
    score, the ML probability, and the technical signal agree?).
    """
    blend = settings.ml_blend_weight if settings else 0.35
    df = stocks_frame(snapshot).copy()
    if df.empty:
        return df

    weights = analysis.adaptive_weights(profile, snapshot.get("ics", {}), adapt)
    df["Score"] = (weights["momentum"] * df["Momentum"]
                   + weights["trend"] * df["TrendScore"]
                   + weights["low_vol"] * df["LowVol"]
                   + weights["rsi"] * df["RSIScore"])

    if use_ml and df["ML %"].notna().any():
        ml_rank = df["ML %"].rank(pct=True) * 100
        df["Score"] = (1 - blend) * df["Score"] + blend * ml_rank.fillna(df["Score"])

    # Cross-engine confidence: factor score, ML prob, technical signal (all 0..1)
    engines = pd.concat([
        df["Score"] / 100,
        df["ML %"] / 100,
        (df["Signal Pts"].fillna(0) + 4) / 8,
    ], axis=1)
    lean = engines.mean(axis=1)
    agreement = 1 - (engines.max(axis=1) - engines.min(axis=1))
    conviction = (lean - 0.5).abs() * 2
    df["Confidence %"] = (100 * (0.6 * agreement + 0.4 * conviction)).round(0)
    df["Lean"] = np.where(lean >= 0.5, "Bullish", "Bearish")
    return df.sort_values("Score", ascending=False)


def allocation_from_snapshot(snapshot: dict, closes: pd.DataFrame, amount: float,
                             profile: str, n_picks: int, adapt: bool,
                             use_ml: bool, settings: Settings) -> dict:
    """Build a dollar/share allocation plan and its risk analysis."""
    scored = score_snapshot(snapshot, profile, adapt, use_ml, settings)
    # metrics frame the allocator needs: prices for ETFs + stocks
    price_map = {r["ticker"]: r["price"] for r in snapshot.get("etfs", [])}
    metrics_rows = [{"Ticker": t, "Price": p, "Type": "ETF"}
                    for t, p in price_map.items()]
    for tkr, row in scored.iterrows():
        metrics_rows.append({"Ticker": tkr, "Price": row["Price"], "Type": "Stock"})
    metrics = pd.DataFrame(metrics_rows).set_index("Ticker")

    alloc = analysis.build_allocation(scored, metrics, amount, profile, n_picks)
    risk = analysis.portfolio_risk(closes, alloc, amount)
    spy_price = float(closes["SPY"].iloc[-1]) if "SPY" in closes.columns else None
    return {"allocation": alloc.to_dict(orient="records"), "risk": risk,
            "spy_price": spy_price}


def load_closes(store: ArtifactStore) -> pd.DataFrame | None:
    raw = store.get_bytes(CLOSES_KEY)
    if raw is None:
        return None
    return joblib.load(io.BytesIO(raw))
