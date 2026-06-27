"""OpenAI-powered narrative layer — the "Desk Note".

Turns the quantitative snapshot into plain-English commentary that is **always
risk-first** and **grounded strictly in the numbers we pass in**. The model is
never given free rein to invent prices, tickers, or facts.

Degrades gracefully: with no OpenAI key (or any API error) it returns a
deterministic summary built from the same numbers, so the endpoints never fail
and the product works offline. Results are cached in the artifact store keyed by
the data date, so we call OpenAI at most once per ticker/market per refresh.

⚠ Educational commentary, not financial advice.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .config import Settings
from .store import ArtifactStore

# The contract the model must follow. Risk-first is deliberate: the product's
# job is to help users lose less, not to cheerlead. No price targets, no hype,
# no guarantees — and every note states what could go wrong.
_SYSTEM = (
    "You are the desk analyst for MarketDesk, a quantitative US-equity research tool. "
    "Write concise, plain-English commentary for a non-expert investor.\n"
    "Hard rules:\n"
    "1. Ground EVERY statement strictly in the numbers provided in the user message. "
    "Never invent prices, tickers, percentages, or facts that are not in the data.\n"
    "2. This is educational commentary, NOT financial advice and NOT a recommendation "
    "to buy or sell.\n"
    "3. Be risk-first. Always state what could go wrong and how a cautious investor "
    "limits downside (position sizing, diversification, not over-concentrating). "
    "Capital preservation matters more than chasing upside.\n"
    "4. No hype, no guarantees, no price targets. Prefer 'shows/suggests/has' over "
    "'will'. Hedge appropriately.\n"
    "Respond ONLY with a JSON object with keys: "
    "headline (string, <= 12 words), "
    "summary (string, 2-4 sentences of what the data says), "
    "risks (string, 1-2 sentences on the main downside and how to manage it)."
)


def _client(settings: Settings):
    """Construct an OpenAI client, or None if unavailable (no key / not installed)."""
    if not settings.openai_api_key:
        return None
    try:  # openai is optional at import time (graceful in minimal envs)
        from openai import OpenAI

        return OpenAI(api_key=settings.openai_api_key)
    except Exception:
        return None


def _complete(settings: Settings, payload: dict) -> dict | None:
    """One grounded JSON completion. Returns None on any failure (→ caller falls back)."""
    client = _client(settings)
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=400,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        out = {
            "headline": str(data.get("headline", "")).strip(),
            "summary": str(data.get("summary", "")).strip(),
            "risks": str(data.get("risks", "")).strip(),
        }
        return out if out["summary"] else None
    except Exception:
        return None


def _wrap(body: dict, used_ai: bool, settings: Settings) -> dict:
    return {
        **body,
        "source": "openai" if used_ai else "fallback",
        "model": settings.openai_model if used_ai else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Educational commentary, not financial advice.",
    }


# ---- Market desk note -------------------------------------------------------

def _market_payload(snap: dict) -> dict:
    sectors = snap.get("sectors", []) or []
    movers = snap.get("movers", {}) or {}

    def _by_1m(rows, reverse):
        return sorted(rows, key=lambda s: s.get("1M %", 0) or 0, reverse=reverse)[:3]

    return {
        "as_of": snap.get("data_through"),
        "regime": snap.get("regime"),
        "vix": snap.get("vix"),
        "breadth_pct": snap.get("breadth", {}),
        "factor_ic": snap.get("ics", {}),
        "leading_sectors": [{"sector": s.get("Sector"), "ret_1m_pct": s.get("1M %")} for s in _by_1m(sectors, True)],
        "lagging_sectors": [{"sector": s.get("Sector"), "ret_1m_pct": s.get("1M %")} for s in _by_1m(sectors, False)],
        "top_gainers": [{"ticker": m.get("Ticker"), "ret_1d_pct": m.get("1D %")} for m in movers.get("gainers", [])[:5]],
        "top_losers": [{"ticker": m.get("Ticker"), "ret_1d_pct": m.get("1D %")} for m in movers.get("losers", [])[:5]],
        "model_skill_ic": (snap.get("model_card") or {}).get("test_ic"),
    }


def _fallback_market(snap: dict) -> dict:
    regime = snap.get("regime", "NEUTRAL")
    vix = snap.get("vix")
    breadth = snap.get("breadth", {}) or {}
    above200 = breadth.get("above_200dma")
    bits = [f"The model reads the tape as {regime}."]
    if above200 is not None:
        bits.append(f"{above200:.0f}% of the universe sits above its 200-day average.")
    if vix is not None:
        bits.append(f"Volatility (VIX) is around {vix:.1f}.")
    return {
        "headline": f"{regime} tape as of {snap.get('data_through', '—')}",
        "summary": " ".join(bits),
        "risks": (
            "Breadth and volatility can turn quickly; size positions so a wrong "
            "call is survivable and avoid concentrating in a single name or sector."
        ),
    }


def market_brief(snap: dict, settings: Settings, store: ArtifactStore | None = None) -> dict:
    """A risk-first narrative of the current market snapshot (cached per date)."""
    cache_key = f"insight/market-{snap.get('data_through')}.json"
    if store is not None:
        cached = store.get_json(cache_key)
        if cached:
            return {**cached, "cached": True}

    ai = _complete(settings, _market_payload(snap))
    result = _wrap(ai or _fallback_market(snap), used_ai=ai is not None, settings=settings)
    result["cached"] = False
    if store is not None and ai is not None:  # only cache real AI output
        store.put_json(cache_key, result)
    return result


# ---- Per-stock thesis -------------------------------------------------------

def _stock_payload(ticker: str, row: dict) -> dict:
    return {
        "ticker": ticker,
        "name": row.get("name"),
        "sector": row.get("sector"),
        "rating_0_100": row.get("score"),
        "ai_odds_pct": row.get("ml_pct"),
        "confidence_pct": row.get("confidence_pct"),
        "signal": row.get("signal"),
        "lean": row.get("lean"),
        "rsi": row.get("rsi"),
        "volatility_pct": row.get("volatility_pct"),
        "momentum_12_1m_pct": row.get("ret_12_1m_pct"),
        "quant_reasons": row.get("reasons"),
    }


def _fallback_stock(ticker: str, row: dict) -> dict:
    score = row.get("score")
    sig = row.get("signal") or "—"
    bits = [f"{ticker} carries a model rating of {score:.0f}/100 with a {sig} signal."
            if isinstance(score, (int, float)) else f"{ticker} has a {sig} signal."]
    if row.get("reasons"):
        bits.append(str(row["reasons"]))
    return {
        "headline": f"{ticker}: {sig} per the model",
        "summary": " ".join(bits),
        "risks": (
            "Single-stock risk is high; the model can be wrong. Keep any position "
            "small relative to the portfolio and pair it with a downside plan."
        ),
    }


def stock_brief(ticker: str, row: dict, settings: Settings,
                store: ArtifactStore | None = None, data_through: str | None = None) -> dict:
    """A risk-first thesis for one ticker, grounded in its snapshot row (cached)."""
    cache_key = f"insight/stock-{ticker}-{data_through}.json"
    if store is not None and data_through:
        cached = store.get_json(cache_key)
        if cached:
            return {**cached, "cached": True}

    ai = _complete(settings, _stock_payload(ticker, row))
    result = _wrap(ai or _fallback_stock(ticker, row), used_ai=ai is not None, settings=settings)
    result["cached"] = False
    if store is not None and data_through and ai is not None:
        store.put_json(cache_key, result)
    return result
