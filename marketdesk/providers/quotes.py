"""Live quotes and cross-source price triangulation.

Quotes are gathered from up to three independent vendors (Finnhub real-time,
Yahoo Finance, Alpha Vantage) and compared, producing the VERIFIED / MINOR
DIVERGENCE / MISMATCH verdict the old Streamlit badge showed — now returned as
plain data the API serializes.
"""

from __future__ import annotations

import yfinance as yf

from . import alphavantage, finnhub


def live_quote(symbol: str, finnhub_key: str) -> tuple[float, float] | None:
    """(price, change_pct) from Finnhub, falling back to Yahoo. None if both fail."""
    q = finnhub.quote(symbol, finnhub_key)
    if q:
        return q["price"], q["change_pct"]
    try:
        fi = yf.Ticker(symbol).fast_info
        last = getattr(fi, "last_price", None)
        prev = getattr(fi, "previous_close", None)
        if last and prev:
            return float(last), (float(last) / float(prev) - 1) * 100
    except Exception:
        return None
    return None


def live_quotes(symbols: list[str], finnhub_key: str) -> dict[str, tuple[float, float]]:
    """Batch of (price, change_pct) keyed by symbol; missing symbols omitted."""
    out: dict[str, tuple[float, float]] = {}
    for s in symbols:
        q = live_quote(s, finnhub_key)
        if q is not None:
            out[s] = q
    return out


def triangulate(symbol: str, yahoo_price: float | None,
                finnhub_key: str, av_key: str) -> dict:
    """Compare a quote across every available source and classify agreement.

    Returns a JSON-friendly dict: status, icon, spread %, detail, and the
    per-source prices that fed the verdict.
    """
    sources: dict[str, float] = {}
    if yahoo_price:
        sources["Yahoo"] = float(yahoo_price)

    fq = finnhub.quote(symbol, finnhub_key)
    if fq:
        sources["Finnhub (real-time)"] = fq["price"]

    av = alphavantage.global_quote(symbol, av_key)
    if av and not av.get("rate_limited"):
        sources["AlphaVantage"] = av["price"]

    if len(sources) < 2:
        return {
            "status": "INSUFFICIENT", "icon": "⚪",
            "detail": "only one source available (API quota reached or offline)",
            "sources": sources,
        }

    spread = (max(sources.values()) / min(sources.values()) - 1) * 100
    if spread < 0.5:
        icon, status, detail = "✅", "VERIFIED", f"{len(sources)} sources agree"
    elif spread < 2.5:
        icon, status, detail = "🟡", "MINOR DIVERGENCE", "one source may be delayed"
    else:
        icon, status, detail = "🔴", "MISMATCH", "large gap — treat quote with caution"
    return {"status": status, "icon": icon, "spread_pct": round(spread, 3),
            "detail": detail, "sources": sources}
