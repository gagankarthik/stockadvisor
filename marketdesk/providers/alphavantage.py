"""Alpha Vantage integration: an independent quote source used to
cross-validate Yahoo Finance / Finnhub. Free tier is ~25 requests/day, so
calls should be used sparingly and cached aggressively at the API layer."""

from __future__ import annotations

import re

import requests

BASE = "https://www.alphavantage.co/query"
TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")


def valid_symbol(sym: str) -> bool:
    """Input validation for ticker symbols before any API call."""
    return bool(sym) and bool(TICKER_RE.match(sym))


def global_quote(symbol: str, api_key: str) -> dict | None:
    """Latest quote from Alpha Vantage. Returns None on error;
    {'rate_limited': True} when the daily free quota is exhausted."""
    if not valid_symbol(symbol) or not api_key:
        return None
    try:
        r = requests.get(BASE, timeout=15, params={
            "function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key})
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return None
    if "Note" in data or "Information" in data:  # daily quota message
        return {"rate_limited": True}
    q = data.get("Global Quote") or {}
    if not q.get("05. price"):
        return None
    return {
        "price": float(q["05. price"]),
        "change_pct": float(q["10. change percent"].rstrip("%")),
        "prev_close": float(q["08. previous close"]),
        "trading_day": q.get("07. latest trading day", ""),
        "volume": int(float(q.get("06. volume", 0))),
    }
