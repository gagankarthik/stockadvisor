"""Finnhub integration: real-time quotes (free tier: 60 calls/min), analyst
recommendations, company profiles, earnings calendar, and company news."""

from __future__ import annotations

from datetime import datetime

import requests

BASE = "https://finnhub.io/api/v1"


def _get(path: str, api_key: str, **params):
    if not api_key:
        return None
    try:
        r = requests.get(f"{BASE}/{path}", timeout=15,
                         params={**params, "token": api_key})
        if r.status_code != 200:
            return None
        return r.json()
    except (requests.RequestException, ValueError):
        return None


def quote(symbol: str, api_key: str) -> dict | None:
    """Real-time quote: c=current, dp=change %, h/l=day high/low, pc=prev close."""
    j = _get("quote", api_key, symbol=symbol)
    if not j or not j.get("c"):
        return None
    return {
        "price": float(j["c"]),
        "change_pct": float(j.get("dp") or 0.0),
        "day_high": float(j.get("h") or 0.0),
        "day_low": float(j.get("l") or 0.0),
        "prev_close": float(j.get("pc") or 0.0),
        "time": datetime.fromtimestamp(j["t"]).strftime("%H:%M:%S") if j.get("t") else "",
    }


def profile(symbol: str, api_key: str) -> dict | None:
    """Company profile: name, industry, market cap (USD millions), IPO date."""
    j = _get("stock/profile2", api_key, symbol=symbol)
    return j if j and j.get("name") else None


def recommendations(symbol: str, api_key: str) -> dict | None:
    """Latest month of analyst ratings: strongBuy/buy/hold/sell/strongSell counts."""
    j = _get("stock/recommendation", api_key, symbol=symbol)
    return j[0] if isinstance(j, list) and j else None


def earnings_calendar(date_from: str, date_to: str, api_key: str) -> list[dict]:
    """Upcoming earnings between two YYYY-MM-DD dates (market-wide)."""
    j = _get("calendar/earnings", api_key, **{"from": date_from, "to": date_to})
    return j.get("earningsCalendar", []) if isinstance(j, dict) else []


def company_news(symbol: str, date_from: str, date_to: str,
                 api_key: str) -> list[dict]:
    """Recent company news, normalized to title/url/provider/date."""
    j = _get("company-news", api_key, symbol=symbol,
             **{"from": date_from, "to": date_to})
    if not isinstance(j, list):
        return []
    out = []
    for it in j[:8]:
        if not it.get("headline"):
            continue
        out.append({
            "title": it["headline"],
            "url": it.get("url", ""),
            "provider": it.get("source", "Finnhub"),
            "date": datetime.fromtimestamp(it["datetime"]).strftime("%Y-%m-%d")
            if it.get("datetime") else "",
        })
    return out
