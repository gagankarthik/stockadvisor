"""Saved allocation plans and their performance vs an S&P 500 buy-and-hold.

Persistence goes through the `ArtifactStore` (local file or S3) rather than a
hard-coded path, because Lambda's filesystem is read-only outside `/tmp`.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from .store import ArtifactStore


def load_plans(store: ArtifactStore, key: str) -> list[dict]:
    plans = store.get_json(key)
    return plans if isinstance(plans, list) else []


def save_plan(store: ArtifactStore, key: str, amount: float, profile: str,
              holdings: list[dict], spy_price: float) -> dict:
    """Append a plan. `holdings` is a list of {ticker, shares, price}."""
    plans = load_plans(store, key)
    plan = {
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "amount": float(amount),
        "profile": profile,
        "spy_price": float(spy_price),
        "holdings": [
            {"ticker": h["ticker"], "shares": float(h["shares"]),
             "price": float(h["price"])}
            for h in holdings
        ],
    }
    plans.append(plan)
    store.put_json(key, plans)
    return plan


def delete_plan(store: ArtifactStore, key: str, index: int) -> bool:
    plans = load_plans(store, key)
    if 0 <= index < len(plans):
        plans.pop(index)
        store.put_json(key, plans)
        return True
    return False


def evaluate_plans(plans: list[dict]) -> list[dict]:
    """Current value and P/L of each saved plan vs an SPY buy-and-hold of the
    same amount on the same day."""
    if not plans:
        return []
    tickers = sorted({h["ticker"] for p in plans for h in p["holdings"]} | {"SPY"})
    quotes = yf.download(tickers, period="5d", interval="1d",
                         auto_adjust=True, progress=False)["Close"].ffill().iloc[-1]

    rows = []
    for i, p in enumerate(plans):
        value = sum(h["shares"] * float(quotes.get(h["ticker"], h["price"]))
                    for h in p["holdings"])
        pnl = value - p["amount"]
        spy_value = p["amount"] / p["spy_price"] * float(quotes["SPY"])
        age_days = (datetime.utcnow()
                    - datetime.strptime(p["saved_at"], "%Y-%m-%d %H:%M")).days
        rows.append({
            "index": i,
            "saved_at": p["saved_at"],
            "days_held": age_days,
            "review": "DUE" if age_days >= 30 else f"in {30 - age_days}d",
            "profile": p["profile"],
            "invested_usd": p["amount"],
            "value_now_usd": round(value, 2),
            "pnl_usd": round(pnl, 2),
            "pnl_pct": round(pnl / p["amount"] * 100, 2),
            "spy_same_day_pct": round((spy_value - p["amount"]) / p["amount"] * 100, 2),
            "vs_market": "BEATING" if value > spy_value else "TRAILING",
        })
    return rows
