"""Insight layer — the no-key fallback path must always return a usable, risk-first
note without any network call."""

from __future__ import annotations

from marketdesk import insight
from marketdesk.config import Settings


def _settings_without_key() -> Settings:
    # Explicit empty key forces the deterministic fallback (no OpenAI call).
    return Settings(openai_api_key="", openai_model="gpt-4o-mini")


SNAP = {
    "data_through": "2026-06-26",
    "regime": "RISK-ON",
    "vix": 18.4,
    "breadth": {"above_200dma": 65.0, "above_50dma": 64.0, "uptrend_share": 41.0,
                "advancing_today": 64.0},
    "ics": {"momentum": 0.04},
    "sectors": [
        {"Sector": "Technology", "1M %": 3.2, "Stocks": 70},
        {"Sector": "Energy", "1M %": -2.1, "Stocks": 22},
    ],
    "movers": {"gainers": [{"Ticker": "NVDA", "1D %": 4.1}],
               "losers": [{"Ticker": "XOM", "1D %": -3.0}]},
    "model_card": {"test_ic": 0.05},
}


def test_market_brief_fallback_is_self_contained():
    out = insight.market_brief(SNAP, _settings_without_key(), store=None)
    assert out["source"] == "fallback"
    assert out["model"] is None
    # All three narrative fields are populated and the note is risk-first.
    assert out["summary"]
    assert out["risks"]
    assert "RISK-ON" in out["headline"] or "RISK-ON" in out["summary"]
    assert "not financial advice" in out["disclaimer"].lower()


def test_stock_brief_fallback_grounds_in_the_row():
    row = {"score": 88.0, "signal": "BUY", "reasons": "Strong 12-month momentum.",
           "name": "Nvidia", "sector": "Technology"}
    out = insight.stock_brief("NVDA", row, _settings_without_key(), store=None,
                              data_through="2026-06-26")
    assert out["source"] == "fallback"
    assert "NVDA" in out["headline"]
    assert "88" in out["summary"]      # grounded in the provided rating
    assert out["risks"]                # always carries a downside note
