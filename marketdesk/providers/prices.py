"""Price history from Yahoo Finance (yfinance).

This is the raw material for the ML model and every indicator. Kept provider-
agnostic at the call site so a different vendor (Polygon, Tiingo, an internal
warehouse) could be slotted in behind the same two functions.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

MIN_HISTORY = 260  # trading days needed for SMA200 + 12-1 momentum metrics


def fetch_prices(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    """Daily adjusted closes, columns = tickers. Drops tickers with no data."""
    data = yf.download(tickers, period=period, interval="1d",
                       auto_adjust=True, progress=False, threads=True)
    closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    if isinstance(closes, pd.Series):  # single ticker -> Series
        closes = closes.to_frame(name=tickers[0])
    closes = closes.dropna(axis=1, how="all").ffill()
    return closes.loc[:, closes.notna().sum() >= MIN_HISTORY]


def ticker_history(symbol: str, period: str = "1y",
                   interval: str = "1d") -> pd.DataFrame:
    """OHLCV history for a single symbol (used by the Stock Lab endpoint)."""
    return yf.Ticker(symbol).history(period=period, interval=interval)
