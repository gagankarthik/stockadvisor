"""Technical BUY/HOLD/SELL engine and overnight gap scanner.

Signals combine classic technical triggers into an explicit verdict per ticker,
with the reasons spelled out so the API can show *why*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .providers.prices import fetch_prices  # noqa: F401  (re-export convenience)


def compute_signals(closes: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker verdict from golden/death cross, trend stack, RSI, MACD
    crossover, and Bollinger band position."""
    sma20 = closes.rolling(20).mean()
    std20 = closes.rolling(20).std()
    upper, lower = sma20 + 2 * std20, sma20 - 2 * std20
    sma50 = closes.rolling(50).mean()
    sma200 = closes.rolling(200).mean()

    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1]

    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    hist = macd - macd.ewm(span=9, adjust=False).mean()

    price = closes.iloc[-1]
    above_now = sma50.iloc[-1] > sma200.iloc[-1]
    above_10d_ago = sma50.iloc[-11] > sma200.iloc[-11]
    golden = above_now & ~above_10d_ago
    death = ~above_now & above_10d_ago
    macd_up = (hist.iloc[-1] > 0) & (hist.iloc[-6] <= 0)
    macd_dn = (hist.iloc[-1] < 0) & (hist.iloc[-6] >= 0)
    uptrend = (price > sma50.iloc[-1]) & (sma50.iloc[-1] > sma200.iloc[-1])
    downtrend = (price < sma50.iloc[-1]) & (sma50.iloc[-1] < sma200.iloc[-1])

    rows = []
    for t in closes.columns:
        pts, why = 0, []
        checks = [
            (bool(golden.get(t, False)), 2, "golden cross (50d crossed above 200d)"),
            (bool(death.get(t, False)), -2, "death cross (50d crossed below 200d)"),
            (bool(uptrend.get(t, False)), 1, "confirmed uptrend (price > 50d > 200d)"),
            (bool(downtrend.get(t, False)), -1, "confirmed downtrend (price < 50d < 200d)"),
            (bool(rsi.get(t, 50) < 35), 1, f"oversold (RSI {rsi.get(t, 0):.0f})"),
            (bool(rsi.get(t, 50) > 70), -1, f"overbought (RSI {rsi.get(t, 0):.0f})"),
            (bool(macd_up.get(t, False)), 1, "MACD bullish crossover (last 5 days)"),
            (bool(macd_dn.get(t, False)), -1, "MACD bearish crossover (last 5 days)"),
            (bool(price.get(t, 0) < lower.iloc[-1].get(t, 0)), 1,
             "below lower Bollinger band (stretched down)"),
            (bool(price.get(t, 0) > upper.iloc[-1].get(t, np.inf)), -1,
             "above upper Bollinger band (stretched up)"),
        ]
        for cond, p, reason in checks:
            if cond:
                pts += p
                why.append(reason)
        verdict = "BUY" if pts >= 2 else "SELL" if pts <= -2 else "HOLD"
        rows.append({"Ticker": t, "Signal": verdict, "Signal Pts": pts,
                     "Reasons": "; ".join(why) if why else "no strong trigger"})
    return pd.DataFrame(rows).set_index("Ticker")


def gap_scan(tickers: list[str], threshold_pct: float = 1.5) -> pd.DataFrame:
    """Overnight gaps: today's open vs yesterday's close, market-wide.
    Also flags whether the gap has already been filled intraday."""
    import yfinance as yf

    data = yf.download(tickers, period="5d", interval="1d",
                       auto_adjust=True, progress=False, threads=True)
    if data.empty or not isinstance(data.columns, pd.MultiIndex):
        return pd.DataFrame()
    opens, closes = data["Open"], data["Close"]
    lows, highs = data["Low"], data["High"]
    if len(closes) < 2:
        return pd.DataFrame()

    prev_close = closes.iloc[-2]
    today_open = opens.iloc[-1]
    gap_pct = (today_open / prev_close - 1) * 100
    filled = pd.Series(
        np.where(gap_pct > 0, lows.iloc[-1] <= prev_close,
                 highs.iloc[-1] >= prev_close),
        index=gap_pct.index)

    df = pd.DataFrame({
        "Gap %": gap_pct,
        "Prev Close": prev_close,
        "Open": today_open,
        "Now": closes.iloc[-1],
        "Gap Filled": np.where(filled, "Yes", "No"),
    }).dropna(subset=["Gap %"])
    return df[df["Gap %"].abs() >= threshold_pct].sort_values("Gap %", ascending=False)
