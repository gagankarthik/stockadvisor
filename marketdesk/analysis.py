"""Indicators, adaptive factor scoring, allocation, and portfolio risk.

Pure functions over a price/metrics frame — no I/O, no global state — so they
are trivially testable and reusable from both the refresh job and the API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .universe import ETFS

# Risk profile -> (core ETF allocation as fraction of total, base metric weights)
PROFILES = {
    "Conservative": {
        "core": {"SPY": 0.25, "SCHD": 0.20, "AGG": 0.15, "GLD": 0.10},
        "weights": {"momentum": 0.25, "trend": 0.25, "low_vol": 0.40, "rsi": 0.10},
    },
    "Balanced": {
        "core": {"SPY": 0.20, "QQQ": 0.15, "SCHD": 0.10, "GLD": 0.05},
        "weights": {"momentum": 0.40, "trend": 0.25, "low_vol": 0.25, "rsi": 0.10},
    },
    "Aggressive": {
        "core": {"QQQ": 0.20, "SPY": 0.10},
        "weights": {"momentum": 0.60, "trend": 0.25, "low_vol": 0.05, "rsi": 0.10},
    },
}


def compute_metrics(closes: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Vectorized per-ticker indicators. `info` maps Ticker -> Name/Sector/Type."""
    price = closes.iloc[-1]
    daily = closes.pct_change()
    sma50 = closes.rolling(50).mean().iloc[-1]
    sma200 = closes.rolling(200).mean().iloc[-1]

    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    df = pd.DataFrame({
        "Price": price,
        "1D %": daily.iloc[-1] * 100,
        "1M %": (price / closes.iloc[-21] - 1) * 100,
        "3M %": (price / closes.iloc[-63] - 1) * 100,
        "6M %": (price / closes.iloc[-126] - 1) * 100,
        # academic-standard momentum: 12 months back to 1 month back,
        # skipping the latest month (short-term reversal noise)
        "12-1M %": (closes.iloc[-22] / closes.iloc[-252] - 1) * 100,
        "Volatility %": daily.std() * np.sqrt(252) * 100,
        "RSI": rsi,
        "Trend": ((price > sma50).astype(int) + (sma50 > sma200).astype(int)
                  + (price > sma200).astype(int)),
        "SMA50": sma50,
        "SMA200": sma200,
    })
    df = df.join(info[["Name", "Sector", "Type"]], how="inner")
    return df.dropna(subset=["Price", "3M %", "Volatility %"])


# ---- Adaptive layer: measure which factors are currently predicting returns ----

def factor_ic(closes: pd.DataFrame, stock_tickers: list[str],
              horizon: int = 63) -> dict[str, float]:
    """Information coefficient per factor: Spearman rank correlation between the
    factor measured `horizon` days ago and the realized return since then.
    Positive IC = the factor has been working in the current market regime."""
    cols = [t for t in stock_tickers if t in closes.columns]
    past = closes[cols].iloc[:-horizon]
    if len(past) < 130:
        return {}
    fwd_ret = closes[cols].iloc[-1] / past.iloc[-1] - 1

    momentum = past.iloc[-22] / past.iloc[-min(252, len(past) - 1)] - 1  # 12-1 style
    low_vol = -past.pct_change().tail(63).std()
    trend = past.iloc[-1] / past.rolling(50).mean().iloc[-1] - 1

    ics = {}
    fwd_rank = fwd_ret.rank()
    for name, factor in {"momentum": momentum, "low_vol": low_vol,
                         "trend": trend}.items():
        ic = factor.rank().corr(fwd_rank)
        if pd.notna(ic):
            ics[name] = float(ic)
    return ics


def adaptive_weights(profile: str, ics: dict[str, float],
                     adapt: bool = True) -> dict[str, float]:
    """Tilt the profile's base factor weights toward factors with positive
    recent IC and away from negative ones, then renormalize."""
    base = dict(PROFILES[profile]["weights"])
    if not adapt or not ics:
        return base
    tilted = {}
    for factor, w in base.items():
        ic = ics.get(factor, 0.0)
        tilted[factor] = w * (1 + float(np.clip(ic, -0.5, 0.5)))
    total = sum(tilted.values())
    return {k: v / total for k, v in tilted.items()}


def score_stocks(metrics: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Composite 0-100 score for stocks (ETFs excluded)."""
    df = metrics[metrics["Type"] == "Stock"].copy()
    # 12-1 momentum weighted highest per the academic standard (Jegadeesh/
    # Fama-French); the most recent month is excluded since it tends to revert
    momentum_raw = 0.45 * df["12-1M %"] + 0.35 * df["6M %"] + 0.20 * df["3M %"]
    df["Momentum"] = momentum_raw.rank(pct=True) * 100
    df["TrendScore"] = df["Trend"] / 3 * 100
    df["LowVol"] = (-df["Volatility %"]).rank(pct=True) * 100
    df["RSIScore"] = (100 - (df["RSI"] - 55).abs() * 2.2).clip(0, 100)

    df["Score"] = (weights["momentum"] * df["Momentum"]
                   + weights["trend"] * df["TrendScore"]
                   + weights["low_vol"] * df["LowVol"]
                   + weights["rsi"] * df["RSIScore"])
    return df.sort_values("Score", ascending=False)


def build_allocation(scored: pd.DataFrame, metrics: pd.DataFrame,
                     amount: float, profile: str, n_picks: int) -> pd.DataFrame:
    """Split `amount` between the profile's core ETFs and the top-scored stocks."""
    core = PROFILES[profile]["core"]
    rows = []
    for ticker, frac in core.items():
        if ticker not in metrics.index:
            continue
        dollars = amount * frac
        price = metrics.loc[ticker, "Price"]
        rows.append({
            "Ticker": ticker, "Name": ETFS[ticker][0], "Type": "ETF (core)",
            "Sector": ETFS[ticker][1], "Allocation $": dollars,
            "Weight %": frac * 100, "Price": price, "Shares": dollars / price,
        })

    stock_budget = amount * (1 - sum(core.values()))
    picks = scored.head(n_picks)
    # risk-balanced sizing: weight by score scaled by inverse volatility, so
    # calm stocks take bigger slices and each pick contributes similar risk
    risk_w = picks["Score"] / picks["Volatility %"].clip(lower=5)
    risk_w = risk_w / risk_w.sum()
    for ticker, row in picks.iterrows():
        dollars = stock_budget * risk_w[ticker]
        rows.append({
            "Ticker": ticker, "Name": row["Name"], "Type": "Stock",
            "Sector": row["Sector"], "Allocation $": dollars,
            "Weight %": dollars / amount * 100, "Price": row["Price"],
            "Shares": dollars / row["Price"],
        })
    return pd.DataFrame(rows)


def portfolio_risk(closes: pd.DataFrame, alloc: pd.DataFrame,
                   amount: float) -> dict:
    """Risk analysis of an allocation: historical backtest of the current
    weights, volatility, worst drawdown, and a plain-dollar 'bad month' VaR."""
    tickers = [t for t in alloc["Ticker"] if t in closes.columns]
    w = (alloc.set_index("Ticker").loc[tickers, "Allocation $"] / amount)
    rets = closes[tickers].pct_change().iloc[1:].fillna(0)
    port_ret = (rets * w).sum(axis=1)

    curve = (1 + port_ret).cumprod() * amount
    drawdown = curve / curve.cummax() - 1
    ann_vol = float(port_ret.std() * np.sqrt(252))
    monthly = port_ret.rolling(21).sum().dropna()

    spy_curve = None
    if "SPY" in closes.columns:
        spy_ret = closes["SPY"].pct_change().iloc[1:].fillna(0)
        spy_curve = (1 + spy_ret).cumprod() * amount

    level = ("LOW" if ann_vol < 0.12 else
             "MEDIUM" if ann_vol < 0.20 else "HIGH")
    stocks_only = alloc[alloc["Type"] == "Stock"]
    stock_cols = [t for t in stocks_only["Ticker"] if t in rets.columns]
    avg_corr = float("nan")
    if len(stock_cols) >= 2:
        cm = rets[stock_cols].corr().values
        avg_corr = float(cm[np.triu_indices_from(cm, k=1)].mean())
    return {
        "ann_vol_pct": ann_vol * 100,
        "risk_level": level,
        "max_drawdown_pct": float(drawdown.min()) * 100,
        "max_drawdown_usd": float(drawdown.min()) * amount,
        "bad_month_usd": float(monthly.quantile(0.05)) * amount,  # 5% worst month
        "best_month_usd": float(monthly.quantile(0.95)) * amount,
        "backtest_return_pct": (float(curve.iloc[-1]) / amount - 1) * 100,
        "largest_position_pct": float(alloc["Weight %"].max()),
        "n_sectors": int(stocks_only["Sector"].nunique()),
        "avg_correlation": avg_corr,
        "etf_pct": float(alloc.loc[alloc["Type"] != "Stock", "Weight %"].sum()),
    }


# ---- Market study helpers ----

def market_breadth(closes: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, float]:
    stocks = metrics[metrics["Type"] == "Stock"]
    return {
        "above_200dma": float((stocks["Price"] > stocks["SMA200"]).mean() * 100),
        "above_50dma": float((stocks["Price"] > stocks["SMA50"]).mean() * 100),
        "advancing_today": float((stocks["1D %"] > 0).mean() * 100),
        "uptrend_share": float((stocks["Trend"] == 3).mean() * 100),
    }


def market_regime(breadth: dict[str, float]) -> str:
    """One-word read of the tape from breadth."""
    score = breadth.get("above_200dma", 50)
    if score >= 65:
        return "RISK-ON"
    if score <= 35:
        return "RISK-OFF"
    return "NEUTRAL"


def sector_performance(metrics: pd.DataFrame) -> pd.DataFrame:
    stocks = metrics[metrics["Type"] == "Stock"]
    out = stocks.groupby("Sector")[["1D %", "1M %", "3M %", "6M %"]].mean()
    out["Stocks"] = stocks.groupby("Sector").size()
    return out.sort_values("1M %", ascending=False)


def top_movers(metrics: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    stocks = metrics[metrics["Type"] == "Stock"]
    cols = ["Name", "Price", "1D %", "1M %"]
    return (stocks.nlargest(n, "1D %")[cols], stocks.nsmallest(n, "1D %")[cols])
