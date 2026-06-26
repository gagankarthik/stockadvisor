"""Vectorized backtesting engine with realistic frictions.

Takes a target-position series (a strategy's desired exposure in ``[-1, 1]``)
and a price series, applies a one-bar execution lag (no look-ahead), charges
transaction cost + slippage on turnover, and reports the full performance suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    cost_bps: float = 5.0       # commission/fees per unit turnover, basis points
    slippage_bps: float = 2.0   # market-impact estimate per unit turnover
    periods_per_year: int = 252
    allow_short: bool = True
    rf_rate: float = 0.0        # annual risk-free, for Sharpe/Sortino


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series        # net per-period strategy returns
    positions: pd.Series
    metrics: dict[str, float]


def _max_drawdown(equity: pd.Series) -> tuple[float, int]:
    peak = equity.cummax()
    dd = equity / peak - 1
    trough = dd.idxmin()
    mdd = float(dd.min())
    # duration: longest stretch below a prior peak
    underwater = (equity < peak).astype(int)
    streak = underwater * (underwater.groupby((underwater == 0).cumsum()).cumcount() + 1)
    return mdd, int(streak.max())


def _metrics(returns: pd.Series, positions: pd.Series, cfg: BacktestConfig) -> dict[str, float]:
    r = returns.dropna()
    ppy = cfg.periods_per_year
    if r.empty or r.std(ddof=0) == 0:
        ann_ret = float((1 + r).prod() ** (ppy / max(len(r), 1)) - 1) if len(r) else 0.0
        return {"ann_return": ann_ret, "ann_vol": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "calmar": 0.0, "max_drawdown": 0.0, "max_dd_duration": 0.0,
                "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0,
                "var_95": 0.0, "cvar_95": 0.0, "turnover": 0.0, "total_return": 0.0}
    rf_per = cfg.rf_rate / ppy
    excess = r - rf_per
    ann_ret = float((1 + r).prod() ** (ppy / len(r)) - 1)
    ann_vol = float(r.std(ddof=0) * np.sqrt(ppy))
    downside = float(r[r < 0].std(ddof=0) * np.sqrt(ppy)) if (r < 0).any() else 0.0
    equity = (1 + r).cumprod()
    mdd, dd_dur = _max_drawdown(equity)
    wins, losses = r[r > 0], r[r < 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    var95 = float(np.percentile(r, 5))
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": float(excess.mean() / r.std(ddof=0) * np.sqrt(ppy)),
        "sortino": float(excess.mean() * np.sqrt(ppy) / downside) if downside else 0.0,
        "calmar": float(ann_ret / abs(mdd)) if mdd != 0 else 0.0,
        "max_drawdown": mdd,
        "max_dd_duration": float(dd_dur),
        "win_rate": float((r > 0).mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "expectancy": float(wins.mean() * (r > 0).mean() + losses.mean() * (r < 0).mean())
        if len(wins) and len(losses) else float(r.mean()),
        "var_95": var95,
        "cvar_95": float(r[r <= var95].mean()) if (r <= var95).any() else var95,
        "turnover": float(positions.diff().abs().sum()),
    }


def run_backtest(prices: pd.Series, signals: pd.Series,
                 config: BacktestConfig | None = None) -> BacktestResult:
    """Backtest a position series against a price series.

    Args:
        prices: Asset price (close) indexed by time.
        signals: Desired position in ``[-1, 1]`` aligned to ``prices``. Executed
            with a one-bar lag, so the position decided at *t* earns the return
            from *t* to *t+1*.
        config: Cost/slippage and annualization settings.

    Returns:
        :class:`BacktestResult` with the equity curve, net returns, and metrics.
    """
    cfg = config or BacktestConfig()
    px = prices.astype(float)
    pos = signals.reindex(px.index).fillna(0.0).clip(-1 if cfg.allow_short else 0, 1)
    asset_ret = px.pct_change().fillna(0.0)

    gross = pos.shift(1).fillna(0.0) * asset_ret
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (cfg.cost_bps + cfg.slippage_bps) / 1e4
    net = gross - cost

    equity = (1 + net).cumprod()
    return BacktestResult(
        equity=equity, returns=net, positions=pos,
        metrics=_metrics(net, pos, cfg),
    )
