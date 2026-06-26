"""Prediction targets (labels).

These intentionally look **forward** — that is what a label is. They must never
be fed back in as features. Keeping them in a separate module from
:mod:`marketdesk.quant.features` makes that boundary explicit and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class TargetConfig:
    direction_horizons: tuple[int, ...] = (1, 5, 21)
    return_horizons: tuple[int, ...] = (1, 5, 21)
    vol_horizon: int = 21
    signal_horizon: int = 5
    signal_sharpe_threshold: float = 0.10
    regime_horizon: int = 21
    regime_return_threshold: float = 0.02
    regime_vol_quantile: float = 0.75


def build_targets(ohlcv: pd.DataFrame, config: TargetConfig | None = None) -> pd.DataFrame:
    """Construct the multi-target label matrix.

    Args:
        ohlcv: OHLCV frame (only ``close`` is required).
        config: Horizon/threshold settings.

    Returns:
        DataFrame of labels aligned to ``ohlcv.index``; the final rows are NaN
        because their forward window extends past the data.
    """
    cfg = config or TargetConfig()
    c = ohlcv["close"] if "close" in ohlcv else ohlcv[[x for x in ohlcv if str(x).lower() == "close"][0]]
    logret = np.log(c / c.shift(1))
    out: dict[str, pd.Series] = {}

    # 1. direction (binary) — masked to NaN where the forward window runs off
    # the end (comparing against a NaN future would silently yield 0/False).
    for h in cfg.direction_horizons:
        fwd_c = c.shift(-h)
        d = (fwd_c > c).astype("float")
        out[f"direction_{h}d"] = d.where(fwd_c.notna())

    # 2. return magnitude (regression)
    fwd_ret: dict[int, pd.Series] = {}
    for h in cfg.return_horizons:
        r = np.log(c.shift(-h) / c)
        fwd_ret[h] = r
        out[f"logret_{h}d"] = r

    # 3. forward realized volatility (regression)
    realized = logret.rolling(cfg.vol_horizon).std() * np.sqrt(TRADING_DAYS)
    out[f"fwd_vol_{cfg.vol_horizon}d"] = realized.shift(-cfg.vol_horizon)

    # 4. risk-adjusted signal in {-1, 0, +1}
    h = cfg.signal_horizon
    fwd = np.log(c.shift(-h) / c)
    vol = logret.rolling(cfg.vol_horizon).std().replace(0, np.nan)
    sharpe = fwd / (vol * np.sqrt(h))
    out["signal"] = np.sign(sharpe).where(sharpe.abs() > cfg.signal_sharpe_threshold, 0.0)

    # 5. quantile targets (a coarse forward-return distribution)
    base = fwd_ret.get(cfg.signal_horizon, fwd)
    out["ret_q10"] = base  # the same realized value; quantile *models* learn the spread
    # (kept as a single realized target; quantile heads predict P10/P50/P90 of it)

    # 6. forward market regime (classification): bull / bear / sideways / volatile
    rr = np.log(c.shift(-cfg.regime_horizon) / c)
    fwd_vol = logret.shift(-cfg.regime_horizon).rolling(cfg.regime_horizon).std()
    hi_vol = fwd_vol > fwd_vol.quantile(cfg.regime_vol_quantile)
    regime = pd.Series("sideways", index=c.index, dtype=object)
    regime[rr > cfg.regime_return_threshold] = "bull"
    regime[rr < -cfg.regime_return_threshold] = "bear"
    regime[hi_vol] = "volatile"
    regime[rr.isna()] = np.nan
    out["regime"] = regime

    return pd.DataFrame(out, index=c.index)
