"""Cross-sectional feature engineering for the ML pattern model.

Design notes (what changed vs the original engine):

1. **Cross-sectional rank normalization.** Every feature is rank-transformed to
   [-1, 1] *within each date* before the model sees it. This is the standard
   trick in the empirical-asset-pricing literature (Gu/Kelly/Xiu 2020): raw
   levels like "20% annualized vol" mean different things in calm vs stormy
   regimes, but a stock's *rank among its peers today* is stationary. It makes
   the model regime-robust and the features directly comparable.

2. **Richer, still-compact feature set.** The academic 12-1 momentum and
   short-term reversal are kept, plus market-relative ("excess") momentum,
   risk-adjusted momentum, a volatility-regime ratio, and MACD histogram — all
   derivable from adjusted closes alone (no extra data dependency).

3. **A single source of truth.** Both the training panel and the live
   prediction snapshot are built from the same `compute_raw_features` +
   `cross_sectional_normalize`, so train and inference can never drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Order matters: this is the model's input contract, persisted in the artifact.
FEATURES = [
    "mom_12_1",         # 12-1 momentum (12 months back to 1 month back)
    "mom_excess",       # 12-1 momentum minus the market's (SPY) 12-1 momentum
    "mom_sharpe",       # 6-month return divided by 6-month volatility
    "st_reversal",      # last month's return (tends to mean-revert)
    "ret_3m",
    "ret_6m",
    "vol_1m",
    "vol_regime",       # 1-month vol / 6-month vol (rising-vol detector)
    "downside_vol",
    "rsi",
    "macd_hist",        # normalized MACD histogram (trend acceleration)
    "price_vs_sma50",
    "sma50_vs_sma200",
    "dist_52w_high",
    "consistency",      # share of recent months with a positive return
    "beta",
]


def compute_raw_features(closes: pd.DataFrame,
                         spy: pd.Series | None) -> dict[str, pd.DataFrame]:
    """Per-feature DataFrames (index=date, columns=ticker) of *raw* values.

    `spy` is the benchmark close series used for market-relative momentum and
    beta; if None those features fall back to neutral values.
    """
    daily = closes.pct_change()
    sma50 = closes.rolling(50).mean()
    sma200 = closes.rolling(200).mean()

    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    # normalize the histogram by price so it's comparable across tickers
    macd_hist = (macd - macd_signal) / closes

    mom_12_1 = closes.shift(21) / closes.shift(252) - 1
    ret_6m = closes.pct_change(126)
    vol_6m = daily.rolling(126).std() * np.sqrt(252)

    feats = {
        "mom_12_1": mom_12_1,
        "mom_sharpe": ret_6m / vol_6m.replace(0, np.nan),
        "st_reversal": closes.pct_change(21),
        "ret_3m": closes.pct_change(63),
        "ret_6m": ret_6m,
        "vol_1m": daily.rolling(21).std() * np.sqrt(252),
        "downside_vol": daily.clip(upper=0).rolling(63).std() * np.sqrt(252),
        "rsi": rsi / 100,
        "macd_hist": macd_hist,
        "price_vs_sma50": closes / sma50 - 1,
        "sma50_vs_sma200": sma50 / sma200 - 1,
        "dist_52w_high": closes / closes.rolling(252).max() - 1,
        "consistency": (closes.pct_change(21) > 0).rolling(231).mean(),
    }
    feats["vol_regime"] = feats["vol_1m"] / feats["downside_vol"].replace(0, np.nan)

    if spy is not None:
        spy_d = spy.pct_change()
        feats["beta"] = daily.rolling(126).cov(spy_d).div(
            spy_d.rolling(126).var(), axis=0)
        spy_mom = spy.shift(21) / spy.shift(252) - 1  # market 12-1 momentum
        feats["mom_excess"] = mom_12_1.sub(spy_mom, axis=0)
    else:
        feats["beta"] = closes * 0 + 1.0
        feats["mom_excess"] = mom_12_1

    return feats


def cross_sectional_normalize(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Rank each column across tickers and map to [-1, 1].

    `snapshot` is one date's features (index=ticker, columns=FEATURES). Ranking
    is done per-snapshot, so the transform is identical at train and inference
    time. Columns that are entirely NaN are left as NaN (dropped downstream).
    """
    ranked = snapshot.rank(pct=True)
    return ranked * 2 - 1


def feature_snapshot(feats: dict[str, pd.DataFrame], date) -> pd.DataFrame:
    """Normalized feature matrix (index=ticker) for a single date."""
    snap = pd.DataFrame({k: feats[k].loc[date] for k in FEATURES})
    return cross_sectional_normalize(snap)


def latest_snapshot(closes: pd.DataFrame,
                    spy: pd.Series | None) -> pd.DataFrame:
    """Normalized features for the most recent date — what inference predicts on."""
    feats = compute_raw_features(closes, spy)
    snap = feature_snapshot(feats, closes.index[-1])
    return snap.dropna()


def build_training_table(closes: pd.DataFrame, spy: pd.Series | None,
                         horizon: int, step: int,
                         warmup: int = 290) -> pd.DataFrame:
    """Stacked, normalized cross-sections with labels for supervised training.

    Columns: FEATURES (normalized), ``y`` (1 if the stock beat the
    cross-sectional median forward return), ``fwd_ret`` (raw forward return, for
    rank-IC scoring), and ``date``.
    """
    feats = compute_raw_features(closes, spy)
    fwd = closes.shift(-horizon) / closes - 1

    sample_dates = closes.index[warmup:-horizon:step]
    parts = []
    for d in sample_dates:
        snap = feature_snapshot(feats, d)
        fwd_d = fwd.loc[d]
        snap["fwd_ret"] = fwd_d
        snap["y"] = (fwd_d > fwd_d.median()).astype(int)
        snap["date"] = d
        parts.append(snap.dropna(subset=FEATURES + ["fwd_ret"]))
    if not parts:
        return pd.DataFrame(columns=FEATURES + ["fwd_ret", "y", "date"])
    return pd.concat(parts, ignore_index=True)
