"""Phase 2 — feature engineering framework.

Two clearly separated stages:

1. :func:`compute_features` — turns an OHLCV frame into a wide, **fully causal**
   feature matrix. Every value at time *t* depends only on data ≤ *t* (rolling
   windows, shifts, EW means). This stage learns nothing, so it can run over the
   whole history before splitting.
2. :class:`FeatureEngineer` — the *learned* post-processing (winsorization
   bounds, correlation pruning, scaling). Parameters are fit on the **train**
   slice only and applied to val/test, so normalization can never leak.

Categories implemented (A–E in the brief): price, volume, technical,
statistical, microstructure — 40+ features in total.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Low-level indicators (all causal)                                           #
# --------------------------------------------------------------------------- #

def _wilder(x: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (the EW mean used by ATR/ADX/RSI)."""
    return x.ewm(alpha=1 / n, adjust=False).mean()


def _true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev = c.shift(1)
    return pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = _wilder(d.clip(lower=0), n)
    loss = _wilder((-d.clip(upper=0)), n)
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.DataFrame:
    up, down = h.diff(), -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _wilder(_true_range(h, l, c), n)
    plus_di = 100 * _wilder(pd.Series(plus_dm, index=h.index), n) / atr
    minus_di = 100 * _wilder(pd.Series(minus_dm, index=h.index), n) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"adx": _wilder(dx, n), "plus_di": plus_di, "minus_di": minus_di})


def _hurst_rs(x: np.ndarray) -> float:
    """Rescaled-range Hurst estimate. >0.5 trending, <0.5 mean-reverting."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 20:
        return np.nan
    z = np.cumsum(x - x.mean())
    r = z.max() - z.min()
    s = x.std(ddof=0)
    if s == 0 or r == 0:
        return np.nan
    return float(np.log(r / s) / np.log(n))


def _shannon_entropy(x: np.ndarray, bins: int = 12) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < bins:
        return np.nan
    hist, _ = np.histogram(x, bins=bins)
    p = hist[hist > 0] / hist.sum()
    return float(stats.entropy(p))


# --------------------------------------------------------------------------- #
# Feature assembly                                                            #
# --------------------------------------------------------------------------- #

def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build the causal feature matrix from an OHLCV frame.

    Args:
        ohlcv: DataFrame indexed by ``DatetimeIndex`` with columns
            ``open, high, low, close, volume`` (case-insensitive).

    Returns:
        Feature DataFrame aligned to ``ohlcv.index``; early rows are NaN until
        each indicator's warm-up window is satisfied.
    """
    df = ohlcv.copy()
    df.columns = [str(c).lower() for c in df.columns]
    o, h, l, c, v = (df["open"], df["high"], df["low"], df["close"], df["volume"])
    logret = np.log(c / c.shift(1))
    f: dict[str, pd.Series] = {}

    # ---- A. Price-based ----
    for hh in (1, 5, 21, 63, 126, 252):
        f[f"logret_{hh}d"] = np.log(c / c.shift(hh))
    f["roc_10"] = c / c.shift(10) - 1
    f["rs_252"] = c / c.rolling(252).mean() - 1
    smas = {n: c.rolling(n).mean() for n in (5, 10, 20, 50, 200)}
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    wma20 = c.rolling(20).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True)
    # Hull MA(20)
    half = c.rolling(10).apply(lambda x: np.dot(x, np.arange(1, 11)) / 55, raw=True)
    full = wma20
    hma_raw = 2 * half - full
    f["hma_20_dist"] = c / hma_raw.rolling(4).mean() - 1
    f["dist_sma20"] = c / smas[20] - 1
    f["dist_sma50"] = c / smas[50] - 1
    f["dist_sma200"] = c / smas[200] - 1
    f["sma10_sma50"] = smas[10] / smas[50] - 1
    f["ema12_ema26"] = ema12 / ema26 - 1
    # Bollinger position
    mb = c.rolling(20).mean()
    sd = c.rolling(20).std()
    f["bb_pos"] = (c - (mb - 2 * sd)) / (4 * sd).replace(0, np.nan)
    # volatilities
    for n in (10, 21, 63):
        f[f"vol_{n}d"] = logret.rolling(n).std() * np.sqrt(TRADING_DAYS)
    f["parkinson_21"] = np.sqrt(
        (np.log(h / l) ** 2).rolling(21).mean() / (4 * np.log(2))) * np.sqrt(TRADING_DAYS)
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    f["garman_klass_21"] = np.sqrt(gk.rolling(21).mean().clip(lower=0)) * np.sqrt(TRADING_DAYS)
    overnight = np.log(o / c.shift(1))
    oc = np.log(c / o)
    rs_vol = (np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)).rolling(21).mean()
    k = 0.34 / (1.34 + 22 / 20)
    f["yang_zhang_21"] = np.sqrt(
        (overnight.rolling(21).var() + k * oc.rolling(21).var()
         + (1 - k) * rs_vol).clip(lower=0)) * np.sqrt(TRADING_DAYS)
    f["atr_14"] = _wilder(_true_range(h, l, c), 14) / c
    f["higher_high"] = (c > c.rolling(20).max().shift(1)).astype(float)
    f["lower_low"] = (c < c.rolling(20).min().shift(1)).astype(float)
    f["consolidation"] = ((c.rolling(20).max() - c.rolling(20).min()) / c < 0.05).astype(float)

    # ---- B. Volume-based ----
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv_slope_5"] = (obv - obv.shift(5)) / v.rolling(20).mean().replace(0, np.nan)
    f["rel_volume"] = v / v.rolling(20).mean().replace(0, np.nan)
    f["vol_trend"] = v.rolling(5).mean() / v.rolling(20).mean().replace(0, np.nan) - 1
    tp = (h + l + c) / 3
    vwap20 = (tp * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    f["close_vs_vwap20"] = c / vwap20 - 1
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    f["cmf_20"] = (mfm * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    rmf = tp * v
    pos = rmf.where(tp > tp.shift(1), 0.0)
    neg = rmf.where(tp < tp.shift(1), 0.0)
    mfr = pos.rolling(14).sum() / neg.rolling(14).sum().replace(0, np.nan)
    f["mfi_14"] = 100 - 100 / (1 + mfr)
    vmean, vstd = v.rolling(20).mean(), v.rolling(20).std(ddof=0)
    f["volume_zscore"] = (v - vmean) / vstd.replace(0, np.nan)

    # ---- C. Technical ----
    f["rsi_14"] = _rsi(c, 14)
    ll14, hh14 = l.rolling(14).min(), h.rolling(14).max()
    stoch_k = 100 * (c - ll14) / (hh14 - ll14).replace(0, np.nan)
    f["stoch_k"] = stoch_k
    f["stoch_d"] = stoch_k.rolling(3).mean()
    f["williams_r"] = -100 * (hh14 - c) / (hh14 - ll14).replace(0, np.nan)
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    f["cci_20"] = (tp - tp.rolling(20).mean()) / (0.015 * mad).replace(0, np.nan)
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    f["macd"] = macd / c
    f["macd_hist"] = (macd - macd_sig) / c
    adx = _adx(h, l, c, 14)
    f["adx_14"] = adx["adx"]
    f["di_spread"] = adx["plus_di"] - adx["minus_di"]

    # ---- D. Statistical ----
    f["skew_20"] = logret.rolling(20).skew()
    f["kurt_20"] = logret.rolling(20).kurt()
    for lag in (1, 2, 3, 5):
        f[f"acf_{lag}"] = logret.rolling(63).apply(
            lambda x, k=lag: pd.Series(x).autocorr(lag=k), raw=False)
    f["hurst_252"] = logret.rolling(252).apply(_hurst_rs, raw=True)
    f["entropy_63"] = logret.rolling(63).apply(_shannon_entropy, raw=True)

    # ---- E. Microstructure ----
    f["spread"] = (h - l) / c
    f["gap"] = (o - c.shift(1)) / c.shift(1)
    f["intraday_vol"] = (h - l) / o
    f["close_location"] = (c - l) / (h - l).replace(0, np.nan)
    f["overnight_ret"] = overnight
    f["intraday_ret"] = oc

    feats = pd.DataFrame(f, index=df.index)
    return feats.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Learned post-processing                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class FeatureConfig:
    """Post-processing options learned on train, applied to all splits."""

    winsorize: bool = True
    winsor_lower: float = 0.005
    winsor_upper: float = 0.995
    drop_correlated: bool = True
    corr_threshold: float = 0.95
    scaler: str = "robust"  # "robust" | "zscore" | "minmax" | "none"


class FeatureEngineer:
    """Sklearn-style transformer that learns winsor bounds, the correlated-column
    drop set, and scaling parameters on :meth:`fit` (train) and applies them on
    :meth:`transform` (val/test) — guaranteeing no normalization leakage."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self.columns_: list[str] = []
        self.dropped_: list[str] = []
        self.bounds_: dict[str, tuple[float, float]] = {}
        self.center_: pd.Series | None = None
        self.scale_: pd.Series | None = None
        self.fitted_ = False

    def fit(self, features: pd.DataFrame) -> "FeatureEngineer":
        """Learn parameters from the training feature matrix."""
        cfg = self.config
        X = features.copy()

        if cfg.winsorize:
            for col in X.columns:
                lo = X[col].quantile(cfg.winsor_lower)
                hi = X[col].quantile(cfg.winsor_upper)
                self.bounds_[col] = (lo, hi)
                X[col] = X[col].clip(lo, hi)

        self.dropped_ = self._correlated_to_drop(X) if cfg.drop_correlated else []
        kept = [c for c in X.columns if c not in self.dropped_]
        self.columns_ = kept
        Xk = X[kept]

        # a zero-variance (constant) column scales by 1.0, not 0 — otherwise the
        # column would become NaN and poison downstream models.
        def _safe(scale: pd.Series) -> pd.Series:
            return scale.replace(0, 1.0).fillna(1.0)

        if cfg.scaler == "robust":
            self.center_ = Xk.median()
            self.scale_ = _safe(Xk.quantile(0.75) - Xk.quantile(0.25))
        elif cfg.scaler == "zscore":
            self.center_ = Xk.mean()
            self.scale_ = _safe(Xk.std(ddof=0))
        elif cfg.scaler == "minmax":
            self.center_ = Xk.min()
            self.scale_ = _safe(Xk.max() - Xk.min())
        else:
            self.center_ = pd.Series(0.0, index=kept)
            self.scale_ = pd.Series(1.0, index=kept)
        self.fitted_ = True
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned winsorization, column drop, and scaling."""
        if not self.fitted_:
            raise RuntimeError("FeatureEngineer must be fit before transform.")
        X = features.copy()
        if self.config.winsorize:
            for col, (lo, hi) in self.bounds_.items():
                if col in X.columns:
                    X[col] = X[col].clip(lo, hi)
        X = X[self.columns_]
        scaled = (X - self.center_) / self.scale_
        return scaled.replace([np.inf, -np.inf], np.nan)

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).transform(features)

    def _correlated_to_drop(self, X: pd.DataFrame) -> list[str]:
        """Greedily drop later columns whose |corr| with an earlier kept column
        exceeds the threshold."""
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        return [c for c in upper.columns if (upper[c] > self.config.corr_threshold).any()]
