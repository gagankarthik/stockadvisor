"""Phase 1 — data validation & cleaning.

Runs a fixed sequence of checks over a raw OHLCV frame and returns a structured
report plus a cleaned frame. Nothing is silently dropped: problems are flagged
in boolean mask columns so a downstream model can decide how to treat them.

Expected input: a DataFrame indexed by a `DatetimeIndex`, with columns
``open, high, low, close, volume`` (case-insensitive). Frequency-agnostic — the
caller declares the expected cadence so the same code serves daily or intraday.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


@dataclass
class ValidationConfig:
    """Tunable thresholds for the validation pipeline."""

    expected_freq: str = "B"  # pandas offset alias for completeness (B = business day)
    outlier_method: str = "zscore"  # "zscore" or "iqr"
    zscore_threshold: float = 3.0
    iqr_k: float = 1.5
    # corporate-action detector: an unflagged close-to-close jump beyond this is
    # treated as a candidate split/dividend that the adjustment should explain.
    corp_action_return: float = 0.20
    common_split_ratios: tuple[float, ...] = (2.0, 3.0, 1.5, 4.0, 0.5, 1 / 3)
    split_ratio_tol: float = 0.03
    # gap-fill tiers, expressed as wall-clock durations (work for any cadence)
    small_gap: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(minutes=5))
    large_gap: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(days=1))


@dataclass
class ValidationReport:
    """Outcome of a validation run. `clean` is the repaired frame."""

    n_rows: int
    missing_timestamps: pd.DatetimeIndex
    n_duplicates: int
    ohlc_violations: int
    negative_volume: int
    zero_volume: int
    price_outliers: int
    volume_outliers: int
    corp_action_candidates: pd.DatetimeIndex
    clean: pd.DataFrame
    flags: pd.DataFrame  # boolean columns, aligned to `clean`

    def summary(self) -> dict[str, object]:
        """JSON-friendly digest for logging/monitoring."""
        return {
            "n_rows": self.n_rows,
            "missing_timestamps": len(self.missing_timestamps),
            "duplicates": self.n_duplicates,
            "ohlc_violations": self.ohlc_violations,
            "negative_volume": self.negative_volume,
            "zero_volume": self.zero_volume,
            "price_outliers": self.price_outliers,
            "volume_outliers": self.volume_outliers,
            "corp_action_candidates": len(self.corp_action_candidates),
        }


class DataValidator:
    """Stateless validator: construct with a config, call :meth:`validate`."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or ValidationConfig()

    # ---- public API ----
    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Run every check in order and return a report with a cleaned frame.

        Args:
            df: Raw OHLCV frame indexed by a timezone-naive or -aware
                ``DatetimeIndex``.

        Returns:
            A :class:`ValidationReport`.
        """
        data = self._normalize(df)
        missing = self._completeness(data)
        data, n_dupes = self._dedupe(data)
        # build the flag frame only after de-duplication so the index is unique
        flags = pd.DataFrame(index=data.index)

        ohlc_mask = self._ohlc_consistency(data)
        neg_vol = data["volume"] < 0
        zero_vol = data["volume"] == 0
        price_out = self._outliers(data["close"].pct_change())
        vol_out = self._outliers(data["volume"].astype(float))
        corp = self._corporate_actions(data)

        flags["ohlc_violation"] = ohlc_mask
        flags["negative_volume"] = neg_vol
        flags["zero_volume"] = zero_vol  # often a trading halt
        flags["price_outlier"] = price_out
        flags["volume_outlier"] = vol_out
        flags["corp_action_candidate"] = data.index.isin(corp)

        return ValidationReport(
            n_rows=len(data),
            missing_timestamps=missing,
            n_duplicates=n_dupes,
            ohlc_violations=int(ohlc_mask.sum()),
            negative_volume=int(neg_vol.sum()),
            zero_volume=int(zero_vol.sum()),
            price_outliers=int(price_out.sum()),
            volume_outliers=int(vol_out.sum()),
            corp_action_candidates=corp,
            clean=data,
            flags=flags.fillna(False),
        )

    def fill_gaps(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Tiered gap filling for a single series on a regular grid.

        - gaps shorter than ``small_gap``: linear interpolation
        - gaps up to ``large_gap``: forward fill, flagged as decayed/synthetic
        - gaps longer than ``large_gap``: left as NaN for the model to handle

        Args:
            series: Values on a complete (reindexed) time grid, with NaNs at the
                missing points.

        Returns:
            ``(filled, synthetic_flag)`` where ``synthetic_flag`` marks
            forward-filled (medium-gap) points.
        """
        cfg = self.config
        filled = series.copy()
        synthetic = pd.Series(False, index=series.index)
        na = series.isna()
        if not na.any():
            return filled, synthetic

        interp = series.interpolate(method="time", limit_area="inside")
        ffilled = series.ffill()
        run_id = (na != na.shift()).cumsum()
        for _, block in series.index.to_series().groupby(run_id):
            block = block.index
            if not bool(na.loc[block].all()):
                continue  # this run isn't a NaN gap
            duration = block[-1] - block[0]
            if duration < cfg.small_gap:
                filled.loc[block] = interp.loc[block]          # small: interpolate
            elif duration <= cfg.large_gap:
                filled.loc[block] = ffilled.loc[block]         # medium: forward-fill
                synthetic.loc[block] = True
            # large: leave NaN for the model to handle
        return filled, synthetic

    # ---- individual checks ----
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lower-case columns, coerce to a sorted DatetimeIndex, keep OHLCV."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataValidator requires a DatetimeIndex.")
        out = df.copy()
        out.columns = [str(c).lower() for c in out.columns]
        missing_cols = [c for c in OHLCV if c not in out.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        out = out[OHLCV].sort_index()
        return out

    def _completeness(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """Timestamps expected on the declared cadence but absent from the data."""
        try:
            full = pd.date_range(df.index.min(), df.index.max(),
                                 freq=self.config.expected_freq)
        except (ValueError, TypeError):
            return pd.DatetimeIndex([])
        return full.difference(df.index)

    def _dedupe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Drop exact duplicate (timestamp + identical row) records, keep first."""
        dup_index = df.index.duplicated(keep="first")
        dup_rows = df.duplicated(keep="first")
        drop = dup_index & dup_rows.to_numpy()
        n = int(drop.sum())
        # also collapse any remaining duplicate timestamps to the last obs
        out = df[~drop]
        out = out[~out.index.duplicated(keep="last")]
        return out, n

    def _ohlc_consistency(self, df: pd.DataFrame) -> pd.Series:
        """Mask of bars where high/low bounds are violated."""
        hi_ok = df["high"] >= df[["open", "close", "low"]].max(axis=1)
        lo_ok = df["low"] <= df[["open", "close", "high"]].min(axis=1)
        return ~(hi_ok & lo_ok)

    def _outliers(self, x: pd.Series) -> pd.Series:
        """Boolean mask of outliers via z-score or IQR (no look-ahead: uses the
        full series only for *flagging*, never for feature values)."""
        x = x.astype(float)
        if self.config.outlier_method == "iqr":
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - q1
            lo = q1 - self.config.iqr_k * iqr
            hi = q3 + self.config.iqr_k * iqr
            return (x < lo) | (x > hi)
        std = x.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(False, index=x.index)
        z = (x - x.mean()) / std
        return z.abs() > self.config.zscore_threshold

    def _corporate_actions(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """Flag close-to-close jumps consistent with an unadjusted split/dividend.

        A jump whose *ratio* is near a common split ratio (or simply exceeds
        ``corp_action_return``) is surfaced for review. This is a detector, not
        an adjuster — adjusted feeds should produce no candidates.
        """
        ratio = df["close"] / df["close"].shift(1)
        ret = ratio - 1
        big = ret.abs() > self.config.corp_action_return
        near_split = pd.Series(False, index=df.index)
        for r in self.config.common_split_ratios:
            near_split |= (ratio - r).abs() < self.config.split_ratio_tol
        return df.index[big | near_split]
