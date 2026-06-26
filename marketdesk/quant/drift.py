"""Drift monitoring: feature drift (PSI), concept drift (error growth), and
data-range checks. Pure functions so they can run in a scheduled job or behind
a monitoring endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def population_stability_index(
    expected: pd.Series, actual: pd.Series, bins: int = 10,
) -> float:
    """PSI between a reference (training) distribution and a current one.

    Bins are quantile edges from ``expected``. Conventional reading:
    ``<0.1`` stable, ``0.1–0.25`` moderate drift, ``>=0.25`` significant.

    Args:
        expected: Reference sample.
        actual: Current sample.
        bins: Number of quantile buckets.

    Returns:
        The PSI value (0 = identical).
    """
    e = expected.dropna().to_numpy()
    a = actual.dropna().to_numpy()
    if len(e) < bins or len(a) == 0:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(e, edges)[0] / len(e)
    a_pct = np.histogram(a, edges)[0] / len(a)
    eps = 1e-6
    e_pct = np.clip(e_pct, eps, None)
    a_pct = np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def _severity(psi: float) -> str:
    if np.isnan(psi):
        return "unknown"
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


@dataclass
class DriftReport:
    feature_psi: dict[str, float]
    drifted_features: list[str]
    max_psi: float
    severity: str

    def needs_retrain(self) -> bool:
        return self.severity == "significant"


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  bins: int = 10) -> DriftReport:
    """PSI per shared column between a reference and current feature frame."""
    cols = [c for c in reference.columns if c in current.columns]
    psi = {c: population_stability_index(reference[c], current[c], bins) for c in cols}
    drifted = [c for c, v in psi.items() if not np.isnan(v) and v >= 0.25]
    valid = [v for v in psi.values() if not np.isnan(v)]
    mx = max(valid) if valid else float("nan")
    return DriftReport(psi, drifted, mx, _severity(mx))


def concept_drift(train_rmse: float, recent_errors: pd.Series,
                  window: int = 30) -> dict[str, object]:
    """Compare recent prediction error to the training baseline.

    Args:
        train_rmse: RMSE achieved on the training/validation set.
        recent_errors: Series of recent residuals (pred − actual).
        window: Trailing window for the rolling RMSE.

    Returns:
        Dict with the rolling RMSE, its ratio to baseline, and a status of
        ``ok`` / ``warning`` (>2×) / ``critical`` (>3×).
    """
    if train_rmse <= 0 or recent_errors.empty:
        return {"rolling_rmse": float("nan"), "ratio": float("nan"), "status": "unknown"}
    rolling = float(np.sqrt((recent_errors.tail(window) ** 2).mean()))
    ratio = rolling / train_rmse
    status = "critical" if ratio > 3 else "warning" if ratio > 2 else "ok"
    return {"rolling_rmse": rolling, "ratio": ratio, "status": status}
