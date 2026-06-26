"""Evaluation metrics: prediction accuracy + interval quality. Financial
performance is delegated to :mod:`marketdesk.quant.backtest`."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of observations where the predicted sign matches the realized sign."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def interval_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Empirical coverage of a prediction interval (should ≈ the nominal level)."""
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def regression_report(y_true: np.ndarray, y_pred: np.ndarray,
                      lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "interval_coverage": interval_coverage(y_true, lo, hi),
        "interval_width": float(np.mean(hi - lo)),
    }
