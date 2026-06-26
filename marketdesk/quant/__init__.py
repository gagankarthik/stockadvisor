"""Quant research toolkit — per-symbol OHLCV time-series pipeline.

Distinct from the cross-sectional serving path in `marketdesk.features` /
`marketdesk.model` (which ranks the whole universe each day). This package is
the research framework: validate raw OHLCV, engineer a causal feature set,
build labels, split without leakage, backtest with costs, and monitor drift.

Design invariants enforced throughout:
- **No look-ahead.** Every feature at time *t* uses only data ≤ *t*. Labels are
  the only thing allowed to peek forward, and they are never fed back as inputs.
- **No hardcoded tickers or dates.** Everything operates on a passed DataFrame.
- **Reproducible.** Transforms learn their parameters on `fit` (train) and apply
  them on `transform` (val/test), so scaling/winsorization never leak.
"""

from .validation import DataValidator, ValidationConfig, ValidationReport  # noqa: F401
from .features import FeatureConfig, FeatureEngineer  # noqa: F401
from .targets import TargetConfig, build_targets  # noqa: F401
from .splits import PurgedWalkForward, time_series_split  # noqa: F401
from .backtest import BacktestConfig, BacktestResult, run_backtest  # noqa: F401
from .drift import DriftReport, concept_drift, population_stability_index  # noqa: F401
from .models import QuantModel, model_zoo, zoo_builders  # noqa: F401
from .pipeline import PipelineConfig, predict_latest, run_pipeline  # noqa: F401

# Optuna is optional; expose HPO entry points only if available.
try:  # pragma: no cover
    from .hpo import HPOResult, optimize, optimize_zoo  # noqa: F401
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "DataValidator",
    "ValidationConfig",
    "ValidationReport",
    "FeatureConfig",
    "FeatureEngineer",
    "TargetConfig",
    "build_targets",
    "PurgedWalkForward",
    "time_series_split",
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "DriftReport",
    "concept_drift",
    "population_stability_index",
    "QuantModel",
    "model_zoo",
    "zoo_builders",
    "PipelineConfig",
    "run_pipeline",
    "predict_latest",
]
