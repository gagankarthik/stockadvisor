"""Tests for Optuna hyperparameter optimization."""

import numpy as np
import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")

from marketdesk.quant.features import FeatureEngineer, compute_features
from marketdesk.quant.hpo import optimize
from marketdesk.quant.pipeline import PipelineConfig, run_pipeline
from marketdesk.quant.targets import build_targets


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2020-01-01", periods=760)
    ret = np.zeros(len(idx))
    for i in range(1, len(idx)):
        ret[i] = 0.12 * ret[i - 1] + rng.normal(0.0003, 0.012)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.002, len(idx)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, len(idx))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, len(idx))))
    vol = rng.integers(1_000_000, 5_000_000, len(idx)).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_optimize_returns_valid_params(ohlcv):
    feats = compute_features(ohlcv)
    y = build_targets(ohlcv)["logret_5d"]
    data = feats.join(y.rename("y")).dropna()
    X = FeatureEngineer().fit_transform(data.drop(columns="y"))
    res = optimize("hist_gbm_quantile", X, data["y"], ohlcv["close"],
                   n_trials=6, cv_splits=2, refine=True)
    assert res.model == "hist_gbm_quantile"
    assert {"max_iter", "max_depth", "learning_rate"}.issubset(res.best_params)
    assert np.isfinite(res.best_value)
    assert res.n_trials >= 1


def test_pipeline_with_optimize(ohlcv):
    cfg = PipelineConfig(target_col="logret_5d", cv_splits=2, optimize=True, hpo_trials=4)
    report = run_pipeline(ohlcv, cfg)
    assert report["best_model"] in report["ranking"]
    # the winning model should carry tuned hyperparameters
    assert isinstance(report["models"][report["best_model"]].get("params"), dict)
