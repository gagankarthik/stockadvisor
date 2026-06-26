"""Tests for the model zoo and the purged-CV training pipeline."""

import io

import joblib
import numpy as np
import pandas as pd
import pytest

from marketdesk.quant.features import FeatureEngineer, compute_features
from marketdesk.quant.targets import build_targets
from marketdesk.quant.models import model_zoo
from marketdesk.quant.pipeline import PipelineConfig, predict_latest, run_pipeline
from marketdesk.store import build_store


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=820)
    # a mild momentum signal so models have something learnable
    ret = np.zeros(len(idx))
    for i in range(1, len(idx)):
        ret[i] = 0.15 * ret[i - 1] + rng.normal(0.0003, 0.012)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.002, len(idx)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, len(idx))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, len(idx))))
    vol = rng.integers(1_000_000, 5_000_000, len(idx)).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


@pytest.fixture(scope="module")
def dataset(ohlcv):
    feats = compute_features(ohlcv)
    y = build_targets(ohlcv)["logret_5d"]
    data = feats.join(y.rename("y")).dropna()
    X = FeatureEngineer().fit_transform(data.drop(columns="y"))
    return X.iloc[:400], data["y"].iloc[:400], X.iloc[400:], data["y"].iloc[400:]


def test_every_model_emits_valid_intervals(dataset):
    X_tr, y_tr, X_te, y_te = dataset
    zoo = model_zoo(alpha=0.1)
    assert len(zoo) >= 3
    for name, model in zoo.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        lo, hi = model.predict_interval(X_te)
        assert pred.shape == (len(X_te),)
        assert np.all(hi >= lo), f"{name} produced inverted intervals"
        coverage = np.mean((y_te.to_numpy() >= lo) & (y_te.to_numpy() <= hi))
        # nominal 90% — allow slack for a finite, noisy sample
        assert 0.6 <= coverage <= 1.0, f"{name} coverage {coverage:.2f} off"


def test_pipeline_trains_ranks_and_saves(ohlcv, tmp_path):
    store = build_store(f"file://{tmp_path.as_posix()}")
    cfg = PipelineConfig(target_col="logret_5d", cv_splits=2)
    report = run_pipeline(ohlcv, cfg, store=store,
                          artifact_key="quant/T/model.joblib",
                          report_key="quant/T/report.json")

    assert report["ranking"], "no models ranked"
    assert report["best_model"] in report["ranking"]
    assert "rmse" in report["test"]["point"]
    assert "sharpe" in report["test"]["financial"]
    # interval coverage reported on the untouched test block
    assert 0.0 <= report["test"]["point"]["interval_coverage"] <= 1.0

    # artifact is reproducible: reload and score the latest bar with uncertainty
    raw = store.get_bytes("quant/T/model.joblib")
    assert raw is not None
    artifact = joblib.load(io.BytesIO(raw))
    out = predict_latest(artifact, ohlcv)
    assert out["prediction"] is not None
    assert out["lower"] <= out["prediction"] <= out["upper"]
    assert store.get_json("quant/T/report.json")["best_model"] == report["best_model"]


def test_pipeline_classifier_target(ohlcv):
    report = run_pipeline(ohlcv, PipelineConfig(target_col="direction_5d", cv_splits=2))
    assert report["best_model"] == "hist_gbm_classifier"
    assert 0.0 <= report["test"]["point"]["directional_accuracy"] <= 1.0
