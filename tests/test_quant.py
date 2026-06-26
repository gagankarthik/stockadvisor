"""Tests for the quant research toolkit: causality (no look-ahead), validation,
feature post-processing, splits, backtest frictions, and drift."""

import numpy as np
import pandas as pd
import pytest

from marketdesk.quant.validation import DataValidator, ValidationConfig
from marketdesk.quant.features import FeatureEngineer, compute_features
from marketdesk.quant.targets import build_targets
from marketdesk.quant.splits import PurgedWalkForward, time_series_split
from marketdesk.quant.backtest import BacktestConfig, run_backtest
from marketdesk.quant.drift import concept_drift, population_stability_index


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2021-01-01", periods=650)
    ret = rng.normal(0.0004, 0.013, len(idx))
    close = 100 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.003, len(idx)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, len(idx))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, len(idx))))
    vol = rng.integers(1_000_000, 5_000_000, len(idx)).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


# ---- no look-ahead: the core invariant ----
def test_features_are_causal(ohlcv):
    full = compute_features(ohlcv)
    k = 520
    truncated = compute_features(ohlcv.iloc[: k + 1])
    a = full.iloc[k]
    b = truncated.iloc[-1]
    assert list(a.index) == list(b.index)
    # every feature at time t must be identical whether or not future rows exist
    assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True, atol=1e-9)


def test_feature_count_and_finiteness(ohlcv):
    feats = compute_features(ohlcv)
    assert feats.shape[1] >= 30
    tail = feats.dropna()
    assert len(tail) > 100
    assert np.isfinite(tail.to_numpy(dtype=float)).all()


def test_targets_are_forward(ohlcv):
    tgt = build_targets(ohlcv)
    assert {"direction_1d", "logret_5d", "regime"}.issubset(tgt.columns)
    # forward labels: the last rows have no future, so must be NaN
    assert tgt["logret_21d"].iloc[-1] != tgt["logret_21d"].iloc[-1] or pd.isna(tgt["logret_21d"].iloc[-1])
    assert pd.isna(tgt["direction_5d"].iloc[-1])


# ---- validation ----
def test_validation_flags_problems(ohlcv):
    df = ohlcv.copy()
    df.iloc[10, df.columns.get_loc("high")] = df.iloc[10]["low"] - 1  # high < low
    df.iloc[20, df.columns.get_loc("volume")] = 0  # halt
    df.iloc[21, df.columns.get_loc("volume")] = -5  # impossible
    df.iloc[30, df.columns.get_loc("close")] *= 1.5  # spike + corp-action candidate
    dup = df.iloc[[40]]
    df = pd.concat([df, dup]).sort_index()  # exact duplicate timestamp+row
    df = df.drop(df.index[100])  # a missing business day

    report = DataValidator(ValidationConfig(expected_freq="B")).validate(df)
    s = report.summary()
    assert s["ohlc_violations"] >= 1
    assert s["zero_volume"] >= 1
    assert s["negative_volume"] >= 1
    assert s["duplicates"] >= 1
    assert s["missing_timestamps"] >= 1
    assert len(report.corp_action_candidates) >= 1


def test_gap_filling_tiers(ohlcv):
    v = DataValidator()
    s = ohlcv["close"].copy()
    s.iloc[200:203] = np.nan  # ~3 business days -> large gap, stays NaN
    filled, synthetic = v.fill_gaps(s)
    assert filled.isna().sum() >= 1  # large gap left for the model
    assert synthetic.dtype == bool


# ---- feature post-processing (learned, no leak) ----
def test_engineer_drops_correlated_and_scales(ohlcv):
    feats = compute_features(ohlcv).dropna()
    train = feats.iloc[:300].copy()
    train["dup_of_rsi"] = train["rsi_14"]  # perfectly correlated -> must be dropped
    eng = FeatureEngineer()
    out_train = eng.fit_transform(train)
    assert "dup_of_rsi" in eng.dropped_
    assert "dup_of_rsi" not in out_train.columns
    # transform of a later slice uses train-learned params (shape-consistent)
    test = feats.iloc[300:].copy()
    test["dup_of_rsi"] = test["rsi_14"]
    out_test = eng.transform(test)
    assert list(out_test.columns) == list(out_train.columns)
    assert np.isfinite(out_test.dropna().to_numpy()).all()


# ---- splits ----
def test_time_series_split_is_ordered_and_disjoint():
    tr, va, te = time_series_split(1000, 0.6, 0.2)
    assert tr.stop == va.start and va.stop == te.start and te.stop == 1000
    assert tr.start == 0


def test_purged_walk_forward_embargo():
    embargo = 21
    folds = list(PurgedWalkForward(n_splits=4, embargo=embargo).split(600))
    assert len(folds) == 4
    for train_idx, test_idx in folds:
        assert train_idx[-1] < test_idx[0]
        assert test_idx[0] - train_idx[-1] >= embargo  # purge gap respected


# ---- backtest frictions ----
def test_backtest_costs_and_foresight(ohlcv):
    px = ohlcv["close"]
    fwd_sign = np.sign(px.shift(-1) / px - 1).fillna(0.0)  # perfect-foresight (engine test only)

    free = run_backtest(px, fwd_sign, BacktestConfig(cost_bps=0, slippage_bps=0))
    costly = run_backtest(px, fwd_sign, BacktestConfig(cost_bps=50, slippage_bps=20))

    assert free.metrics["turnover"] > 0
    assert costly.metrics["total_return"] < free.metrics["total_return"]  # frictions bite
    assert free.metrics["sharpe"] > 0
    assert set(["sharpe", "sortino", "calmar", "max_drawdown", "var_95", "cvar_95"]).issubset(free.metrics)


def test_backtest_no_lookahead_in_execution(ohlcv):
    # a constant position must equal buy & hold minus zero turnover cost
    px = ohlcv["close"]
    res = run_backtest(px, pd.Series(1.0, index=px.index), BacktestConfig(cost_bps=0, slippage_bps=0))
    bh = px.iloc[-1] / px.iloc[0] - 1
    assert abs(res.metrics["total_return"] - bh) < 1e-6


# ---- drift ----
def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0, 1, 5000))
    same = pd.Series(rng.normal(0, 1, 5000))
    shifted = pd.Series(rng.normal(1.5, 1, 5000))
    assert population_stability_index(base, same) < 0.1
    assert population_stability_index(base, shifted) >= 0.25


def test_concept_drift_status():
    errs = pd.Series(np.full(30, 0.5))
    assert concept_drift(0.1, errs)["status"] == "critical"
    assert concept_drift(0.5, errs)["status"] == "ok"
