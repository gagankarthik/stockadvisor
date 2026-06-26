"""Model training, calibration, and serialization (no network)."""

import numpy as np

from marketdesk import features as F
from marketdesk.config import get_settings
from marketdesk.model import MarketModel, load_history, log_history, train_model
from marketdesk.store import build_store


def test_features_are_rank_normalized(synthetic_closes, stock_tickers):
    feats = F.compute_raw_features(synthetic_closes[stock_tickers],
                                   synthetic_closes["SPY"])
    snap = F.latest_snapshot(synthetic_closes[stock_tickers], synthetic_closes["SPY"])
    assert set(F.FEATURES).issubset(feats)
    # rank-normalized features live in [-1, 1]
    assert snap.to_numpy().min() >= -1.0001 and snap.to_numpy().max() <= 1.0001


def test_train_and_predict(synthetic_closes, stock_tickers):
    settings = get_settings()
    model = train_model(synthetic_closes, stock_tickers, settings)
    assert model is not None
    card = model.card
    assert card.features == F.FEATURES
    assert 0.0 <= card.test_auc <= 1.0
    assert card.train_samples > 1000

    probs = model.predict_latest(synthetic_closes, synthetic_closes["SPY"])
    assert len(probs) > 0
    assert probs.between(0.0, 1.0).all()  # calibrated probabilities


def test_save_load_roundtrip(synthetic_closes, stock_tickers):
    settings = get_settings()
    store = build_store(settings.artifact_uri)
    model = train_model(synthetic_closes, stock_tickers, settings)
    model.save(store)
    log_history(store, model.card)

    reloaded = MarketModel.load(store)
    assert reloaded is not None
    a = model.predict_latest(synthetic_closes, synthetic_closes["SPY"])
    b = reloaded.predict_latest(synthetic_closes, synthetic_closes["SPY"])
    assert np.allclose(a.values, b.values)

    hist = load_history(store)
    assert isinstance(hist, list) and hist
    assert hist[-1]["data_through"] == card_data_through(model)


def card_data_through(model) -> str:
    return model.card.data_through
