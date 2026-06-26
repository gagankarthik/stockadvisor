"""The ML pattern model: estimate each stock's probability of beating the
cross-sectional median over the next ~month.

What makes this version stronger and more production-ready than the original
single-split RF+GB ensemble:

* **Calibrated probabilities.** Each base learner is wrapped in isotonic
  calibration so "ML %" is an honest probability, not just a ranking score —
  the app interprets ">60%" literally, so calibration matters.
* **Purged, embargoed walk-forward CV.** Because the 21-day forward label of
  nearby samples overlaps, a naive split leaks. Evaluation uses expanding
  walk-forward folds with an embargo gap, and reports the metric that actually
  governs ranking quality: the **rank information coefficient (IC)**, alongside
  AUC and accuracy.
* **Recency-weighted training.** Recent regimes count more via exponentially
  decaying sample weights.
* **A three-learner ensemble** (Random Forest + Extra Trees + Hist Gradient
  Boosting) for variance reduction.
* **Self-describing artifact.** The fitted model serializes (joblib) together
  with a model card — feature contract, horizon, metrics, sklearn version — so
  the serving side can load it from S3 and predict with zero retraining.
"""

from __future__ import annotations

import io
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import accuracy_score, roc_auc_score

from . import features as F
from .config import Settings
from .store import ArtifactStore

MODEL_KEY = "model.joblib"
HISTORY_KEY = "model_history.json"


@dataclass
class ModelCard:
    trained_at: str
    data_through: str
    horizon_days: int
    n_stocks: int
    train_samples: int
    test_samples: int
    train_until: str
    test_auc: float
    test_accuracy: float
    test_ic: float
    cv_auc: float
    cv_ic: float
    model_name: str
    sklearn_version: str
    features: list[str] = field(default_factory=lambda: list(F.FEATURES))
    feature_importances: dict[str, float] = field(default_factory=dict)

    def to_history_entry(self) -> dict:
        # Backward-compatible with the old model_history.json schema, plus extras.
        return {
            "trained_at": self.trained_at,
            "data_through": self.data_through,
            "auc": round(self.test_auc, 4),
            "accuracy": round(self.test_accuracy, 4),
            "ic": round(self.test_ic, 4),
            "cv_auc": round(self.cv_auc, 4),
            "cv_ic": round(self.cv_ic, 4),
            "n_stocks": self.n_stocks,
            "train_samples": self.train_samples,
        }


class MarketModel:
    """A fitted, serializable ensemble plus its model card."""

    def __init__(self, estimators: list, card: ModelCard):
        self._estimators = estimators
        self.card = card
        self.features = card.features
        self.horizon = card.horizon_days

    # ---- inference ----
    def _proba(self, X: pd.DataFrame) -> np.ndarray:
        cols = X[self.features]
        return np.mean([e.predict_proba(cols)[:, 1] for e in self._estimators], axis=0)

    def predict(self, snapshot: pd.DataFrame) -> pd.Series:
        """Outperformance probability per ticker for a normalized feature snapshot."""
        clean = snapshot.dropna(subset=self.features)
        if clean.empty:
            return pd.Series(dtype=float, name="ML Prob")
        return pd.Series(self._proba(clean), index=clean.index, name="ML Prob")

    def predict_latest(self, closes: pd.DataFrame,
                       spy: pd.Series | None) -> pd.Series:
        """Convenience: build the latest snapshot from prices and predict."""
        return self.predict(F.latest_snapshot(closes, spy))

    # ---- persistence ----
    def save(self, store: ArtifactStore, key: str = MODEL_KEY) -> None:
        buf = io.BytesIO()
        joblib.dump({"estimators": self._estimators, "card": asdict(self.card)}, buf)
        store.put_bytes(key, buf.getvalue())

    @classmethod
    def load(cls, store: ArtifactStore, key: str = MODEL_KEY) -> "MarketModel | None":
        raw = store.get_bytes(key)
        if raw is None:
            return None
        payload = joblib.load(io.BytesIO(raw))
        return cls(payload["estimators"], ModelCard(**payload["card"]))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _base_estimators(settings: Settings) -> dict:
    rs = settings.random_state
    return {
        "rf": RandomForestClassifier(
            n_estimators=settings.rf_estimators, max_depth=6,
            min_samples_leaf=20, n_jobs=-1, random_state=rs),
        "et": ExtraTreesClassifier(
            n_estimators=settings.rf_estimators, max_depth=8,
            min_samples_leaf=20, n_jobs=-1, random_state=rs),
        "gb": HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05,
            l2_regularization=1.0, random_state=rs),
    }


def _recency_weights(dates: pd.Series, half_life_days: float = 252.0) -> np.ndarray:
    """Exponential-decay sample weights: a sample `half_life_days` older than the
    most recent one counts half as much."""
    d = pd.to_datetime(dates)
    age = (d.max() - d).dt.days.to_numpy(dtype=float)
    return np.power(0.5, age / half_life_days)


def _rank_ic(proba: np.ndarray, fwd_ret: np.ndarray,
             dates: np.ndarray) -> float:
    """Mean per-date Spearman correlation between predicted probability and the
    realized forward return — the metric that reflects ranking skill."""
    df = pd.DataFrame({"p": proba, "r": fwd_ret, "d": dates})
    ics = []
    for _, g in df.groupby("d"):
        if len(g) >= 5:
            ic = g["p"].rank().corr(g["r"].rank())
            if pd.notna(ic):
                ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


def _ensemble_fit_proba(estimators: dict, train: pd.DataFrame,
                        test: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """Fit fresh (uncalibrated) clones on `train`, average proba on `test`.
    Used inside CV where calibration (monotonic) wouldn't change AUC/IC."""
    from sklearn.base import clone

    probas = []
    for est in estimators.values():
        m = clone(est)
        m.fit(train[F.FEATURES], train["y"], sample_weight=weights)
        probas.append(m.predict_proba(test[F.FEATURES])[:, 1])
    return np.mean(probas, axis=0)


def _walk_forward(data: pd.DataFrame, settings: Settings,
                  embargo: int) -> tuple[float, float]:
    """Expanding-window walk-forward CV with an embargo gap between train and
    test. Returns (mean AUC, mean IC) across folds."""
    dates = np.array(sorted(data["date"].unique()))
    n_folds = settings.n_walk_forward_folds
    if len(dates) < (n_folds + 2):
        return float("nan"), float("nan")

    bounds = np.linspace(0, len(dates), n_folds + 2, dtype=int)
    estimators = _base_estimators(settings)
    aucs, ics = [], []
    for k in range(1, n_folds + 1):
        train_end = bounds[k]
        test_start, test_end = bounds[k], bounds[k + 1]
        if test_end <= test_start or train_end <= embargo:
            continue
        train_dates = set(dates[: max(0, train_end - embargo)])
        test_dates = set(dates[test_start:test_end])
        train = data[data["date"].isin(train_dates)]
        test = data[data["date"].isin(test_dates)]
        if len(train) < 500 or len(test) < 100 or test["y"].nunique() < 2:
            continue
        w = _recency_weights(train["date"])
        proba = _ensemble_fit_proba(estimators, train, test, w)
        aucs.append(roc_auc_score(test["y"], proba))
        ics.append(_rank_ic(proba, test["fwd_ret"].to_numpy(),
                            test["date"].to_numpy()))
    return (float(np.nanmean(aucs)) if aucs else float("nan"),
            float(np.nanmean(ics)) if ics else float("nan"))


def train_model(closes: pd.DataFrame, stock_tickers: list[str],
                settings: Settings) -> MarketModel | None:
    """Train the calibrated ensemble and return a ready-to-serve MarketModel."""
    cols = [t for t in stock_tickers if t in closes.columns]
    if len(cols) < 30:
        return None
    px = closes[cols]
    spy = closes["SPY"] if "SPY" in closes.columns else None

    data = F.build_training_table(px, spy, settings.horizon_days,
                                  settings.sample_step)
    if len(data) < 1000:
        return None

    # sample-date embargo so train/test label windows can't overlap
    embargo = max(1, math.ceil(settings.horizon_days / settings.sample_step))

    # ---- walk-forward evaluation (purged) ----
    cv_auc, cv_ic = _walk_forward(data, settings, embargo)

    # ---- final hold-out (most recent 20%) for a single headline test number ----
    dates_sorted = sorted(data["date"].unique())
    split_idx = int(len(dates_sorted) * 0.8)
    split = dates_sorted[split_idx]
    train_cut = dates_sorted[max(0, split_idx - embargo)]
    train = data[data["date"] < train_cut]
    test = data[data["date"] >= split]

    test_auc = test_acc = test_ic = float("nan")
    if len(train) >= 500 and len(test) >= 100 and test["y"].nunique() >= 2:
        w = _recency_weights(train["date"])
        proba = _ensemble_fit_proba(_base_estimators(settings), train, test, w)
        test_auc = float(roc_auc_score(test["y"], proba))
        test_acc = float(accuracy_score(test["y"], proba > 0.5))
        test_ic = _rank_ic(proba, test["fwd_ret"].to_numpy(),
                           test["date"].to_numpy())

    # ---- production fit: calibrated ensemble on ALL data ----
    weights = _recency_weights(data["date"])
    calibrated, importances = [], np.zeros(len(F.FEATURES))
    n_tree_models = 0
    for name, est in _base_estimators(settings).items():
        cal = CalibratedClassifierCV(est, method="isotonic", cv=3)
        cal.fit(data[F.FEATURES], data["y"], sample_weight=weights)
        calibrated.append(cal)
        if hasattr(est, "feature_importances_") or name in ("rf", "et"):
            # pull importances from the underlying fitted tree ensembles
            for cc in cal.calibrated_classifiers_:
                base = cc.estimator
                if hasattr(base, "feature_importances_"):
                    importances += base.feature_importances_
                    n_tree_models += 1
    if n_tree_models:
        importances /= n_tree_models

    card = ModelCard(
        trained_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        data_through=str(px.index[-1].date()),
        horizon_days=settings.horizon_days,
        n_stocks=len(cols),
        train_samples=len(data),
        test_samples=len(test),
        train_until=str(pd.Timestamp(split).date()),
        test_auc=test_auc, test_accuracy=test_acc, test_ic=test_ic,
        cv_auc=cv_auc, cv_ic=cv_ic,
        model_name="Calibrated ensemble: Random Forest + Extra Trees + HistGradientBoosting",
        sklearn_version=sklearn.__version__,
        feature_importances={f: float(round(v, 5))
                             for f, v in zip(F.FEATURES, importances)},
    )
    return MarketModel(calibrated, card)


# ---- training history (append-only, stored alongside the artifact) ----

def log_history(store: ArtifactStore, card: ModelCard) -> None:
    hist = store.get_json(HISTORY_KEY) or []
    if not isinstance(hist, list):
        hist = []
    hist = [h for h in hist if h.get("data_through") != card.data_through]
    hist.append(card.to_history_entry())
    store.put_json(HISTORY_KEY, hist[-200:])


def load_history(store: ArtifactStore) -> list[dict]:
    hist = store.get_json(HISTORY_KEY)
    return hist if isinstance(hist, list) else []
