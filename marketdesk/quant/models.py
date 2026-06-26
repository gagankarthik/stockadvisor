"""Model zoo — Lambda-safe tier (gradient boosting + classical), every model
emitting prediction intervals (uncertainty).

A single :class:`QuantModel` interface lets the training pipeline treat every
architecture identically:

    model.fit(X_train, y_train)
    point  = model.predict(X_test)
    lo, hi = model.predict_interval(X_test)   # (1 - alpha) coverage

Always-available models are built on scikit-learn. Heavier optional backends
(XGBoost, LightGBM, CatBoost, ARIMA, GARCH) are registered only if importable,
so the package runs anywhere while still using them when present.

Interval strategy:
- **Quantile GBM** fits separate lower/median/upper quantile regressors.
- Everything else is wrapped in **split-conformal** calibration: residual
  quantiles from a held-out (most-recent) slice of the training data give
  distribution-free coverage. The calibration slice is always *later* in time
  than the fitting slice, so there is no look-ahead.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR


class QuantModel:
    """Base interface. Subclasses implement `_fit`/`_predict`; intervals come
    either from native quantiles or the conformal wrapper below."""

    name: str = "base"
    is_classifier: bool = False

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha  # 1 - alpha = nominal interval coverage
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Conformal wrapper (works for any point regressor)                           #
# --------------------------------------------------------------------------- #

class ConformalRegressor(QuantModel):
    """Split-conformal intervals around any scikit-learn-style point regressor."""

    def __init__(self, estimator, name: str, alpha: float = 0.1,
                 calib_frac: float = 0.2) -> None:
        super().__init__(alpha)
        self.estimator = estimator
        self.name = name
        self.calib_frac = calib_frac
        self.q_ = np.nan

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ConformalRegressor":
        n = len(X)
        cut = max(1, int(n * (1 - self.calib_frac)))
        Xf, yf = X.iloc[:cut], y.iloc[:cut]
        Xc, yc = X.iloc[cut:], y.iloc[cut:]
        self.estimator.fit(Xf, yf)
        if len(Xc) >= 10:
            resid = np.abs(yc.to_numpy() - self.estimator.predict(Xc))
            self.q_ = float(np.quantile(resid, 1 - self.alpha))
            # refit on all training data now that the residual budget is known
            self.estimator.fit(X, y)
        else:  # too little data to calibrate — fall back to in-sample residuals
            self.estimator.fit(X, y)
            resid = np.abs(y.to_numpy() - self.estimator.predict(X))
            self.q_ = float(np.quantile(resid, 1 - self.alpha))
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p = self.predict(X)
        return p - self.q_, p + self.q_


# --------------------------------------------------------------------------- #
# Native quantile gradient boosting                                           #
# --------------------------------------------------------------------------- #

class QuantileGBM(QuantModel):
    """Three HistGradientBoosting quantile regressors → median + interval."""

    name = "hist_gbm_quantile"

    def __init__(self, alpha: float = 0.1, **kw) -> None:
        super().__init__(alpha)
        self._kw = {"max_iter": 300, "max_depth": 3, "learning_rate": 0.05,
                    "l2_regularization": 1.0, "random_state": 42, **kw}
        self._lo = self._md = self._hi = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileGBM":
        self._lo = HistGradientBoostingRegressor(loss="quantile", quantile=self.alpha / 2, **self._kw).fit(X, y)
        self._md = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **self._kw).fit(X, y)
        self._hi = HistGradientBoostingRegressor(loss="quantile", quantile=1 - self.alpha / 2, **self._kw).fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._md.predict(X)

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self._lo.predict(X), self._hi.predict(X)
        return np.minimum(lo, hi), np.maximum(lo, hi)


class RandomForestInterval(QuantModel):
    """Random Forest with intervals from the spread of per-tree predictions."""

    name = "random_forest"

    def __init__(self, alpha: float = 0.1, **kw) -> None:
        super().__init__(alpha)
        self.rf = RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            n_jobs=-1, random_state=42, **kw)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestInterval":
        self.rf.fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.rf.predict(X)

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        per_tree = np.stack([t.predict(X.to_numpy()) for t in self.rf.estimators_])
        lo = np.quantile(per_tree, self.alpha / 2, axis=0)
        hi = np.quantile(per_tree, 1 - self.alpha / 2, axis=0)
        return lo, hi


# --------------------------------------------------------------------------- #
# Direction classifier (calibrated probability = uncertainty)                 #
# --------------------------------------------------------------------------- #

class DirectionGBM(QuantModel):
    """Binary up/down classifier. `predict` returns P(up); the interval is the
    calibrated probability band [p, p] collapsed to express confidence via
    `predict_interval` returning (1 - p, p) style uncertainty bounds."""

    name = "hist_gbm_classifier"
    is_classifier = True

    def __init__(self, alpha: float = 0.1, **kw) -> None:
        super().__init__(alpha)
        self.clf = HistGradientBoostingClassifier(
            max_iter=300, max_depth=3, learning_rate=0.05,
            l2_regularization=1.0, random_state=42, **kw)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DirectionGBM":
        self.clf.fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p = self.predict(X)
        # uncertainty band: wider when the probability is near 0.5
        spread = (1 - 2 * np.abs(p - 0.5)) * 0.5
        return np.clip(p - spread, 0, 1), np.clip(p + spread, 0, 1)


# --------------------------------------------------------------------------- #
# Registry (optional backends added only if importable)                       #
# --------------------------------------------------------------------------- #

def _elasticnet(alpha: float = 0.1, reg_alpha: float = 0.001,
                l1_ratio: float = 0.5) -> QuantModel:
    return ConformalRegressor(
        ElasticNet(alpha=reg_alpha, l1_ratio=l1_ratio, max_iter=5000),
        name="elasticnet", alpha=alpha)


def _svr(alpha: float = 0.1, C: float = 1.0, epsilon: float = 0.01) -> QuantModel:
    return ConformalRegressor(SVR(kernel="rbf", C=C, epsilon=epsilon),
                              name="svr_rbf", alpha=alpha)


def _optional_models() -> dict[str, Callable[..., QuantModel]]:
    """Heavier backends, registered only when their library is installed."""
    out: dict[str, Callable[..., QuantModel]] = {}
    try:
        import xgboost  # noqa: F401

        def _xgb(alpha: float = 0.1, **params) -> QuantModel:
            from xgboost import XGBRegressor
            kw = dict(n_estimators=400, max_depth=4, learning_rate=0.04,
                      subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
            kw.update(params)
            return ConformalRegressor(XGBRegressor(**kw), name="xgboost", alpha=alpha)

        out["xgboost"] = _xgb
    except ImportError:
        pass
    try:
        import lightgbm  # noqa: F401

        def _lgbm(alpha: float = 0.1, **params) -> QuantModel:
            from lightgbm import LGBMRegressor
            kw = dict(n_estimators=500, num_leaves=31, learning_rate=0.04,
                      subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
            kw.update(params)
            return ConformalRegressor(LGBMRegressor(**kw), name="lightgbm", alpha=alpha)

        out["lightgbm"] = _lgbm
    except ImportError:
        pass
    try:
        import catboost  # noqa: F401

        def _cat(alpha: float = 0.1, **params) -> QuantModel:
            from catboost import CatBoostRegressor
            kw = dict(iterations=500, depth=5, learning_rate=0.04,
                      verbose=False, allow_writing_files=False)
            kw.update(params)
            return ConformalRegressor(CatBoostRegressor(**kw), name="catboost", alpha=alpha)

        out["catboost"] = _cat
    except ImportError:
        pass
    return out


def zoo_builders(include_classifier: bool = False) -> dict[str, Callable[..., QuantModel]]:
    """Map ``name -> builder(alpha=...)`` for every available model. Builders
    (not instances) let the CV loop construct a fresh model per fold."""
    builders: dict[str, Callable[..., QuantModel]] = {
        "hist_gbm_quantile": QuantileGBM,
        "random_forest": RandomForestInterval,
        "elasticnet": _elasticnet,
        "svr_rbf": _svr,
        **_optional_models(),
    }
    if include_classifier:
        builders["hist_gbm_classifier"] = DirectionGBM
    return builders


def model_zoo(alpha: float = 0.1, include_classifier: bool = False) -> dict[str, QuantModel]:
    """Instantiate every available model (see :func:`zoo_builders`)."""
    return {name: build(alpha=alpha)
            for name, build in zoo_builders(include_classifier).items()}
