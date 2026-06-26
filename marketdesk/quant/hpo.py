"""Hyperparameter optimization (Optuna).

Bayesian search (TPE) with a MedianPruner, objective = **purged walk-forward
Sharpe** on the training data only (never the test block). A two-phase
coarse→fine schedule is supported: phase 1 explores the full space, phase 2
re-seeds the search around the incumbent best.

Optuna is an optional dependency — importing this module without it raises a
clear error, and the rest of the package keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .models import zoo_builders
from .splits import PurgedWalkForward

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:  # pragma: no cover
    _HAS_OPTUNA = False


# Per-model search spaces. Keys must match the builder kwargs in models.py.
def _space(name: str, trial) -> dict:
    if name == "hist_gbm_quantile":
        return {
            "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        }
    if name == "elasticnet":
        return {
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
        }
    if name == "svr_rbf":
        return {
            "C": trial.suggest_float("C", 0.1, 100.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 0.1, log=True),
        }
    if name in ("xgboost", "lightgbm", "catboost"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
    return {}


@dataclass
class HPOResult:
    model: str
    best_params: dict
    best_value: float          # mean purged-CV Sharpe
    n_trials: int


def _require_optuna() -> None:
    if not _HAS_OPTUNA:
        raise ImportError("Optuna is not installed. `pip install optuna` to use HPO.")


def _fold_sharpe(model, X_te, y_te, prices, is_classifier, cost_bps, slippage_bps) -> float:
    pred = model.predict(X_te)
    pos = pd.Series(np.where(pred > 0.5, 1.0, -1.0) if is_classifier else np.sign(pred),
                    index=X_te.index)
    res = run_backtest(prices.reindex(X_te.index), pos,
                       BacktestConfig(cost_bps=cost_bps, slippage_bps=slippage_bps))
    return res.metrics["sharpe"]


def optimize(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    prices: pd.Series,
    *,
    n_trials: int = 40,
    alpha: float = 0.1,
    cv_splits: int = 4,
    embargo: int = 21,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    refine: bool = True,
    seed: int = 42,
) -> HPOResult:
    """Tune one model by maximizing mean purged-CV Sharpe on ``(X, y)``.

    Args:
        model_name: A key from :func:`marketdesk.quant.models.zoo_builders`.
        X, y: Training features and target (no test data).
        prices: Price series for the Sharpe objective.
        n_trials: Total trials (split across coarse/fine if ``refine``).
        refine: If True, run a second phase seeded with the best params.

    Returns:
        :class:`HPOResult` with the best params and CV Sharpe.
    """
    _require_optuna()
    builders = zoo_builders(include_classifier=model_name == "hist_gbm_classifier")
    if model_name not in builders:
        raise ValueError(f"Unknown model {model_name!r}.")
    builder = builders[model_name]
    is_classifier = model_name == "hist_gbm_classifier"
    folds = list(PurgedWalkForward(cv_splits, embargo).split(len(X)))
    if not folds:
        raise ValueError("Not enough data for the requested CV folds.")

    def objective(trial) -> float:
        params = _space(model_name, trial)
        scores = []
        for step, (tr, te) in enumerate(folds):
            if len(tr) < 50 or len(te) < 10:
                continue
            model = builder(alpha=alpha, **params)
            model.fit(X.iloc[tr], y.iloc[tr])
            s = _fold_sharpe(model, X.iloc[te], y.iloc[te], prices,
                             is_classifier, cost_bps, slippage_bps)
            scores.append(s)
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores)) if scores else -np.inf

    pruner = MedianPruner(n_startup_trials=max(3, n_trials // 8))
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=seed), pruner=pruner)

    if refine and n_trials >= 8:
        coarse = n_trials // 2
        study.optimize(objective, n_trials=coarse)
        # phase 2: focus the search around the incumbent best
        study.enqueue_trial(study.best_params)
        study.optimize(objective, n_trials=n_trials - coarse)
    else:
        study.optimize(objective, n_trials=n_trials)

    return HPOResult(model=model_name, best_params=dict(study.best_params),
                     best_value=float(study.best_value),
                     n_trials=len(study.trials))


def optimize_zoo(
    X: pd.DataFrame, y: pd.Series, prices: pd.Series,
    models: list[str] | None = None, n_trials: int = 40, **kw,
) -> dict[str, HPOResult]:
    """Run :func:`optimize` for several models; returns ``{name: HPOResult}``."""
    _require_optuna()
    names = models or [n for n in zoo_builders() if n != "hist_gbm_classifier"]
    out: dict[str, HPOResult] = {}
    for name in names:
        try:
            out[name] = optimize(name, X, y, prices, n_trials=n_trials, **kw)
        except Exception:  # keep going if one backend's search fails
            continue
    return out
