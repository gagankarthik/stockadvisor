"""End-to-end training pipeline (Lambda-safe tier).

Wiring, in order, with no look-ahead at any step:

    OHLCV ─▶ causal features ─▶ forward targets ─▶ time-ordered train/val/test
          ─▶ fit FeatureEngineer on TRAIN only
          ─▶ per model: purged walk-forward CV on train  +  val/test evaluation
                         (point metrics AND financial metrics via the backtester)
          ─▶ rank by validation Sharpe (primary), RMSE (secondary)
          ─▶ refit winner on train+val, evaluate on the untouched test block
          ─▶ save a reproducible artifact (model + feature engineer + config + report)

Everything is reproducible from the saved artifact; nothing is fit on data that
postdates what it is evaluated against.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import metrics as M
from .backtest import BacktestConfig, run_backtest
from .features import FeatureConfig, FeatureEngineer, compute_features
from .models import QuantModel, zoo_builders
from .splits import PurgedWalkForward, time_series_split
from .targets import TargetConfig, build_targets

try:  # store is optional (tests run without it)
    from ..store import ArtifactStore
except Exception:  # pragma: no cover
    ArtifactStore = object  # type: ignore


@dataclass
class PipelineConfig:
    target_col: str = "logret_5d"
    alpha: float = 0.1                 # 1 - alpha interval coverage
    train_frac: float = 0.6
    val_frac: float = 0.2
    cv_splits: int = 4
    embargo: int = 21
    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    optimize: bool = False        # run Optuna HPO per model before final fit
    hpo_trials: int = 30
    features: FeatureConfig = field(default_factory=FeatureConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)


def _signal_from_prediction(pred: np.ndarray, is_classifier: bool) -> np.ndarray:
    """Map a model output to a position in {-1, 0, +1}."""
    if is_classifier:
        return np.where(pred > 0.5, 1.0, -1.0)
    return np.sign(pred)


def _financial(prices: pd.Series, idx: pd.Index, pred: np.ndarray,
               is_classifier: bool, cfg: PipelineConfig) -> dict[str, float]:
    pos = pd.Series(_signal_from_prediction(pred, is_classifier), index=idx)
    px = prices.reindex(idx)
    res = run_backtest(px, pos, BacktestConfig(cost_bps=cfg.cost_bps,
                                               slippage_bps=cfg.slippage_bps))
    return res.metrics


def _evaluate(model: QuantModel, X: pd.DataFrame, y: pd.Series, prices: pd.Series,
              is_classifier: bool, cfg: PipelineConfig) -> dict:
    pred = model.predict(X)
    lo, hi = model.predict_interval(X)
    point = M.regression_report(y.to_numpy(), pred, lo, hi)
    fin = _financial(prices, X.index, pred, is_classifier, cfg)
    return {"point": point, "financial": fin}


def _cross_validate(builder, X: pd.DataFrame, y: pd.Series, cfg: PipelineConfig,
                    params: dict | None = None) -> dict:
    """Purged walk-forward CV; returns averaged out-of-fold point metrics."""
    params = params or {}
    oof_y, oof_p, oof_lo, oof_hi = [], [], [], []
    for tr, te in PurgedWalkForward(cfg.cv_splits, cfg.embargo).split(len(X)):
        if len(tr) < 50 or len(te) < 10:
            continue
        m = builder(alpha=cfg.alpha, **params)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[te])
        lo, hi = m.predict_interval(X.iloc[te])
        oof_y.append(y.iloc[te].to_numpy()); oof_p.append(p)
        oof_lo.append(lo); oof_hi.append(hi)
    if not oof_y:
        return {}
    return M.regression_report(np.concatenate(oof_y), np.concatenate(oof_p),
                               np.concatenate(oof_lo), np.concatenate(oof_hi))


def _tune(name: str, X: pd.DataFrame, y: pd.Series, prices: pd.Series,
          cfg: PipelineConfig) -> dict:
    """Optuna search for one model's hyperparameters (empty dict on any issue)."""
    from .hpo import optimize  # lazy: optuna is optional
    try:
        return optimize(name, X, y, prices, n_trials=cfg.hpo_trials, alpha=cfg.alpha,
                        cv_splits=cfg.cv_splits, embargo=cfg.embargo,
                        cost_bps=cfg.cost_bps, slippage_bps=cfg.slippage_bps).best_params
    except Exception:
        return {}


def run_pipeline(ohlcv: pd.DataFrame, config: PipelineConfig | None = None,
                 prices: pd.Series | None = None,
                 store: "ArtifactStore | None" = None,
                 artifact_key: str = "quant/model.joblib",
                 report_key: str = "quant/report.json") -> dict:
    """Train every model in the zoo and return a ranked, reproducible report.

    Args:
        ohlcv: OHLCV frame for a single instrument.
        config: Pipeline settings.
        prices: Price series for financial metrics (defaults to ``ohlcv.close``).
        store: Optional artifact store; if given, the winner and report are saved.
        artifact_key / report_key: Keys under the store.

    Returns:
        JSON-friendly report: per-model val/CV metrics, the winner, and its
        test-set performance.
    """
    cfg = config or PipelineConfig()
    ohlcv = ohlcv.rename(columns=str.lower)
    px = prices if prices is not None else ohlcv["close"]
    is_classifier = cfg.target_col.startswith("direction")

    feats = compute_features(ohlcv)
    targets = build_targets(ohlcv, cfg.targets)
    if cfg.target_col not in targets:
        raise ValueError(f"Unknown target_col {cfg.target_col!r}.")

    data = feats.join(targets[cfg.target_col].rename("y")).dropna()
    if len(data) < 200:
        raise ValueError("Not enough complete rows to train (need >= 200).")
    y_all = data["y"]
    X_all = data.drop(columns="y")

    tr_s, va_s, te_s = time_series_split(len(data), cfg.train_frac, cfg.val_frac)

    eng = FeatureEngineer(cfg.features).fit(X_all.iloc[tr_s])
    Xs = eng.transform(X_all)
    X_tr, X_va, X_te = Xs.iloc[tr_s], Xs.iloc[va_s], Xs.iloc[te_s]
    y_tr, y_va, y_te = y_all.iloc[tr_s], y_all.iloc[va_s], y_all.iloc[te_s]

    builders = zoo_builders(include_classifier=is_classifier)
    results: dict[str, dict] = {}
    fitted: dict[str, QuantModel] = {}
    for name, builder in builders.items():
        if is_classifier and name != "hist_gbm_classifier":
            continue
        if not is_classifier and name == "hist_gbm_classifier":
            continue
        try:
            params = _tune(name, X_tr, y_tr, px, cfg) if cfg.optimize else {}
            cv = _cross_validate(builder, X_tr, y_tr, cfg, params)
            model = builder(alpha=cfg.alpha, **params).fit(X_tr, y_tr)
            val = _evaluate(model, X_va, y_va, px, is_classifier, cfg)
            results[name] = {"cv": cv, "val": val, "params": params,
                             "val_sharpe": val["financial"]["sharpe"],
                             "val_rmse": val["point"]["rmse"]}
            fitted[name] = model
        except Exception as exc:  # never let one bad backend sink the run
            results[name] = {"error": str(exc)}

    ranked = sorted(
        (n for n in results if "error" not in results[n]),
        key=lambda n: (-results[n]["val_sharpe"], results[n]["val_rmse"]),
    )
    if not ranked:
        raise RuntimeError("No model trained successfully.")
    best = ranked[0]

    # refit winner (with its tuned params) on train+val, judge on the test block
    best_params = results[best].get("params", {})
    final = zoo_builders(include_classifier=is_classifier)[best](alpha=cfg.alpha, **best_params)
    final.fit(pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]))
    test = _evaluate(final, X_te, y_te, px, is_classifier, cfg)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": cfg.target_col,
        "n_samples": len(data),
        "n_features": X_tr.shape[1],
        "split": {"train": int(tr_s.stop), "val": int(va_s.stop - va_s.start),
                  "test": int(te_s.stop - te_s.start)},
        "ranking": ranked,
        "best_model": best,
        "models": results,
        "test": test,
        "config": asdict(cfg),
    }

    if store is not None:
        buf = io.BytesIO()
        joblib.dump({"model": final, "feature_engineer": eng,
                     "columns": list(Xs.columns), "config": asdict(cfg),
                     "best_model": best}, buf)
        store.put_bytes(artifact_key, buf.getvalue())
        store.put_json(report_key, report)

    return report


def predict_latest(artifact: dict, ohlcv: pd.DataFrame) -> dict:
    """Score the most recent bar with a saved pipeline artifact.

    Returns the point prediction and its interval — every prediction carries
    uncertainty, as required.
    """
    eng: FeatureEngineer = artifact["feature_engineer"]
    model: QuantModel = artifact["model"]
    feats = compute_features(ohlcv.rename(columns=str.lower))
    X = eng.transform(feats).dropna()
    if X.empty:
        return {"prediction": None, "lower": None, "upper": None}
    last = X.iloc[[-1]]
    pred = float(model.predict(last)[0])
    lo, hi = model.predict_interval(last)
    return {"date": str(X.index[-1].date()), "prediction": pred,
            "lower": float(lo[0]), "upper": float(hi[0]),
            "model": artifact.get("best_model")}
