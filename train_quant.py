"""CLI: train the per-symbol quant model zoo and save a reproducible artifact.

No hardcoded tickers or dates — the symbol and window are arguments. OHLCV is
pulled from the existing provider layer, then handed to the purged-CV pipeline.

    python train_quant.py --ticker AAPL --period 5y --target logret_5d
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from marketdesk.config import get_settings
from marketdesk.providers.prices import ticker_history
from marketdesk.quant.pipeline import PipelineConfig, run_pipeline
from marketdesk.store import build_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("marketdesk.quant.train")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the quant model zoo for one symbol.")
    parser.add_argument("--ticker", required=True, help="symbol to train on (e.g. AAPL)")
    parser.add_argument("--period", default="5y", help="yfinance period window")
    parser.add_argument("--target", default="logret_5d",
                        help="target column from build_targets (e.g. direction_5d)")
    parser.add_argument("--optimize", action="store_true",
                        help="run Optuna hyperparameter search per model")
    parser.add_argument("--trials", type=int, default=30, help="HPO trials per model")
    parser.add_argument("--no-save", action="store_true", help="skip writing artifacts")
    args = parser.parse_args(argv)

    settings = get_settings()
    log.info("Fetching OHLCV for %s (%s)...", args.ticker, args.period)
    ohlcv = ticker_history(args.ticker, period=args.period, interval="1d")
    if ohlcv.empty:
        log.error("No data returned for %s.", args.ticker)
        return 1

    store = None if args.no_save else build_store(settings.artifact_uri)
    report = run_pipeline(
        ohlcv, PipelineConfig(target_col=args.target, optimize=args.optimize,
                              hpo_trials=args.trials),
        store=store,
        artifact_key=f"quant/{args.ticker}/model.joblib",
        report_key=f"quant/{args.ticker}/report.json",
    )

    best = report["best_model"]
    test = report["test"]
    log.info("Winner: %s | test RMSE=%.5f dir_acc=%.3f coverage=%.2f Sharpe=%.2f",
             best, test["point"]["rmse"], test["point"]["directional_accuracy"],
             test["point"]["interval_coverage"], test["financial"]["sharpe"])
    print(json.dumps({"best_model": best, "ranking": report["ranking"],
                      "test": test}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
