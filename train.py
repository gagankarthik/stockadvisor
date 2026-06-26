"""Decoupled refresh job: fetch data, (re)train the ML model, score the
universe, and write the artifacts the API serves.

Run it three ways:

    python train.py                 # local CLI (uses .env / environment)
    docker run marketdesk-train     # container (CMD below)
    -> EventBridge schedule          # as a Lambda using `lambda_handler`

Keeping training here — separate from the request path — is the whole point of
the architecture: heavy work runs on a schedule and writes a small artifact to
S3; the API stays fast, cheap, and within Lambda's limits.
"""

from __future__ import annotations

import argparse
import logging
import sys

from marketdesk.config import get_settings
from marketdesk.model import train_model
from marketdesk.providers.prices import fetch_prices
from marketdesk.service import build_snapshot
from marketdesk.store import build_store
from marketdesk.universe import full_info
from marketdesk import analysis
from marketdesk.model import MarketModel, log_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("marketdesk.train")


def run(force_retrain: bool = True) -> dict:
    """Train + snapshot. Returns a summary dict (also used as the Lambda result)."""
    settings = get_settings()
    store = build_store(settings.artifact_uri)
    log.info("Artifact store: %s", settings.artifact_uri)

    model = None
    if force_retrain:
        log.info("Fetching universe %s (%s)...", settings.indices, settings.price_period)
        info = full_info(settings.indices)
        closes = fetch_prices(list(info.index), settings.price_period)
        metrics = analysis.compute_metrics(closes, info)
        stock_tickers = list(metrics[metrics["Type"] == "Stock"].index)
        log.info("Training model on %d stocks...", len(stock_tickers))
        model = train_model(closes, stock_tickers, settings)
        if model is None:
            log.error("Training produced no model (insufficient data).")
            return {"status": "error", "reason": "insufficient_data"}
        model.save(store)
        log_history(store, model.card)
        log.info("Model trained: test AUC=%.4f IC=%.4f | CV AUC=%.4f IC=%.4f",
                 model.card.test_auc, model.card.test_ic,
                 model.card.cv_auc, model.card.cv_ic)

    log.info("Building market snapshot...")
    snap = build_snapshot(settings, store, model=model)
    log.info("Snapshot written: data_through=%s, %d stocks.",
             snap["data_through"], snap["n_stocks"])
    return {"status": "ok", "data_through": snap["data_through"],
            "n_stocks": snap["n_stocks"],
            "model": snap.get("model_card", {}).get("model_name")
            if snap.get("model_card") else None}


def lambda_handler(event, context):  # noqa: ANN001
    """EventBridge / manual invoke entry point. `event` may set {"retrain": bool}."""
    force = bool((event or {}).get("retrain", True))
    return run(force_retrain=force)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MarketDesk refresh job")
    parser.add_argument("--no-retrain", action="store_true",
                        help="reuse the stored model; only rebuild the snapshot")
    args = parser.parse_args(argv)
    result = run(force_retrain=not args.no_retrain)
    log.info("Done: %s", result)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
