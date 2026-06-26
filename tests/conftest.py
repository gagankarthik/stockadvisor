"""Shared fixtures. Points the artifact store at a temp dir *before* importing
anything that reads settings, and builds a synthetic, network-free dataset."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# Must be set before marketdesk.config.get_settings() is first called.
_TMP = tempfile.mkdtemp()
os.environ.setdefault("MARKETDESK_ARTIFACT_URI", "file://" + _TMP.replace("\\", "/"))
os.environ.setdefault("FINNHUB_KEY", "")
os.environ.setdefault("ALPHAVANTAGE_KEY", "")


@pytest.fixture(scope="session")
def synthetic_closes() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-01", periods=650)
    stocks = [f"S{i:02d}" for i in range(50)]
    etfs = ["SPY", "QQQ", "SCHD", "GLD", "AGG"]
    data = {}
    for t in stocks + etfs:
        rets = rng.normal(rng.normal(0.0004, 0.0004), 0.015, len(dates))
        data[t] = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=dates)


@pytest.fixture(scope="session")
def stock_tickers(synthetic_closes) -> list[str]:
    return [c for c in synthetic_closes.columns if c.startswith("S")]


@pytest.fixture(scope="session")
def synthetic_info(synthetic_closes) -> pd.DataFrame:
    cols = list(synthetic_closes.columns)
    etfs = {"SPY", "QQQ", "SCHD", "GLD", "AGG"}
    return pd.DataFrame({
        "Name": {t: f"Company {t}" for t in cols},
        "Sector": {t: ("Broad Market" if t in etfs
                       else ["Tech", "Health", "Energy", "Financials"][i % 4])
                   for i, t in enumerate(cols)},
        "Type": {t: ("ETF" if t in etfs else "Stock") for t in cols},
    })
