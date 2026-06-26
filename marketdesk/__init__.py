"""MarketDesk — US market analysis engine.

A Streamlit-free, importable package powering a Lambda-hosted API:

- `config`     — environment-based settings (12-factor; no Streamlit secrets).
- `providers`  — market data sources (yfinance prices, Finnhub, Alpha Vantage).
- `features`   — cross-sectional, rank-normalized feature engineering.
- `model`      — the ML pattern model (train / predict / persist as an artifact).
- `analysis`   — indicators, adaptive factor scoring, allocation, risk.
- `signals`    — technical BUY/HOLD/SELL engine and gap scanner.
- `service`    — orchestration: builds the daily snapshot the API serves.
- `store`      — artifact/state persistence over local FS or S3.
"""

__version__ = "2.0.0"
