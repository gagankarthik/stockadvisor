"""Market data providers.

- `prices`        — yfinance price history (the model's training data).
- `finnhub`       — real-time quotes, profiles, recommendations, earnings, news.
- `alphavantage`  — independent third quote source for cross-validation.
- `quotes`        — live quotes + cross-source triangulation badge.
"""

from . import alphavantage, finnhub, prices, quotes  # noqa: F401
