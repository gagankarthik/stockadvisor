"""Environment-based configuration (12-factor).

Replaces Streamlit's `st.secrets`. Every value can be supplied via an
environment variable, which is how Lambda, ECS, and local `.env` files all
inject configuration. Provider keys keep their familiar names
(`FINNHUB_KEY`, `ALPHAVANTAGE_KEY`); everything else is namespaced with the
`MARKETDESK_` prefix to avoid collisions.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MARKETDESK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Provider credentials (unprefixed, matching the old secrets names) ----
    finnhub_key: str = Field(default="", validation_alias="FINNHUB_KEY")
    alphavantage_key: str = Field(default="", validation_alias="ALPHAVANTAGE_KEY")
    # OpenAI key (e.g. for news-sentiment / narrative features). Read straight
    # from OPENAI_API_KEY so the same value works locally and in Lambda.
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")

    # ---- Storage (local path or s3://bucket/prefix) ----
    # Where trained model artifacts and the daily market snapshot live. The API
    # reads from here; the refresh job writes to it.
    artifact_uri: str = "file://.artifacts"
    # Saved allocation plans (a small JSON document under the artifact store).
    plans_key: str = "plans.json"

    # ---- Universe + data window ----
    indices: tuple[str, ...] = ("S&P 500",)
    price_period: str = "2y"

    # ---- Model hyper-parameters (overridable without code changes) ----
    horizon_days: int = 21          # forward window the model predicts (~1 month)
    sample_step: int = 5            # sample the panel every N trading days
    rf_estimators: int = 300
    n_walk_forward_folds: int = 4
    ml_blend_weight: float = 0.35   # how much ML rank tilts the composite score
    random_state: int = 42

    # ---- API behaviour ----
    api_title: str = "MarketDesk API"
    cors_origins: tuple[str, ...] = ("*",)
    # If the snapshot in the store is older than this many seconds, the API
    # flags it as stale (so clients/monitoring can trigger a refresh).
    snapshot_max_age_seconds: int = 24 * 3600


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton. Cached so Lambda warm starts don't re-parse env."""
    return Settings()
