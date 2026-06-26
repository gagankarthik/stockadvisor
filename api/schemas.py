"""Pydantic response/request models — the API's typed contract and the source
of the auto-generated OpenAPI docs at ``/docs``."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str
    version: str
    snapshot_available: bool
    snapshot_age_seconds: float | None = None
    snapshot_stale: bool = False
    data_through: str | None = None


class ModelCardOut(BaseModel):
    trained_at: str | None = None
    data_through: str | None = None
    horizon_days: int | None = None
    n_stocks: int | None = None
    test_auc: float | None = None
    test_accuracy: float | None = None
    test_ic: float | None = None
    cv_auc: float | None = None
    cv_ic: float | None = None
    model_name: str | None = None
    sklearn_version: str | None = None
    feature_importances: dict[str, float] = Field(default_factory=dict)


class DashboardOut(BaseModel):
    generated_at: str
    data_through: str
    regime: str
    vix: float | None = None
    breadth: dict[str, float]
    sectors: list[dict]
    movers: dict[str, list[dict]]
    etfs: list[dict]
    ics: dict[str, float]
    model: ModelCardOut | None = None


class StockRow(BaseModel):
    ticker: str
    name: str | None = None
    sector: str | None = None
    price: float | None = None
    score: float | None = None
    ml_pct: float | None = None
    confidence_pct: float | None = None
    lean: str | None = None
    signal: str | None = None
    rsi: float | None = None
    volatility_pct: float | None = None
    ret_1m_pct: float | None = None
    ret_12_1m_pct: float | None = None
    reasons: str | None = None


class ScreenerOut(BaseModel):
    profile: str
    adaptive: bool
    use_ml: bool
    count: int
    data_through: str
    stocks: list[StockRow]


class AllocationRequest(BaseModel):
    amount: float = Field(gt=0, default=10000)
    profile: str = "Balanced"
    n_picks: int = Field(ge=1, le=25, default=8)
    adaptive: bool = True
    use_ml: bool = True


class AllocationOut(BaseModel):
    amount: float
    profile: str
    spy_price: float | None = None
    allocation: list[dict]
    risk: dict


class SavePlanRequest(BaseModel):
    amount: float = Field(gt=0)
    profile: str = "Balanced"
    n_picks: int = Field(ge=1, le=25, default=8)
    adaptive: bool = True
    use_ml: bool = True
