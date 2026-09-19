"""Request models for the ForecastIQ API."""
from __future__ import annotations

from datetime import date as Date
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Overrides(BaseModel):
    """Optional what-if values. Anything left empty is auto-filled from historical data."""

    model_config = ConfigDict(extra="forbid")

    temperature: Optional[float] = Field(default=None, ge=-60, le=140, description="Degrees Fahrenheit")
    fuel_price: Optional[float] = Field(default=None, gt=0, le=20, description="USD per gallon")
    cpi: Optional[float] = Field(default=None, gt=0, le=1000)
    unemployment: Optional[float] = Field(default=None, ge=0, le=100, description="Percent")
    markdown1: Optional[float] = Field(default=None, ge=0)
    markdown2: Optional[float] = Field(default=None, ge=0)
    markdown3: Optional[float] = Field(default=None, ge=0)
    markdown4: Optional[float] = Field(default=None, ge=0)
    markdown5: Optional[float] = Field(default=None, ge=0)
    is_holiday: Optional[bool] = None


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    store: int = Field(ge=1, description="Store ID")
    dept: int = Field(ge=1, description="Department ID")
    date: Date = Field(description="Any date in the target week (snapped to the week's Friday)")
    horizon: int = Field(default=1, ge=1, le=13, description="Number of consecutive weeks to forecast")
    overrides: Optional[Overrides] = None


class SingleForecastRequest(BaseModel):
    """Legacy 'bring your own data' request (kept for API compatibility).

    The caller supplies the recent sales history and the economic context
    instead of relying on the workspace's reference data.
    """

    store: int = Field(ge=1, le=45)
    dept: int = Field(ge=1, le=99)
    forecast_date: Date
    is_holiday: bool = False
    store_type: str = Field(default="A", pattern="^[ABC]$")
    size: float = Field(gt=0)
    temperature: float
    fuel_price: float = Field(gt=0)
    cpi: float = Field(gt=0)
    unemployment: float = Field(ge=0)
    markdown1: float = 0
    markdown2: float = 0
    markdown3: float = 0
    markdown4: float = 0
    markdown5: float = 0
    # Newest first. Needs >= 3 values; supply >= 55 to enable the seasonal (last-year) features.
    recent_sales_history: List[float] = Field(min_length=3, max_length=156)
    is_holiday_prev_week: bool = False
    is_holiday_next_week: bool = False
    store_avg_sales: Optional[float] = None
    dept_avg_sales: Optional[float] = None
    store_avg_fuel_price: Optional[float] = None
