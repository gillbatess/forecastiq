"""Central configuration for ForecastIQ.

Everything that is a path, a constant or a feature contract lives here so the
training script, the inference service and the tests can never drift apart.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Raw Kaggle-style CSVs. Only needed for (re)training; NOT needed at runtime.
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data" / "raw"))

# Everything the running service needs is inside ARTIFACT_DIR (model files +
# compact reference data), so the deployed image does not ship the 12 MB train.csv.
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT / "artifacts"))
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
REFERENCE_DIR = ARTIFACT_DIR / "reference"

WEB_DIR = ROOT / "web"

MODEL_VERSION = "2.0.0"
MODEL_NAME = "ForecastIQ XGBoost"

# --------------------------------------------------------------------------
# Feature contract (order matters: it is the column order the booster sees).
# The first 28 are the notebook's features. Three seasonal features are added:
#   Lag52                    same week last year
#   Seasonal_Ratio           last year's growth from the 3 preceding weeks into that week
#   Seasonal_Naive_Forecast  this year's 3-week average x that ratio
# They are NaN when a year of history is not available (XGBoost handles NaN natively).
# --------------------------------------------------------------------------
MARKDOWNS = [f"MarkDown{i}" for i in range(1, 6)]

FEATURES = [
    "IsHoliday", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Size",
    "StoreType_A", "StoreType_B", "StoreType_C",
    *MARKDOWNS,
    "Total_MarkDown", "Week_sin", "Week_cos", "Month_sin", "Month_cos",
    "Weekly_Sales_Lag1", "Weekly_Sales_Lag2", "Weekly_Sales_MA3",
    "Weekly_Sales_STD3", "IsHoliday_PrevWeek", "IsHoliday_NextWeek",
    "Rel_Fuel_Price", "Store_Avg_Sales", "Dept_Avg_Sales",
    "Weekly_Sales_Lag52", "Seasonal_Ratio", "Seasonal_Naive_Forecast",
]

# Human-readable labels used by the UI ("what drove this forecast").
FEATURE_LABELS = {
    "IsHoliday": "Holiday week",
    "Temperature": "Temperature",
    "Fuel_Price": "Fuel price",
    "CPI": "Consumer price index",
    "Unemployment": "Unemployment rate",
    "Size": "Store size",
    "StoreType_A": "Store type A",
    "StoreType_B": "Store type B",
    "StoreType_C": "Store type C",
    "MarkDown1": "Promo markdown 1",
    "MarkDown2": "Promo markdown 2",
    "MarkDown3": "Promo markdown 3",
    "MarkDown4": "Promo markdown 4",
    "MarkDown5": "Promo markdown 5",
    "Total_MarkDown": "Total promo markdown",
    "Week_sin": "Seasonality (week of year)",
    "Week_cos": "Seasonality (week of year, phase)",
    "Month_sin": "Seasonality (month)",
    "Month_cos": "Seasonality (month, phase)",
    "Weekly_Sales_Lag1": "Last week's sales",
    "Weekly_Sales_Lag2": "Sales two weeks ago",
    "Weekly_Sales_MA3": "3-week average sales",
    "Weekly_Sales_STD3": "3-week sales volatility",
    "IsHoliday_PrevWeek": "Holiday last week",
    "IsHoliday_NextWeek": "Holiday next week",
    "Rel_Fuel_Price": "Fuel price vs store norm",
    "Store_Avg_Sales": "Store average sales",
    "Dept_Avg_Sales": "Department average sales",
    "Weekly_Sales_Lag52": "Same week last year",
    "Seasonal_Ratio": "Last year's seasonal lift",
    "Seasonal_Naive_Forecast": "Seasonally-adjusted run-rate",
}

# Indicator columns that come from features.csv.
INDICATORS = ["Temperature", "Fuel_Price", "CPI", "Unemployment", *MARKDOWNS]

# Validation design ---------------------------------------------------------
# Hold-out: fit on everything before 2012-01-01, score Jan-Oct 2012.
HOLDOUT_START = "2012-01-01"
# Peak-season back-test: fit on data before 2011-11-01, score Nov 2011-Jan 2012
# (the Thanksgiving/Christmas peak, which the Jan-Oct hold-out never sees).
PEAK_FIT_END = "2011-11-01"
PEAK_START = "2011-11-01"
PEAK_END = "2012-01-31"

# Nominal coverage of the prediction interval that is shown in the UI.
INTERVAL_QUANTILES = (0.10, 0.90)

# Service limits -------------------------------------------------------------
MAX_HORIZON_WEEKS = 13
MAX_BULK_ROWS = 100_000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
