"""Feature engineering shared by training and inference.

Design rule: there is exactly ONE function that turns "raw ingredients" into the
model's feature matrix (`make_model_frame`). Training and serving both call it,
so the two can never disagree (train/serve skew was one of the bugs in v1).

Leakage rule: every lag / rolling feature for week *t* is computed only from
weeks *t-1, t-2, t-3 (and t-52)*. The week being predicted never contributes to
its own features. (The research notebook computed the 3-week mean/std WITHOUT a
one-week shift, which leaks the target and produces the unrealistic R2 = 0.998.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import FEATURES, INDICATORS, MARKDOWNS

# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------
_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d",
    "%d/%m/%y", "%d/%m/%Y", "%d-%m-%Y",
    "%m/%d/%Y", "%m/%d/%y",
)


def parse_dates(values: Iterable) -> pd.Series:
    """Parse a column of dates *deterministically*.

    Why this exists: the Kaggle ``train.csv`` stores dates as ``dd/mm/yy``
    (``05/02/10`` = 5 Feb 2010). ``pd.to_datetime`` silently guesses month-first
    for the ones that look valid that way (-> 10 May 2010!) and day-first for the
    rest, which scrambles the whole time axis. We try explicit formats and, when
    several parse cleanly, prefer the one that lands on Fridays (Walmart weeks
    end on Friday). ISO ``YYYY-MM-DD`` always wins ties.
    """
    s = pd.Series(list(values) if not isinstance(values, pd.Series) else values.values)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.astype("datetime64[ns]").reset_index(drop=True)
    s = s.astype(str).str.strip()
    best = None
    for fmt in _DATE_FORMATS:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if parsed.isna().any():
            continue
        score = float((parsed.dt.weekday == 4).mean())
        if best is None or score > best[0] + 1e-9:
            best = (score, parsed)
    if best is None:
        bad = s[pd.to_datetime(s, format="%Y-%m-%d", errors="coerce").isna()].head(3).tolist()
        raise ValueError(
            f"Could not read dates such as {bad}. Use ISO format YYYY-MM-DD (e.g. 2012-11-02)."
        )
    return best[1].astype("datetime64[ns]").reset_index(drop=True)


def week_friday(dates: pd.Series | pd.Timestamp | str | date):
    """Snap any date to the Friday of its Mon-Sun week (Walmart's week key)."""
    if isinstance(dates, pd.Series):
        d = pd.to_datetime(dates).dt.normalize()
        return d + pd.to_timedelta(4 - d.dt.weekday, unit="D")
    d = pd.Timestamp(dates).normalize()
    return d + pd.Timedelta(days=4 - d.weekday())


# --------------------------------------------------------------------------
# Holidays
# --------------------------------------------------------------------------
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def holiday_rule(friday: pd.Timestamp) -> bool:
    """Rule-based holiday-week flag, used only outside the dataset's calendar.

    Reproduces the Kaggle definition (verified against 2010-2013 in the tests):
    week ending Friday after the Super Bowl, Labor Day, Thanksgiving, and the
    Christmas week (Friday between 27 Dec and 2 Jan).
    """
    f = pd.Timestamp(friday).date()
    y = f.year
    # Super Bowl: first Sunday of Feb (second Sunday from 2022 on)
    sb = _nth_weekday(y, 2, 6, 2 if y >= 2022 else 1)
    if f == sb + timedelta(days=5):
        return True
    if f == _nth_weekday(y, 9, 0, 1) + timedelta(days=4):      # Labor Day (Mon) -> Fri
        return True
    if f == _nth_weekday(y, 11, 3, 4) + timedelta(days=1):     # Thanksgiving (Thu) -> Fri
        return True
    if (f.month == 12 and f.day >= 27) or (f.month == 1 and f.day <= 2):
        return True
    return False


class HolidayCalendar:
    """Holiday-week lookup: dataset calendar first, rule as fallback."""

    def __init__(self, known: dict[pd.Timestamp, bool] | None = None):
        self.known = known or {}

    @classmethod
    def from_indicators(cls, ind: pd.DataFrame) -> "HolidayCalendar":
        cal = ind.drop_duplicates("Date").set_index("Date")["IsHoliday"].astype(bool)
        return cls({pd.Timestamp(k): bool(v) for k, v in cal.items()})

    def is_holiday(self, friday) -> bool:
        f = pd.Timestamp(friday)
        if f in self.known:
            return self.known[f]
        return holiday_rule(f)

    def series(self, dates: pd.Series) -> pd.Series:
        """Vectorised lookup (computed once per unique date, not per row)."""
        lookup = {d: int(self.is_holiday(d)) for d in pd.Series(dates).unique()}
        return pd.Series(dates).map(lookup).astype(int).set_axis(dates.index)


# --------------------------------------------------------------------------
# Raw data loading / cleaning
# --------------------------------------------------------------------------
def load_raw(data_dir: Path):
    """Read the four Kaggle files with correct dtypes and dates."""
    train = pd.read_csv(data_dir / "train.csv")
    features = pd.read_csv(data_dir / "features.csv")
    stores = pd.read_csv(data_dir / "stores.csv")
    train["Date"] = parse_dates(train["Date"])
    features["Date"] = parse_dates(features["Date"])
    train["IsHoliday"] = train["IsHoliday"].astype(bool)
    features["IsHoliday"] = features["IsHoliday"].astype(bool)
    return train, features, stores


def clean_indicators(features: pd.DataFrame) -> pd.DataFrame:
    """One clean row per (Store, Date) with no missing values.

    * MarkDown NaN -> 0  ("no promotion recorded"; the columns are blank before
      Nov 2011 and sparse afterwards - interpolating them, as the notebook did,
      invents promotions that never ran).
    * CPI / Unemployment are *store level* (regional CPI ranges 126-229) so they
      are filled per store: interpolate inside the series, carry the last known
      value forward at the end. Rows that needed a fill are flagged
      ``Macro_Estimated`` so the UI can say "estimated" instead of "historical".
    """
    f = features.copy()
    f["Date"] = pd.to_datetime(f["Date"])
    f = f.sort_values(["Store", "Date"]).reset_index(drop=True)
    f[MARKDOWNS] = f[MARKDOWNS].fillna(0.0)
    f["Macro_Estimated"] = f["CPI"].isna() | f["Unemployment"].isna()
    g = f.groupby("Store", sort=False)
    for col in ("CPI", "Unemployment", "Temperature", "Fuel_Price"):
        f[col] = g[col].transform(lambda x: x.interpolate(limit_direction="both").ffill().bfill())
    cols = ["Store", "Date", *INDICATORS, "IsHoliday", "Macro_Estimated"]
    return f[cols]


# --------------------------------------------------------------------------
# Sales matrix + lags
# --------------------------------------------------------------------------
def weekly_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="7D")


def sales_matrix(train: pd.DataFrame) -> pd.DataFrame:
    """(Store, Dept) x week matrix of Weekly_Sales on a regular Friday grid."""
    weeks = weekly_grid(train["Date"].min(), train["Date"].max())
    m = train.pivot_table(index=["Store", "Dept"], columns="Date", values="Weekly_Sales", aggfunc="sum")
    return m.reindex(columns=weeks)


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    if k < a.shape[1]:
        out[:, k:] = a[:, :-k]
    return out


def lag_frame(m: pd.DataFrame) -> pd.DataFrame:
    """Long frame (Store, Dept, Date, Weekly_Sales, Lag1..Lag3, Lag52) from a matrix.

    Lags are *calendar* lags (t-1 week, t-2 weeks ...), so a series with a gap
    gets NaN rather than silently using a stale value like a row-shift would.
    """
    a = m.to_numpy(dtype=float)
    s, w = a.shape
    stores = np.repeat(m.index.get_level_values(0).to_numpy(), w)
    depts = np.repeat(m.index.get_level_values(1).to_numpy(), w)
    dates = np.tile(m.columns.to_numpy(), s)
    return pd.DataFrame({
        "Store": stores.astype(int), "Dept": depts.astype(int), "Date": dates,
        "Weekly_Sales": a.ravel(),
        "Lag1": _shift(a, 1).ravel(), "Lag2": _shift(a, 2).ravel(),
        "Lag3": _shift(a, 3).ravel(), "Lag52": _shift(a, 52).ravel(),
        "Lag53": _shift(a, 53).ravel(), "Lag54": _shift(a, 54).ravel(),
        "Lag55": _shift(a, 55).ravel(),
    })


# --------------------------------------------------------------------------
# Aggregates (fit on the training window only -> no look-ahead)
# --------------------------------------------------------------------------
@dataclass
class Aggregates:
    store_avg: dict[int, float] = field(default_factory=dict)
    dept_avg: dict[int, float] = field(default_factory=dict)
    store_fuel_avg: dict[int, float] = field(default_factory=dict)
    overall_avg: float = 0.0
    overall_fuel: float = 3.0

    @classmethod
    def fit(cls, panel: pd.DataFrame) -> "Aggregates":
        """panel needs Store, Dept, Weekly_Sales, Fuel_Price."""
        return cls(
            store_avg={int(k): float(v) for k, v in panel.groupby("Store")["Weekly_Sales"].mean().items()},
            dept_avg={int(k): float(v) for k, v in panel.groupby("Dept")["Weekly_Sales"].mean().items()},
            store_fuel_avg={int(k): float(v) for k, v in panel.groupby("Store")["Fuel_Price"].mean().items()},
            overall_avg=float(panel["Weekly_Sales"].mean()),
            overall_fuel=float(panel["Fuel_Price"].mean()),
        )

    def to_dict(self) -> dict:
        return {
            "store_avg": {str(k): v for k, v in self.store_avg.items()},
            "dept_avg": {str(k): v for k, v in self.dept_avg.items()},
            "store_fuel_avg": {str(k): v for k, v in self.store_fuel_avg.items()},
            "overall_avg": self.overall_avg,
            "overall_fuel": self.overall_fuel,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Aggregates":
        return cls(
            store_avg={int(k): float(v) for k, v in d["store_avg"].items()},
            dept_avg={int(k): float(v) for k, v in d["dept_avg"].items()},
            store_fuel_avg={int(k): float(v) for k, v in d["store_fuel_avg"].items()},
            overall_avg=float(d["overall_avg"]),
            overall_fuel=float(d["overall_fuel"]),
        )


# --------------------------------------------------------------------------
# The single feature assembly used by training AND serving
# --------------------------------------------------------------------------
BASE_COLUMNS = [
    "Store", "Dept", "Date", "Type", "Size",
    "IsHoliday", "IsHoliday_PrevWeek", "IsHoliday_NextWeek",
    "Temperature", "Fuel_Price", "CPI", "Unemployment", *MARKDOWNS,
    "Lag1", "Lag2", "Lag3", "Lag52", "Lag53", "Lag54", "Lag55",
]


def make_model_frame(base: pd.DataFrame, agg: Aggregates) -> pd.DataFrame:
    """Turn raw ingredients (see BASE_COLUMNS) into the FEATURES matrix."""
    missing = [c for c in BASE_COLUMNS if c not in base.columns]
    if missing:
        raise ValueError(f"make_model_frame missing columns: {missing}")
    b = base.reset_index(drop=True)
    dates = pd.to_datetime(b["Date"])
    week = dates.dt.isocalendar().week.astype(int).to_numpy()
    month = dates.dt.month.to_numpy()

    out = pd.DataFrame(index=b.index)
    out["IsHoliday"] = b["IsHoliday"].astype(int)
    for c in ("Temperature", "Fuel_Price", "CPI", "Unemployment", "Size"):
        out[c] = b[c].astype(float)
    for t in "ABC":
        out[f"StoreType_{t}"] = (b["Type"] == t).astype(int)
    for c in MARKDOWNS:
        out[c] = b[c].astype(float)
    out["Total_MarkDown"] = out[MARKDOWNS].sum(axis=1)
    out["Week_sin"] = np.sin(2 * np.pi * week / 52)
    out["Week_cos"] = np.cos(2 * np.pi * week / 52)
    out["Month_sin"] = np.sin(2 * np.pi * month / 12)
    out["Month_cos"] = np.cos(2 * np.pi * month / 12)

    lags = np.column_stack([b["Lag1"], b["Lag2"], b["Lag3"]]).astype(float)
    out["Weekly_Sales_Lag1"] = lags[:, 0]
    out["Weekly_Sales_Lag2"] = lags[:, 1]
    out["Weekly_Sales_MA3"] = lags.mean(axis=1)                 # NaN if any lag missing
    out["Weekly_Sales_STD3"] = lags.std(axis=1, ddof=1)         # same definition as pandas rolling().std()
    out["IsHoliday_PrevWeek"] = b["IsHoliday_PrevWeek"].astype(int)
    out["IsHoliday_NextWeek"] = b["IsHoliday_NextWeek"].astype(int)

    fuel_avg = b["Store"].map(agg.store_fuel_avg).fillna(agg.overall_fuel).astype(float)
    out["Rel_Fuel_Price"] = out["Fuel_Price"] / fuel_avg
    out["Store_Avg_Sales"] = b["Store"].map(agg.store_avg).fillna(agg.overall_avg).astype(float)
    dept_fallback = float(np.mean(list(agg.dept_avg.values()))) if agg.dept_avg else agg.overall_avg
    out["Dept_Avg_Sales"] = b["Dept"].map(agg.dept_avg).fillna(dept_fallback).astype(float)
    out["Weekly_Sales_Lag52"] = b["Lag52"].astype(float)

    # Seasonal lift: how much last year's sales rose from the 3 weeks before into
    # this week of the year, applied to this year's run-rate.
    prev = np.column_stack([b["Lag53"], b["Lag54"], b["Lag55"]]).astype(float).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(prev > 0, out["Weekly_Sales_Lag52"].to_numpy() / prev, np.nan)
    ratio = np.clip(ratio, 0.2, 5.0)
    out["Seasonal_Ratio"] = ratio
    out["Seasonal_Naive_Forecast"] = out["Weekly_Sales_MA3"].to_numpy() * ratio
    return out[FEATURES].astype(float)


# --------------------------------------------------------------------------
# Training panel
# --------------------------------------------------------------------------
def build_training_base(train: pd.DataFrame, indicators: pd.DataFrame, stores: pd.DataFrame,
                        calendar: HolidayCalendar) -> pd.DataFrame:
    """Every observed (Store, Dept, Week) with lags, indicators and holiday flags."""
    m = sales_matrix(train)
    long = lag_frame(m)
    long = long[long["Weekly_Sales"].notna()].copy()
    long["Date"] = pd.to_datetime(long["Date"])

    ind = indicators.drop(columns=["IsHoliday"])
    long = long.merge(ind, on=["Store", "Date"], how="left").merge(stores, on="Store", how="left")

    long["IsHoliday"] = calendar.series(long["Date"]).to_numpy()
    long["IsHoliday_PrevWeek"] = calendar.series(long["Date"] - pd.Timedelta(days=7)).to_numpy()
    long["IsHoliday_NextWeek"] = calendar.series(long["Date"] + pd.Timedelta(days=7)).to_numpy()
    return long.sort_values(["Date", "Store", "Dept"]).reset_index(drop=True)


def training_matrix(base: pd.DataFrame, agg: Aggregates):
    """(X, y, kept_rows) - rows need Lag1..Lag3 so MA3/STD3 exist."""
    keep = base[["Lag1", "Lag2", "Lag3"]].notna().all(axis=1) & base["Weekly_Sales"].notna()
    kept = base.loc[keep].reset_index(drop=True)
    X = make_model_frame(kept, agg)
    y = kept["Weekly_Sales"].astype(float)
    return X, y, kept
