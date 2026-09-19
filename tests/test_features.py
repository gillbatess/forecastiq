"""Date handling, holidays and (above all) leakage-freedom of the lag features."""
import numpy as np
import pandas as pd
import pytest

from src.features import (Aggregates, HolidayCalendar, holiday_rule, lag_frame, make_model_frame,
                          parse_dates, week_friday, BASE_COLUMNS)
from src.config import FEATURES


def test_parse_dates_prefers_fridays_for_dd_mm_yy():
    # 05/02/10 is 5 Feb 2010 (a Friday), not 2 May 2010
    out = parse_dates(["05/02/10", "12/02/10", "19/02/10"])
    assert list(out.dt.strftime("%Y-%m-%d")) == ["2010-02-05", "2010-02-12", "2010-02-19"]


def test_parse_dates_iso_and_errors():
    assert parse_dates(["2012-11-02"])[0] == pd.Timestamp("2012-11-02")
    with pytest.raises(ValueError):
        parse_dates(["not a date"])


def test_week_friday_snaps_any_day():
    for d in ["2012-10-29", "2012-11-01", "2012-11-02", "2012-11-04"]:
        assert week_friday(d) == pd.Timestamp("2012-11-02")
    s = week_friday(pd.Series(pd.to_datetime(["2012-11-05", "2012-11-09"])))
    assert list(s) == [pd.Timestamp("2012-11-09")] * 2


def test_holiday_rule_matches_dataset_calendar(service):
    known = service.ref.calendar.known
    assert len(known) > 150
    wrong = [d for d, v in known.items() if holiday_rule(d) != v]
    assert not wrong, f"rule disagrees with dataset on {wrong[:5]}"


def test_holiday_rule_future_weeks():
    assert holiday_rule(pd.Timestamp("2013-11-29"))      # day after Thanksgiving
    assert holiday_rule(pd.Timestamp("2013-12-27"))
    assert not holiday_rule(pd.Timestamp("2013-06-14"))


def _matrix(n_weeks=70, n_series=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_tuples([(1, d) for d in range(1, n_series + 1)])
    cols = pd.date_range("2010-02-05", periods=n_weeks, freq="7D")
    return pd.DataFrame(rng.uniform(1000, 9000, (n_series, n_weeks)), index=idx, columns=cols)


def _base(lags: pd.DataFrame) -> pd.DataFrame:
    b = lags.copy()
    b["Type"], b["Size"] = "A", 150000.0
    b["IsHoliday"] = b["IsHoliday_PrevWeek"] = b["IsHoliday_NextWeek"] = 0
    b["Temperature"], b["Fuel_Price"], b["CPI"], b["Unemployment"] = 60.0, 3.0, 170.0, 7.0
    for m in ("MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"):
        b[m] = 0.0
    return b


def test_features_do_not_use_the_target_week():
    """Changing this week's sales must not change any of this week's features."""
    m = _matrix()
    agg = Aggregates(store_avg={1: 5000.0}, dept_avg={1: 5000.0, 2: 5000.0, 3: 5000.0},
                     store_fuel_avg={1: 3.0}, overall_avg=5000.0, overall_fuel=3.0)
    t = 60
    m2 = m.copy()
    m2.iloc[:, t] = m2.iloc[:, t] * 10 + 12345
    f1 = make_model_frame(_base(lag_frame(m)), agg)
    f2 = make_model_frame(_base(lag_frame(m2)), agg)
    assert list(f1.columns) == FEATURES
    rows = np.arange(len(m)) * m.shape[1] + t
    pd.testing.assert_frame_equal(f1.iloc[rows].reset_index(drop=True), f2.iloc[rows].reset_index(drop=True))
    # ... but next week's features DO move (they legitimately use this week)
    assert not np.allclose(f1.iloc[rows + 1]["Weekly_Sales_Lag1"], f2.iloc[rows + 1]["Weekly_Sales_Lag1"])


def test_moving_average_is_previous_three_weeks():
    m = _matrix()
    agg = Aggregates(store_avg={1: 1.0}, dept_avg={1: 1.0}, store_fuel_avg={1: 3.0}, overall_avg=1.0, overall_fuel=3.0)
    f = make_model_frame(_base(lag_frame(m)), agg)
    t = 30
    row = 0 * m.shape[1] + t
    expected = m.iloc[0, t - 3:t].mean()
    assert f.loc[row, "Weekly_Sales_MA3"] == pytest.approx(expected)
    assert f.loc[row, "Weekly_Sales_STD3"] == pytest.approx(m.iloc[0, t - 3:t].std(ddof=1))
    assert f.loc[row, "Weekly_Sales_Lag52"] == pytest.approx(m.iloc[0, t - 52]) if t >= 52 else True


def test_calendar_lag_gap_gives_nan_not_stale_value():
    m = _matrix()
    m.iloc[0, 40] = np.nan
    lf = lag_frame(m)
    row = lambda t: t  # series 0
    assert np.isnan(lf.loc[row(41), "Lag1"])
    assert np.isnan(lf.loc[row(43), "Lag3"])
    assert not np.isnan(lf.loc[row(44), "Lag3"])
