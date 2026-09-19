"""Indicator auto-fill and forecasting behaviour."""
import numpy as np
import pandas as pd
import pytest

from src.forecaster import ForecastError


def test_indicators_are_filled_from_history(service):
    snap = service.ref.indicator_snapshot(1, pd.Timestamp("2011-11-25"))
    assert snap["is_holiday"]["value"] is True             # Thanksgiving week
    assert snap["temperature"]["source"] == "historical"
    assert 30 < snap["temperature"]["value"] < 90
    assert snap["fuel_price"]["value"] > 2


def test_indicators_beyond_dataset_are_estimated(service):
    snap = service.ref.indicator_snapshot(1, pd.Timestamp("2014-03-07"))
    assert snap["temperature"]["source"] == "estimated"
    assert snap["markdowns"]["value"] == [0, 0, 0, 0, 0]


def test_indicators_differ_by_store(service):
    a = service.ref.indicator_snapshot(1, pd.Timestamp("2012-01-06"))["temperature"]["value"]
    b = service.ref.indicator_snapshot(20, pd.Timestamp("2012-01-06"))["temperature"]["value"]
    assert a != b


def test_backtest_week_returns_actual_and_interval(service):
    r = service.forecast_series(1, 1, "2012-08-03", 1)
    w = r["weeks"][0]
    assert w["actual"] is not None and w["forecast"] > 0
    assert w["low"] <= w["forecast"] <= w["high"]
    assert w["mode"] == "backtest"


def test_future_horizon_recurses_and_widens(service):
    r = service.forecast_series(1, 1, "2012-11-02", 8)
    assert len(r["weeks"]) == 8
    assert all(w["mode"] == "forecast" and w["actual"] is None for w in r["weeks"])
    widths = [w["high"] - w["low"] for w in r["weeks"]]
    assert widths[-1] > widths[0]
    assert all(np.isfinite(w["forecast"]) for w in r["weeks"])


def test_overrides_move_the_forecast_but_not_wildly(service):
    base = service.forecast_series(1, 1, "2012-08-03", 1)["weeks"][0]["forecast"]
    hot = service.forecast_series(1, 1, "2012-08-03", 1, {"markdown1": 50000})["weeks"][0]["forecast"]
    assert hot > 0 and abs(hot / base - 1) < 1.0


@pytest.mark.parametrize("args,msg", [
    ((99, 1, "2012-08-03", 1), "Unknown store"),
    ((1, 15, "2012-08-03", 1), "no sales history"),
    ((1, 1, "2012-08-03", 99), "Horizon"),
    ((1, 1, "1999-01-01", 1), "outside the supported range"),
    ((1, 1, "2099-01-01", 1), "outside the supported range"),
    ((1, 1, "garbage", 1), "Invalid date"),
])
def test_errors_are_friendly(service, args, msg):
    with pytest.raises(ForecastError, match=msg):
        service.forecast_series(*args)


def test_bulk_rows(service):
    df = pd.DataFrame({"Store": [1, 1, 99], "Dept": [1, 1, 1],
                       "Date": ["2012-08-03", "2012-11-09", "2012-08-03"]})
    out = service.forecast_rows(df)
    assert len(out) == 3
    assert out.loc[0, "Forecast_Type"] == "Out-of-sample back-test"
    assert out.loc[1, "Forecast_Type"] == "Forecast"
    assert out.loc[2, "Status"] != "OK" and np.isnan(out.loc[2, "Predicted_Weekly_Sales"])


def test_bulk_requires_columns(service):
    with pytest.raises(ForecastError, match="Missing required column"):
        service.forecast_rows(pd.DataFrame({"Store": [1]}))


def test_backtest_accuracy_is_sane(service):
    """Guards against a regression to broken features (hold-out WAPE ~8.4%; naive ~10.9%)."""
    v = service.metadata["validation"]
    assert v["holdout"]["WAPE"] < 12
    assert v["holdout"]["WAPE"] < v["baselines"]["Last week (naive)"]["WAPE"]
    assert 75 <= v["interval"]["calibrated_coverage_pct_out_of_sample"] <= 92
    assert v["holdout"]["R2"] < 0.995        # 0.998 would mean target leakage (the v1 notebook bug)


def test_bulk_accepts_lowercase_headers(service):
    df = pd.DataFrame({"store": [1], "dept": [1], "date": ["2012-08-03"]})
    out = service.forecast_rows(df)
    assert list(out.columns[:3]) == ["Store", "Dept", "Date"]
    assert out.loc[0, "Status"] == "OK"
