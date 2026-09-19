"""Train ForecastIQ and export everything the web service needs.

    python scripts/train_model.py            # full grid search (a few minutes)
    python scripts/train_model.py --fast     # tiny grid, for smoke tests / CI

What it does (and why it differs from v1 / the notebook):

1. Parses the Kaggle dates correctly (train.csv is dd/mm/yy).
2. Joins store-level indicators on Store+Date (v1 averaged them across stores).
3. Builds lag features with a one-week shift so a week never sees its own sales.
4. Tunes with a genuinely chronological CV (rows sorted by date).
5. Reports *honest* metrics on two out-of-sample windows and compares them with
   naive baselines - not MAPE, which explodes on near-zero/negative sales.
6. Fits a second, quantile model for the 80 % prediction interval and calibrates
   it on held-out data (conformalised quantile regression).
7. Refits on ALL data for the production model.
8. Writes compact reference data so the deployed service never needs train.csv.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    ARTIFACT_DIR, DATA_DIR, FEATURE_LABELS, FEATURES, HOLDOUT_START, INTERVAL_QUANTILES,
    MODEL_NAME, MODEL_VERSION, PEAK_END, PEAK_FIT_END, PEAK_START, REFERENCE_DIR,
)
from src.features import (  # noqa: E402
    Aggregates, HolidayCalendar, build_training_base, clean_indicators, load_raw, training_matrix,
)
from src.logging_config import logger  # noqa: E402


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def metrics(y, p) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    err = y - p
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "R2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        # Weighted absolute % error: sum|err| / sum|actual|. Robust to tiny/negative weeks.
        "WAPE": float(np.abs(err).sum() / np.abs(y).sum() * 100),
        "Bias_pct": float(err.sum() / np.abs(y).sum() * 100),
        "rows": int(len(y)),
    }


def wape(y, p) -> float:
    y = np.asarray(y, float)
    return float(np.abs(y - p).sum() / np.abs(y).sum() * 100)


def clean_json(o):
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


# --------------------------------------------------------------------------
# model fitting
# --------------------------------------------------------------------------
def fit_point(X, y, fast: bool):
    """Grid search with chronological CV. Rows must already be sorted by date."""
    base = xgb.XGBRegressor(
        objective="reg:squarederror", subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    grid = {"n_estimators": [150], "learning_rate": [0.1], "max_depth": [6]} if fast else {
        "n_estimators": [300, 600], "learning_rate": [0.05], "max_depth": [6, 8, 10],
    }
    gs = GridSearchCV(base, grid, scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=3), n_jobs=1)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


def fit_quantile(X, y, params: dict):
    q = xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=np.array(INTERVAL_QUANTILES),
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=-1,
        n_estimators=params["n_estimators"], learning_rate=params["learning_rate"],
        max_depth=min(params["max_depth"], 8),
    )
    q.fit(X, y)
    return q


def refit_point(X, y, params):
    m = xgb.XGBRegressor(
        objective="reg:squarederror", subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, n_jobs=-1, **params,
    )
    m.fit(X, y)
    return m


def interval_from(qmodel, X, adjust: float = 0.0, point=None):
    q = np.asarray(qmodel.predict(X))
    lo, hi = np.minimum(q[:, 0], q[:, 1]), np.maximum(q[:, 0], q[:, 1])
    w = np.maximum(hi - lo, 1.0)
    lo, hi = lo - adjust * w, hi + adjust * w
    if point is not None:
        lo, hi = np.minimum(lo, point), np.maximum(hi, point)
    return lo, hi


def conformal_adjust(y, lo, hi, alpha: float) -> float:
    """Normalised CQR: how much (in interval widths) to widen/narrow to hit 1-alpha."""
    w = np.maximum(hi - lo, 1.0)
    score = np.maximum(lo - y, y - hi) / w
    n = len(score)
    level = min(1.0, (1 - alpha) * (1 + 1 / n))
    return float(np.clip(np.quantile(score, level), -0.4, 5.0))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="tiny grid (smoke test)")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out-dir", default=str(ARTIFACT_DIR))
    args = ap.parse_args(argv)
    t0 = time.time()

    out = Path(args.out_dir)
    ref_dir = out / "reference"
    train, features, stores = load_raw(Path(args.data_dir))
    indicators = clean_indicators(features)
    calendar = HolidayCalendar.from_indicators(indicators)
    base = build_training_base(train, indicators, stores, calendar)
    logger.info("panel rows=%s series=%s weeks=%s..%s", len(base), train.groupby(["Store", "Dept"]).ngroups,
                base["Date"].min().date(), base["Date"].max().date())

    last_actual = base["Date"].max()
    hold_start = pd.Timestamp(HOLDOUT_START)

    # ---------------- hold-out phase (fit < 2012-01-01) -----------------------
    fit_rows = base[base["Date"] < hold_start]
    agg_h = Aggregates.fit(fit_rows)
    X_fit, y_fit, _ = training_matrix(fit_rows, agg_h)
    logger.info("hold-out fit rows=%s", len(X_fit))
    point_h, best = fit_point(X_fit, y_fit, args.fast)
    logger.info("best params %s", best)
    q_h = fit_quantile(X_fit, y_fit, best)

    val_rows = base[base["Date"] >= hold_start]
    X_val, y_val, val_kept = training_matrix(val_rows, agg_h)
    p_val = np.clip(point_h.predict(X_val), 0, None)
    m_model = metrics(y_val, p_val)
    logger.info("hold-out metrics %s", m_model)

    # baselines on exactly the same rows
    lag1 = val_kept["Lag1"].to_numpy()
    lag52 = val_kept["Lag52"].to_numpy()
    baselines = {
        "Last week (naive)": metrics(y_val, lag1),
        "Same week last year (seasonal naive)": metrics(y_val, np.where(np.isnan(lag52), lag1, lag52)),
        "3-week average": metrics(y_val, X_val["Weekly_Sales_MA3"]),
        "Department average": metrics(y_val, X_val["Dept_Avg_Sales"]),
    }

    # interval calibration: calibrate on first half of hold-out, report coverage on second half
    lo_raw, hi_raw = interval_from(q_h, X_val)
    dates_val = val_kept["Date"].to_numpy()
    mid = np.sort(np.unique(dates_val))[len(np.unique(dates_val)) // 2]
    cal, tst = dates_val < mid, dates_val >= mid
    alpha = INTERVAL_QUANTILES[0] + (1 - INTERVAL_QUANTILES[1])           # 0.20
    adj_half = conformal_adjust(y_val.to_numpy()[cal], lo_raw[cal], hi_raw[cal], alpha)
    lo_t, hi_t = interval_from(q_h, X_val[tst], adj_half, p_val[tst])
    yt = y_val.to_numpy()[tst]
    cov_raw = float(((yt >= lo_raw[tst]) & (yt <= hi_raw[tst])).mean() * 100)
    cov_cal = float(((yt >= lo_t) & (yt <= hi_t)).mean() * 100)
    adjust = conformal_adjust(y_val.to_numpy(), lo_raw, hi_raw, alpha)  # final: all hold-out
    logger.info("interval coverage raw=%.1f%% calibrated(out-of-sample half)=%.1f%% adjust=%.3f", cov_raw, cov_cal, adjust)

    # breakdowns (hold-out)
    vk = val_kept.assign(pred=p_val, err=np.abs(val_kept["Weekly_Sales"].to_numpy() - p_val),
                         ab=np.abs(val_kept["Weekly_Sales"].to_numpy()))
    by_type = {t: wape(g["Weekly_Sales"], g["pred"]) for t, g in vk.groupby("Type")}
    by_month = [
        {"month": pd.Timestamp(k).strftime("%b %Y"), "WAPE": wape(g["Weekly_Sales"], g["pred"]), "rows": int(len(g))}
        for k, g in vk.groupby(vk["Date"].dt.to_period("M").dt.to_timestamp())
    ]
    holiday = {
        "holiday_weeks": wape(vk.loc[vk["IsHoliday"] == 1, "Weekly_Sales"], vk.loc[vk["IsHoliday"] == 1, "pred"])
        if (vk["IsHoliday"] == 1).any() else None,
        "regular_weeks": wape(vk.loc[vk["IsHoliday"] == 0, "Weekly_Sales"], vk.loc[vk["IsHoliday"] == 0, "pred"]),
    }
    top_depts = vk.groupby("Dept")["ab"].sum().sort_values(ascending=False).head(10).index
    by_dept = [{"dept": int(d), "WAPE": wape(g["Weekly_Sales"], g["pred"]), "sales": float(g["Weekly_Sales"].sum())}
               for d, g in vk[vk["Dept"].isin(top_depts)].groupby("Dept")]
    by_dept.sort(key=lambda r: -r["sales"])

    # weekly total: actual vs forecast (hold-out) for the overview chart
    wk = vk.groupby("Date").agg(actual=("Weekly_Sales", "sum"), forecast=("pred", "sum")).reset_index()
    backtest_series = [{"date": d.strftime("%Y-%m-%d"), "actual": float(a), "forecast": float(f)}
                       for d, a, f in zip(wk["Date"], wk["actual"], wk["forecast"])]

    # ---------------- peak-season back-test ----------------------------------
    pk_fit = base[base["Date"] < pd.Timestamp(PEAK_FIT_END)]
    agg_p = Aggregates.fit(pk_fit)
    Xp, yp, _ = training_matrix(pk_fit, agg_p)
    pk_model = refit_point(Xp, yp, best)
    pk_rows = base[(base["Date"] >= pd.Timestamp(PEAK_START)) & (base["Date"] <= pd.Timestamp(PEAK_END))]
    Xpv, ypv, pk_kept = training_matrix(pk_rows, agg_p)
    pk_pred = np.clip(pk_model.predict(Xpv), 0, None)
    peak = {
        "window": f"{PEAK_START} to {PEAK_END}",
        "model": metrics(ypv, pk_pred),
        "last_week_naive": metrics(ypv, pk_kept["Lag1"]),
        "seasonal_naive": metrics(ypv, np.where(pk_kept["Lag52"].isna(), pk_kept["Lag1"], pk_kept["Lag52"])),
    }
    logger.info("peak-season back-test %s", peak["model"])

    # ---------------- production fit (all data) ------------------------------
    agg_all = Aggregates.fit(base)
    X_all, y_all, _ = training_matrix(base, agg_all)
    point_p = refit_point(X_all, y_all, best)
    q_p = fit_quantile(X_all, y_all, best)
    logger.info("production fit rows=%s", len(X_all))

    # ---------------- feature importance (gain) ------------------------------
    gain = point_p.get_booster().get_score(importance_type="total_gain")
    tot = sum(gain.values()) or 1.0
    importance = sorted(
        ({"feature": f, "label": FEATURE_LABELS.get(f, f), "importance": v / tot} for f, v in gain.items()),
        key=lambda r: -r["importance"],
    )[:15]

    # ---------------- persist ------------------------------------------------
    for phase, point, qm, agg in (("holdout", point_h, q_h, agg_h), ("production", point_p, q_p, agg_all)):
        d = out / phase
        d.mkdir(parents=True, exist_ok=True)
        # gzip keeps every file < 10 MB, so the repo needs no Git-LFS to be pushed to Hugging Face.
        for stem, model in (("point", point), ("quantile", qm)):
            (d / f"{stem}.ubj.gz").write_bytes(gzip.compress(bytes(model.get_booster().save_raw("ubj")), 9))
        (d / "aggregates.json").write_text(json.dumps(agg.to_dict()))

    ref_dir.mkdir(parents=True, exist_ok=True)
    train_out = train[["Store", "Dept", "Date", "Weekly_Sales"]].copy()
    train_out["Date"] = train_out["Date"].dt.strftime("%Y-%m-%d")
    train_out.to_csv(ref_dir / "sales.csv.gz", index=False)
    ind_out = indicators.copy()
    ind_out["Date"] = ind_out["Date"].dt.strftime("%Y-%m-%d")
    ind_out.to_csv(ref_dir / "indicators.csv.gz", index=False)
    stores.to_csv(ref_dir / "stores.csv", index=False)

    metadata = {
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "trained_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "algorithm": "XGBoost gradient-boosted trees (point + 10/90 % quantile models)",
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "best_parameters": best,
        "data": {
            "rows": int(len(base)),
            "series": int(train.groupby(["Store", "Dept"]).ngroups),
            "stores": int(train["Store"].nunique()),
            "departments": int(train["Dept"].nunique()),
            "first_week": base["Date"].min().strftime("%Y-%m-%d"),
            "last_actual_week": last_actual.strftime("%Y-%m-%d"),
            "indicator_last_week": indicators["Date"].max().strftime("%Y-%m-%d"),
        },
        "validation": {
            "holdout_window": f"{HOLDOUT_START} to {last_actual.strftime('%Y-%m-%d')}",
            "fit_window": f"{base['Date'].min().strftime('%Y-%m-%d')} to 2011-12-30",
            "design": "Chronological hold-out. Every lag feature uses only earlier weeks; store/department "
                      "averages and all model fitting use the fit window only.",
            "holdout": m_model,
            "baselines": baselines,
            "peak_season_backtest": peak,
            "interval": {
                "nominal_coverage": 80.0,
                "raw_quantile_coverage_pct": cov_raw,
                "calibrated_coverage_pct_out_of_sample": cov_cal,
                "calibration_adjust": adjust,
            },
            "by_store_type": by_type,
            "by_month": by_month,
            "holiday_vs_regular": holiday,
            "top_departments": by_dept,
            "backtest_weekly_totals": backtest_series,
        },
        "feature_importance": importance,
        "interval_adjust": adjust,
        "notes": [
            "Metrics are out-of-sample one-step-ahead forecasts: each week is predicted from the actual "
            "sales of the previous weeks.",
            "Multi-week forecasts are recursive (predictions are fed back as lags), so error grows with horizon.",
            "The research notebook's R2 0.998 / MAE 335 came from a 3-week rolling mean that included the week "
            "being predicted (target leakage) and is NOT reproducible on a leak-free pipeline.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_metadata.json").write_text(json.dumps(clean_json(metadata), indent=2, allow_nan=False))
    logger.info("done in %.0fs -> %s", time.time() - t0, out)
    return metadata


if __name__ == "__main__":
    main()
