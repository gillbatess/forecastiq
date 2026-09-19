"""Inference engine: turns (store, dept, week) into a forecast with an interval.

How a forecast is produced
--------------------------
* Weeks that lie inside the sales history are scored one-step-ahead from the
  *actual* previous weeks (a true back-test). If the week is inside the hold-out
  window (2012) the *hold-out* model is used, so the number is genuinely
  out-of-sample; weeks before that are in-sample and flagged as such.
* Weeks after the last actual week are forecast recursively with the
  *production* model: week t+1 is predicted from actuals, t+2 from actuals and
  the t+1 forecast, and so on. Intervals widen with sqrt(weeks ahead).
* Economic indicators are looked up automatically from the historical dataset by
  store and week (`ReferenceData.indicators_for`); the caller may override any of
  them (what-if scenarios).
"""
from __future__ import annotations

import gzip
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (
    ARTIFACT_DIR, FEATURE_LABELS, FEATURES, HOLDOUT_START, MARKDOWNS, MAX_HORIZON_WEEKS,
)
from src.features import Aggregates, make_model_frame, week_friday
from src.logging_config import logger
from src.reference import ReferenceData

OVERRIDE_COLS = ["Temperature", "Fuel_Price", "CPI", "Unemployment", *MARKDOWNS, "IsHoliday"]
# API/CSV spelling -> internal column
OVERRIDE_ALIASES = {
    "temperature": "Temperature", "fuel_price": "Fuel_Price", "cpi": "CPI", "unemployment": "Unemployment",
    "markdown1": "MarkDown1", "markdown2": "MarkDown2", "markdown3": "MarkDown3", "markdown4": "MarkDown4",
    "markdown5": "MarkDown5", "is_holiday": "IsHoliday",
}


@dataclass
class Phase:
    name: str
    point: xgb.Booster
    quant: xgb.Booster
    agg: Aggregates


@dataclass
class Pred:
    point: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    X: pd.DataFrame


class ForecastError(ValueError):
    """A problem the caller can fix (bad store, date out of range ...)."""


class ForecastService:
    def __init__(self, artifact_dir: Path | None = None):
        d = Path(artifact_dir or ARTIFACT_DIR)
        meta_path = d / "model_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found. Run `python scripts/train_model.py` first.")
        self.metadata = json.loads(meta_path.read_text())
        self.adjust = float(self.metadata.get("interval_adjust", 0.0))
        self.ref = ReferenceData(d / "reference")
        self.holdout = self._load_phase(d, "holdout")
        self.production = self._load_phase(d, "production")
        self._lock = threading.Lock()          # xgboost Booster.predict is not thread-safe
        self._overview: dict | None = None
        self.holdout_start = pd.Timestamp(HOLDOUT_START)
        logger.info("ForecastService ready: %d series, history %s..%s",
                    len(self.ref.series_keys), self.ref.week0.date(), self.ref.last_actual.date())

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _load_phase(d: Path, name: str) -> Phase:
        def load(stem: str) -> xgb.Booster:
            b = xgb.Booster()
            b.load_model(bytearray(gzip.decompress((d / name / f"{stem}.ubj.gz").read_bytes())))
            return b

        pt, q = load("point"), load("quantile")
        agg = Aggregates.from_dict(json.loads((d / name / "aggregates.json").read_text()))
        return Phase(name, pt, q, agg)

    # --------------------------------------------------------- feature building
    def _recent(self, E: np.ndarray, erows: np.ndarray, c: int, window: int = 13):
        """Last three observed weekly values before column ``c`` (most recent first).

        Normally these are exactly weeks c-1, c-2, c-3. If a series has a gap (a
        week with no sales row) we fall back to the last three *observed* values
        within ``window`` weeks and report how stale Lag1 is (``gap`` weeks).
        """
        n = len(erows)
        nan = np.full(n, np.nan)
        lo = max(0, c - window)
        if lo >= c:
            return nan, nan.copy(), nan.copy(), np.full(n, -1)
        blk = E[erows, lo:c][:, ::-1]
        valid = ~np.isnan(blk)
        csum = valid.cumsum(axis=1)
        out, gap = [], None
        for k in (1, 2, 3):
            m = valid & (csum == k)
            has = m.any(axis=1)
            idx = m.argmax(axis=1)
            out.append(np.where(has, blk[np.arange(n), idx], np.nan))
            if k == 1:
                gap = np.where(has, idx, -1)
        return out[0], out[1], out[2], gap

    def _build_base(self, c: int, sidx: np.ndarray, E: np.ndarray, erows: np.ndarray,
                    ov: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
        ref = self.ref
        n = len(sidx)
        date = ref.week_date(c)
        stores = ref.series_store[sidx]
        depts = ref.series_dept[sidx]
        ind = ref.indicators_for(date).reindex(stores).reset_index(drop=True)

        overridden = {col: np.zeros(n, dtype=bool) for col in OVERRIDE_COLS}
        if ov is not None:
            for col in ov.columns:
                vals = ov[col].to_numpy()
                mask = ~pd.isna(vals)
                if mask.any():
                    if col == "IsHoliday":
                        ind.loc[mask, col] = vals[mask].astype(bool)
                    else:
                        ind.loc[mask, col] = vals[mask].astype(float)
                    overridden[col] = mask

        lag1, lag2, lag3, gap = self._recent(E, erows, c)

        def lag(k):
            return E[erows, c - k] if c - k >= 0 else np.full(n, np.nan)

        base = pd.DataFrame({
            "Store": stores, "Dept": depts, "Date": date,
            "Type": [ref.store_type[s] for s in stores],
            "Size": [ref.store_size[s] for s in stores],
            "IsHoliday": ind["IsHoliday"].astype(bool).to_numpy(),
            "IsHoliday_PrevWeek": int(ref.calendar.is_holiday(date - pd.Timedelta(days=7))),
            "IsHoliday_NextWeek": int(ref.calendar.is_holiday(date + pd.Timedelta(days=7))),
            "Temperature": ind["Temperature"].to_numpy(), "Fuel_Price": ind["Fuel_Price"].to_numpy(),
            "CPI": ind["CPI"].to_numpy(), "Unemployment": ind["Unemployment"].to_numpy(),
            **{m: ind[m].to_numpy() for m in MARKDOWNS},
            "Lag1": lag1, "Lag2": lag2, "Lag3": lag3,
            "Lag52": lag(52), "Lag53": lag(53), "Lag54": lag(54), "Lag55": lag(55),
        })
        meta = {
            "gap": gap, "overridden": overridden,
            "est_macro": ind["est_macro"].to_numpy(dtype=bool), "est_other": ind["est_other"].to_numpy(dtype=bool),
        }
        return base, meta

    # -------------------------------------------------------------- prediction
    def _predict(self, base: pd.DataFrame, phase: Phase, hstep: int, contribs: bool = False):
        X = make_model_frame(base, phase.agg)
        ok = X["Weekly_Sales_MA3"].notna().to_numpy()
        n = len(X)
        point = np.full(n, np.nan)
        lo = np.full(n, np.nan)
        hi = np.full(n, np.nan)
        contrib = None
        if ok.any():
            Xo = X.loc[ok]
            with self._lock:
                dm = xgb.DMatrix(Xo, feature_names=FEATURES)
                p = np.clip(phase.point.predict(dm), 0, None)
                q = np.asarray(phase.quant.predict(dm))
                if contribs:
                    contrib = np.full((n, len(FEATURES) + 1), np.nan)
                    contrib[ok] = phase.point.predict(dm, pred_contribs=True)
            l, h = np.minimum(q[:, 0], q[:, 1]), np.maximum(q[:, 0], q[:, 1])
            w = np.maximum(h - l, 1.0)
            l, h = l - self.adjust * w, h + self.adjust * w
            l, h = np.minimum(l, p), np.maximum(h, p)
            f = float(np.sqrt(max(1, hstep)))                    # uncertainty grows with horizon
            l, h = p - (p - l) * f, p + (h - p) * f
            point[ok], lo[ok], hi[ok] = p, np.maximum(l, 0.0), h
        return Pred(point, lo, hi, X), contrib

    def _recurse(self, sidx: np.ndarray, c_end: int, ov_at=None, detail: bool = False):
        """Recursive multi-week forecast for the given series, from last actual to ``c_end``."""
        ref = self.ref
        W = ref.n_hist
        n_ext = max(0, c_end - W + 1)
        E = np.full((len(sidx), W + n_ext), np.nan)
        E[:, :W] = ref.A[sidx]
        erows = np.arange(len(sidx))
        out = {}
        for c in range(W, c_end + 1):
            ov = ov_at(c) if ov_at else None
            base, meta = self._build_base(c, sidx, E, erows, ov)
            pred, contrib = self._predict(base, self.production, hstep=c - W + 1, contribs=detail)
            E[:, c] = pred.point                                    # NaN stays NaN -> flagged downstream
            out[c] = (pred, base, meta, contrib)
        return out

    # ---------------------------------------------------------------- engine
    def _engine(self, rows: pd.DataFrame, detail: bool = False) -> tuple[pd.DataFrame, dict]:
        """rows: columns ``sidx`` (-1 = unknown), ``c`` (week column) [+ override columns].

        Returns (results, details). ``results`` is aligned to ``rows``.
        """
        ref = self.ref
        W = ref.n_hist
        n = len(rows)
        res = pd.DataFrame({
            "point": np.full(n, np.nan), "lo": np.full(n, np.nan), "hi": np.full(n, np.nan),
            "mode": [""] * n, "hstep": np.zeros(n, dtype=int), "status": ["ok"] * n,
            "actual": np.full(n, np.nan), "gap": np.zeros(n, dtype=int),
        }, index=rows.index)
        details: dict = {}
        ov_cols = [c for c in OVERRIDE_COLS if c in rows.columns]
        lo_c, hi_c = 3, W - 1 + 104                     # need 3 weeks of history; 2 years beyond the data
        sidx = rows["sidx"].to_numpy()
        cc = rows["c"].to_numpy()

        res.loc[sidx < 0, "status"] = "unknown_series"
        res.loc[(sidx >= 0) & ((cc < lo_c) | (cc > hi_c)), "status"] = "date_out_of_range"
        live = (res["status"] == "ok").to_numpy()
        pos = np.arange(n)

        # ---- 1) back-test weeks: one-step-ahead from actuals ------------------
        bt = live & (cc < W)
        for c in np.unique(cc[bt]):
            sel = pos[bt & (cc == c)]
            s = sidx[sel]
            ov = rows.iloc[sel][ov_cols].reset_index(drop=True) if ov_cols else None
            base, meta = self._build_base(int(c), s, ref.A, s, ov)
            pred, contrib = self._predict(base, self.holdout, hstep=1, contribs=detail)
            res.iloc[sel, res.columns.get_loc("point")] = pred.point
            res.iloc[sel, res.columns.get_loc("lo")] = pred.lo
            res.iloc[sel, res.columns.get_loc("hi")] = pred.hi
            res.iloc[sel, res.columns.get_loc("hstep")] = 1
            res.iloc[sel, res.columns.get_loc("gap")] = meta["gap"]
            is_oos = ref.week_date(int(c)) >= self.holdout_start
            res.iloc[sel, res.columns.get_loc("mode")] = "backtest" if is_oos else "in_sample"
            res.iloc[sel, res.columns.get_loc("actual")] = ref.A[s, int(c)]
            if detail:
                self._collect_detail(details, sel, np.arange(len(sel)), pred, base, meta, contrib)

        # ---- 2) forecast weeks: recursion with the production model ----------
        fut = live & (cc >= W)
        if fut.any():
            fsel = pos[fut]
            uniq, local = np.unique(sidx[fsel], return_inverse=True)
            c_end = int(cc[fsel].max())
            by_c: dict[int, np.ndarray] = {int(c): fsel[cc[fsel] == c] for c in np.unique(cc[fsel])}
            local_of = dict(zip(fsel, local))

            def ov_at(c):
                if not ov_cols or c not in by_c:
                    return None
                js = by_c[c]
                frame = pd.DataFrame(np.nan, index=range(len(uniq)), columns=ov_cols, dtype=object)
                frame.iloc[[local_of[j] for j in js]] = rows.iloc[js][ov_cols].to_numpy()
                return frame

            out = self._recurse(uniq, c_end, ov_at, detail)
            for c, js in by_c.items():
                pred, base, meta, contrib = out[c]
                li = np.array([local_of[j] for j in js])
                res.iloc[js, res.columns.get_loc("point")] = pred.point[li]
                res.iloc[js, res.columns.get_loc("lo")] = pred.lo[li]
                res.iloc[js, res.columns.get_loc("hi")] = pred.hi[li]
                res.iloc[js, res.columns.get_loc("hstep")] = c - W + 1
                res.iloc[js, res.columns.get_loc("gap")] = meta["gap"][li]
                res.iloc[js, res.columns.get_loc("mode")] = "forecast"
                if detail:
                    self._collect_detail(details, js, li, pred, base, meta, contrib)

        # ---- status for rows that produced nothing ---------------------------
        empty = (res["status"] == "ok") & res["point"].isna()
        res.loc[empty, "status"] = "insufficient_history"
        return res, details

    @staticmethod
    def _collect_detail(details, js, li, pred: Pred, base, meta, contrib):
        """Store per-row diagnostics (features, driver contributions, provenance)."""
        for k, j in enumerate(js):
            i = int(li[k])
            details[int(j)] = {
                "X": pred.X.iloc[i].to_dict(),
                "base": base.iloc[i].to_dict(),
                "contrib": None if contrib is None else contrib[i],
                "est_macro": bool(meta["est_macro"][i]), "est_other": bool(meta["est_other"][i]),
                "overridden": {c: bool(v[i]) for c, v in meta["overridden"].items()},
            }

    # -------------------------------------------------------------- public API
    def _resolve_date(self, value) -> pd.Timestamp:
        try:
            d = pd.Timestamp(value)
        except Exception as exc:
            raise ForecastError(f"Invalid date: {value!r}. Use YYYY-MM-DD.") from exc
        if pd.isna(d):
            raise ForecastError("Date is required (YYYY-MM-DD).")
        return week_friday(d)

    def check_range(self, friday: pd.Timestamp) -> None:
        lo, hi = self.ref.supported_range
        if friday < lo or friday > hi:
            raise ForecastError(
                f"Week {friday.date()} is outside the supported range "
                f"({lo.date()} to {hi.date()}) for this workspace's data."
            )

    def forecast_series(self, store: int, dept: int, date, horizon: int = 1,
                        overrides: dict | None = None) -> dict:
        ref = self.ref
        if store not in ref.store_type:
            raise ForecastError(f"Unknown store {store}. Valid stores are 1-{max(ref.store_type)}.")
        sidx = ref.series_pos.get((int(store), int(dept)))
        if sidx is None:
            raise ForecastError(f"Store {store} has no sales history for department {dept} in this workspace.")
        horizon = int(horizon)
        if not 1 <= horizon <= MAX_HORIZON_WEEKS:
            raise ForecastError(f"Horizon must be between 1 and {MAX_HORIZON_WEEKS} weeks.")
        start = self._resolve_date(date)
        end = start + pd.Timedelta(days=7 * (horizon - 1))
        self.check_range(start)
        self.check_range(end)

        overrides = {OVERRIDE_ALIASES[k]: v for k, v in (overrides or {}).items()
                     if k in OVERRIDE_ALIASES and v is not None}
        weeks = [start + pd.Timedelta(days=7 * i) for i in range(horizon)]
        rows = pd.DataFrame({"sidx": sidx, "c": [ref.week_index(w) for w in weeks]})
        for col, v in overrides.items():
            rows[col] = v
        res, details = self._engine(rows, detail=True)

        out_weeks = []
        for j, w in enumerate(weeks):
            r = res.iloc[j]
            d = details.get(j)
            if r["status"] != "ok":
                raise ForecastError(_status_message(r["status"], store, dept, w))
            out_weeks.append(self._week_payload(r, d, w, j))

        first = out_weeks[0]
        total = float(sum(x["forecast"] for x in out_weeks))
        ly = [x["last_year"] for x in out_weeks]
        yoy = None
        if all(v is not None for v in ly) and sum(ly) != 0:
            yoy = (total / sum(ly) - 1) * 100
        notes = self._notes(out_weeks, res, first)
        conf = self._confidence(out_weeks, res)

        hist_from = start - pd.Timedelta(days=7 * 52)
        hist_to = min(ref.last_actual, end)
        return {
            "store": int(store), "dept": int(dept),
            "store_type": ref.store_type[int(store)], "store_size": int(ref.store_size[int(store)]),
            "start": start.strftime("%Y-%m-%d"), "horizon": horizon,
            "weeks": out_weeks,
            "history": ref.history(store, dept, hist_from, hist_to),
            "summary": {
                "total_forecast": total, "avg_weekly": total / horizon,
                "vs_last_year_pct": yoy, "confidence": conf, "mode": first["mode"],
            },
            "notes": notes,
            "model": {"name": self.metadata["model_name"], "version": self.metadata["version"]},
            "data_range": {"first_week": ref.week0.strftime("%Y-%m-%d"),
                           "last_actual_week": ref.last_actual.strftime("%Y-%m-%d"),
                           "indicators_through": ref.ind_last.strftime("%Y-%m-%d")},
        }

    def _week_payload(self, r, d, w: pd.Timestamp, j: int) -> dict:
        ref = self.ref
        X = d["X"]
        base = d["base"]
        est_o, est_m = d["est_other"], d["est_macro"]
        ovr = d["overridden"]

        def src(col, est):
            return "override" if ovr.get(col) else ("estimated" if est else "historical")

        lag52 = X.get("Weekly_Sales_Lag52")
        drivers = []
        if d["contrib"] is not None and not np.isnan(d["contrib"]).any():
            c = d["contrib"]
            feats = list(FEATURES) + ["__bias__"]
            pairs = sorted(zip(feats[:-1], c[:-1]), key=lambda t: -abs(t[1]))
            top = pairs[:6]
            drivers = [{"feature": f, "label": FEATURE_LABELS.get(f, f), "impact": float(v)} for f, v in top]
            rest = float(sum(v for _, v in pairs[6:]))
            drivers.append({"feature": "other", "label": "All other factors", "impact": rest})
            baseline = float(c[-1])
        else:
            baseline = None
        actual = None if np.isnan(r["actual"]) else float(r["actual"])
        return {
            "date": w.strftime("%Y-%m-%d"),
            "forecast": round(float(r["point"]), 2),
            "low": round(float(r["lo"]), 2),
            "high": round(float(r["hi"]), 2),
            "actual": actual,
            "mode": r["mode"],
            "weeks_ahead": int(r["hstep"]),
            "last_year": None if lag52 is None or np.isnan(lag52) else float(lag52),
            "inputs": {
                "last_week": _num(X.get("Weekly_Sales_Lag1")), "two_weeks_ago": _num(X.get("Weekly_Sales_Lag2")),
                "avg_3wk": _num(X.get("Weekly_Sales_MA3")), "volatility_3wk": _num(X.get("Weekly_Sales_STD3")),
                "history_gap_weeks": int(r["gap"]),
            },
            "indicators": {
                "temperature": {"value": round(float(base["Temperature"]), 2), "source": src("Temperature", est_o)},
                "fuel_price": {"value": round(float(base["Fuel_Price"]), 3), "source": src("Fuel_Price", est_o)},
                "cpi": {"value": round(float(base["CPI"]), 3), "source": src("CPI", est_m)},
                "unemployment": {"value": round(float(base["Unemployment"]), 3), "source": src("Unemployment", est_m)},
                "markdowns": {
                    "value": [round(float(base[m]), 2) for m in MARKDOWNS],
                    "source": "override" if any(ovr.get(m) for m in MARKDOWNS) else ("estimated" if est_o else "historical"),
                },
                "is_holiday": {"value": bool(base["IsHoliday"]),
                               "source": "override" if ovr.get("IsHoliday") else ("estimated" if est_o else "historical")},
            },
            "drivers": drivers,
            "baseline": baseline,
        }

    def _notes(self, weeks, res, first) -> list[str]:
        notes = []
        m = first["mode"]
        if m == "in_sample":
            notes.append("This week is inside the model's training period, so the fit shown is optimistic "
                         "(in-sample). Pick a week in 2012 for a genuine out-of-sample back-test, or a later week "
                         "for a true forecast.")
        elif m == "backtest":
            notes.append("Out-of-sample back-test: this week was not used to train the model that produced it.")
        if any(w["mode"] == "forecast" for w in weeks):
            ahead = max(w["weeks_ahead"] for w in weeks)
            notes.append(f"Forecast weeks are projected recursively from the last actual sales week "
                         f"({self.ref.last_actual.date()}), up to {ahead} week(s) ahead; uncertainty grows with distance.")
        if any(w["indicators"]["cpi"]["source"] == "estimated" for w in weeks):
            notes.append("CPI / unemployment are not published for some of these weeks; the last known value for "
                         "the store was carried forward.")
        if any(w["indicators"]["temperature"]["source"] == "estimated" for w in weeks):
            notes.append("This week is beyond the historical dataset: temperature uses the store's seasonal average, "
                         "fuel price is carried forward, and no promotions are assumed.")
        if any(w["inputs"]["history_gap_weeks"] > 0 for w in weeks):
            notes.append("This department has gaps in its recent sales history; the most recent observed weeks "
                         "were used, so the forecast is less reliable.")
        return notes

    def _confidence(self, weeks, res) -> str:
        ahead = max(w["weeks_ahead"] for w in weeks)
        stale = any(w["inputs"]["history_gap_weeks"] > 0 for w in weeks)
        est = any(w["indicators"]["temperature"]["source"] == "estimated" for w in weeks)
        if weeks[0]["mode"] != "forecast":
            return "backtest"
        if stale or est or ahead > 26:
            return "low"
        if ahead > 8:
            return "medium"
        return "high"

    # ------------------------------------------------------------- bulk scoring
    def forecast_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a table of Store, Dept, Date (+ optional indicator overrides)."""
        from src.features import parse_dates

        ref = self.ref
        cols = {c.lower(): c for c in df.columns}
        for need in ("store", "dept", "date"):
            if need not in cols:
                raise ForecastError(
                    f"Missing required column '{need.title()}'. Required: Store, Dept, Date "
                    f"(optional: IsHoliday, Temperature, Fuel_Price, CPI, Unemployment, MarkDown1-5)."
                )
        work = pd.DataFrame({
            "Store": pd.to_numeric(df[cols["store"]], errors="coerce"),
            "Dept": pd.to_numeric(df[cols["dept"]], errors="coerce"),
        })
        if work[["Store", "Dept"]].isna().any().any():
            bad = int(work[["Store", "Dept"]].isna().any(axis=1).sum())
            raise ForecastError(f"{bad} row(s) have a non-numeric Store or Dept.")
        raw_dates = df[cols["date"]]
        try:
            parsed = parse_dates(raw_dates)
        except ValueError as exc:
            raise ForecastError(str(exc)) from exc
        work["Date"] = week_friday(parsed).to_numpy()
        work["sidx"] = [ref.series_pos.get((int(s), int(d)), -1) for s, d in zip(work["Store"], work["Dept"])]
        work["c"] = ((work["Date"] - ref.week0).dt.days // 7).astype(int)

        for low, internal in OVERRIDE_ALIASES.items():
            if low in cols:
                col = df[cols[low]]
                work[internal] = pd.to_numeric(col.astype(str).str.upper().replace({"TRUE": "1", "FALSE": "0"}),
                                               errors="coerce") if internal == "IsHoliday" else pd.to_numeric(col, errors="coerce")
                if internal == "IsHoliday":
                    work[internal] = work[internal].map(lambda v: np.nan if pd.isna(v) else bool(v)).astype(object)
        res, _ = self._engine(work)
        out = df.copy().reset_index(drop=True)
        out["Week_Ending"] = work["Date"].dt.strftime("%Y-%m-%d").to_numpy()
        out["Predicted_Weekly_Sales"] = res["point"].round(2).to_numpy()
        out["Lower_80"] = res["lo"].round(2).to_numpy()
        out["Upper_80"] = res["hi"].round(2).to_numpy()
        out["Actual_Weekly_Sales"] = res["actual"].round(2).to_numpy()
        out["Forecast_Type"] = res["mode"].map({"backtest": "Out-of-sample back-test", "in_sample": "In-sample fit",
                                                 "forecast": "Forecast"}).fillna("").to_numpy()
        out["Weeks_Ahead"] = np.where(res["mode"] == "forecast", res["hstep"], np.nan)
        out["Status"] = res["status"].map(STATUS_TEXT).to_numpy()
        return out

    # ------------------------------------------------- bring-your-own-data API
    def predict_manual(self, p: dict) -> float:
        """One forecast from caller-supplied history and indicators (legacy /predict_single).

        ``recent_sales_history`` is newest-first. Three values are enough; with
        55+ values the seasonal (last-year) features are populated as well.
        """
        ref = self.ref
        store = int(p["store"])
        if store not in ref.store_type:
            raise ForecastError(f"Unknown store {store}.")
        hist = [float(v) for v in p["recent_sales_history"]]
        if len(hist) < 3:
            raise ForecastError("recent_sales_history must contain at least 3 weekly values (newest first).")
        date = week_friday(pd.Timestamp(p["forecast_date"]))

        def h(i):                       # history[i] is the sales i+1 weeks before the forecast week
            return hist[i] if i < len(hist) else np.nan

        base = pd.DataFrame([{
            "Store": store, "Dept": int(p["dept"]), "Date": date,
            "Type": p.get("store_type") or ref.store_type[store],
            "Size": float(p.get("size") or ref.store_size[store]),
            "IsHoliday": bool(p.get("is_holiday", False)),
            "IsHoliday_PrevWeek": int(bool(p.get("is_holiday_prev_week", False))),
            "IsHoliday_NextWeek": int(bool(p.get("is_holiday_next_week", False))),
            "Temperature": float(p["temperature"]), "Fuel_Price": float(p["fuel_price"]),
            "CPI": float(p["cpi"]), "Unemployment": float(p["unemployment"]),
            **{m: float(p.get(m.lower(), 0.0) or 0.0) for m in MARKDOWNS},
            "Lag1": h(0), "Lag2": h(1), "Lag3": h(2),
            "Lag52": h(51), "Lag53": h(52), "Lag54": h(53), "Lag55": h(54),
        }])
        X = make_model_frame(base, self.production.agg)
        if p.get("store_avg_sales"):
            X["Store_Avg_Sales"] = float(p["store_avg_sales"])
        if p.get("dept_avg_sales"):
            X["Dept_Avg_Sales"] = float(p["dept_avg_sales"])
        if p.get("store_avg_fuel_price"):
            X["Rel_Fuel_Price"] = float(p["fuel_price"]) / float(p["store_avg_fuel_price"])
        with self._lock:
            pred = float(self.production.point.predict(xgb.DMatrix(X, feature_names=FEATURES))[0])
        return max(0.0, pred)

    # ---------------------------------------------------------------- overview
    def overview(self) -> dict:
        if self._overview is None:
            self._overview = self._compute_overview()
        return self._overview

    def _compute_overview(self) -> dict:
        ref = self.ref
        W = ref.n_hist
        A = ref.A
        N = int((ref.ind_last - ref.last_actual).days // 7)
        active = np.where(~np.isnan(A[:, -4:]).all(axis=1))[0]
        out = self._recurse(active, W - 1 + N)
        fc = np.column_stack([out[c][0].point for c in range(W, W + N)])          # (active, N)
        fc_dates = [ref.week_date(c) for c in range(W, W + N)]
        fc_total = np.nansum(fc, axis=0)
        actual_total = np.nansum(A, axis=0)
        last_year_cols = [c - 52 for c in range(W, W + N)]
        ly = np.nansum(A[np.ix_(active, last_year_cols)], axis=0)

        keys_dept = ref.series_dept[active]
        keys_store = ref.series_store[active]
        ly13 = np.nansum(A[np.ix_(active, last_year_cols[:13])], axis=1)          # per active series
        top_depts, top_stores = [], []
        for k in np.unique(keys_dept):
            m = keys_dept == k
            f13 = float(np.nansum(fc[m][:, :13]))
            l13 = float(np.nansum(ly13[m]))
            top_depts.append({"dept": int(k), "forecast_13w": f13, "last_year_13w": l13,
                              "growth_pct": None if l13 <= 0 else (f13 / l13 - 1) * 100,
                              "trailing_52w": float(np.nansum(A[active][m][:, -52:]))})
        for k in np.unique(keys_store):
            m = keys_store == k
            f13 = float(np.nansum(fc[m][:, :13]))
            l13 = float(np.nansum(ly13[m]))
            top_stores.append({"store": int(k), "type": ref.store_type[int(k)], "forecast_13w": f13,
                               "last_year_13w": l13,
                               "growth_pct": None if l13 <= 0 else (f13 / l13 - 1) * 100,
                               "trailing_52w": float(np.nansum(A[active][m][:, -52:]))})
        top_depts.sort(key=lambda r: -r["trailing_52w"])
        top_stores.sort(key=lambda r: -r["trailing_52w"])

        hist = [{"date": ref.weeks[i].strftime("%Y-%m-%d"), "actual": float(actual_total[i])} for i in range(W)]
        fcast = [{"date": d.strftime("%Y-%m-%d"), "forecast": float(v), "last_year": float(l)}
                 for d, v, l in zip(fc_dates, fc_total, ly)]
        f13 = float(fc_total[:13].sum())
        l13 = float(ly[:13].sum())
        meta_val = self.metadata["validation"]
        return {
            "kpis": {
                "last_week": ref.last_actual.strftime("%Y-%m-%d"),
                "last_week_sales": float(actual_total[-1]),
                "trailing_52w_sales": float(actual_total[-52:].sum()),
                "forecast_13w": f13,
                "forecast_13w_vs_last_year_pct": None if l13 <= 0 else (f13 / l13 - 1) * 100,
                "stores": int(len(ref.store_type)),
                "active_series": int(len(active)),
                "holdout_wape": meta_val["holdout"]["WAPE"],
                "naive_wape": meta_val["baselines"]["Last week (naive)"]["WAPE"],
            },
            "history": hist,
            "forecast": fcast,
            "backtest": meta_val["backtest_weekly_totals"],
            "top_departments": top_depts[:10],
            "top_stores": top_stores[:10],
            "store_growth": sorted(top_stores, key=lambda r: (r["growth_pct"] is None, -(r["growth_pct"] or 0)))[:5],
        }


STATUS_TEXT = {
    "ok": "OK",
    "unknown_series": "No sales history for this Store/Dept in the workspace",
    "date_out_of_range": "Week outside the supported date range",
    "insufficient_history": "Fewer than 3 recent weeks of history",
}


def _status_message(status: str, store, dept, week) -> str:
    base = STATUS_TEXT.get(status, status)
    return f"Store {store} / Dept {dept} / week {pd.Timestamp(week).date()}: {base}."


def _num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
