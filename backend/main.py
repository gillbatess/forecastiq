"""ForecastIQ API + web app (single process, single port).

    uvicorn backend.main:app --port 7860

* ``/api/*``   JSON API used by the web app (and available to clients)
* ``/``        marketing landing page
* ``/app``     the product UI (single page app)
* ``/docs``    interactive API docs (FastAPI/Swagger)

Legacy endpoints ``/health``, ``/model``, ``/predict_single`` and ``/predict_batch``
are kept for API compatibility with the earlier Streamlit prototype.
"""
from __future__ import annotations

import io
import os
import secrets
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from threading import Lock

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.schemas import ForecastRequest, SingleForecastRequest
from src.config import MAX_BULK_ROWS, MAX_UPLOAD_BYTES, WEB_DIR
from src.forecaster import ForecastError, ForecastService
from src.logging_config import logger

state: dict = {"service": None, "error": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        svc = ForecastService()
        svc.overview()                       # warm the dashboard cache so the first visit is instant
        state["service"] = svc
        logger.info("ForecastIQ model service initialised")
    except Exception as exc:                 # keep the process up so /api/health can explain
        state["error"] = str(exc)
        logger.exception("Model service unavailable: %s", exc)
    yield


app = FastAPI(
    title="ForecastIQ API",
    version="2.0.0",
    description="Weekly retail demand forecasting with automatic economic-indicator lookup.",
    lifespan=lifespan,
)

_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:                                  # same-origin by default; opt in for a separate front end
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    if request.url.path in ("/", "/app") or request.url.path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "no-cache")
    return resp


# ----------------------------------------------------------------------- helpers
def svc() -> ForecastService:
    s = state["service"]
    if s is None:
        raise HTTPException(503, "Model service is not ready. Train the model (python scripts/train_model.py) "
                                 f"or check the logs. Detail: {state['error'] or 'starting up'}")
    return s


def clean(o):
    """Recursively make a structure JSON-safe (NaN/inf -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.strftime("%Y-%m-%d")
    return o


def frame_records(df: pd.DataFrame) -> list[dict]:
    d = df.replace([np.inf, -np.inf], np.nan)
    d = d.astype(object).where(pd.notna(d), None)
    return clean(d.to_dict(orient="records"))


class BulkStore:
    """Keeps the last few scored files in memory so the UI can page and download them."""

    def __init__(self, keep: int = 6):
        self.keep = keep
        self._d: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._lock = Lock()

    def put(self, df: pd.DataFrame) -> str:
        key = secrets.token_urlsafe(9)
        with self._lock:
            self._d[key] = df
            while len(self._d) > self.keep:
                self._d.popitem(last=False)
        return key

    def get(self, key: str) -> pd.DataFrame:
        with self._lock:
            df = self._d.get(key)
        if df is None:
            raise HTTPException(404, "This result has expired. Run the forecast again.")
        return df


bulk_store = BulkStore()


# -------------------------------------------------------------------------- API
@app.get("/api/health")
def api_health():
    s = state["service"]
    return {
        "status": "ok" if s else "degraded",
        "model_loaded": s is not None,
        "service": "forecastiq",
        "version": s.metadata["version"] if s else None,
        "error": state["error"],
    }


@app.get("/api/meta")
def api_meta():
    s = svc()
    ref = s.ref
    lo, hi = ref.supported_range
    return clean({
        "stores": ref.store_list(),
        "first_week": ref.week0.strftime("%Y-%m-%d"),
        "last_actual_week": ref.last_actual.strftime("%Y-%m-%d"),
        "indicators_through": ref.ind_last.strftime("%Y-%m-%d"),
        "supported_from": lo.strftime("%Y-%m-%d"),
        "supported_to": hi.strftime("%Y-%m-%d"),
        "holdout_start": s.holdout_start.strftime("%Y-%m-%d"),
        "max_horizon": 13,
        "workspace": {"name": "Demo workspace", "dataset": "Walmart weekly sales (Kaggle), 45 stores"},
        "model": {"name": s.metadata["model_name"], "version": s.metadata["version"]},
    })


@app.get("/api/stores/{store}/departments")
def api_departments(store: int):
    s = svc()
    if store not in s.ref.store_type:
        raise HTTPException(404, f"Unknown store {store}")
    return clean({"store": store, "type": s.ref.store_type[store], "size": int(s.ref.store_size[store]),
                  "departments": s.ref.dept_list(store)})


@app.get("/api/indicators")
def api_indicators(store: int = Query(ge=1), date: str = Query(description="Any date, YYYY-MM-DD")):
    """Economic indicators for a store and week, auto-filled from historical data."""
    s = svc()
    try:
        fri = s._resolve_date(date)
        s.check_range(fri)
        snap = s.ref.indicator_snapshot(store, fri)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ForecastError as exc:
        raise HTTPException(422, str(exc)) from exc
    snap["context"] = s.ref.indicator_series(store, fri - pd.Timedelta(weeks=52), fri + pd.Timedelta(weeks=13))
    return clean(snap)


@app.post("/api/forecast")
def api_forecast(req: ForecastRequest):
    s = svc()
    t0 = time.perf_counter()
    try:
        ov = req.overrides.model_dump(exclude_none=True) if req.overrides else {}
        out = s.forecast_series(req.store, req.dept, req.date.isoformat(), req.horizon, ov)
    except ForecastError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Forecast failed")
        raise HTTPException(500, "Forecast failed") from exc
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return clean(out)


@app.get("/api/overview")
def api_overview():
    return clean(svc().overview())


@app.get("/api/model")
def api_model():
    return clean(svc().metadata)


def _read_csv(content: bytes) -> pd.DataFrame:
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(422, "Could not read the file as CSV.") from exc
    if df.empty:
        raise HTTPException(422, "The file has no rows.")
    if len(df) > MAX_BULK_ROWS:
        raise HTTPException(413, f"Too many rows ({len(df):,}); the limit is {MAX_BULK_ROWS:,} per upload.")
    return df


@app.post("/api/bulk")
async def api_bulk(file: UploadFile = File(...)):
    s = svc()
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(415, "Only .csv files are supported.")
    df = _read_csv(await file.read())
    t0 = time.perf_counter()
    try:
        scored = s.forecast_rows(df)
    except ForecastError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Bulk forecast failed")
        raise HTTPException(500, "Bulk forecast failed") from exc

    key = bulk_store.put(scored)
    ok = scored["Status"] == "OK"
    good = scored[ok]
    tl = []
    if len(good):
        g = good.groupby("Week_Ending")
        fc = g["Predicted_Weekly_Sales"].sum()
        act_rows = good[good["Actual_Weekly_Sales"].notna()]
        act = act_rows.groupby("Week_Ending")["Actual_Weekly_Sales"].sum()
        fc_m = act_rows.groupby("Week_Ending")["Predicted_Weekly_Sales"].sum()
        for d in fc.index:
            tl.append({"date": d, "forecast": float(fc[d]),
                       "actual": float(act[d]) if d in act.index else None,
                       "forecast_matched": float(fc_m[d]) if d in fc_m.index else None})
    acc = None
    if len(good) and good["Actual_Weekly_Sales"].notna().any():
        m = good.dropna(subset=["Actual_Weekly_Sales"])
        denom = np.abs(m["Actual_Weekly_Sales"]).sum()
        acc = {"rows": int(len(m)),
               "wape": float(np.abs(m["Actual_Weekly_Sales"] - m["Predicted_Weekly_Sales"]).sum() / denom * 100)
               if denom else None,
               "in_sample_rows": int((m["Forecast_Type"] == "In-sample fit").sum())}
    return clean({
        "id": key,
        "rows": int(len(scored)),
        "ok_rows": int(ok.sum()),
        "failed_rows": int((~ok).sum()),
        "status_counts": scored["Status"].value_counts().to_dict(),
        "type_counts": scored["Forecast_Type"].replace("", "n/a").value_counts().to_dict(),
        "summary": {
            "total_forecast": float(good["Predicted_Weekly_Sales"].sum()) if len(good) else 0.0,
            "stores": int(good["Store"].nunique()) if len(good) else 0,
            "departments": int(good["Dept"].nunique()) if len(good) else 0,
            "weeks": int(good["Week_Ending"].nunique()) if len(good) else 0,
            "first_week": good["Week_Ending"].min() if len(good) else None,
            "last_week": good["Week_Ending"].max() if len(good) else None,
        },
        "accuracy": acc,
        "timeline": tl,
        "columns": list(scored.columns),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    })


@app.get("/api/bulk/{key}/rows")
def api_bulk_rows(key: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500),
                  only_issues: bool = False):
    df = bulk_store.get(key)
    if only_issues:
        df = df[df["Status"] != "OK"]
    page = df.iloc[offset: offset + limit]
    return clean({"total": int(len(df)), "offset": offset, "rows": frame_records(page)})


@app.get("/api/bulk/{key}/download")
def api_bulk_download(key: str):
    df = bulk_store.get(key)
    return Response(df.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="forecastiq_results.csv"'})


@app.get("/api/bulk-template")
def api_bulk_template():
    """A ready-to-run example: 8 weeks of the 2012 hold-out plus 4 forecast weeks for two departments."""
    rows = []
    for store, dept in ((1, 1), (1, 3), (2, 1)):
        for d in pd.date_range("2012-09-07", periods=8, freq="7D"):
            rows.append((store, dept, d.strftime("%Y-%m-%d")))
    df = pd.DataFrame(rows, columns=["Store", "Dept", "Date"])
    return Response(df.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="forecastiq_template.csv"'})


# ---------------------------------------------------------------- legacy endpoints
@app.get("/health")
def health():
    s = state["service"]
    return {"status": "ok", "model_loaded": s is not None, "service": "forecastiq-api"}


@app.get("/model")
def model_info():
    return clean(svc().metadata)


@app.post("/predict_single")
def predict_single(request: SingleForecastRequest):
    s = svc()
    t0 = time.perf_counter()
    payload = request.model_dump()
    payload["forecast_date"] = request.forecast_date.isoformat()
    try:
        value = s.predict_manual(payload)
    except ForecastError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Single prediction failed")
        raise HTTPException(500, "Prediction failed") from exc
    return {
        "predicted_weekly_sales": round(value, 2),
        "currency": "USD",
        "model": s.metadata["model_name"],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    s = svc()
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(415, "Only CSV files are supported.")
    df = _read_csv(await file.read())
    try:
        out = s.forecast_rows(df)
    except ForecastError as exc:
        raise HTTPException(422, str(exc)) from exc
    return frame_records(out)


# ------------------------------------------------------------------- web front end
_static = WEB_DIR / "static"
if _static.exists():
    app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/", include_in_schema=False)
def landing():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/app", include_in_schema=False)
def product():
    return FileResponse(WEB_DIR / "app.html")
