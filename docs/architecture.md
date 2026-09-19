# Architecture

How ForecastIQ is put together: one process, no database, XGBoost boosters (a point
model and a quantile model, each trained in two phases), and a static front end served by the same app.

## System overview

```
                              ┌─────────────────────────────────────────┐
                              │   Docker container (Render / anywhere)   │
                              │                                           │
  Browser  ── HTTPS ──────────▶  FastAPI app  (backend/main.py)          │
                              │      │                                   │
                              │      ├── "/"        landing page (HTML)  │
                              │      ├── "/app"      product SPA (HTML)  │
                              │      ├── "/static/*" CSS / JS / fonts    │
                              │      ├── "/docs"      Swagger UI         │
                              │      └── "/api/*"     JSON API           │
                              │             │                            │
                              │             ▼                            │
                              │   ForecastService (src/forecaster.py)    │
                              │      │            │                      │
                              │      ▼            ▼                      │
                              │  ReferenceData   XGBoost boosters        │
                              │  (src/reference)  (2 phases, loaded      │
                              │             │      from artifacts/)      │
                              │             ▼                            │
                              │   artifacts/  (models + reference data,  │
                              │                baked into the image)     │
                              └─────────────────────────────────────────┘
```

Everything runs in a single container behind a single port (`$PORT`, default
`7860`). There is no database and no external service call on the request
path — the model files and the reference dataset are loaded into memory once
at startup and served from there.

## Components

| Path | Responsibility |
|---|---|
| [`backend/main.py`](../backend/main.py) | FastAPI app: routes, request validation glue, error mapping, static file / SPA serving, in-memory bulk-result store. |
| [`backend/schemas.py`](../backend/schemas.py) | Pydantic request models (`/api/forecast` body, legacy `/predict_single`). |
| [`src/config.py`](../src/config.py) | Single source of truth for paths, the model's feature contract, hold-out windows and service limits (max horizon, max bulk rows/bytes). Training and inference both import from here so they can't drift apart. |
| [`src/reference.py`](../src/reference.py) | `ReferenceData`: the compact, pre-aggregated history (per store/department sales, per store/week economic indicators, holiday calendar) that the "auto-fill" feature reads from. Loaded once from `artifacts/reference/`. |
| [`src/features.py`](../src/features.py) | The one shared feature builder (`make_model_frame`) — turns a row of (store, dept, week, indicators, recent sales) into the exact model-input columns, used identically by training and inference so there's no train/serve skew. |
| [`src/forecaster.py`](../src/forecaster.py) | `ForecastService`: the inference engine. Picks the right model phase for a date, resolves indicators (historical or estimated), recurses multi-week forecasts, and produces the 80% interval. |
| [`artifacts/`](../artifacts) | Build output of `scripts/train_model.py`: gzipped XGBoost boosters (point + quantile, two training phases) and `model_metadata.json`. Shipped inside the Docker image; nothing is trained at request time. |
| [`web/`](../web) | Static front end: `index.html` (landing page) and `app.html` (the product shell) plus `static/css` and `static/js`. Vanilla JS (no build step) renders views client-side against `/api/*`; Chart.js and the Inter font are self-hosted so there's no CDN dependency at runtime. |
| [`scripts/train_model.py`](../scripts/train_model.py) | Offline training pipeline. Reads the raw Kaggle CSVs, builds features, fits both model phases, and writes `artifacts/`. Run locally/CI, never in the deployed container. |

## Why two model "phases"

`ForecastService` loads two trained boosters (see `Phase` in `src/forecaster.py`):

- **hold-out phase** — fit only on data before 2012-01-01, used to score
  weeks inside the 2012 hold-out so the accuracy numbers shown in *Model
  performance* are genuinely out-of-sample.
- **production phase** — fit on all available history, used for any week
  after the last actual data point (real forecasting) and for in-sample
  weeks before the hold-out.

A request picks whichever phase applies to the requested week; the caller
never has to know which one was used.

## Request lifecycle (`POST /api/forecast`)

1. `backend/main.py` validates the body against `ForecastRequest` (store,
   dept, date, horizon 1–13, optional overrides) — FastAPI/Pydantic reject
   malformed input before it reaches the service.
2. `ForecastService.forecast_series` resolves the target date to that week's
   Friday and picks the model phase.
3. For each week in the horizon: indicators are pulled from `ReferenceData`
   (historical value if the week is inside the dataset, otherwise a seasonal
   estimate — see the README's "How the indicator auto-fill works") unless
   the caller supplied an override; `make_model_frame` builds the feature
   row; the booster produces a point prediction plus a quantile-based 80%
   interval. Weeks beyond the last actual data point feed each prediction
   back in as the next week's lag/rolling features (recursive forecasting),
   which is why the interval widens with horizon.
4. The API layer converts NumPy/pandas types to JSON-safe values (`clean`)
   and returns the series plus per-week feature contributions.

`POST /api/bulk` follows the same per-row path (`forecast_rows`) but scores a
whole CSV, keeps the scored `DataFrame` in an in-memory `BulkStore` (last 6
results, keyed by a random token) so the UI can page/download it, then calls
`trim_memory()` to return freed heap to the OS before the next request —
important on a memory-capped free host.

## Data flow: training vs. serving

```
Kaggle CSVs (data/raw/, not shipped)
        │  scripts/train_model.py
        ▼
artifacts/  (boosters + compact reference data + model_metadata.json)
        │  copied into the Docker image at build time
        ▼
ForecastService (loaded once at process startup)
        │  serves every request from memory
        ▼
JSON API  →  web/static/js views  →  browser
```

Retraining (`python scripts/train_model.py`) only ever touches `artifacts/`;
redeploying is "replace `artifacts/`, rebuild the image" — the API and the
front end don't change.

## Deployment shape

- **One Dockerfile, one process.** `Dockerfile` installs dependencies, copies
  `backend/`, `src/`, `web/` and `artifacts/`, and starts
  `uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}`. The same
  image runs on Render, Hugging Face Spaces, or any Docker host — the
  `${PORT:-7860}` fallback exists because Render (and most PaaS hosts) inject
  `$PORT`, while Hugging Face Spaces expects the fixed port 7860.
- **No database, no persistence.** Reference data and models are read-only
  artifacts baked into the image. Bulk-upload results live in an in-memory
  dict and are lost on restart (by design — see the README's limits section).
- **Memory-conscious by construction.** `trim_memory()` calls `malloc_trim`
  after startup warm-up and after every bulk upload so the process stays
  comfortably inside a ~512 MB host; `MAX_BULK_ROWS` (100,000) and
  `MAX_UPLOAD_BYTES` (25 MB) in `src/config.py` cap how much a single request
  can pull into memory.
- **Stateless horizontally.** Because there's no shared state besides the
  in-process `BulkStore`, the only cost of running multiple instances is that
  a bulk result fetched from instance A won't exist on instance B — fine for
  a single free-tier instance, worth knowing before scaling out.
- **Same-origin by default.** CORS is off unless `CORS_ORIGINS` is set, since
  the API and the web app are served from the same origin.

See the README's [Deploy on the web](../README.md#deploy-on-the-web) section
for the actual Render/Hugging Face steps.
