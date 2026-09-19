---
title: ForecastIQ
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Weekly retail demand forecasting with auto-filled economic indicators
---

# ForecastIQ — weekly demand forecasting

ForecastIQ is a client-facing web product built on the Walmart weekly-sales project
(`notebooks/MBA944_v3.ipynb`): pick a store, a department and a week, and get a forecast with an
80% range, the reasons behind it, and the **economic indicators filled in automatically from historical data**.

One container serves everything: a marketing landing page (`/`), the product (`/app`), a JSON API (`/api/*`)
and interactive API docs (`/docs`). No database, no external services — it runs on free hosting.

## What the product does

| Area | What the user gets |
|---|---|
| **Overview** | Portfolio dashboard: full weekly history, 13-week outlook, back-test vs. actual, top departments and stores. |
| **Forecast studio** | Store + department + date + horizon (1–13 weeks). Temperature, fuel price, CPI, unemployment, markdowns and holiday flag are **auto-filled for that store and week** and labelled *historical* or *estimated*. Optional "Adjust" panel for what-if overrides. Results: chart with range, weekly table, "what drove it" (per-week feature contributions), economic-context charts. |
| **Bulk forecast** | Drag-and-drop a CSV (`Store, Dept, Date`; indicators optional), get a scored table with status per row, paging, and CSV download. Up to 100,000 rows / 25 MB per upload. |
| **Model performance** | Hold-out accuracy, baselines, peak-season back-test, interval calibration, breakdowns — all read from `artifacts/model_metadata.json`. |

### How the indicator auto-fill works

Weeks inside the dataset (Feb 2010 – Jul 2013): the store's own recorded temperature, fuel price, CPI, unemployment, markdowns and holiday flag for that week.
Weeks beyond it: temperature = that store's typical value for the same week of the year; fuel/CPI/unemployment carried forward from the latest value; markdowns assumed 0;
holidays from a calendar rule (Super Bowl, Labor Day, Thanksgiving, Christmas week). Anything estimated is flagged in the UI, and the confidence label drops accordingly.

## Honest accuracy

The original notebook reports R² 0.998 / MAE ≈ 335. **That number is not reproducible**: its 3-week rolling mean included the week being predicted (target leakage).
The rebuilt, leak-free pipeline gives (chronological hold-out, Jan–Oct 2012, models fitted only on data before 2012):

| | WAPE | R² |
|---|---:|---:|
| ForecastIQ (one-week-ahead) | **8.4%** | 0.981 |
| Last-week naive | 10.9% | 0.968 |
| Same week last year | 11.2% | 0.970 |
| Department average | 52.8% | 0.574 |

* In the Nov 2011 – Jan 2012 peak season the model (WAPE 11.5%) is roughly on par with seasonal-naive (11.9%) and far better than last-week-naive (23.2%).
* The 80% range covers ≈85% of out-of-sample points after calibration.
* Multi-week forecasts feed predictions back as inputs, so error grows with horizon (the ranges widen accordingly).
* Indicator overrides move a forecast only slightly — the model is driven mainly by recent sales, seasonality and store/department level. The UI says so rather than implying otherwise.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --port 7860
# open http://localhost:7860
```

Or with Docker: `docker compose up --build` (then http://localhost:7860).

Tests: `pip install -r requirements-dev.txt && pytest -q`

## Deploy on the web

The app is one Docker container (`Dockerfile`) that listens on `$PORT` (default 7860) and needs about 300-380 MB of RAM.

### Render (free web service) - recommended

1. Push this folder to a GitHub repo.
2. In Render: **New -> Web Service**, connect the repo.
3. Language **Docker**, instance type **Free**, everything else default, then **Deploy**.
4. When the build finishes, open the `https://<name>.onrender.com` link. Free services sleep after ~15 minutes idle; the first visit afterwards takes about a minute, so open it before a client demo.

### Hugging Face Spaces (Docker) - requires a paid (PRO) plan

Docker Spaces are no longer available on the free plan (free accounts can only create *Static* Spaces, which cannot run this Python backend).
With a plan that allows it: create a **Docker -> Blank** Space, create a write token, then

```bash
pip install -U huggingface_hub
export HF_TOKEN=hf_xxx HF_SPACE=<username>/<space-name>
python scripts/deploy_hf.py --dry-run     # optional: list what will be sent
python scripts/deploy_hf.py
```

(Plain `git push` to a Space is rejected for binary files unless they use Git-LFS/Xet; the script avoids that.)
`README.md` must keep its `---` header - Spaces reads `sdk: docker` and `app_port: 7860` from it.
`docs/github-actions-sync.md` shows how to redeploy automatically from GitHub.

## Retrain on new data

1. Put the four Kaggle files (`train.csv`, `test.csv`, `features.csv`, `stores.csv`) in `data/raw/`.
2. `pip install -r requirements-dev.txt`
3. `python scripts/train_model.py` (≈5 min; `--fast` for a 1-minute smoke run).

This rewrites everything in `artifacts/` (both model phases, reference data, metadata/metrics). Redeploy by pushing the new `artifacts/` folder.
To use *your* data: keep the same four-file schema (weekly, week ending Friday).

## API

Interactive docs at `/docs`. Main endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness + model status |
| GET | `/api/meta` | stores, supported date range |
| GET | `/api/indicators?store=1&date=2012-08-03` | auto-filled indicators with provenance |
| POST | `/api/forecast` | `{"store":1,"dept":1,"date":"2012-11-02","horizon":4,"overrides":{…}}` |
| POST | `/api/bulk` | CSV upload; then `/api/bulk/{id}/rows` and `/download` |
| GET | `/api/overview`, `/api/model` | dashboard data, model card |

`/predict_single`, `/predict_batch`, `/health`, `/model` from the earlier prototype still work.

## Project layout

```
backend/    FastAPI app + request schemas
src/        features (single shared feature builder), reference data, forecasting engine
scripts/    train_model.py
web/        landing page, app shell, css/js (Chart.js and Inter font are self-hosted)
artifacts/  trained models (gz), reference data, model_metadata.json
tests/      pytest suite (dates, holidays, leakage, indicators, forecasts, API)
```

See [`docs/architecture.md`](docs/architecture.md) for how the pieces fit together (request lifecycle, the two model phases, training vs. serving, deployment shape).

## Limits worth knowing before a client pitch

* The demo workspace is the public Walmart dataset (2010–2012). A real client needs their own history in the same schema, then a retrain.
* Forecasts beyond the data are recursive and use estimated indicators; the UI marks them and lowers the confidence label.
* New store/department combinations with fewer than three weeks of history cannot be forecast (the app says why).
* Free hosting tiers (Render, Hugging Face Spaces) have no authentication or persistent storage; uploaded files live in memory only for the life of the process and are not stored. Add login and a database before real client data goes in.

## Licenses and attributions

Application code: MIT (see `LICENSE`). Chart.js — MIT. Inter font — SIL Open Font License 1.1.
Walmart dataset from a Kaggle competition; check its terms before redistributing the raw files (they are deliberately not part of the deployable repo).
ForecastIQ is an independent project and is not affiliated with or endorsed by Walmart.
