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
| **Bulk forecast** | Drag-and-drop a CSV (`Store, Dept, Date`; indicators optional), get a scored table with status per row, paging, and CSV download. Up to 250,000 rows / 25 MB. |
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

## Deploy free on Hugging Face Spaces

Hugging Face Spaces runs Docker apps on a free CPU tier (2 vCPU / 16 GB at the time of writing; the Space goes to sleep after a period of inactivity and wakes on the next visit).
Check the current limits in the Spaces documentation before a client demo.

1. Create a free account at <https://huggingface.co/join>.
2. **New → Space.** Name it (e.g. `forecastiq`), choose **Docker → Blank**, hardware **CPU basic (free)**, visibility *Public* (or *Private* to demo from your own login).
3. Get the code into the Space — either:
   * **Web upload:** *Files → Add file → Upload files*, drag in the contents of this folder (`Dockerfile`, `README.md`, `requirements.txt`, `backend/`, `src/`, `web/`, `artifacts/`). Skip `data/`, `legacy/`, `notebooks/`, `tests/`.
   * **Git:** `git clone https://huggingface.co/spaces/<your-username>/forecastiq`, copy the same files in, then `git add . && git commit -m "ForecastIQ" && git push`. When asked for a password use an access token from *Settings → Access Tokens* (write permission).
4. The Space builds automatically (2–4 minutes). Watch the **Logs** tab; when it says *Running*, the app is live at
   `https://<your-username>-forecastiq.hf.space`.
5. For a custom look for clients, embed or link that URL, or attach your own domain later (paid feature). The first request after sleeping takes ~30 s.

Notes: every model file in `artifacts/` is under 10 MB (they are gzip-compressed), so no Git-LFS setup is needed. `README.md` must keep its `---` header — Spaces reads `sdk: docker` and `app_port: 7860` from it.

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

## Limits worth knowing before a client pitch

* The demo workspace is the public Walmart dataset (2010–2012). A real client needs their own history in the same schema, then a retrain.
* Forecasts beyond the data are recursive and use estimated indicators; the UI marks them and lowers the confidence label.
* New store/department combinations with fewer than three weeks of history cannot be forecast (the app says why).
* Free Spaces have no authentication or persistence; uploaded files live in memory only and are not stored. Add login and a database before real client data goes in.

## Licenses and attributions

Application code: MIT (see `LICENSE`). Chart.js — MIT. Inter font — SIL Open Font License 1.1.
Walmart dataset from a Kaggle competition; check its terms before redistributing the raw files (they are deliberately not part of the deployable repo).
ForecastIQ is an independent project and is not affiliated with or endorsed by Walmart.
