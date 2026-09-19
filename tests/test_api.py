import io

import pandas as pd


def test_health_and_meta(client):
    assert client.get("/api/health").json()["model_loaded"] is True
    m = client.get("/api/meta").json()
    assert len(m["stores"]) == 45 and m["supported_from"] < m["supported_to"]


def test_pages_and_static(client):
    assert "ForecastIQ" in client.get("/").text
    assert "ForecastIQ" in client.get("/app").text
    assert client.get("/static/js/core.js").status_code == 200
    r = client.get("/")
    assert r.headers["x-content-type-options"] == "nosniff"


def test_departments_and_indicators(client):
    d = client.get("/api/stores/1/departments").json()
    assert d["departments"]
    ind = client.get("/api/indicators", params={"store": 1, "date": "2012-08-01"}).json()
    assert ind["week"] == "2012-08-03" and ind["temperature"]["source"] == "historical"
    assert client.get("/api/indicators", params={"store": 999, "date": "2012-08-01"}).status_code == 404
    assert client.get("/api/indicators", params={"store": 1, "date": "1990-01-01"}).status_code == 422


def test_forecast_endpoint(client):
    r = client.post("/api/forecast", json={"store": 1, "dept": 1, "date": "2012-11-02", "horizon": 4})
    assert r.status_code == 200
    body = r.json()
    assert len(body["weeks"]) == 4 and "summary" in body and "history" in body
    bad = client.post("/api/forecast", json={"store": 1, "dept": 1, "date": "2012-11-02", "horizon": 40})
    assert bad.status_code == 422
    extra = client.post("/api/forecast", json={"store": 1, "dept": 1, "date": "2012-11-02", "nope": 1})
    assert extra.status_code == 422
    unk = client.post("/api/forecast", json={"store": 45, "dept": 99, "date": "2012-11-02"})
    assert unk.status_code == 422


def test_overview_and_model(client):
    o = client.get("/api/overview").json()
    assert o
    m = client.get("/api/model").json()
    assert m["feature_count"] == len(m["features"])


def test_bulk_flow(client):
    tpl = client.get("/api/bulk-template")
    assert tpl.status_code == 200
    r = client.post("/api/bulk", files={"file": ("t.csv", tpl.content, "text/csv")})
    assert r.status_code == 200, r.text
    key = r.json()["id"]
    rows = client.get(f"/api/bulk/{key}/rows", params={"limit": 5}).json()
    assert rows["total"] > 0 and len(rows["rows"]) == 5
    dl = client.get(f"/api/bulk/{key}/download")
    df = pd.read_csv(io.BytesIO(dl.content))
    assert "Predicted_Weekly_Sales" in df.columns
    assert client.get("/api/bulk/doesnotexist/rows").status_code == 404


def test_bulk_rejects_bad_uploads(client):
    assert client.post("/api/bulk", files={"file": ("x.txt", b"a,b", "text/plain")}).status_code == 415
    assert client.post("/api/bulk", files={"file": ("x.csv", b"", "text/csv")}).status_code == 422
    assert client.post("/api/bulk", files={"file": ("x.csv", b"A,B\n1,2\n", "text/csv")}).status_code == 422


def test_legacy_endpoints(client):
    assert client.get("/health").json()["status"] == "ok"
    payload = {
        "store": 1, "dept": 1, "forecast_date": "2012-11-02", "size": 151315,
        "temperature": 55, "fuel_price": 3.5, "cpi": 171.5, "unemployment": 7.8,
        "recent_sales_history": [15000, 14800, 14500],
    }
    r = client.post("/predict_single", json=payload)
    assert r.status_code == 200 and r.json()["predicted_weekly_sales"] > 0
    csv = b"Store,Dept,Date\n1,1,2012-08-03\n"
    r = client.post("/predict_batch", files={"file": ("a.csv", csv, "text/csv")})
    assert r.status_code == 200 and r.json()[0]["Predicted_Weekly_Sales"] > 0
