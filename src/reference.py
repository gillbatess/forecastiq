"""Compact reference data the running service needs (no raw Kaggle files).

Loads, from ``artifacts/reference``:

* ``sales.csv.gz``       - weekly sales history per Store/Dept (the client's "warehouse")
* ``indicators.csv.gz``  - cleaned economic indicators per Store+Date
* ``stores.csv``         - store type / size

and answers the questions the UI asks: which departments does store X have, what
were the economic conditions in store X in the week of D, what was the recent
sales history of store X / dept Y ...
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import INDICATORS, MARKDOWNS, REFERENCE_DIR
from src.features import HolidayCalendar, parse_dates, sales_matrix, week_friday

IND_COLS = ["Temperature", "Fuel_Price", "CPI", "Unemployment", *MARKDOWNS]


class ReferenceData:
    def __init__(self, ref_dir: Path | None = None):
        ref_dir = Path(ref_dir or REFERENCE_DIR)
        for name in ("sales.csv.gz", "indicators.csv.gz", "stores.csv"):
            if not (ref_dir / name).exists():
                raise FileNotFoundError(
                    f"Reference file {ref_dir / name} not found. Run `python scripts/train_model.py` first."
                )

        sales = pd.read_csv(ref_dir / "sales.csv.gz")
        sales["Date"] = parse_dates(sales["Date"])
        self.matrix = sales_matrix(sales)                       # (Store, Dept) x Friday grid
        self.A = self.matrix.to_numpy(dtype=float)
        self.weeks: pd.DatetimeIndex = self.matrix.columns
        self.week0: pd.Timestamp = self.weeks[0]
        self.last_actual: pd.Timestamp = self.weeks[-1]
        self.n_hist = len(self.weeks)
        self.series_keys = [(int(s), int(d)) for s, d in self.matrix.index]
        self.series_pos = {k: i for i, k in enumerate(self.series_keys)}
        self.series_store = np.array([k[0] for k in self.series_keys])
        self.series_dept = np.array([k[1] for k in self.series_keys])

        stores = pd.read_csv(ref_dir / "stores.csv")
        self.stores = stores.set_index("Store")
        self.store_type = self.stores["Type"].to_dict()
        self.store_size = self.stores["Size"].astype(float).to_dict()

        ind = pd.read_csv(ref_dir / "indicators.csv.gz")
        ind["Date"] = parse_dates(ind["Date"])
        ind["IsHoliday"] = ind["IsHoliday"].astype(bool)
        ind["Macro_Estimated"] = ind["Macro_Estimated"].astype(bool)
        self.indicators = ind
        self.ind_first: pd.Timestamp = ind["Date"].min()
        self.ind_last: pd.Timestamp = ind["Date"].max()
        self.calendar = HolidayCalendar.from_indicators(ind)
        self._ind_by_date = {d: g.set_index("Store") for d, g in ind.groupby("Date")}
        self._first_row = ind.sort_values("Date").groupby("Store").head(1).set_index("Store")
        # last row with *observed* macro values, for carry-forward beyond the table
        self._last_row = ind.sort_values("Date").groupby("Store").tail(1).set_index("Store")
        ind["_wk"] = ind["Date"].dt.isocalendar().week.astype(int)
        self._clim = ind.groupby(["Store", "_wk"])["Temperature"].mean()
        self._clim_all = ind.groupby("_wk")["Temperature"].mean()
        self._est_cache: dict[pd.Timestamp, pd.DataFrame] = {}

    # ------------------------------------------------------------------ dates
    def week_index(self, friday: pd.Timestamp) -> int:
        """Column index of a Friday on the (extended) weekly grid."""
        return int((pd.Timestamp(friday) - self.week0).days // 7)

    def week_date(self, c: int) -> pd.Timestamp:
        return self.week0 + pd.Timedelta(days=7 * int(c))

    @property
    def supported_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Earliest / latest week the service will forecast (needs 3 weeks of history)."""
        return self.week0 + pd.Timedelta(days=21), self.last_actual + pd.Timedelta(days=7 * 104)

    # ------------------------------------------------------------- indicators
    def indicators_for(self, friday: pd.Timestamp) -> pd.DataFrame:
        """Indicators for every store in the week of ``friday`` (index = Store).

        Columns: the 9 indicators, IsHoliday, ``est_macro`` (CPI/unemployment were
        carried forward) and ``est_other`` (the week is outside the dataset, so
        temperature / fuel / promotions are estimated).
        """
        f = pd.Timestamp(friday)
        hit = self._ind_by_date.get(f)
        if hit is not None:
            out = hit[[*IND_COLS, "IsHoliday", "Macro_Estimated"]].rename(columns={"Macro_Estimated": "est_macro"})
            out = out.copy()
            out["est_other"] = False
            return out
        if f in self._est_cache:
            return self._est_cache[f]

        ref = self._last_row if f > self.ind_last else self._first_row
        out = pd.DataFrame(index=ref.index)
        wk = int(f.isocalendar().week)
        clim = [self._clim.get((s, wk), self._clim_all.get(wk, np.nan)) for s in out.index]
        out["Temperature"] = clim
        out["Fuel_Price"] = ref["Fuel_Price"]
        out["CPI"] = ref["CPI"]
        out["Unemployment"] = ref["Unemployment"]
        for m in MARKDOWNS:
            out[m] = 0.0
        out["IsHoliday"] = bool(self.calendar.is_holiday(f))
        out["est_macro"] = True
        out["est_other"] = True
        if len(self._est_cache) > 400:
            self._est_cache.clear()
        self._est_cache[f] = out
        return out

    def indicator_snapshot(self, store: int, friday: pd.Timestamp) -> dict:
        """What the UI auto-fills for a store + week, with provenance per field."""
        if store not in self.store_type:
            raise KeyError(f"Unknown store {store}")
        row = self.indicators_for(friday).loc[store]
        est_o, est_m = bool(row["est_other"]), bool(row["est_macro"])
        src = lambda est: "estimated" if est else "historical"  # noqa: E731
        return {
            "week": pd.Timestamp(friday).strftime("%Y-%m-%d"),
            "store": int(store),
            "temperature": {"value": round(float(row["Temperature"]), 2), "source": src(est_o), "unit": "F"},
            "fuel_price": {"value": round(float(row["Fuel_Price"]), 3), "source": src(est_o), "unit": "USD/gal"},
            "cpi": {"value": round(float(row["CPI"]), 3), "source": src(est_m), "unit": "index"},
            "unemployment": {"value": round(float(row["Unemployment"]), 3), "source": src(est_m), "unit": "%"},
            "markdowns": {
                "value": [round(float(row[m]), 2) for m in MARKDOWNS],
                "source": src(est_o),
                "unit": "USD",
            },
            "is_holiday": {"value": bool(row["IsHoliday"]), "source": "historical" if not est_o else "estimated"},
        }

    def indicator_series(self, store: int, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
        """Weekly indicator values for a store (used for the context sparkline)."""
        g = self.indicators[self.indicators["Store"] == store]
        g = g[(g["Date"] >= start) & (g["Date"] <= end)]
        return [
            {"date": d.strftime("%Y-%m-%d"), "temperature": float(t), "fuel_price": float(f), "cpi": float(c),
             "unemployment": float(u)}
            for d, t, f, c, u in zip(g["Date"], g["Temperature"], g["Fuel_Price"], g["CPI"], g["Unemployment"])
        ]

    # ------------------------------------------------------------------ lists
    def store_list(self) -> list[dict]:
        rows = []
        for s in sorted(self.store_type):
            mask = self.series_store == s
            active = mask & ~np.isnan(self.A[:, -4:]).all(axis=1)
            weekly_total = np.nansum(self.A[mask][:, -52:], axis=0).mean() if mask.any() else 0.0
            rows.append({
                "store": int(s), "type": self.store_type[s], "size": int(self.store_size[s]),
                "departments": int(active.sum()),
                "avg_weekly_sales": float(weekly_total),
            })
        return rows

    def dept_list(self, store: int) -> list[dict]:
        rows = []
        for i in np.where(self.series_store == store)[0]:
            a = self.A[i]
            obs = np.where(~np.isnan(a))[0]
            if len(obs) == 0:
                continue
            last = int(obs[-1])
            rows.append({
                "dept": int(self.series_dept[i]),
                "weeks": int(len(obs)),
                "last_week": self.weeks[last].strftime("%Y-%m-%d"),
                "active": bool(self.n_hist - 1 - last <= 4),
                "avg_weekly_sales": float(np.nanmean(a[-52:])) if not np.isnan(a[-52:]).all() else float(np.nanmean(a)),
            })
        rows.sort(key=lambda r: r["dept"])
        return rows

    def history(self, store: int, dept: int, start: pd.Timestamp | None = None,
                end: pd.Timestamp | None = None) -> list[dict]:
        i = self.series_pos.get((int(store), int(dept)))
        if i is None:
            return []
        s = pd.Series(self.A[i], index=self.weeks)
        if start is not None:
            s = s[s.index >= start]
        if end is not None:
            s = s[s.index <= end]
        s = s.dropna()
        return [{"date": d.strftime("%Y-%m-%d"), "sales": round(float(v), 2)} for d, v in s.items()]
