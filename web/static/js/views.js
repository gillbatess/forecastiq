/* ForecastIQ front end - views */
(function () {
  "use strict";
  const FIQ = window.FIQ;
  const { $, $$, esc, money, int, pct, pctPlain, dShort, dFull, dMon, icon, api } = FIQ;
  FIQ.views = {};
  FIQ.state = { meta: null, cache: {}, ind: null, lastForecast: null };

  const cached = async (path) => (FIQ.state.cache[path] ||= await api(path));
  const setBusy = (btn, busy, label) => {
    btn.disabled = busy;
    if (busy) { btn.dataset.label = btn.innerHTML; btn.innerHTML = `<span class="spin"></span> ${label || "Working..."}`; }
    else if (btn.dataset.label) btn.innerHTML = btn.dataset.label;
  };
  const deltaHtml = (v) => v == null ? "" : `<span class="delta ${v >= 0 ? "up" : "down"}">${v >= 0 ? "&#9650;" : "&#9660;"} ${pct(v)}</span>`;
  const kpi = (label, value, sub, tip, cls) =>
    `<div class="card kpi"><div class="l">${label}${tip ? ` <span class="info-tip" title="${esc(tip)}" tabindex="0" aria-label="${esc(tip)}">i</span>` : ""}</div><div class="v num ${cls || ""}">${value}</div><div class="s">${sub || ""}</div></div>`;
  const skeleton = (n = 4) => `<div class="grid g4">${Array.from({ length: n }, () => `<div class="card"><div class="skeleton" style="height:14px;width:50%"></div><div class="skeleton" style="height:30px;width:70%;margin-top:14px"></div></div>`).join("")}</div><div class="card mt"><div class="skeleton" style="height:340px"></div></div>`;
  const pageHead = (title, sub, right) => `<div class="page-head"><div><h1>${title}</h1><p>${sub}</p></div><div>${right || ""}</div></div>`;
  const weekFriday = (iso) => { const [y, m, d] = iso.split("-").map(Number); const t = new Date(Date.UTC(y, m - 1, d)); const wd = (t.getUTCDay() + 6) % 7; return FIQ.addDays(iso, 4 - wd); };
  const srcBadge = (s) => s === "historical" ? `<span class="badge good"><span class="dot"></span>Historical data</span>`
    : s === "override" ? `<span class="badge brand"><span class="dot"></span>Your value</span>`
    : `<span class="badge warn"><span class="dot"></span>Estimated</span>`;
  const modeBadge = (m) => m === "forecast" ? `<span class="badge brand">${icon("sparkle", 13)} Forecast</span>`
    : m === "backtest" ? `<span class="badge info">${icon("check", 13)} Out-of-sample back-test</span>`
    : `<span class="badge warn">${icon("alert", 13)} In-sample (training period)</span>`;
  const confBadge = (c) => ({ high: `<span class="badge good"><span class="dot"></span>High confidence</span>`,
    medium: `<span class="badge warn"><span class="dot"></span>Medium confidence</span>`,
    low: `<span class="badge bad"><span class="dot"></span>Low confidence</span>`,
    backtest: "" }[c] || "");
  const chartCard = (id, title, sub, legendHtml) => `
    <div class="card">
      <div class="card-head"><div><div class="card-title">${title}</div><div class="card-sub">${sub || ""}</div></div>
        <div class="chart-tools"><div class="seg" role="group" aria-label="Chart or table view"><button type="button" data-view="chart" aria-pressed="true">Chart</button><button type="button" data-view="table" aria-pressed="false">Table</button></div></div></div>
      ${legendHtml || ""}
      <div class="chart-box" id="${id}-box" style="margin-top:10px"><canvas id="${id}" role="img" aria-label="${esc(title)}"></canvas></div>
      <div id="${id}-table" hidden></div>
    </div>`;
  const wireChartToggle = (id, labels, datasets) => {
    const card = $("#" + id).closest(".card");
    $$("[data-view]", card).forEach((b) => b.addEventListener("click", () => {
      const table = b.dataset.view === "table";
      $$("[data-view]", card).forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      $("#" + id + "-box").hidden = table;
      const t = $("#" + id + "-table"); t.hidden = !table;
      if (table && !t.innerHTML) t.innerHTML = FIQ.dataTable(labels, datasets);
    }));
  };

  // ============================================================== OVERVIEW
  FIQ.views.overview = async function (root) {
    root.innerHTML = pageHead("Overview", "Portfolio demand across every store and department, with the forecast for the next quarter.") + skeleton();
    const ov = await cached("/api/overview");
    const k = ov.kpis, meta = FIQ.state.meta;
    const H = ov.history.length;
    const labels = [...ov.history.map((r) => r.date), ...ov.forecast.map((r) => r.date)];
    const idx = Object.fromEntries(labels.map((l, i) => [l, i]));
    const blank = () => labels.map(() => null);
    const actual = blank(), fc = blank(), ly = blank(), bt = blank();
    ov.history.forEach((r) => { actual[idx[r.date]] = r.actual; });
    ov.backtest.forEach((r) => { if (idx[r.date] != null) bt[idx[r.date]] = r.forecast; });
    ov.forecast.forEach((r) => { fc[idx[r.date]] = r.forecast; ly[idx[r.date]] = r.last_year; });
    fc[H - 1] = actual[H - 1];   // visually join the lines
    const datasets = [
      { label: "Actual sales", data: actual, color: FIQ.css("--c-actual"), width: 2 },
      { label: "Back-test forecast (out-of-sample)", data: bt, color: FIQ.css("--c-forecast"), width: 1.5, dash: [5, 4] },
      { label: "Forecast", data: fc, color: FIQ.css("--c-forecast"), width: 2.5 },
      { label: "Same weeks last year", data: ly, color: FIQ.css("--c-ly"), width: 1.75, dash: [2, 4] },
    ];
    const acc = 100 - k.holdout_wape, naiveAcc = 100 - k.naive_wape;

    root.innerHTML = pageHead("Overview", "Portfolio demand across every store and department, with the forecast for the next quarter.",
      `<a class="btn btn-primary" href="#/forecast">${icon("forecast", 16)} Open forecast studio</a>`) + `
      <div class="grid g4">
        ${kpi("Latest week's sales", money(k.last_week_sales), `Week ending ${dFull(k.last_week)}`)}
        ${kpi("Next 13 weeks (forecast)", money(k.forecast_13w), `${deltaHtml(k.forecast_13w_vs_last_year_pct)} vs same weeks last year`)}
        ${kpi("Forecast accuracy", pctPlain(acc), `Out-of-sample &middot; last-week naive: ${pctPlain(naiveAcc)}`, "Accuracy = 100% minus WAPE (weighted absolute percentage error) on Jan-Oct 2012, weeks the model never saw.")}
        ${kpi("Active series", int(k.active_series), `store &times; department pairs in ${k.stores} stores`)}
      </div>
      <div class="mt">${chartCard("ov-chart", "Total weekly sales &amp; forecast",
        `All ${k.stores} stores &middot; actuals to ${dFull(k.last_week)}, forecast to ${dFull(ov.forecast[ov.forecast.length - 1].date)}`,
        FIQ.legend([{ label: "Actual sales", color: datasets[0].color }, { label: "Back-test forecast (out-of-sample)", color: datasets[1].color, dash: true },
          { label: "Forecast", color: datasets[2].color }, { label: "Same weeks last year", color: datasets[3].color, dash: true }]))}</div>
      <div class="grid g-main mt">
        <div class="card">
          <div class="card-head"><div><div class="card-title">Where the demand is</div><div class="card-sub">Trailing 52-week sales and the next 13-week forecast</div></div>
            <div class="seg" role="tablist"><button role="tab" aria-selected="true" data-t="dept">Departments</button><button role="tab" aria-selected="false" data-t="store">Stores</button></div></div>
          <div id="ov-table"></div>
        </div>
        <div class="card">
          <div class="card-head"><div><div class="card-title">Fastest-growing stores</div><div class="card-sub">Next 13 weeks vs the same weeks last year</div></div></div>
          <div id="ov-growth"></div>
        </div>
      </div>`;

    FIQ.destroyCharts();
    FIQ.lineChart($("#ov-chart"), { labels, datasets, xFmt: dMon, maxTicks: 9, beginAtZero: false,
      marker: { index: H, label: "Forecast" }, tooltipFilter: (i) => !(i.datasetIndex === 2 && i.dataIndex === H - 1) });
    wireChartToggle("ov-chart", labels, datasets);

    const tbl = (rows, cols) => `<div class="table-wrap"><table class="tbl"><thead><tr>${cols.map((c) => `<th class="${c.r ? "r" : ""}">${c.h}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`;
    const growthPill = (g) => g == null ? "-" : `<span class="badge ${g >= 0 ? "good" : "bad"}">${g >= 0 ? "&#9650;" : "&#9660;"} ${pct(g)}</span>`;
    const drawTable = (which) => {
      if (which === "dept") {
        $("#ov-table").innerHTML = tbl(ov.top_departments.map((d) => `<tr><td>Dept ${d.dept}</td><td class="r num">${money(d.trailing_52w)}</td><td class="r num">${money(d.forecast_13w)}</td><td class="r">${growthPill(d.growth_pct)}</td></tr>`).join(""),
          [{ h: "Department" }, { h: "52-wk sales", r: 1 }, { h: "Next 13 wks", r: 1 }, { h: "vs last yr", r: 1 }]);
      } else {
        $("#ov-table").innerHTML = tbl(ov.top_stores.map((d) => `<tr style="cursor:pointer" data-store="${d.store}" tabindex="0"><td>Store ${d.store} <span class="badge">Type ${d.type}</span></td><td class="r num">${money(d.trailing_52w)}</td><td class="r num">${money(d.forecast_13w)}</td><td class="r">${growthPill(d.growth_pct)}</td></tr>`).join(""),
          [{ h: "Store" }, { h: "52-wk sales", r: 1 }, { h: "Next 13 wks", r: 1 }, { h: "vs last yr", r: 1 }]);
        $$("#ov-table tr[data-store]").forEach((tr) => {
          const go = () => { location.hash = `#/forecast?store=${tr.dataset.store}&dept=1`; };
          tr.addEventListener("click", go); tr.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
        });
      }
    };
    drawTable("dept");
    $$("[data-t]").forEach((b) => b.addEventListener("click", () => {
      $$("[data-t]").forEach((x) => x.setAttribute("aria-selected", String(x === b))); drawTable(b.dataset.t);
    }));
    $("#ov-growth").innerHTML = ov.store_growth.map((s) => `
      <a href="#/forecast?store=${s.store}&dept=1" style="display:flex;align-items:center;justify-content:space-between;padding:11px 0;border-bottom:1px solid var(--line-2);color:inherit">
        <span><b>Store ${s.store}</b> <span class="badge">Type ${s.type}</span><br><span class="hint">${money(s.forecast_13w)} next 13 weeks</span></span>${growthPill(s.growth_pct)}</a>`).join("");
  };

  // ============================================================ FORECAST STUDIO
  FIQ.views.forecast = async function (root, q) {
    const meta = FIQ.state.meta;
    const nextWeek = FIQ.addDays(meta.last_actual_week, 7);
    const st = { store: +q.store || 1, dept: +q.dept || 1, date: q.date || nextWeek, h: +q.h || 4, adjust: false, ind: null };
    const presets = [
      { label: "Next 4 weeks", store: 1, dept: 1, date: nextWeek, h: 4 },
      { label: "Holiday peak", store: 20, dept: 92, date: "2012-11-23", h: 4 },
      { label: "Back-test (2012)", store: 10, dept: 7, date: "2012-08-03", h: 4 },
    ];
    root.innerHTML = pageHead("Forecast studio", "Pick a store, department and week. Economic indicators, sales history and holiday effects are filled in automatically from your data.") + `
      <div class="studio">
        <form class="panel" id="fc-form" autocomplete="off" novalidate>
          <div class="card" style="display:flex;flex-direction:column;gap:14px">
            <div class="field"><label for="f-store">Store</label><select class="select" id="f-store">${meta.stores.map((s) => `<option value="${s.store}">Store ${s.store} &middot; Type ${s.type} &middot; ${int(s.size / 1000)}k sq ft</option>`).join("")}</select></div>
            <div class="field"><label for="f-dept">Department</label><select class="select" id="f-dept"><option>Loading...</option></select></div>
            <div class="field"><label for="f-date">Forecast from week of</label>
              <input class="input" type="date" id="f-date" min="${meta.supported_from}" max="${meta.supported_to}" value="${st.date}">
              <span class="hint" id="f-week"></span></div>
            <div class="field"><span class="label" id="f-h-l">Weeks to forecast</span>
              <div class="seg" role="group" aria-labelledby="f-h-l" id="f-h">${[1, 4, 8, 13].map((n) => `<button type="button" data-h="${n}" aria-pressed="${n === st.h}">${n}</button>`).join("")}</div></div>
            <div class="presets" aria-label="Examples">${presets.map((p, i) => `<button type="button" class="chip-btn" data-p="${i}">${p.label}</button>`).join("")}</div>
          </div>
          <div class="card" style="display:flex;flex-direction:column;gap:12px">
            <div class="card-head" style="margin:0"><div><div class="card-title">Economic indicators</div><div class="card-sub" id="ind-sub">Auto-filled for the selected week</div></div>
              <label class="hint" style="display:flex;gap:6px;align-items:center;cursor:pointer"><input type="checkbox" id="f-adjust"> Adjust</label></div>
            <div class="ind-list" id="ind-list"><div class="skeleton" style="height:180px"></div></div>
            <p class="hint" id="ind-help">Looked up from historical records for this store and week. Turn on <b>Adjust</b> to test a what-if scenario.</p>
          </div>
          <button class="btn btn-primary btn-lg" id="f-run" type="submit" style="width:100%">Generate forecast ${icon("arrow", 16)}</button>
        </form>
        <div id="fc-out" aria-live="polite"></div>
      </div>`;

    const el = { store: $("#f-store"), dept: $("#f-dept"), date: $("#f-date"), week: $("#f-week"), list: $("#ind-list"), run: $("#f-run"), out: $("#fc-out"), adjust: $("#f-adjust") };
    el.store.value = st.store;

    const emptyState = () => { el.out.innerHTML = `<div class="card empty"><span class="ico">${icon("forecast", 26)}</span><h3>Your forecast will appear here</h3><p>Choose a store, department and week, then generate a forecast.</p></div>`; };
    emptyState();

    async function loadDepts(keep) {
      el.dept.innerHTML = `<option>Loading...</option>`;
      const r = await cached(`/api/stores/${st.store}/departments`);
      el.dept.innerHTML = r.departments.map((d) => `<option value="${d.dept}">Dept ${d.dept} &middot; avg ${money(d.avg_weekly_sales)}/wk${d.active ? "" : " (inactive)"}</option>`).join("");
      const has = r.departments.some((d) => d.dept === keep);
      st.dept = has ? keep : (r.departments.find((d) => d.active) || r.departments[0]).dept;
      el.dept.value = st.dept;
    }

    function renderInd() {
      const ind = st.ind;
      if (!ind) return;
      const rows = [
        ["temperature", "Temperature", ind.temperature, "&deg;F", 1], ["fuel_price", "Fuel price", ind.fuel_price, "$/gal", 0.001],
        ["cpi", "Consumer price index", ind.cpi, "", 0.001], ["unemployment", "Unemployment rate", ind.unemployment, "%", 0.001],
      ];
      const show = (k, n, o, u, step) => st.adjust
        ? `<div class="ind"><span class="n">${n}</span><input class="input" type="number" step="any" data-k="${k}" data-orig="${o.value}" value="${o.value}" aria-label="${n} ${u}"><span class="src">${srcBadge(o.source)}</span></div>`
        : `<div class="ind"><span class="n">${n}</span><span class="v num">${k === "fuel_price" ? "$" + o.value.toFixed(3) : k === "unemployment" ? o.value.toFixed(2) + "%" : k === "temperature" ? o.value.toFixed(1) + "&deg;F" : o.value.toFixed(2)}</span><span class="src">${srcBadge(o.source)}</span></div>`;
      const md = ind.markdowns;
      const mdTotal = md.value.reduce((a, b) => a + b, 0);
      const mdHtml = st.adjust
        ? `<div class="ind"><span class="n">Promo markdowns ($)</span><span></span><div style="grid-column:1/-1;display:grid;grid-template-columns:repeat(3,1fr);gap:6px">${md.value.map((v, i) => `<label class="hint">MD${i + 1}<input class="input" type="number" step="any" min="0" data-k="markdown${i + 1}" data-orig="${v}" value="${v}" aria-label="Markdown ${i + 1}"></label>`).join("")}</div><span class="src">${srcBadge(md.source)}</span></div>`
        : `<div class="ind"><span class="n">Promo markdowns</span><span class="v num">${money(mdTotal)}</span><span class="src">${srcBadge(md.source)}</span></div>`;
      const hol = ind.is_holiday;
      el.list.innerHTML = rows.map((r) => show(...r)).join("") + mdHtml +
        `<div class="ind"><span class="n">Holiday week</span><span class="v">${hol.value ? "Yes" : "No"}</span><span class="src">${srcBadge(hol.source)}</span></div>`;
      $("#ind-sub").textContent = `Week ending ${dFull(ind.week)}`;
      $("#ind-help").innerHTML = st.adjust
        ? "Edit any value to test a scenario. Changed values apply to <b>every</b> forecast week; untouched values stay automatic."
        : "Looked up from historical records for this store and week. Turn on <b>Adjust</b> to test a what-if scenario.";
    }

    async function refreshInd() {
      el.week.textContent = st.date ? `Weeks end on Friday: using week ending ${dFull(weekFriday(st.date))}` : "";
      el.list.innerHTML = `<div class="skeleton" style="height:180px"></div>`;
      try {
        st.ind = await api(`/api/indicators?store=${st.store}&date=${st.date}`);
        FIQ.state.ind = st.ind; renderInd();
      } catch (e) { st.ind = null; el.list.innerHTML = `<div class="callout warn">${icon("alert")}<p>${esc(e.message)}</p></div>`; }
    }
    const refreshIndSoon = FIQ.debounce(refreshInd, 250);

    el.store.addEventListener("change", async () => { st.store = +el.store.value; await loadDepts(st.dept); refreshInd(); });
    el.dept.addEventListener("change", () => { st.dept = +el.dept.value; });
    el.date.addEventListener("change", () => { if (el.date.value) { st.date = el.date.value; refreshIndSoon(); } });
    el.adjust.addEventListener("change", () => { st.adjust = el.adjust.checked; renderInd(); });
    $$("#f-h [data-h]").forEach((b) => b.addEventListener("click", () => { st.h = +b.dataset.h; $$("#f-h [data-h]").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); }));
    $$("[data-p]").forEach((b) => b.addEventListener("click", async () => {
      const p = presets[+b.dataset.p]; Object.assign(st, { store: p.store, date: p.date, h: p.h });
      el.store.value = p.store; el.date.value = p.date;
      $$("#f-h [data-h]").forEach((x) => x.setAttribute("aria-pressed", String(+x.dataset.h === p.h)));
      await loadDepts(p.dept); await refreshInd(); run();
    }));

    function collectOverrides() {
      const o = {};
      $$("#ind-list input[data-k]").forEach((i) => {
        if (i.value === "" ) return;
        const v = Number(i.value), orig = Number(i.dataset.orig);
        if (Math.abs(v - orig) > 1e-9) o[i.dataset.k] = v;
      });
      return o;
    }

    async function run(e) {
      if (e) e.preventDefault();
      const overrides = st.adjust ? collectOverrides() : {};
      const body = { store: st.store, dept: st.dept, date: st.date, horizon: st.h };
      if (Object.keys(overrides).length) body.overrides = overrides;
      history.replaceState(null, "", `#/forecast?store=${st.store}&dept=${st.dept}&date=${st.date}&h=${st.h}`);
      setBusy(el.run, true, "Forecasting...");
      el.out.innerHTML = `<div class="card"><div class="skeleton" style="height:420px"></div></div>`;
      try {
        const data = await api("/api/forecast", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        FIQ.state.lastForecast = data; renderResults(data, Object.keys(overrides).length > 0);
      } catch (err) {
        el.out.innerHTML = `<div class="card"><div class="callout bad">${icon("alert")}<p><b>Couldn't generate this forecast.</b><br>${esc(err.message)}</p></div></div>`;
      } finally { setBusy(el.run, false); }
    }
    $("#fc-form").addEventListener("submit", run);

    function renderResults(d, overridden) {
      FIQ.destroyCharts();
      const W = d.weeks, first = W[0];
      const total = d.summary.total_forecast;
      const lows = W.reduce((a, w) => a + w.low, 0), highs = W.reduce((a, w) => a + w.high, 0);
      const withActual = W.filter((w) => w.actual != null);
      const actTotal = withActual.reduce((a, w) => a + w.actual, 0);
      const errTotal = withActual.length ? withActual.reduce((a, w) => a + Math.abs(w.forecast - w.actual), 0) / Math.max(1, Math.abs(actTotal)) * 100 : null;
      const lastHist = d.history.length ? d.history[d.history.length - 1].date : null;

      const cut = FIQ.addDays(first.date, -7 * 30);
      d.history = d.history.filter((h) => h.date >= cut);
      const labels = [...new Set([...d.history.map((h) => h.date), ...W.map((w) => w.date)])].sort();
      const ix = Object.fromEntries(labels.map((l, i) => [l, i])), blank = () => labels.map(() => null);
      const actual = blank(), fc = blank(), lo = blank(), hi = blank(), ly = blank();
      d.history.forEach((h) => { actual[ix[h.date]] = h.sales; });
      W.forEach((w) => { fc[ix[w.date]] = w.forecast; lo[ix[w.date]] = w.low; hi[ix[w.date]] = w.high; if (w.last_year != null) ly[ix[w.date]] = w.last_year; });
      let linkIdx = -1;
      if (first.mode === "forecast" && lastHist && (new Date(first.date) - new Date(lastHist)) / 864e5 <= 14) {
        linkIdx = ix[lastHist]; fc[linkIdx] = actual[linkIdx]; lo[linkIdx] = actual[linkIdx]; hi[linkIdx] = actual[linkIdx];
      }
      const cA = FIQ.css("--c-actual"), cF = FIQ.css("--c-forecast"), cL = FIQ.css("--c-ly");
      const bandFill = cF + "30";
      const datasets = [
        { label: "Actual sales", data: actual, color: cA, width: 2.25, points: labels.length < 30 },
        { label: "Forecast", data: fc, color: cF, width: 2.5, points: true },
        { label: "Range low", data: lo, color: "transparent", width: 0, tooltipHidden: true, order: 5 },
        { label: "80% range", data: hi, color: "transparent", fillColor: bandFill, width: 0, fill: "-1", tooltipHidden: true, order: 5 },
        { label: "Same week last year", data: ly, color: cL, width: 1.75, dash: [2, 4], points: true },
      ];
      const showActualForecastWeeks = withActual.length > 0;

      const notes = d.notes.map((n) => `<div class="callout ${n.startsWith("This week is inside") ? "warn" : "info"}">${icon(n.startsWith("This week is inside") ? "alert" : "info")}<p>${esc(n)}</p></div>`).join("");
      const range = W.length === 1 ? `${money(first.low)}&thinsp;&ndash;&thinsp;${money(first.high)}` : `${money(lows)}&thinsp;&ndash;&thinsp;${money(highs)}`;

      el.out.innerHTML = `
        <div class="res-head"><div><h2>Store ${d.store} &middot; Dept ${d.dept}</h2>
          <div class="meta">${modeBadge(first.mode)}${confBadge(d.summary.confidence)}<span class="badge">Week${W.length > 1 ? "s" : ""} ending ${dShort(W[0].date)}${W.length > 1 ? " &ndash; " + dFull(W[W.length - 1].date) : " " + W[0].date.slice(0, 4)}</span>${overridden ? `<span class="badge brand">Scenario</span>` : ""}</div></div>
          <div style="display:flex;gap:8px"><button class="btn btn-sm" id="r-link" type="button">${icon("link", 15)} Copy link</button><button class="btn btn-sm" id="r-csv" type="button">${icon("download", 15)} CSV</button></div></div>
        <div class="grid g4">
          ${kpi(W.length > 1 ? `Total forecast (${W.length} wks)` : "Forecast", money(total), W.length > 1 ? `Average ${money(d.summary.avg_weekly)} per week` : `Week ending ${dFull(first.date)}`)}
          ${kpi("Likely range (80%)", range, W.length > 1 ? "Sum of weekly ranges" : "80% prediction interval", "There is roughly an 80% chance the actual result falls inside this range.", "sm")}
          ${kpi("vs same weeks last year", d.summary.vs_last_year_pct == null ? "-" : pct(d.summary.vs_last_year_pct), d.summary.vs_last_year_pct == null ? "No prior-year data" : `${money(W.reduce((a, w) => a + (w.last_year || 0), 0))} a year ago`)}
          ${withActual.length ? kpi("Actual (recorded)", money(actTotal), `Forecast error ${pctPlain(errTotal)}${first.mode === "in_sample" ? " (in-sample)" : ""}`)
            : kpi("Furthest week", `${Math.max(...W.map((w) => w.weeks_ahead))} wk${Math.max(...W.map((w) => w.weeks_ahead)) > 1 ? "s" : ""}`, "ahead of the last recorded week")}
        </div>
        ${notes ? `<div class="grid mt" style="gap:10px">${notes}</div>` : ""}
        <div class="mt">${chartCard("fc-chart", "Sales history &amp; forecast", `Store ${d.store}, department ${d.dept}`,
          FIQ.legend([{ label: "Actual sales", color: cA }, { label: "Forecast", color: cF }, { label: "80% range", color: cF, band: true }, { label: "Same week last year", color: cL, dash: true }]))}</div>
        <div class="card mt">
          <div class="tabs" role="tablist"><button role="tab" aria-selected="true" data-tab="weeks">Weekly detail</button><button role="tab" aria-selected="false" data-tab="drivers">What drove it</button><button role="tab" aria-selected="false" data-tab="context">Economic context</button></div>
          <div id="tab-body"></div>
        </div>`;

      FIQ.lineChart($("#fc-chart"), { labels, datasets, xFmt: dShort, maxTicks: 8,
        marker: first.mode === "forecast" && lastHist ? { index: ix[lastHist], label: "Forecast" } : null,
        tooltipFilter: (i) => !(linkIdx >= 0 && i.dataIndex === linkIdx && i.datasetIndex === 1) });
      wireChartToggle("fc-chart", labels, datasets);

      $("#r-link").addEventListener("click", () => { navigator.clipboard && navigator.clipboard.writeText(location.href).then(() => FIQ.toast("Link copied to clipboard", "good")); });
      $("#r-csv").addEventListener("click", () => {
        const head = ["Week_Ending", "Forecast", "Low_80", "High_80", "Actual", "Same_Week_Last_Year", "Type", "Weeks_Ahead", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Markdown_Total"];
        const rows = W.map((w) => [w.date, w.forecast, w.low, w.high, w.actual ?? "", w.last_year ?? "", w.mode, w.weeks_ahead, w.indicators.temperature.value, w.indicators.fuel_price.value, w.indicators.cpi.value, w.indicators.unemployment.value, w.indicators.markdowns.value.reduce((a, b) => a + b, 0)]);
        const csv = [head, ...rows].map((r) => r.join(",")).join("\n");
        const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
        a.download = `forecastiq_store${d.store}_dept${d.dept}.csv`; a.click();
      });

      const tabBody = $("#tab-body");
      const tabs = {
        weeks() {
          tabBody.innerHTML = `<div class="table-wrap"><table class="tbl"><thead><tr><th>Week ending</th><th>Type</th><th class="r">Forecast</th><th class="r">80% range</th><th class="r">Actual</th><th class="r">Error</th><th class="r">Last year</th><th>Holiday</th></tr></thead><tbody>${
            W.map((w) => { const e = w.actual != null && w.actual !== 0 ? (w.forecast - w.actual) / Math.abs(w.actual) * 100 : null;
              return `<tr><td>${dFull(w.date)}</td><td>${w.mode === "forecast" ? `<span class="badge brand">+${w.weeks_ahead}w</span>` : w.mode === "backtest" ? `<span class="badge info">Back-test</span>` : `<span class="badge warn">In-sample</span>`}</td>
              <td class="r num"><b>${money(w.forecast, { full: true })}</b></td><td class="r num">${money(w.low, { full: true })} &ndash; ${money(w.high, { full: true })}</td>
              <td class="r num">${w.actual == null ? "&ndash;" : money(w.actual, { full: true })}</td><td class="r num">${e == null ? "&ndash;" : pct(e)}</td>
              <td class="r num">${w.last_year == null ? "&ndash;" : money(w.last_year, { full: true })}</td><td>${w.indicators.is_holiday.value ? `<span class="badge warn">Holiday</span>` : ""}</td></tr>`; }).join("")}</tbody></table></div>`;
        },
        drivers() {
          tabBody.innerHTML = `<div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:6px">
            <p class="card-sub" style="max-width:62ch">How each factor moved the forecast for the selected week, relative to the model's average prediction. Recent sales dominate; economic indicators refine the result.</p>
            ${W.length > 1 ? `<select class="select" id="drv-week" style="width:auto" aria-label="Week">${W.map((w, i) => `<option value="${i}">${dFull(w.date)}</option>`).join("")}</select>` : ""}</div><div id="drv-body"></div>`;
          const draw = (i) => {
            const w = W[i]; const dr = w.drivers || [];
            if (!dr.length) { $("#drv-body").innerHTML = `<p class="hint">Driver breakdown is unavailable for this week.</p>`; return; }
            const mx = Math.max(...dr.map((x) => Math.abs(x.impact)), 1);
            $("#drv-body").innerHTML = `<div class="metric-row"><span class="k">Model average (baseline)</span><span class="v num">${money(w.baseline, { full: true })}</span></div>` +
              dr.map((x) => { const wpct = Math.abs(x.impact) / mx * 50, pos = x.impact >= 0;
                return `<div class="drv"><span>${esc(x.label)}</span><span class="bar" aria-hidden="true"><i style="background:${pos ? "var(--c-pos)" : "var(--c-neg)"};${pos ? "left:50%" : `right:50%`};width:${wpct}%"></i></span><span class="val num">${pos ? "&#9650; +" : "&#9660; -"}${money(Math.abs(x.impact), { full: true })}</span></div>`; }).join("") +
              `<div class="metric-row" style="border-top:1px solid var(--line);margin-top:8px"><span class="k"><b>Forecast</b></span><span class="v num">${money(w.forecast, { full: true })}</span></div>`;
          };
          draw(0); const sel = $("#drv-week"); if (sel) sel.addEventListener("change", () => draw(+sel.value));
        },
        context() {
          const ctx = (st.ind && st.ind.context) || [];
          const w0 = W[0];
          const defs = [["temperature", "Temperature", (v) => v.toFixed(1) + "°F", w0.indicators.temperature],
            ["fuel_price", "Fuel price ($/gal)", (v) => "$" + v.toFixed(3), w0.indicators.fuel_price], ["cpi", "Consumer price index", (v) => v.toFixed(1), w0.indicators.cpi],
            ["unemployment", "Unemployment rate", (v) => v.toFixed(2) + "%", w0.indicators.unemployment]];
          tabBody.innerHTML = `<p class="card-sub" style="margin-bottom:12px">Conditions in Store ${d.store}'s market around the first forecast week (52 weeks before to 13 weeks after). All values were filled automatically.</p>
            <div class="grid g2">${defs.map(([k, n, f, o]) => `<div class="card ctx"><div class="h"><span class="n">${n}</span>${srcBadge(o.source)}</div><div class="big num">${f(o.value)}</div><div class="chart-box xs"><canvas id="cx-${k}" role="img" aria-label="${n} trend"></canvas></div></div>`).join("")}</div>`;
          defs.forEach(([k, n]) => {
            const labels2 = ctx.map((c) => c.date), i0 = labels2.indexOf(w0.date);
            const ch = FIQ.lineChart($("#cx-" + k), { labels: labels2, xFmt: dMon, maxTicks: 4, beginAtZero: false, yFmt: (v) => (k === "fuel_price" ? v.toFixed(2) : k === "unemployment" ? v.toFixed(1) : Math.round(v)),
              tipFmt: (v) => v.toFixed(2), marker: i0 >= 0 ? { index: i0, color: "--brand" } : null,
              datasets: [{ label: n, data: ctx.map((c) => c[k]), color: FIQ.css("--c-actual"), width: 2 }] });
          });
        },
      };
      const show = (name) => { $$("[data-tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name))); tabs[name](); };
      $$("[data-tab]").forEach((b) => b.addEventListener("click", () => { if (b.dataset.tab !== "context") { FIQ.charts.slice(1).forEach((c) => c.destroy()); FIQ.charts = FIQ.charts.slice(0, 1); } show(b.dataset.tab); }));
      show("weeks");
    }

    await loadDepts(st.dept);
    await refreshInd();
    run();
    FIQ.rerender = () => run();
  };

  // ================================================================== BULK
  FIQ.views.bulk = async function (root) {
    let file = null, resultId = null, page = 0, onlyIssues = false;
    const PAGE = 50;
    root.innerHTML = pageHead("Bulk forecast", "Upload a list of stores, departments and weeks. ForecastIQ looks up history and economic indicators for every row and returns a forecast with a range.",
      `<a class="btn" href="/api/bulk-template">${icon("download", 16)} Download template</a>`) + `
      <div class="grid g-main">
        <div>
          <label class="drop" id="drop" for="file" tabindex="0">
            <span class="ico">${icon("bulk", 26)}</span>
            <h3 id="drop-t">Drop a CSV here, or click to browse</h3>
            <p id="drop-s">Up to 100,000 rows &middot; 25 MB</p>
            <input id="file" type="file" accept=".csv,text/csv" class="sr-only">
          </label>
          <div style="display:flex;gap:10px;margin-top:12px;flex-wrap:wrap">
            <button class="btn btn-primary" id="b-run" disabled>Run bulk forecast ${icon("arrow", 16)}</button>
            <button class="btn" id="b-sample" type="button">${icon("sparkle", 16)} Try the example file</button>
          </div>
        </div>
        <div class="card">
          <div class="card-title">File format</div>
          <p class="card-sub">Only three columns are required. Everything else is looked up automatically.</p>
          <div class="cols"><span class="code">Store</span><span class="code">Dept</span><span class="code">Date</span></div>
          <p class="card-sub" style="margin-top:12px">Optional overrides (leave blank to auto-fill):</p>
          <div class="cols"><span class="code">Temperature</span><span class="code">Fuel_Price</span><span class="code">CPI</span><span class="code">Unemployment</span><span class="code">MarkDown1</span><span class="code">&hellip;</span><span class="code">MarkDown5</span><span class="code">IsHoliday</span></div>
          <p class="card-sub" style="margin-top:12px">Dates use <span class="code">YYYY-MM-DD</span>. Any day of the week is fine; it is matched to that week's Friday.</p>
        </div>
      </div>
      <div id="b-out" class="mt" aria-live="polite"></div>`;
    const out = $("#b-out");
    const setFile = (f) => {
      file = f; $("#b-run").disabled = !f;
      $("#drop-t").textContent = f ? f.name : "Drop a CSV here, or click to browse";
      $("#drop-s").textContent = f ? `${Math.max(1, Math.round(f.size / 1024))} KB ready to score` : "Up to 100,000 rows · 25 MB";
    };
    $("#file").addEventListener("change", (e) => setFile(e.target.files[0] || null));
    const drop = $("#drop");
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) setFile(f); });
    drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#file").click(); } });
    $("#b-sample").addEventListener("click", async () => {
      const csv = await api("/api/bulk-template"); setFile(new File([csv], "forecastiq_template.csv", { type: "text/csv" })); $("#b-run").click();
    });

    $("#b-run").addEventListener("click", async () => {
      if (!file) return;
      const btn = $("#b-run"); setBusy(btn, true, "Scoring...");
      out.innerHTML = `<div class="card"><div class="skeleton" style="height:300px"></div></div>`;
      const fd = new FormData(); fd.append("file", file, file.name);
      try {
        const r = await api("/api/bulk", { method: "POST", body: fd });
        resultId = r.id; page = 0; onlyIssues = false; render(r);
      } catch (e) { out.innerHTML = `<div class="card"><div class="callout bad">${icon("alert")}<p><b>Couldn't score this file.</b><br>${esc(e.message)}</p></div></div>`; }
      finally { setBusy(btn, false); $("#b-run").disabled = !file; }
    });

    function render(r) {
      FIQ.destroyCharts();
      const s = r.summary, a = r.accuracy;
      const labels = r.timeline.map((t) => t.date);
      const hasActual = r.timeline.some((t) => t.actual != null);
      const datasets = hasActual
        ? [{ label: "Actual sales", data: r.timeline.map((t) => t.actual), color: FIQ.css("--c-actual"), width: 2 }, { label: "Forecast", data: r.timeline.map((t) => t.forecast_matched), color: FIQ.css("--c-forecast"), width: 2.5 }]
        : [{ label: "Forecast", data: r.timeline.map((t) => t.forecast), color: FIQ.css("--c-forecast"), width: 2.5 }];
      const issues = Object.entries(r.status_counts).filter(([k]) => k !== "OK");
      out.innerHTML = `
        <div class="grid g4">
          ${kpi("Rows scored", `${int(r.ok_rows)} <span style="font-size:15px;color:var(--muted);font-weight:600">/ ${int(r.rows)}</span>`, r.failed_rows ? `<span class="badge warn">${int(r.failed_rows)} need attention</span>` : `<span class="badge good">${icon("check", 13)} All rows scored</span>`)}
          ${kpi("Total forecast", money(s.total_forecast), s.first_week ? `${dFull(s.first_week)} &ndash; ${dFull(s.last_week)}` : "")}
          ${kpi("Coverage", `${int(s.stores)} stores`, `${int(s.departments)} departments &middot; ${int(s.weeks)} weeks`)}
          ${a ? kpi("Accuracy on known weeks", pctPlain(100 - a.wape), `WAPE ${pctPlain(a.wape)} across ${int(a.rows)} rows${a.in_sample_rows ? ` &middot; ${int(a.in_sample_rows)} in-sample` : ""}`, "Rows whose actual sales are on record are compared with the forecast. In-sample rows were part of training and flatter the result.")
              : kpi("Scored in", `${(r.latency_ms / 1000).toFixed(1)}s`, "Forecast, range and indicators per row")}
        </div>
        ${issues.length ? `<div class="callout warn mt">${icon("alert")}<p><b>Some rows could not be scored:</b> ${issues.map(([k, v]) => `${int(v)} &times; ${esc(k)}`).join(" &middot; ")}. They are kept in the download with a status message.</p></div>` : ""}
        <div class="mt">${chartCard("b-chart", "Forecast by week", hasActual ? "Total across scored rows that have recorded sales" : "Total forecast across all scored rows", FIQ.legend(datasets.map((d) => ({ label: d.label, color: d.color }))))}</div>
        <div class="card mt">
          <div class="card-head"><div><div class="card-title">Results</div><div class="card-sub">Every row with its forecast, range and status</div></div>
            <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">${issues.length ? `<label class="hint" style="display:flex;gap:6px;align-items:center;cursor:pointer"><input type="checkbox" id="b-issues"> Only rows needing attention</label>` : ""}
            <a class="btn btn-sm btn-primary" href="/api/bulk/${r.id}/download">${icon("download", 15)} Download CSV</a></div></div>
          <div id="b-rows"></div>
        </div>`;
      if (labels.length) {
        FIQ.lineChart($("#b-chart"), { labels, datasets, xFmt: dShort, maxTicks: 10 });
        wireChartToggle("b-chart", labels, datasets);
      } else { $("#b-chart-box").innerHTML = `<div class="empty">No rows could be scored.</div>`; }
      const cb = $("#b-issues"); if (cb) cb.addEventListener("change", () => { onlyIssues = cb.checked; page = 0; loadRows(); });
      loadRows();
    }

    async function loadRows() {
      const box = $("#b-rows"); box.innerHTML = `<div class="skeleton" style="height:200px"></div>`;
      const r = await api(`/api/bulk/${resultId}/rows?offset=${page * PAGE}&limit=${PAGE}&only_issues=${onlyIssues}`);
      const cols = ["Store", "Dept", "Week_Ending", "Predicted_Weekly_Sales", "Lower_80", "Upper_80", "Actual_Weekly_Sales", "Forecast_Type", "Weeks_Ahead", "Status"];
      const label = { Week_Ending: "Week ending", Predicted_Weekly_Sales: "Forecast", Lower_80: "Low (80%)", Upper_80: "High (80%)", Actual_Weekly_Sales: "Actual", Forecast_Type: "Type", Weeks_Ahead: "Weeks ahead" };
      const cell = (c, v) => v == null ? "&ndash;" : ["Predicted_Weekly_Sales", "Lower_80", "Upper_80", "Actual_Weekly_Sales"].includes(c) ? money(v, { full: true })
        : c === "Status" ? (v === "OK" ? `<span class="badge good">OK</span>` : `<span class="badge warn">${esc(v)}</span>`) : c === "Week_Ending" ? dFull(v) : esc(v);
      const right = ["Predicted_Weekly_Sales", "Lower_80", "Upper_80", "Actual_Weekly_Sales", "Weeks_Ahead"];
      box.innerHTML = `<div class="table-wrap"><table class="tbl"><thead><tr>${cols.map((c) => `<th class="${right.includes(c) ? "r" : ""}">${label[c] || c}</th>`).join("")}</tr></thead><tbody>${
        r.rows.map((row) => `<tr>${cols.map((c) => `<td class="${right.includes(c) ? "r num" : ""}">${cell(c, row[c])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
        <div class="pager"><span>${int(r.offset + 1)}&ndash;${int(Math.min(r.offset + PAGE, r.total))} of ${int(r.total)} rows</span>
        <span style="display:flex;gap:8px"><button class="btn btn-sm" id="pg-prev" ${page === 0 ? "disabled" : ""}>Previous</button><button class="btn btn-sm" id="pg-next" ${(page + 1) * PAGE >= r.total ? "disabled" : ""}>Next</button></span></div>`;
      $("#pg-prev").addEventListener("click", () => { page--; loadRows(); });
      $("#pg-next").addEventListener("click", () => { page++; loadRows(); });
    }
  };

  // ================================================================= MODEL
  FIQ.views.model = async function (root) {
    root.innerHTML = pageHead("Model performance", "How accurate the forecasts are, measured only on weeks the model never saw during training.") + skeleton();
    const m = await cached("/api/model");
    const v = m.validation, h = v.holdout, base = v.baselines, pk = v.peak_season_backtest, iv = v.interval;
    const naive = base["Last week (naive)"].WAPE, better = (1 - h.WAPE / naive) * 100;
    const bars = (rows, fmt, max, stack) => `<div class="bars${stack ? " stack" : ""}">${rows.map((r) => `<div class="b"><span>${esc(r.label)}</span><span class="track"><span class="fill" style="display:block;width:${Math.max(2, r.value / max * 100)}%;background:${r.color || "var(--c-ly)"}"></span></span><span class="val num">${fmt(r.value)}</span></div>`).join("")}</div>`;
    const cM = FIQ.css("--c-actual"), cG = FIQ.css("--c-ly");
    const rows1 = [{ label: "ForecastIQ model", value: h.WAPE, color: cM }, ...["Last week (naive)", "Same week last year (seasonal naive)", "3-week average"].map((k) => ({ label: k, value: base[k].WAPE }))];
    const rows2 = [{ label: "ForecastIQ model", value: pk.model.WAPE, color: cM }, { label: "Same week last year (seasonal naive)", value: pk.seasonal_naive.WAPE }, { label: "Last week (naive)", value: pk.last_week_naive.WAPE }];
    const imp = m.feature_importance.slice(0, 8), mxI = imp[0].importance;
    const row = (k, val) => `<div class="metric-row"><span class="k">${k}</span><span class="v num">${val}</span></div>`;
    root.innerHTML = pageHead("Model performance", "How accurate the forecasts are, measured only on weeks the model never saw during training.",
      `<span class="badge brand">${esc(m.model_name)} v${esc(m.version)}</span>`) + `
      <div class="grid g4">
        ${kpi("Forecast accuracy", pctPlain(100 - h.WAPE), `WAPE ${pctPlain(h.WAPE, 2)} &middot; out-of-sample`, "WAPE = total absolute error divided by total actual sales. Accuracy = 100% - WAPE.")}
        ${kpi("Error vs naive forecast", `-${better.toFixed(0)}%`, `Naive (repeat last week): WAPE ${pctPlain(naive)}`, "How much lower the model's error is than simply repeating last week's sales.")}
        ${kpi("Typical miss (MAE)", money(h.MAE, { full: true }), `per store-department-week &middot; R&sup2; ${h.R2.toFixed(3)}`)}
        ${kpi("Range reliability", pctPlain(iv.calibrated_coverage_pct_out_of_sample, 0), `of actuals fell inside the 80% range`, "Measured on the second half of the hold-out window, using a range calibrated on the first half.")}
      </div>
      <div class="grid g2 mt">
        <div class="card"><div class="card-head"><div><div class="card-title">Accuracy vs simple baselines</div><div class="card-sub">Jan &ndash; Oct 2012 hold-out &middot; error (WAPE), lower is better &middot; ${int(h.rows)} forecasts</div></div></div>${bars(rows1, (x) => pctPlain(x), Math.max(...rows1.map((r) => r.value)))}</div>
        <div class="card"><div class="card-head"><div><div class="card-title">Holiday-peak back-test</div><div class="card-sub">${esc(pk.window)} &middot; trained only on earlier data</div></div></div>${bars(rows2, (x) => pctPlain(x), Math.max(...rows2.map((r) => r.value)))}
          <p class="hint" style="margin-top:10px">Peak weeks are the hardest. The model matches the strongest baseline here and is clearly better in regular weeks.</p></div>
      </div>
      <div class="grid g-main mt">
        ${chartCard("m-month", "Error by month", "WAPE per month of the 2012 hold-out (lower is better)")}
        <div class="card"><div class="card-head"><div><div class="card-title">What the model relies on</div><div class="card-sub">Share of predictive power (gain)</div></div></div>
          ${bars(imp.map((f) => ({ label: f.label, value: f.importance * 100, color: cM })), (x) => x.toFixed(1) + "%", mxI * 100, true)}</div>
      </div>
      <div class="grid g3 mt">
        <div class="card"><div class="card-title">Segments</div><div class="card-sub" style="margin-bottom:6px">WAPE on the hold-out</div>
          ${Object.entries(v.by_store_type).map(([t, x]) => row(`Store type ${t}`, pctPlain(x))).join("")}
          ${row("Regular weeks", pctPlain(v.holiday_vs_regular.regular_weeks))}${v.holiday_vs_regular.holiday_weeks != null ? row("Holiday weeks", pctPlain(v.holiday_vs_regular.holiday_weeks)) : ""}</div>
        <div class="card"><div class="card-title">Model card</div><div class="card-sub" style="margin-bottom:6px">${esc(m.algorithm)}</div>
          ${row("Training history", `${dFull(m.data.first_week)} &ndash; ${dFull(m.data.last_actual_week)}`)}${row("Series", `${int(m.data.series)} store &times; dept`)}${row("Observations", int(m.data.rows))}${row("Input features", m.feature_count)}${row("Trees / depth", `${m.best_parameters.n_estimators} / ${m.best_parameters.max_depth}`)}</div>
        <div class="card"><div class="card-title">Good to know</div>
          <ul style="margin:8px 0 0;padding-left:18px;color:var(--ink-2);display:grid;gap:8px">
            <li>Multi-week forecasts are built one week at a time, so uncertainty grows with distance.</li>
            <li>Recent sales are the strongest signal; indicator changes refine rather than drive a forecast.</li>
            <li>Economic indicators are auto-filled from this workspace's historical data.</li>
            <li>The demo workspace uses public retail data (2010&ndash;2012).</li></ul></div>
      </div>`;
    FIQ.destroyCharts();
    const bm = v.by_month, ds = [{ label: "WAPE", data: bm.map((b) => b.WAPE), color: cM }];
    const ch = new Chart($("#m-month"), { type: "bar", data: { labels: bm.map((b) => b.month), datasets: [{ label: "WAPE", data: bm.map((b) => b.WAPE), backgroundColor: cM, borderRadius: 4, borderSkipped: "bottom", maxBarThickness: 34 }] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, crosshair: false, tooltip: { backgroundColor: FIQ.css("--surface"), titleColor: FIQ.css("--ink"), bodyColor: FIQ.css("--ink-2"), borderColor: FIQ.css("--line"), borderWidth: 1, callbacks: { label: (i) => ` WAPE: ${i.raw.toFixed(1)}%` } } },
        scales: { x: { grid: { display: false }, ticks: { color: FIQ.css("--muted") } }, y: { beginAtZero: true, grid: { color: FIQ.css("--grid") }, border: { display: false }, ticks: { color: FIQ.css("--muted"), callback: (x) => x + "%" } } } } });
    FIQ.charts.push(ch);
    const card = $("#m-month").closest(".card");
    $$("[data-view]", card).forEach((b) => b.addEventListener("click", () => {
      const t = b.dataset.view === "table"; $$("[data-view]", card).forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      $("#m-month-box").hidden = t; const tb = $("#m-month-table"); tb.hidden = !t;
      tb.innerHTML = `<div class="table-wrap"><table class="tbl"><thead><tr><th>Month</th><th class="r">WAPE</th><th class="r">Forecasts</th></tr></thead><tbody>${bm.map((b2) => `<tr><td>${esc(b2.month)}</td><td class="r num">${pctPlain(b2.WAPE)}</td><td class="r num">${int(b2.rows)}</td></tr>`).join("")}</tbody></table></div>`;
    }));
  };
})();
