/* ForecastIQ front end - boot + hash router */
(function () {
  "use strict";
  const FIQ = window.FIQ;
  const { $, $$, esc, icon, api } = FIQ;

  const routes = {
    overview: { title: "Overview", view: "overview" },
    forecast: { title: "Forecast studio", view: "forecast" },
    bulk: { title: "Bulk forecast", view: "bulk" },
    model: { title: "Model performance", view: "model" },
  };

  function parseHash() {
    const raw = location.hash.replace(/^#\/?/, "");
    const [name, qs] = raw.split("?");
    const q = Object.fromEntries(new URLSearchParams(qs || ""));
    return { name: routes[name] ? name : "overview", q };
  }

  async function render() {
    const { name, q } = parseHash();
    const r = routes[name];
    FIQ.destroyCharts();
    $$(".nav a").forEach((a) => (a.getAttribute("href") === "#/" + name ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
    $("#crumb").innerHTML = `Demo workspace / <b>${esc(r.title)}</b>`;
    document.title = `${r.title} - ForecastIQ`;
    $("#side").classList.remove("open");
    const root = $("#view");
    try {
      await FIQ.views[r.view](root, q);
    } catch (e) {
      root.innerHTML = `<div class="card"><div class="callout bad">${icon("alert")}<p><b>Something went wrong.</b><br>${esc(e.message)}</p></div></div>`;
    }
    window.scrollTo({ top: 0 });
  }

  async function boot() {
    $("#theme-btn").addEventListener("click", () => FIQ.theme.toggle());
    window.addEventListener("fiq-theme", () => { syncThemeBtn(); render(); });
    $("#menu-btn").addEventListener("click", () => $("#side").classList.toggle("open"));
    window.addEventListener("hashchange", render);
    syncThemeBtn();
    try {
      const [health, meta] = await Promise.all([api("/api/health"), api("/api/meta")]);
      FIQ.state.meta = meta;
      $("#ws-data").textContent = `Data through ${FIQ.dFull(meta.last_actual_week)}`;
      $("#status").innerHTML = `<span class="dot"></span>Model v${esc(health.version)} online`;
    } catch (e) {
      $("#status").classList.add("off"); $("#status").innerHTML = `<span class="dot"></span>Service unavailable`;
      $("#view").innerHTML = `<div class="card"><div class="callout bad">${icon("alert")}<p><b>The forecasting service isn't ready.</b><br>${esc(e.message)}</p></div></div>`;
      return;
    }
    render();
  }

  function syncThemeBtn() {
    const dark = FIQ.theme.isDark();
    $("#theme-btn").innerHTML = `${icon(dark ? "sun" : "moon", 15)} ${dark ? "Light mode" : "Dark mode"}`;
  }
  document.addEventListener("DOMContentLoaded", boot);
})();
