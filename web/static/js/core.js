/* ForecastIQ front end - shared helpers: formatting, API client, icons, chart helpers */
(function () {
  "use strict";
  const FIQ = (window.FIQ = window.FIQ || {});

  // ---------------------------------------------------------------- DOM utils
  FIQ.$ = (sel, root) => (root || document).querySelector(sel);
  FIQ.$$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  FIQ.esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ------------------------------------------------------------- formatting
  const nf0 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
  const nf1 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
  const nf2 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2, minimumFractionDigits: 2 });
  FIQ.int = (v) => (v == null || isNaN(v) ? "-" : nf0.format(v));
  FIQ.num1 = (v) => (v == null || isNaN(v) ? "-" : nf1.format(v));
  FIQ.num2 = (v) => (v == null || isNaN(v) ? "-" : nf2.format(v));
  FIQ.money = (v, opts) => {
    if (v == null || isNaN(v)) return "-";
    const a = Math.abs(v), sign = v < 0 ? "-" : "";
    if (opts && opts.full) return sign + "$" + nf0.format(a);
    if (a >= 1e9) return sign + "$" + nf2.format(a / 1e9).replace(/\.?0+$/, "") + "B";
    if (a >= 1e6) return sign + "$" + nf1.format(a / 1e6) + "M";
    if (a >= 1e4) return sign + "$" + nf1.format(a / 1e3) + "K";
    return sign + "$" + nf0.format(a);
  };
  FIQ.pct = (v, d = 1) => (v == null || isNaN(v) ? "-" : (v > 0 ? "+" : "") + v.toFixed(d) + "%");
  FIQ.pctPlain = (v, d = 1) => (v == null || isNaN(v) ? "-" : v.toFixed(d) + "%");
  const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const parts = (iso) => iso.slice(0, 10).split("-").map(Number);
  FIQ.dShort = (iso) => { const [y, m, d] = parts(iso); return `${d} ${MON[m - 1]}`; };
  FIQ.dMon = (iso) => { const [y, m] = parts(iso); return `${MON[m - 1]} ${String(y).slice(2)}`; };
  FIQ.dFull = (iso) => { const [y, m, d] = parts(iso); return `${d} ${MON[m - 1]} ${y}`; };
  FIQ.addDays = (iso, n) => {
    const [y, m, d] = parts(iso);
    const t = new Date(Date.UTC(y, m - 1, d + n));
    return t.toISOString().slice(0, 10);
  };
  FIQ.debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

  // ------------------------------------------------------------------- icons
  const P = {
    overview: '<rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/>',
    forecast: '<path d="M3 17l6-6 4 4 8-9"/><path d="M15 6h6v6"/>',
    bulk: '<path d="M12 3v12"/><path d="M7 8l5-5 5 5"/><path d="M4 15v4a2 2 0 002 2h12a2 2 0 002-2v-4"/>',
    model: '<path d="M12 3v3"/><path d="M5.6 5.6l2.1 2.1"/><path d="M3 12h3"/><circle cx="12" cy="13" r="7"/><path d="M12 13l3.5-3.5"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    moon: '<path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z"/>',
    external: '<path d="M14 4h6v6"/><path d="M10 14L20 4"/><path d="M20 14v5a1 1 0 01-1 1H5a1 1 0 01-1-1V5a1 1 0 011-1h5"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/>',
    alert: '<path d="M12 3l10 18H2L12 3z"/><path d="M12 10v5"/><path d="M12 18h.01"/>',
    check: '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.7 2.7L16 9.5"/>',
    download: '<path d="M12 4v11"/><path d="M7 11l5 5 5-5"/><path d="M4 20h16"/>',
    link: '<path d="M10 14a4 4 0 005.7 0l3-3a4 4 0 00-5.7-5.7l-1 1"/><path d="M14 10a4 4 0 00-5.7 0l-3 3A4 4 0 0011 18.7l1-1"/>',
    arrow: '<path d="M5 12h14"/><path d="M13 6l6 6-6 6"/>',
    menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
    file: '<path d="M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8z"/><path d="M14 3v5h5"/>',
    sparkle: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/>',
    up: '<path d="M12 19V5"/><path d="M6 11l6-6 6 6"/>',
    home: '<path d="M3 11l9-8 9 8"/><path d="M5 10v10h5v-6h4v6h5V10"/>',
  };
  FIQ.icon = (name, size = 18) =>
    `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${P[name] || ""}</svg>`;

  // --------------------------------------------------------------------- API
  FIQ.api = async function (path, opts) {
    let res;
    try {
      res = await fetch(path, opts);
    } catch (e) {
      throw new Error("Cannot reach the ForecastIQ server. Check your connection and try again.");
    }
    if (!res.ok) {
      let msg = `Request failed (${res.status})`;
      try {
        const j = await res.json();
        if (typeof j.detail === "string") msg = j.detail;
        else if (Array.isArray(j.detail)) msg = j.detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
      } catch (_) { /* ignore */ }
      const err = new Error(msg); err.status = res.status; throw err;
    }
    const ct = res.headers.get("content-type") || "";
    return ct.includes("json") ? res.json() : res.text();
  };

  // ------------------------------------------------------------------- theme
  FIQ.theme = {
    get() { try { return localStorage.getItem("fiq-theme"); } catch (_) { return null; } },
    apply(t) {
      if (t) document.documentElement.setAttribute("data-theme", t); else document.documentElement.removeAttribute("data-theme");
    },
    isDark() {
      const t = document.documentElement.getAttribute("data-theme");
      return t ? t === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    },
    toggle() {
      const next = this.isDark() ? "light" : "dark";
      try { localStorage.setItem("fiq-theme", next); } catch (_) { /* private mode */ }
      this.apply(next);
      window.dispatchEvent(new Event("fiq-theme"));
    },
    init() { this.apply(this.get()); },
  };
  FIQ.theme.init();

  FIQ.css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  // ------------------------------------------------------------ chart helpers
  const crosshair = {
    id: "crosshair",
    afterDatasetsDraw(chart) {
      const a = chart.tooltip && chart.tooltip._active;
      if (!a || !a.length || chart.config.options.plugins.crosshair === false) return;
      const x = a[0].element.x, { top, bottom } = chart.chartArea, ctx = chart.ctx;
      ctx.save(); ctx.strokeStyle = FIQ.css("--muted"); ctx.globalAlpha = .45; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke(); ctx.restore();
    },
  };
  // vertical marker + label, e.g. "Forecast starts"
  const marker = {
    id: "marker",
    afterDatasetsDraw(chart, _a, opts) {
      if (!opts || opts.index == null) return;
      const meta = chart.getDatasetMeta(0), pt = meta.data[opts.index];
      if (!pt) return;
      const { top, bottom } = chart.chartArea, ctx = chart.ctx, x = pt.x;
      ctx.save(); ctx.strokeStyle = FIQ.css(opts.color || "--muted"); ctx.globalAlpha = .55; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke(); ctx.setLineDash([]);
      if (opts.label) {
        ctx.globalAlpha = 1; ctx.fillStyle = FIQ.css("--muted"); ctx.font = "600 11px Inter, sans-serif";
        ctx.textAlign = "left"; ctx.fillText(opts.label, x + 6, top + 12);
      }
      ctx.restore();
    },
  };
  if (window.Chart) Chart.register(crosshair, marker);

  FIQ.charts = [];
  FIQ.destroyCharts = () => { FIQ.charts.forEach((c) => { try { c.destroy(); } catch (_) { /* gone */ } }); FIQ.charts = []; };

  /** Line chart on a category (ISO date) axis. datasets: [{label,data,color,dash,fill,width,points,hidden}] */
  FIQ.lineChart = function (canvas, cfg) {
    const ink = FIQ.css("--muted"), grid = FIQ.css("--grid"), surface = FIQ.css("--surface");
    const ds = cfg.datasets.map((d) => ({
      label: d.label, data: d.data, borderColor: d.color, backgroundColor: d.fillColor || d.color,
      borderWidth: d.width || 2, borderDash: d.dash || [], pointRadius: d.points ? 3.5 : 0, pointHoverRadius: 5,
      pointBackgroundColor: d.color, pointBorderColor: surface, pointBorderWidth: 2,
      tension: 0.15, spanGaps: d.spanGaps !== false, fill: d.fill == null ? false : d.fill, order: d.order || 0,
      hidden: !!d.hidden, tooltipHidden: !!d.tooltipHidden,
    }));
    const chart = new Chart(canvas, {
      type: "line",
      data: { labels: cfg.labels, datasets: ds },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 350 },
        interaction: { mode: "index", intersect: false },
        layout: { padding: { top: 4, right: 6 } },
        scales: {
          x: { grid: { display: false }, border: { color: grid }, ticks: { color: ink, maxTicksLimit: canvas.parentElement.clientWidth < 520 ? 4 : (cfg.maxTicks || 8), maxRotation: 0, autoSkip: true,
                callback(v) { const l = this.getLabelForValue(v); return cfg.xFmt ? cfg.xFmt(l) : l; } } },
          y: { grid: { color: grid }, border: { display: false }, ticks: { color: ink, maxTicksLimit: 6, callback: (v) => (cfg.yFmt || FIQ.money)(v) },
               beginAtZero: cfg.beginAtZero !== false, suggestedMax: cfg.suggestedMax },
        },
        plugins: {
          legend: { display: false },
          marker: cfg.marker || null,
          tooltip: {
            backgroundColor: FIQ.css("--surface"), titleColor: FIQ.css("--ink"), bodyColor: FIQ.css("--ink-2"),
            borderColor: FIQ.css("--line"), borderWidth: 1, padding: 10, boxPadding: 4, usePointStyle: true,
            titleFont: { weight: "650" },
            filter: (item) => !item.dataset.tooltipHidden && item.raw != null && !(cfg.tooltipFilter && !cfg.tooltipFilter(item)),
            callbacks: {
              title: (items) => (items.length ? FIQ.dFull(items[0].label) : ""),
              label: (item) => ` ${item.dataset.label}: ${(cfg.tipFmt || ((v) => FIQ.money(v, { full: true })))(item.raw)}`,
              labelColor: (item) => ({ borderColor: item.dataset.borderColor, backgroundColor: item.dataset.borderColor, borderRadius: 2 }),
            },
          },
        },
      },
    });
    FIQ.charts.push(chart);
    return chart;
  };

  FIQ.legend = (items) =>
    `<div class="legend" role="list">${items.map((i) =>
      `<span class="k" role="listitem"><span class="sw ${i.band ? "band" : ""} ${i.dash ? "dash" : ""}" style="background:${i.color};color:${i.color}"></span>${FIQ.esc(i.label)}</span>`).join("")}</div>`;

  /** Accessible table view for a chart's data. */
  FIQ.dataTable = (labels, datasets, fmt) => {
    const rows = labels.map((l, i) => {
      const cells = datasets.filter((d) => !d.tooltipHidden && !d.band).map((d) => `<td class="r num">${d.data[i] == null ? "" : (fmt || FIQ.money)(d.data[i], { full: true })}</td>`).join("");
      return `<tr><td>${FIQ.dFull(l)}</td>${cells}</tr>`;
    }).join("");
    const head = datasets.filter((d) => !d.tooltipHidden && !d.band).map((d) => `<th class="r">${FIQ.esc(d.label)}</th>`).join("");
    return `<div class="table-wrap" style="max-height:340px;overflow:auto"><table class="tbl"><thead><tr><th>Week ending</th>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
  };

  FIQ.toast = (msg, kind) => {
    let host = FIQ.$("#toasts");
    if (!host) { host = document.createElement("div"); host.id = "toasts"; host.className = "toast"; host.setAttribute("role", "status"); document.body.appendChild(host); }
    host.innerHTML = `<div class="callout ${kind || "bad"}">${FIQ.icon(kind === "good" ? "check" : "alert", 18)}<p>${FIQ.esc(msg)}</p></div>`;
    clearTimeout(FIQ._tt); FIQ._tt = setTimeout(() => { host.innerHTML = ""; }, 7000);
  };
})();
