/* TradingCrew — Chart.js bindings for price + indicators + RSI + MACD. */

(function () {
  const charts = { price: null, rsi: null, macd: null };

  const indicatorColors = {
    close_10_ema: "#facc15",
    close_50_sma: "#60a5fa",
    close_200_sma: "#a855f7",
    boll_ub: "#22c55e",
    boll_lb: "#22c55e",
    boll: "#22c55e",
    atr: "#f97316",
  };

  function commonScaleOptions() {
    return {
      x: {
        type: "time",
        time: { unit: "month", tooltipFormat: "yyyy-MM-dd" },
        ticks: { color: "#64748b", maxRotation: 0 },
        grid: { color: "rgba(31,41,66,0.4)" },
      },
      y: {
        ticks: { color: "#94a3b8" },
        grid: { color: "rgba(31,41,66,0.4)" },
      },
    };
  }

  function destroy(name) {
    if (charts[name]) { charts[name].destroy(); charts[name] = null; }
  }

  function buildPriceChart(canvas, payload, enabledIndicators) {
    destroy("price");
    if (!canvas || !payload?.candles?.length) return null;

    const candles = payload.candles;
    const closeData = candles
      .filter((c) => c.close != null)
      .map((c) => ({ x: c.date, y: c.close }));

    const datasets = [
      {
        label: "Close",
        data: closeData,
        borderColor: "#e2e8f0",
        backgroundColor: "rgba(226, 232, 240, 0.08)",
        tension: 0.18,
        pointRadius: 0,
        borderWidth: 1.6,
        fill: true,
      },
    ];

    const inds = payload.indicators || {};
    enabledIndicators.forEach((key) => {
      if (!inds[key]) return;
      // RSI / MACD / ATR get rendered in dedicated panels
      if (["rsi", "macd", "macds", "macdh", "atr"].includes(key)) return;
      datasets.push({
        label: payload.indicator_labels?.[key] || key,
        data: inds[key].map((p) => ({ x: p.date, y: p.value })),
        borderColor: indicatorColors[key] || "#94a3b8",
        backgroundColor: "transparent",
        pointRadius: 0,
        borderWidth: 1.2,
        borderDash: key.startsWith("boll") ? [4, 4] : [],
        tension: 0.2,
        fill: false,
      });
    });

    charts.price = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: { datasets },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { labels: { color: "#cbd5e1", boxWidth: 14, font: { size: 11 } } },
          tooltip: {
            backgroundColor: "#0a0e1a",
            borderColor: "#1f2942",
            borderWidth: 1,
            titleColor: "#e2e8f0",
            bodyColor: "#cbd5e1",
          },
        },
        scales: commonScaleOptions(),
      },
    });
    return charts.price;
  }

  function buildRsiChart(canvas, payload) {
    destroy("rsi");
    if (!canvas || !payload?.indicators?.rsi?.length) return null;
    const data = payload.indicators.rsi.map((p) => ({ x: p.date, y: p.value }));
    charts.rsi = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        datasets: [{
          label: "RSI(14)",
          data,
          borderColor: "#06b6d4",
          backgroundColor: "rgba(6, 182, 212, 0.15)",
          pointRadius: 0,
          borderWidth: 1.5,
          fill: true,
          tension: 0.2,
        }],
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#cbd5e1", font: { size: 11 } } },
          tooltip: { backgroundColor: "#0a0e1a", borderColor: "#1f2942", borderWidth: 1 },
        },
        scales: {
          ...commonScaleOptions(),
          y: { min: 0, max: 100, ticks: { color: "#94a3b8", stepSize: 25 },
               grid: { color: "rgba(31,41,66,0.4)" } },
        },
      },
    });
    return charts.rsi;
  }

  function buildMacdChart(canvas, payload) {
    destroy("macd");
    const inds = payload?.indicators || {};
    if (!canvas || (!inds.macd && !inds.macds && !inds.macdh)) return null;
    const datasets = [];
    if (inds.macd) {
      datasets.push({
        label: "MACD",
        data: inds.macd.map((p) => ({ x: p.date, y: p.value })),
        borderColor: "#60a5fa",
        backgroundColor: "transparent",
        pointRadius: 0, borderWidth: 1.5, tension: 0.2, fill: false,
      });
    }
    if (inds.macds) {
      datasets.push({
        label: "Signal",
        data: inds.macds.map((p) => ({ x: p.date, y: p.value })),
        borderColor: "#f59e0b",
        backgroundColor: "transparent",
        pointRadius: 0, borderWidth: 1.2, tension: 0.2, fill: false,
      });
    }
    if (inds.macdh) {
      datasets.push({
        type: "bar",
        label: "Histogram",
        data: inds.macdh.map((p) => ({ x: p.date, y: p.value })),
        backgroundColor: (ctx) => {
          const v = ctx.raw?.y ?? 0;
          return v >= 0 ? "rgba(34,197,94,0.55)" : "rgba(239,68,68,0.55)";
        },
        borderWidth: 0,
        barThickness: 2,
      });
    }
    charts.macd = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: { datasets },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#cbd5e1", font: { size: 11 } } },
          tooltip: { backgroundColor: "#0a0e1a", borderColor: "#1f2942", borderWidth: 1 },
        },
        scales: commonScaleOptions(),
      },
    });
    return charts.macd;
  }

  window.TcCharts = {
    buildPriceChart, buildRsiChart, buildMacdChart,
    destroyAll() { destroy("price"); destroy("rsi"); destroy("macd"); },
  };
})();
