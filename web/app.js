const simCanvas = document.getElementById("simCanvas");
const simCtx = simCanvas.getContext("2d");
const frameImage = simCtx.createImageData(1920, 1080);

const chartCanvas = document.getElementById("chartCanvas");
const chartCtx = chartCanvas.getContext("2d");

const history = [];
const MAX_HISTORY = 240;

let isPaused = false;
let fetchingFrame = false;

let lastFrameTime = 0;
async function fetchFrameLoop(now) {
  if (!fetchingFrame && (now - lastFrameTime) >= 33) {
    lastFrameTime = now;
    fetchingFrame = true;
    try {
      const res = await fetch("/api/frame", { cache: "no-store" });
      if (res.ok) {
        const buf = await res.arrayBuffer();
        if (buf.byteLength === 1920 * 1080 * 4) {
          frameImage.data.set(new Uint8ClampedArray(buf));
          simCtx.putImageData(frameImage, 0, 0);
        }
      }
    } catch (_) {}
    fetchingFrame = false;
  }
  requestAnimationFrame(fetchFrameLoop);
}

async function fetchStats() {
  try {
    const res = await fetch("/api/stats", { cache: "no-store" });
    if (!res.ok) return;
    const d = await res.json();

    document.getElementById("valDay").textContent = d.sim_day.toFixed(1);
    document.getElementById("valStepMs").textContent = d.step_ms.toFixed(2);
    document.getElementById("valRt").textContent = d.rt.toFixed(2);
    document.getElementById("valZ").textContent = d.max_cell_z.toFixed(2);
    document.getElementById("valHotspots").textContent = d.hotspot_cells;

    document.getElementById("cntS").textContent = d.s.toLocaleString();
    document.getElementById("cntE").textContent = d.e.toLocaleString();
    document.getElementById("cntI").textContent = d.i.toLocaleString();
    document.getElementById("cntR").textContent = d.r.toLocaleString();
    document.getElementById("cntD").textContent = d.d.toLocaleString();
    document.getElementById("valCfr").textContent = (d.cfr * 100).toFixed(2) + "%";

    const alertBadge = document.getElementById("alertBadge");
    if (d.outbreak_alert) {
      alertBadge.textContent = "BROTE ACTIVO";
      alertBadge.classList.add("active");
    } else {
      alertBadge.textContent = "NORMAL";
      alertBadge.classList.remove("active");
    }

    isPaused = d.paused;
    document.getElementById("btnPause").textContent = isPaused ? "Reanudar" : "Pausar";

    history.push({ s: d.s, e: d.e, i: d.i, r: d.r, d: d.d, total: d.total || 1 });
    if (history.length > MAX_HISTORY) history.shift();
    drawChart();
  } catch (_) {}
}

function drawChart() {
  const w = chartCanvas.width;
  const h = chartCanvas.height;
  chartCtx.clearRect(0, 0, w, h);
  if (history.length < 2) return;

  const series = [
    { key: "s", color: "#38bdf8" },
    { key: "e", color: "#facc15" },
    { key: "i", color: "#ef4444" },
    { key: "r", color: "#22c55e" },
    { key: "d", color: "#64748b" },
  ];

  for (const ser of series) {
    chartCtx.beginPath();
    chartCtx.strokeStyle = ser.color;
    chartCtx.lineWidth = 1.8;
    for (let idx = 0; idx < history.length; ++idx) {
      const x = (idx / (history.length - 1)) * w;
      const frac = history[idx][ser.key] / history[idx].total;
      const y = h - frac * (h - 8) - 4;
      if (idx === 0) chartCtx.moveTo(x, y);
      else chartCtx.lineTo(x, y);
    }
    chartCtx.stroke();
  }
}

async function pushParams() {
  const q = document.getElementById("rngQuarantine").value;
  const b = document.getElementById("rngBeta").value;
  const f = document.getElementById("rngFatality").value;
  const r = document.getElementById("rngRadius").value;

  document.getElementById("lblQuarantine").textContent = Number(q).toFixed(2);
  document.getElementById("lblBeta").textContent = Number(b).toFixed(2);
  document.getElementById("lblFatality").textContent = Number(f).toFixed(3);
  document.getElementById("lblRadius").textContent = Number(r).toFixed(2);

  await fetch(
    `/api/params?quarantine_factor=${q}&beta=${b}&fatality_rate=${f}&infection_radius=${r}`,
    { method: "POST" }
  );
}

for (const id of ["rngQuarantine", "rngBeta", "rngFatality", "rngRadius"]) {
  document.getElementById(id).addEventListener("input", pushParams);
}

document.getElementById("btnPause").addEventListener("click", async () => {
  const action = isPaused ? "resume" : "pause";
  await fetch(`/api/control?action=${action}`, { method: "POST" });
  fetchStats();
});

document.getElementById("btnReset").addEventListener("click", async () => {
  history.length = 0;
  await fetch("/api/control?action=reset", { method: "POST" });
  fetchStats();
});

document.getElementById("btnQuarantine").addEventListener("click", () => {
  document.getElementById("rngQuarantine").value = "0.10";
  pushParams();
});

document.getElementById("btnVaccine").addEventListener("click", () => {
  document.getElementById("rngBeta").value = "0.06";
  pushParams();
});

document.getElementById("btnNormal").addEventListener("click", () => {
  document.getElementById("rngQuarantine").value = "1.00";
  document.getElementById("rngBeta").value = "0.25";
  pushParams();
});

requestAnimationFrame(fetchFrameLoop);
setInterval(fetchStats, 200);
