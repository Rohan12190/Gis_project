/**
 * GreenPlan GIS - Client Application
 * Pure real-data interface with offline snapshot support and live GEE execution.
 */

// State
const state = {
  map: null,
  regionLayer: null,
  activeOverlay: null,
  currentBounds: null,
  currentOverlays: {},
  currentResults: null,
  studyAreaMarker: null,
  studyAreaCircle: null,
  activeLayerName: "planning_priority",
  opacity: 0.85,
  isOnline: false,
  snapshots: []
};

// Legends definition
const LEGENDS = {
  lulc_baseline: [
    { label: "Built-up", color: "#e11d48" },
    { label: "Green Space", color: "#16a34a" },
    { label: "Farmland", color: "#ca8a04" },
    { label: "Water", color: "#2563eb" },
    { label: "Bare Land", color: "#9ca3af" }
  ],
  lulc_recent: [
    { label: "Built-up", color: "#e11d48" },
    { label: "Green Space", color: "#16a34a" },
    { label: "Farmland", color: "#ca8a04" },
    { label: "Water", color: "#2563eb" },
    { label: "Bare Land", color: "#9ca3af" }
  ],
  urban_growth: [
    { label: "Historical Urban (Baseline)", color: "#94a3b8" },
    { label: "New Urban Expansion", color: "#ef4444" },
    { label: "Non-Urban", color: "#1e293b" }
  ],
  thermal_hotspots: [
    { label: "Extreme Hotspot (99% Conf.)", color: "#b91c1c" },
    { label: "Hotspot (95% Conf.)", color: "#ea580c" },
    { label: "Moderate Hotspot (90% Conf.)", color: "#f59e0b" },
    { label: "Insignificant / Neutral", color: "#475569" },
    { label: "Cool Island (90-99% Conf.)", color: "#0284c7" }
  ],
  planning_priority: [
    { label: "High Risk + High Heat (Protect & Infill)", color: "#ef4444" },
    { label: "High Risk + Med Heat (Buffer)", color: "#f97316" },
    { label: "High Risk + Low Heat (Corridor)", color: "#f59e0b" },
    { label: "Med Risk + High Heat (Thermal Infill)", color: "#ec4899" },
    { label: "Med Risk + Med Heat (Landscaping)", color: "#8b5cf6" },
    { label: "Med Risk + Low Heat (Monitoring)", color: "#3b82f6" },
    { label: "Low Risk + High Heat (Pocket Parks)", color: "#06b6d4" },
    { label: "Low Risk + Med Heat (Canopy Maint.)", color: "#10b981" },
    { label: "Low Risk + Low Heat (Baseline Pres.)", color: "#22c55e" }
  ]
};

// Initialize Map
function initMap() {
  state.map = L.map("map", {
    zoomControl: true,
    attributionControl: false
  }).setView([23.25, 77.41], 7);

  // OpenStreetMap basemap tile layer (free and open; zero API key required)
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(state.map);

  // Map Click Listener
  state.map.on("click", (e) => {
    setCustomPoint(e.latlng.lat, e.latlng.lng);
  });
}

// Load Region Boundary
async function loadRegion() {
  try {
    const res = await fetch("/api/region");
    if (!res.ok) return;
    const geojson = await res.json();
    state.regionLayer = L.geoJSON(geojson, {
      style: {
        color: "#3b82f6",
        weight: 2,
        fillColor: "#3b82f6",
        fillOpacity: 0.04,
        dashArray: "4, 4"
      }
    }).addTo(state.map);
  } catch (err) {
    console.warn("Could not load region boundary:", err);
  }
}

// Check Backend Status
async function checkStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    state.isOnline = data.gee_available;
    state.snapshots = data.snapshots || [];

    const badge = document.getElementById("status-badge");
    if (state.isOnline) {
      badge.className = "badge badge-online";
      badge.innerHTML = `<span class="dot"></span> Online (GEE Live)`;
    } else {
      badge.className = "badge badge-offline";
      badge.innerHTML = `<span class="dot"></span> Offline (Snapshots Only)`;
      document.getElementById("btn-analyze").title = "GEE not configured - use offline snapshots";
    }
  } catch (err) {
    console.warn("API status check failed:", err);
  }
}

// Set custom point from click or inputs
function setCustomPoint(lat, lon) {
  document.getElementById("input-lat").value = lat.toFixed(4);
  document.getElementById("input-lon").value = lon.toFixed(4);

  // De-activate snapshot buttons
  document.querySelectorAll(".quick-btn-group button").forEach(b => b.classList.remove("active"));

  const radius = parseFloat(document.getElementById("input-radius").value);
  updateStudyAreaMarker(lat, lon, radius);
}

function updateStudyAreaMarker(lat, lon, radiusKm) {
  if (state.studyAreaMarker) state.map.removeLayer(state.studyAreaMarker);
  if (state.studyAreaCircle) state.map.removeLayer(state.studyAreaCircle);

  state.studyAreaMarker = L.marker([lat, lon], { draggable: true }).addTo(state.map);
  state.studyAreaCircle = L.circle([lat, lon], {
    radius: radiusKm * 1000,
    color: "#10b981",
    fillColor: "#10b981",
    fillOpacity: 0.1,
    weight: 2
  }).addTo(state.map);

  state.studyAreaMarker.on("dragend", (e) => {
    const pos = e.target.getLatLng();
    setCustomPoint(pos.lat, pos.lng);
  });
}

// Load a saved snapshot
async function loadSnapshot(snapshotId) {
  try {
    const btn = document.getElementById(`btn-${snapshotId}`);
    document.querySelectorAll(".quick-btn-group button").forEach(b => b.classList.remove("active"));
    if (btn) btn.classList.add("active");

    const res = await fetch(`/api/snapshots/${snapshotId}`);
    if (!res.ok) throw new Error("Snapshot not found");
    const snap = await res.json();

    const lat = snap.lat ?? snap.metadata?.lat;
    const lon = snap.lon ?? snap.metadata?.lon;
    const radius = snap.radius_km ?? snap.metadata?.radius_km ?? 15;

    if (lat !== undefined && lon !== undefined) {
      document.getElementById("input-lat").value = Number(lat).toFixed(4);
      document.getElementById("input-lon").value = Number(lon).toFixed(4);
      document.getElementById("input-radius").value = radius;
      document.getElementById("radius-val").textContent = `${radius} km`;
      updateStudyAreaMarker(Number(lat), Number(lon), Number(radius));
    }

    // Bounds [min_lon, min_lat, max_lon, max_lat]
    const b = snap.bounds;
    if (b && b.length === 4) {
      state.currentBounds = [[b[1], b[0]], [b[3], b[2]]];
      state.map.fitBounds(state.currentBounds, { padding: [20, 20] });
    }
    state.currentOverlays = snap.overlays || {};
    state.currentResults = snap.results || snap;

    renderOverlays();
    renderResults(state.currentResults);
  } catch (err) {
    alert("Error loading snapshot: " + err.message);
  }
}

// Render Active Overlay on Map
function renderOverlays() {
  if (state.activeOverlay) {
    state.map.removeLayer(state.activeOverlay);
    state.activeOverlay = null;
  }

  const url = state.currentOverlays[state.activeLayerName];
  if (!url || !state.currentBounds) return;

  state.activeOverlay = L.imageOverlay(url, state.currentBounds, {
    opacity: state.opacity,
    interactive: true
  }).addTo(state.map);

  renderLegend(state.activeLayerName);
}

// Render dynamic legend
function renderLegend(layerName) {
  const container = document.getElementById("legend-container");
  const items = LEGENDS[layerName] || [];
  let html = `<div class="legend-box">`;
  items.forEach(item => {
    html += `
      <div class="legend-row">
        <span class="legend-color" style="background-color: ${item.color}"></span>
        <span>${item.label}</span>
      </div>`;
  });
  html += `</div>`;
  container.innerHTML = html;
}

// Render Results (KPIs, Charts, Matrix, Transparency)
function renderResults(data) {
  const h = data.headline || {};

  // KPIs
  document.getElementById("kpi-baseline-green").textContent = `${(h.green_cover_pct_baseline || 0).toFixed(1)}%`;
  document.getElementById("kpi-recent-green").textContent = `${(h.green_cover_pct_recent || 0).toFixed(1)}%`;
  
  const netLoss = h.net_green_loss_sqkm || 0;
  const netEl = document.getElementById("kpi-net-green");
  netEl.textContent = `${netLoss >= 0 ? "+" : ""}${netLoss.toFixed(2)} km²`;
  netEl.style.color = netLoss > 0 ? "#ef4444" : "#10b981";

  document.getElementById("kpi-urban-growth").textContent = `+${(h.builtup_growth_sqkm || 0).toFixed(2)} km²`;
  document.getElementById("kpi-mean-lst").textContent = `${(h.mean_lst_celsius_recent || 0).toFixed(1)} °C`;
  document.getElementById("kpi-conservation-priority").textContent = `${(h.high_priority_conservation_sqkm || 0).toFixed(2)} km²`;

  // Render Charts
  renderTrendChart(data.annual_greenness || []);
  renderScenarioChart(data.scenarios || []);

  // Render 3x3 Planning Matrix
  renderMatrix(data.planning_matrix || {});

  // Methodological Disclosure
  const acc = data.accuracy || {};
  const t = data.timings || {};
  document.getElementById("meta-overall-acc").textContent = `${((acc.overall_accuracy || 0) * 100).toFixed(1)}%`;
  document.getElementById("meta-kappa").textContent = (acc.kappa || 0).toFixed(3);
  document.getElementById("meta-fom").textContent = (data.hindcast_validation?.figure_of_merit || 0).toFixed(3);
  document.getElementById("meta-timing-total").textContent = `${(t.total_seconds || 0).toFixed(1)}s`;
}

// Render SVG Annual Greenness Trend
function renderTrendChart(data) {
  const container = document.getElementById("trend-chart-container");
  if (!data || data.length === 0) {
    container.innerHTML = `<p style="color: var(--text-muted); font-size:12px;">No historical NDVI data available</p>`;
    return;
  }

  const w = 480, h = 180, pad = 35;
  const years = data.map(d => d.year);
  const vals = data.map(d => d.mean_ndvi);
  const minVal = Math.min(...vals) * 0.9;
  const maxVal = Math.max(...vals) * 1.1;

  const xScale = (i) => pad + (i / (years.length - 1)) * (w - 2 * pad);
  const yScale = (v) => h - pad - ((v - minVal) / (maxVal - minVal)) * (h - 2 * pad);

  let pathD = "";
  let circles = "";
  data.forEach((d, i) => {
    const cx = xScale(i);
    const cy = yScale(d.mean_ndvi);
    if (i === 0) pathD += `M ${cx} ${cy}`;
    else pathD += ` L ${cx} ${cy}`;
    circles += `<circle cx="${cx}" cy="${cy}" r="3" fill="#10b981"><title>${d.year}: NDVI ${d.mean_ndvi.toFixed(3)}</title></circle>`;
  });

  const svg = `
    <svg class="chart-svg" viewBox="0 0 ${w} ${h}">
      <g class="chart-grid">
        <line x1="${pad}" y1="${pad}" x2="${w - pad}" y2="${pad}" />
        <line x1="${pad}" y1="${h / 2}" x2="${w - pad}" y2="${h / 2}" />
        <line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}" />
      </g>
      <path d="${pathD}" fill="none" stroke="#10b981" stroke-width="2.5" />
      ${circles}
      <g class="chart-axis">
        <text x="${pad}" y="${h - 10}">${years[0]}</text>
        <text x="${w / 2 - 15}" y="${h - 10}">${years[Math.floor(years.length / 2)]}</text>
        <text x="${w - pad - 10}" y="${h - 10}">${years[years.length - 1]}</text>
        <text x="5" y="${pad + 10}">${maxVal.toFixed(2)}</text>
        <text x="5" y="${h - pad}">${minVal.toFixed(2)}</text>
      </g>
    </svg>`;
  container.innerHTML = svg;
}

// Render SVG Scenarios Chart (BAU vs Conservation vs Infill)
function renderScenarioChart(scenarios) {
  const container = document.getElementById("scenario-chart-container");
  if (!scenarios || scenarios.length === 0) {
    container.innerHTML = `<p style="color: var(--text-muted); font-size:12px;">No projection data</p>`;
    return;
  }

  // Filter 2035 & 2045 scenarios
  const w = 480, h = 180, pad = 40;
  const targetYear = 2035;
  const yearScenarios = scenarios.filter(s => s.Year === targetYear);
  if (yearScenarios.length === 0) return;

  const maxVal = Math.max(...yearScenarios.map(s => Math.max(s.Builtup_sqkm, s.Green_sqkm))) * 1.15;
  const barWidth = 35;
  const groupWidth = 110;

  let barsHtml = "";
  yearScenarios.forEach((s, idx) => {
    const groupX = pad + 30 + idx * groupWidth;
    const hBuilt = (s.Builtup_sqkm / maxVal) * (h - 2 * pad);
    const hGreen = (s.Green_sqkm / maxVal) * (h - 2 * pad);

    const yBuilt = h - pad - hBuilt;
    const yGreen = h - pad - hGreen;

    barsHtml += `
      <!-- Builtup bar -->
      <rect x="${groupX}" y="${yBuilt}" width="${barWidth}" height="${hBuilt}" fill="#ef4444" rx="2">
        <title>${s.Scenario} Built-up: ${s.Builtup_sqkm.toFixed(1)} km²</title>
      </rect>
      <!-- Green bar -->
      <rect x="${groupX + barWidth + 4}" y="${yGreen}" width="${barWidth}" height="${hGreen}" fill="#10b981" rx="2">
        <title>${s.Scenario} Green: ${s.Green_sqkm.toFixed(1)} km²</title>
      </rect>
      <text x="${groupX + barWidth}" y="${h - pad + 18}" text-anchor="middle" font-size="11" fill="#94a3b8">${s.Scenario}</text>
    `;
  });

  const svg = `
    <svg class="chart-svg" viewBox="0 0 ${w} ${h}">
      <g class="chart-grid">
        <line x1="${pad}" y1="${pad}" x2="${w - pad}" y2="${pad}" />
        <line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}" />
      </g>
      ${barsHtml}
      <g class="chart-axis">
        <text x="5" y="${pad + 10}">${maxVal.toFixed(0)} km²</text>
        <text x="5" y="${h - pad}">0</text>
      </g>
      <!-- Legend -->
      <g transform="translate(${w - 180}, 10)">
        <rect x="0" y="0" width="10" height="10" fill="#ef4444" />
        <text x="14" y="9" font-size="10" fill="#cbd5e1">Built-up</text>
        <rect x="70" y="0" width="10" height="10" fill="#10b981" />
        <text x="84" y="9" font-size="10" fill="#cbd5e1">Green Space</text>
      </g>
    </svg>`;
  container.innerHTML = svg;
}

// Render 3x3 Planning Matrix
function renderMatrix(matrixData) {
  const container = document.getElementById("matrix-grid-container");
  const areas = matrixData.area_breakdown_ha || {};

  // Standard ordered cell definitions (Row: Loss Risk H->L, Col: Thermal Stress H->L)
  const cells = [
    { title: "Protect & Infill Canopy (Urgent Priority)", class: "cell-high-high", stress: "High", risk: "High" },
    { title: "Designate Green Buffer", class: "cell-high-med", stress: "Med", risk: "High" },
    { title: "Preventive Green Corridor Protection", class: "cell-high-low", stress: "Low", risk: "High" },
    { title: "Thermal Infill & Cool Pavements", class: "cell-med-high", stress: "High", risk: "Med" },
    { title: "Sustainable Urban Landscaping", class: "cell-med-med", stress: "Med", risk: "Med" },
    { title: "Conservation Monitoring", class: "cell-med-low", stress: "Low", risk: "Med" },
    { title: "Urban Heat Island Mitigation & Pocket Parks", class: "cell-low-high", stress: "High", risk: "Low" },
    { title: "General Canopy Maintenance", class: "cell-low-med", stress: "Med", risk: "Low" },
    { title: "Baseline Ecological Preservation", class: "cell-low-low", stress: "Low", risk: "Low" }
  ];

  let html = "";
  cells.forEach(c => {
    const ha = areas[c.title] || 0;
    html += `
      <div class="matrix-cell ${c.class}">
        <div class="matrix-cell-title">${c.title}</div>
        <div class="matrix-cell-meta">
          <span class="matrix-stress">T:${c.stress} / R:${c.risk}</span>
          <span class="matrix-area">${ha.toFixed(1)} ha</span>
        </div>
      </div>`;
  });
  container.innerHTML = html;
}

// Live Run Analysis
async function runAnalysis() {
  const lat = parseFloat(document.getElementById("input-lat").value);
  const lon = parseFloat(document.getElementById("input-lon").value);
  const radius = parseFloat(document.getElementById("input-radius").value);
  const name = document.getElementById("input-name").value || "Custom_AOI";

  const progressBox = document.getElementById("progress-box");
  const progressBar = document.getElementById("progress-bar");
  const progressText = document.getElementById("progress-status-text");
  const btnAnalyze = document.getElementById("btn-analyze");

  progressBox.style.display = "block";
  progressBar.style.width = "10%";
  progressText.textContent = "Submitting satellite analysis job to GEE...";
  btnAnalyze.disabled = true;

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lon, radius_km: radius, name })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Analysis request rejected");
    }

    const { job_id } = await res.json();

    // Poll job status
    const pollInterval = setInterval(async () => {
      try {
        const jRes = await fetch(`/api/jobs/${job_id}`);
        const job = await jRes.json();

        progressBar.style.width = `${Math.min(95, (job.progress || 0.1) * 100)}%`;
        progressText.textContent = job.message || `Processing in Earth Engine (${job.status})...`;

        if (job.status === "completed") {
          clearInterval(pollInterval);
          progressBar.style.width = "100%";
          progressText.textContent = "Analysis complete! Rendering results...";

          const r = job.result;
          state.currentBounds = [[r.bounds[1], r.bounds[0]], [r.bounds[3], r.bounds[2]]];
          state.currentOverlays = r.overlays;
          state.currentResults = r.results;

          state.map.fitBounds(state.currentBounds, { padding: [20, 20] });
          renderOverlays();
          renderResults(r.results);

          setTimeout(() => {
            progressBox.style.display = "none";
            btnAnalyze.disabled = false;
          }, 1500);
        } else if (job.status === "failed") {
          clearInterval(pollInterval);
          progressBar.style.width = "100%";
          progressBar.style.backgroundColor = "#ef4444";
          progressText.textContent = `Analysis failed: ${job.error || "Unknown error"}`;
          btnAnalyze.disabled = false;
        }
      } catch (pollErr) {
        console.warn("Polling error:", pollErr);
      }
    }, 2000);
  } catch (err) {
    progressBox.style.display = "none";
    btnAnalyze.disabled = false;
    alert("Live analysis error: " + err.message);
  }
}

// DOM Setup & Event Listeners
window.addEventListener("DOMContentLoaded", async () => {
  initMap();
  await checkStatus();
  await loadRegion();

  // Load default snapshot (Bhopal)
  await loadSnapshot("bhopal");

  // Snapshot buttons
  document.getElementById("btn-bhopal").addEventListener("click", () => loadSnapshot("bhopal"));
  document.getElementById("btn-kothri").addEventListener("click", () => loadSnapshot("kothri"));

  // Radius slider
  const radiusSlider = document.getElementById("input-radius");
  radiusSlider.addEventListener("input", (e) => {
    const val = e.target.value;
    document.getElementById("radius-val").textContent = `${val} km`;
    const lat = parseFloat(document.getElementById("input-lat").value);
    const lon = parseFloat(document.getElementById("input-lon").value);
    if (!isNaN(lat) && !isNaN(lon)) updateStudyAreaMarker(lat, lon, parseFloat(val));
  });

  // Layer buttons
  document.querySelectorAll(".layer-item").forEach(item => {
    item.addEventListener("click", () => {
      document.querySelectorAll(".layer-item").forEach(i => i.classList.remove("active"));
      item.classList.add("active");
      state.activeLayerName = item.dataset.layer;
      renderOverlays();
    });
  });

  // Opacity slider
  document.getElementById("input-opacity").addEventListener("input", (e) => {
    state.opacity = parseFloat(e.target.value);
    if (state.activeOverlay) state.activeOverlay.setOpacity(state.opacity);
  });

  // Analyze button
  document.getElementById("btn-analyze").addEventListener("click", runAnalysis);
});
