"""
dashboard.py
Simple Flask-based UI dashboard for the ML Autoscaler.

Displays:
  1. Current CPU usage
  2. ML Model information
  3. Scaled CPU recommendation (predicted CPU)
  4. Current number of pods
  5. Number of pods added

Run standalone:
    python dashboard.py

By default the dashboard tries to reach the autoscaler API at
http://localhost:5000/api/metrics.  When the autoscaler is not
running it falls back to demo data so the UI can always be previewed.
"""

import os
from datetime import datetime
from flask import Flask, jsonify, render_template_string
import requests

app = Flask(__name__)

# Where the autoscaler metrics API lives
AUTOSCALER_API = os.environ.get("AUTOSCALER_API_URL", "http://localhost:5000/api/metrics")

# ---------------------------------------------------------------------------
# Demo / fallback data (used when the live autoscaler is unreachable)
# ---------------------------------------------------------------------------
_demo_pods_added = 2

DEMO_METRICS = {
    "current_cpu": 0.4231,
    "predicted_cpu": 0.5102,
    "current_replicas": 3,
    "pods_added": _demo_pods_added,
    "status": "STABLE",
    "model": {
        "type": "GRU",
        "r2_score": 0.9138,
        "rmse": 19.7157,
        "mae": 9.1969,
    },
    "scaling_events": [],
    "timestamp": datetime.now().isoformat(),
    "source": "demo",
}


def _count_pods_added(scaling_events: list) -> int:
    """Return the total number of pods added across all scale-up events."""
    total = 0
    for ev in scaling_events:
        old = ev.get("old_replicas", 0)
        new = ev.get("new_replicas", 0)
        if new > old:
            total += new - old
    return total


def _fetch_live_metrics() -> dict | None:
    """Try to fetch metrics from the running autoscaler; return None on failure."""
    try:
        resp = requests.get(AUTOSCALER_API, timeout=3)
        resp.raise_for_status()
        data = resp.json()

        scaling_events = data.get("scaling_events", [])
        pods_added = _count_pods_added(scaling_events)

        # ML model info is static (matches the GRU model used in the project)
        model_info = {
            "type": "GRU",
            "r2_score": 0.9138,
            "rmse": 19.7157,
            "mae": 9.1969,
        }

        return {
            "current_cpu": round(data.get("current_cpu", 0.0), 4),
            "predicted_cpu": round(data.get("predicted_cpu", 0.0), 4),
            "current_replicas": data.get("current_replicas", 1),
            "pods_added": pods_added,
            "status": data.get("status", "UNKNOWN"),
            "model": model_info,
            "scaling_events": scaling_events,
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "source": "live",
        }
    except (requests.RequestException, requests.exceptions.JSONDecodeError, KeyError, ValueError):
        return None


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ML Autoscaler Dashboard</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
      min-height: 100vh;
      padding: 24px 16px;
      color: #e0e0e0;
    }

    /* ── header ── */
    header {
      text-align: center;
      margin-bottom: 32px;
    }
    header h1 {
      font-size: 2rem;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: 1px;
    }
    header .subtitle {
      margin-top: 6px;
      font-size: 0.95rem;
      color: #94a3b8;
    }
    .source-badge {
      display: inline-block;
      margin-top: 10px;
      padding: 4px 12px;
      border-radius: 12px;
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.5px;
    }
    .source-live   { background: #166534; color: #bbf7d0; }
    .source-demo   { background: #78350f; color: #fde68a; }

    /* ── metrics grid ── */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 20px;
      max-width: 1200px;
      margin: 0 auto 32px;
    }

    .card {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 16px;
      padding: 24px 20px;
      text-align: center;
      transition: transform 0.25s, box-shadow 0.25s;
      backdrop-filter: blur(8px);
    }
    .card:hover {
      transform: translateY(-4px);
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }

    .card-icon {
      font-size: 2rem;
      margin-bottom: 10px;
    }
    .card-label {
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: #94a3b8;
      margin-bottom: 12px;
    }
    .card-value {
      font-size: 2.4rem;
      font-weight: 700;
      line-height: 1;
      margin-bottom: 6px;
    }
    .card-unit {
      font-size: 0.82rem;
      color: #64748b;
    }

    /* accent colours per card */
    .accent-cpu    { color: #60a5fa; }
    .accent-model  { color: #a78bfa; }
    .accent-scaled { color: #34d399; }
    .accent-pods   { color: #fb923c; }
    .accent-added  { color: #f472b6; }

    /* model info sub-lines */
    .model-sub {
      margin-top: 10px;
      font-size: 0.82rem;
      color: #94a3b8;
      line-height: 1.8;
    }
    .model-sub span { color: #e2e8f0; font-weight: 600; }

    /* ── status section ── */
    .status-section {
      max-width: 1200px;
      margin: 0 auto 32px;
      display: flex;
      justify-content: center;
    }
    .status-pill {
      padding: 10px 28px;
      border-radius: 40px;
      font-size: 1rem;
      font-weight: 700;
      letter-spacing: 1px;
    }
    .status-STABLE        { background: #14532d; color: #86efac; }
    .status-SCALING_UP    { background: #78350f; color: #fde68a; }
    .status-SCALING_DOWN  { background: #1e3a5f; color: #93c5fd; }
    .status-default       { background: #1e293b; color: #94a3b8; }

    /* ── events table ── */
    .events-section {
      max-width: 1200px;
      margin: 0 auto 32px;
    }
    .section-title {
      font-size: 1.1rem;
      font-weight: 600;
      color: #e2e8f0;
      margin-bottom: 16px;
      padding-left: 4px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: rgba(255,255,255,0.04);
      border-radius: 12px;
      overflow: hidden;
    }
    th, td {
      padding: 12px 16px;
      text-align: left;
      font-size: 0.87rem;
    }
    th {
      background: rgba(255,255,255,0.08);
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      font-weight: 600;
    }
    td { color: #e2e8f0; }
    tr:not(:last-child) td { border-bottom: 1px solid rgba(255,255,255,0.06); }
    tr:hover td { background: rgba(255,255,255,0.04); }
    .no-events { text-align: center; color: #475569; padding: 20px; }

    /* ── footer ── */
    footer {
      text-align: center;
      font-size: 0.78rem;
      color: #334155;
      margin-top: 8px;
    }
    #last-updated { color: #475569; }

    /* ── responsive ── */
    @media (max-width: 600px) {
      header h1 { font-size: 1.4rem; }
      .card-value { font-size: 1.9rem; }
    }
  </style>
</head>
<body>
  <header>
    <h1>&#128640; ML Autoscaler Dashboard</h1>
    <p class="subtitle">Real-time Kubernetes resource management powered by GRU</p>
    <span class="source-badge source-demo" id="source-badge">&#9679; demo data</span>
  </header>

  <!-- 5 metric cards -->
  <div class="metrics-grid">

    <!-- 1. Current CPU -->
    <div class="card">
      <div class="card-icon">&#128202;</div>
      <div class="card-label">Current CPU</div>
      <div class="card-value accent-cpu" id="current-cpu">—</div>
      <div class="card-unit">cores</div>
    </div>

    <!-- 2. ML Model -->
    <div class="card">
      <div class="card-icon">&#129302;</div>
      <div class="card-label">ML Model</div>
      <div class="card-value accent-model" id="model-type">GRU</div>
      <div class="model-sub" id="model-sub">
        R² <span id="model-r2">—</span>&nbsp;&nbsp;
        RMSE <span id="model-rmse">—</span>&nbsp;&nbsp;
        MAE <span id="model-mae">—</span>
      </div>
    </div>

    <!-- 3. Scaled CPU recommendation -->
    <div class="card">
      <div class="card-icon">&#128270;</div>
      <div class="card-label">Scaled CPU Recommendation</div>
      <div class="card-value accent-scaled" id="predicted-cpu">—</div>
      <div class="card-unit">cores (predicted)</div>
    </div>

    <!-- 4. Current pods -->
    <div class="card">
      <div class="card-icon">&#9741;</div>
      <div class="card-label">Current Pods</div>
      <div class="card-value accent-pods" id="current-replicas">—</div>
      <div class="card-unit">replicas</div>
    </div>

    <!-- 5. Pods added -->
    <div class="card">
      <div class="card-icon">&#43;</div>
      <div class="card-label">Pods Added</div>
      <div class="card-value accent-added" id="pods-added">—</div>
      <div class="card-unit">cumulative scale-ups</div>
    </div>

  </div>

  <!-- Status pill -->
  <div class="status-section">
    <div class="status-pill status-default" id="status-pill">INITIALIZING</div>
  </div>

  <!-- Scaling events -->
  <div class="events-section">
    <div class="section-title">&#9889; Recent Scaling Events</div>
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Action</th>
          <th>From</th>
          <th>To</th>
        </tr>
      </thead>
      <tbody id="events-body">
        <tr><td colspan="4" class="no-events">Loading…</td></tr>
      </tbody>
    </table>
  </div>

  <footer>
    Last updated: <span id="last-updated">—</span>
  </footer>

  <script>
    function applyStatus(status) {
      const pill = document.getElementById('status-pill');
      pill.textContent = status;
      pill.className = 'status-pill';
      if (status.includes('SCALING_UP'))   pill.classList.add('status-SCALING_UP');
      else if (status.includes('SCALING_DOWN')) pill.classList.add('status-SCALING_DOWN');
      else if (status === 'STABLE')        pill.classList.add('status-STABLE');
      else                                 pill.classList.add('status-default');
    }

    function renderEvents(events) {
      const tbody = document.getElementById('events-body');
      if (!events || events.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="no-events">No scaling events recorded yet.</td></tr>';
        return;
      }
      tbody.innerHTML = events.map(ev => {
        const t = new Date(ev.timestamp).toLocaleString();
        return `<tr>
          <td>${t}</td>
          <td>${ev.action}</td>
          <td>${ev.old_replicas !== undefined && ev.old_replicas !== null ? ev.old_replicas : '—'}</td>
          <td>${ev.new_replicas !== undefined && ev.new_replicas !== null ? ev.new_replicas : '—'}</td>
        </tr>`;
      }).join('');
    }

    function update(data) {
      document.getElementById('current-cpu').textContent      = data.current_cpu.toFixed(4);
      document.getElementById('predicted-cpu').textContent    = data.predicted_cpu.toFixed(4);
      document.getElementById('current-replicas').textContent = data.current_replicas;
      document.getElementById('pods-added').textContent       = data.pods_added;

      // Model info
      document.getElementById('model-type').textContent  = data.model.type;
      document.getElementById('model-r2').textContent    = data.model.r2_score.toFixed(4);
      document.getElementById('model-rmse').textContent  = data.model.rmse.toFixed(2);
      document.getElementById('model-mae').textContent   = data.model.mae.toFixed(2);

      applyStatus(data.status);
      renderEvents(data.scaling_events);

      // Source badge
      const badge = document.getElementById('source-badge');
      if (data.source === 'live') {
        badge.className = 'source-badge source-live';
        badge.textContent = '● live';
      } else {
        badge.className = 'source-badge source-demo';
        badge.textContent = '● demo data';
      }

      document.getElementById('last-updated').textContent =
        new Date(data.timestamp).toLocaleString();
    }

    function poll() {
      fetch('/api/metrics')
        .then(r => r.json())
        .then(data => update(data))
        .catch(err => console.error('Fetch error:', err));
    }

    // Initial load + auto-refresh every 5 s
    poll();
    setInterval(poll, 5000);
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the dashboard."""
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/metrics")
def metrics():
    """
    Return current autoscaler metrics as JSON.
    Falls back to demo data when the autoscaler process is not reachable.
    """
    data = _fetch_live_metrics()
    if data is None:
        # Return demo data with a fresh timestamp
        demo = dict(DEMO_METRICS)
        demo["timestamp"] = datetime.now().isoformat()
        return jsonify(demo)
    return jsonify(data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 8080))
    print("=" * 60)
    print(" ML AUTOSCALER DASHBOARD")
    print("=" * 60)
    print(f" URL  : http://localhost:{port}")
    print(f" API  : {AUTOSCALER_API}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
