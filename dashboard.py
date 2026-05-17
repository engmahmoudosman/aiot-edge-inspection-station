#!/usr/bin/env python3
"""
dashboard.py — Web dashboard for the Inspection Station.

Exposes a global `dashboard` (DashboardState) that other modules update,
and `start_dashboard(port)` to launch the Flask server in a background thread.

Open http://<pi-ip>:5000 in any browser on the same network.
"""

import logging
import os
import threading
import time
from collections import deque
from datetime import datetime


# ── Shared state ──────────────────────────────────────────────────────────────

class DashboardState:

    def __init__(self):
        self._lock         = threading.Lock()
        self.start_time    = time.time()
        # Set by main() after arg parsing
        self.sensor        = "?"
        self.proximity_m   = 1.0
        self.arc_deg       = 45.0
        self.motion_m      = 0.5
        self.cooldown_s    = 8.0
        self.min_points    = 3
        # Updated by gate loop
        self.zone_occupied = False
        self.current_dist  = None
        # Updated after each capture
        self.last_capture_path = None
        self.last_capture_time = None
        self.total_captures    = 0
        self.authorized_count  = 0
        self.alert_count       = 0
        self._log = deque(maxlen=30)

    def update_zone(self, occupied: bool, dist):
        with self._lock:
            self.zone_occupied = occupied
            self.current_dist  = dist

    def record_capture(self, path: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            self.last_capture_path = path
            self.last_capture_time = ts
            self.total_captures   += 1

    def record_authorized(self, name: str):
        with self._lock:
            self.authorized_count += 1

    def record_alert(self):
        with self._lock:
            self.alert_count += 1

    def push_event(self, msg: str, kind: str = "info"):
        ts = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self._log.appendleft({"time": ts, "msg": msg, "kind": kind})

    def snapshot(self) -> dict:
        elapsed = int(time.time() - self.start_time)
        h, r = divmod(elapsed, 3600)
        m, s = divmod(r, 60)
        with self._lock:
            return {
                "zone_occupied":     self.zone_occupied,
                "current_dist":      self.current_dist,
                "last_capture_time": self.last_capture_time,
                "total_captures":    self.total_captures,
                "authorized_count":  self.authorized_count,
                "alert_count":       self.alert_count,
                "event_log":         list(self._log),
                "uptime":            f"{h:02d}:{m:02d}:{s:02d}",
                "sensor":            self.sensor,
                "proximity_m":       self.proximity_m,
                "arc_deg":           self.arc_deg,
                "motion_m":          self.motion_m,
                "cooldown_s":        self.cooldown_s,
                "min_points":        self.min_points,
            }


dashboard = DashboardState()


# ── HTML template ─────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Inspection Station</title>
<style>
  :root{--bg:#0d0d0d;--card:#1a1a1a;--border:#2a2a2a;--green:#00e676;
        --red:#ff1744;--blue:#40c4ff;--text:#e0e0e0;--muted:#666}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);
       font-family:'Segoe UI',system-ui,monospace;padding:16px}
  header{display:flex;justify-content:space-between;align-items:center;
         margin-bottom:18px;padding-bottom:12px;border-bottom:1px solid var(--border)}
  h1{font-size:1.2rem;letter-spacing:3px;text-transform:uppercase;color:var(--blue)}
  .dot{display:inline-block;width:9px;height:9px;border-radius:50%;
       background:var(--green);margin-right:8px;animation:pulse 1.5s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;
        max-width:960px;margin:0 auto}
  .card{background:var(--card);border:1px solid var(--border);
        border-radius:10px;padding:18px}
  .lbl{font-size:.65rem;letter-spacing:2px;text-transform:uppercase;
       color:var(--muted);margin-bottom:10px}
  .full{grid-column:1/-1}
  #zone{font-size:2.8rem;font-weight:700;text-align:center;padding:22px;
        border-radius:8px;transition:all .4s;border:2px solid}
  .clear{background:rgba(0,230,118,.1);color:var(--green);border-color:var(--green)}
  .occupied{background:rgba(255,23,68,.15);color:var(--red);border-color:var(--red)}
  #dist{font-size:3.5rem;font-weight:700;text-align:center;
        color:var(--blue);font-variant-numeric:tabular-nums}
  .sub{text-align:center;font-size:.75rem;color:var(--muted);margin-top:6px}
  .stats{display:flex;justify-content:space-around;text-align:center;padding:6px 0}
  .stat-val{font-size:2.4rem;font-weight:700}
  .stat-lbl{font-size:.65rem;text-transform:uppercase;color:var(--muted);margin-top:2px}
  #photo-wrap img{width:100%;border-radius:6px}
  .no-photo{text-align:center;color:var(--muted);padding:48px 0;font-size:.85rem}
  #photo-time{font-size:.7rem;color:var(--muted);margin-top:6px}
  #log-wrap{font-size:.78rem;font-family:monospace;max-height:320px;overflow-y:auto}
  .entry{display:flex;gap:8px;padding:5px 0;border-bottom:1px solid var(--border)}
  .entry:last-child{border-bottom:none}
  .ts{color:var(--muted);flex-shrink:0}
  .info{color:var(--blue)}.ok{color:var(--green)}.alert{color:var(--red)}
  .cfg{display:flex;flex-wrap:wrap;gap:16px;font-size:.78rem}
  .cfg-item{color:var(--muted)}.cfg-item span{color:var(--text);margin-left:4px}
  @media(max-width:600px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div style="max-width:960px;margin:0 auto">
  <header>
    <h1><span class="dot"></span>Inspection Station</h1>
    <div id="uptime" style="font-size:.75rem;color:var(--muted)">--</div>
  </header>
  <div class="grid">

    <div class="card">
      <div class="lbl">Zone Status</div>
      <div id="zone" class="clear">CLEAR</div>
    </div>

    <div class="card">
      <div class="lbl">Forward Distance</div>
      <div id="dist">-- m</div>
      <div class="sub" id="sub-dist">nothing in zone</div>
    </div>

    <div class="card full">
      <div class="lbl">Session Totals</div>
      <div class="stats">
        <div><div class="stat-val" id="captures">0</div>
             <div class="stat-lbl">Captures</div></div>
        <div><div class="stat-val ok" id="authorized">0</div>
             <div class="stat-lbl">Authorized</div></div>
        <div><div class="stat-val alert" id="alerts">0</div>
             <div class="stat-lbl">Alerts</div></div>
      </div>
    </div>

    <div class="card">
      <div class="lbl">Last Capture</div>
      <div id="photo-wrap"><div class="no-photo">No captures yet</div></div>
      <div id="photo-time"></div>
    </div>

    <div class="card">
      <div class="lbl">Event Log</div>
      <div id="log-wrap"></div>
    </div>

    <div class="card full">
      <div class="lbl">Configuration</div>
      <div class="cfg">
        <div class="cfg-item">Sensor<span id="c-sensor">--</span></div>
        <div class="cfg-item">Zone<span id="c-zone">--</span></div>
        <div class="cfg-item">Arc<span id="c-arc">--</span></div>
        <div class="cfg-item">Motion threshold<span id="c-motion">--</span></div>
        <div class="cfg-item">Cooldown<span id="c-cool">--</span></div>
        <div class="cfg-item">Min points<span id="c-pts">--</span></div>
      </div>
    </div>

  </div>
</div>
<script>
let lastPhotoTs = null;
async function refresh() {
  try {
    const d = await fetch('/api/status').then(r => r.json());

    const z = document.getElementById('zone');
    z.textContent = d.zone_occupied ? 'OCCUPIED' : 'CLEAR';
    z.className   = d.zone_occupied ? 'occupied'  : 'clear';

    document.getElementById('dist').textContent =
      d.current_dist !== null ? d.current_dist.toFixed(2) + ' m' : '-- m';
    document.getElementById('sub-dist').textContent =
      d.current_dist !== null ? (d.zone_occupied ? 'object in zone' : 'nearest forward object')
                               : 'nothing in zone';

    document.getElementById('captures').textContent  = d.total_captures;
    document.getElementById('authorized').textContent = d.authorized_count;
    document.getElementById('alerts').textContent     = d.alert_count;
    document.getElementById('uptime').textContent =
      'Uptime ' + d.uptime + '  •  ' + d.sensor;

    if (d.last_capture_time && d.last_capture_time !== lastPhotoTs) {
      lastPhotoTs = d.last_capture_time;
      document.getElementById('photo-wrap').innerHTML =
        '<img src="/api/photo?t=' + Date.now() + '" alt="capture">';
      document.getElementById('photo-time').textContent = d.last_capture_time;
    }

    document.getElementById('log-wrap').innerHTML = d.event_log.map(e =>
      '<div class="entry"><span class="ts">' + e.time + '</span>' +
      '<span class="' + e.kind + '">' + e.msg + '</span></div>'
    ).join('');

    document.getElementById('c-sensor').textContent = ' ' + d.sensor;
    document.getElementById('c-zone').textContent   = ' ' + d.proximity_m + ' m';
    document.getElementById('c-arc').textContent    = ' ±' + d.arc_deg + '°';
    document.getElementById('c-motion').textContent = ' ' + d.motion_m + ' m';
    document.getElementById('c-cool').textContent   = ' ' + d.cooldown_s + ' s';
    document.getElementById('c-pts').textContent    = ' ' + d.min_points;
  } catch(e) {}
}
setInterval(refresh, 1000);
refresh();
</script>
</body>
</html>"""


# ── Flask server ──────────────────────────────────────────────────────────────

def start_dashboard(port: int = 5000):
    """Launch Flask in a background daemon thread. Safe to call from main."""
    try:
        from flask import Flask, jsonify, send_file
    except ImportError:
        print("[DASH] Flask not installed — run: pip install flask", flush=True)
        return

    app = Flask(__name__)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.route("/")
    def index():
        return HTML

    @app.route("/api/status")
    def api_status():
        return jsonify(dashboard.snapshot())

    @app.route("/api/photo")
    def api_photo():
        path = dashboard.last_capture_path
        if path and os.path.exists(path):
            return send_file(path, mimetype="image/jpeg")
        return "", 204

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False),
        daemon=True,
        name="Dashboard",
    ).start()

    print(f"[DASH] http://0.0.0.0:{port}  →  open http://<pi-ip>:{port}", flush=True)
