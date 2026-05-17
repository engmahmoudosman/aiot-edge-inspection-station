#!/usr/bin/env python3
"""
sensor_fusion_m3.py — Milestone 3: LiDAR gate + face recognition + Telegram + web dashboard

Usage:
    python sensor_fusion_m3.py --sensor hokuyo
    python sensor_fusion_m3.py --sensor rplidar --proximity 1.5 --port 8080

Dashboard: http://<pi-ip>:5000
"""

import argparse
import json
import os
import pickle
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

from dashboard import dashboard, start_dashboard

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE           = os.path.dirname(os.path.abspath(__file__))
USER_HOME      = os.path.expanduser("~")
HAILO_APPS_DIR = os.path.join(USER_HOME, "hailo-apps")
LIDAR_SERVER   = os.path.join(HERE, "lidar_server.py")
CAPTURES_DIR   = os.path.join(HERE, "captures")
ENCODINGS_FILE = os.path.join(HERE, "encodings.pickle")
CONFIG_FILE    = os.path.join(HERE, "config.json")

# ── Parameters ────────────────────────────────────────────────────────────────
UDP_PORT           = 5001
PROXIMITY_M        = 1.0
FORWARD_ARC_DEG    = 45.0
MIN_POINTS         = 3
COOLDOWN_S         = 8.0
CAMERA_WARMUP_S    = 0.5
MOTION_THRESHOLD_M = 0.5
FACE_SCALE         = 4

_running = True


def log(msg, tag="M3"):
    print(f"[{tag}] {msg}", flush=True)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config():
    cfg = {"telegram_token": "", "telegram_chat_id": "", "telegram_enabled": False}
    if not os.path.exists(CONFIG_FILE):
        log("config.json not found — Telegram disabled", "WARN")
        return cfg
    try:
        with open(CONFIG_FILE) as f:
            cfg.update(json.load(f))
    except Exception as e:
        log(f"Config error: {e}", "WARN")
    return cfg


# ── Telegram (from door_security.py) ─────────────────────────────────────────

def tg_send_message(config, text):
    if not config.get("telegram_enabled"):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{config['telegram_token']}/sendMessage",
            json={"chat_id": config["telegram_chat_id"], "text": text, "parse_mode": "HTML"},
            timeout=5,
        )
    except Exception as e:
        log(f"Telegram error: {e}", "WARN")


def tg_send_photo(config, photo_path, caption=""):
    if not config.get("telegram_enabled"):
        return
    try:
        with open(photo_path, "rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{config['telegram_token']}/sendPhoto",
                data={"chat_id": config["telegram_chat_id"], "caption": caption, "parse_mode": "HTML"},
                files={"photo": f},
                timeout=10,
            )
    except Exception as e:
        log(f"Telegram photo error: {e}", "WARN")


# ── Face recognition ──────────────────────────────────────────────────────────

def load_face_encodings():
    if not os.path.exists(ENCODINGS_FILE):
        log("encodings.pickle not found — all captures will alert", "WARN")
        return [], []
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.loads(f.read())
        log(f"Enrolled: {sorted(set(data['names']))}")
        return data["encodings"], data["names"]
    except Exception as e:
        log(f"Failed to load encodings: {e}", "WARN")
        return [], []


def identify_face(photo_path, known_encodings, known_names):
    """Returns (face_found, is_known, name|None). Mirrors facial_recognition.py."""
    try:
        import face_recognition
    except ImportError:
        log("face_recognition not installed — alerting by default", "WARN")
        return False, False, None

    image = cv2.imread(photo_path)
    if image is None:
        return False, False, None

    rgb = cv2.cvtColor(
        cv2.resize(image, (0, 0), fx=1/FACE_SCALE, fy=1/FACE_SCALE),
        cv2.COLOR_BGR2RGB,
    )
    locations = face_recognition.face_locations(rgb)
    if not locations:
        return False, False, None

    if not known_encodings:
        return True, False, None

    for enc in face_recognition.face_encodings(rgb, locations, model="large"):
        matches   = face_recognition.compare_faces(known_encodings, enc)
        distances = face_recognition.face_distance(known_encodings, enc)
        best      = int(np.argmin(distances))
        if matches[best]:
            return True, True, known_names[best]

    return True, False, None


def handle_capture(photo_path, known_encodings, known_names, config):
    if photo_path is None:
        return
    dashboard.record_capture(photo_path)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    t0 = time.perf_counter()
    face_found, is_known, name = identify_face(photo_path, known_encodings, known_names)
    face_ms = (time.perf_counter() - t0) * 1000
    log(f"Face recognition: {face_ms:.0f} ms", "PERF")

    if is_known:
        log(f"Known: {name} — no alert")
        dashboard.record_authorized(name)
        dashboard.push_event(f"Authorized — {name}", "ok")
        t1 = time.perf_counter()
        tg_send_message(config, f"✅ <b>Authorized — {name}</b>\n🕐 {ts}")
        log(f"Telegram message: {(time.perf_counter()-t1)*1000:.0f} ms", "PERF")
    elif face_found:
        log("Unknown face — alerting")
        dashboard.record_alert()
        dashboard.push_event("Unknown person — alert sent", "alert")
        t1 = time.perf_counter()
        tg_send_photo(config, photo_path, f"⚠️ <b>Unknown Person</b>\n🕐 {ts}")
        log(f"Telegram photo:   {(time.perf_counter()-t1)*1000:.0f} ms", "PERF")
    else:
        log("No face — alerting")
        dashboard.record_alert()
        dashboard.push_event("Motion, no face — alert sent", "alert")
        t1 = time.perf_counter()
        tg_send_photo(config, photo_path, f"🔔 <b>Motion Detected</b>\n🕐 {ts}\n(no face visible)")
        log(f"Telegram photo:   {(time.perf_counter()-t1)*1000:.0f} ms", "PERF")


# ── Camera ────────────────────────────────────────────────────────────────────

def capture_photo() -> str | None:
    try:
        from picamera2 import Picamera2
    except ImportError:
        log("picamera2 not available", "WARN")
        return None

    Path(CAPTURES_DIR).mkdir(exist_ok=True)
    filepath = os.path.join(CAPTURES_DIR,
                            f"capture_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
    cam = None
    try:
        cam = Picamera2()
        cam.configure(cam.create_still_configuration(main={"size": (1280, 720)}))
        cam.start()
        time.sleep(CAMERA_WARMUP_S)
        cam.capture_file(filepath)
        log(f"Photo saved → {filepath}")
        return filepath
    except Exception as e:
        log(f"Capture failed: {e}", "ERR")
        return None
    finally:
        if cam:
            try: cam.stop(); cam.close()
            except Exception: pass


# ── LiDAR helpers ─────────────────────────────────────────────────────────────

def in_forward_arc(angle_deg: float) -> bool:
    a = angle_deg % 360.0
    return a <= FORWARD_ARC_DEG or a >= (360.0 - FORWARD_ARC_DEG)


def get_forward_distance(points: list, proximity_m: float):
    close = [p[1] for p in points
             if isinstance(p, (list, tuple)) and len(p) >= 2
             and 0 < p[1] <= proximity_m and in_forward_arc(p[0])]
    return sum(close) / len(close) if len(close) >= MIN_POINTS else None


def drain_socket(sock: socket.socket):
    sock.settimeout(0.0)
    while True:
        try: sock.recvfrom(65535)
        except OSError: break
    sock.settimeout(1.0)


def start_lidar_server(sensor: str) -> subprocess.Popen:
    setup_env = os.path.join(HAILO_APPS_DIR, "setup_env.sh")
    prefix = f"cd {HAILO_APPS_DIR} && source setup_env.sh && " if os.path.exists(setup_env) else ""
    return subprocess.Popen(
        ["bash", "-c", f"{prefix}cd {HERE} && python {LIDAR_SERVER} --lidar {sensor}"],
        preexec_fn=os.setsid,
    )


def kill_proc(proc, grace=3.0):
    if proc is None or proc.poll() is not None:
        return
    try:
        import os as _os
        pgid = _os.getpgid(proc.pid)
        import signal as _sig
        _os.killpg(pgid, _sig.SIGTERM)
        try: proc.wait(timeout=grace)
        except subprocess.TimeoutExpired: _os.killpg(pgid, _sig.SIGKILL)
    except ProcessLookupError:
        pass


# ── Proximity gate ────────────────────────────────────────────────────────────

def proximity_gate(proximity_m: float, known_encodings, known_names, config):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", UDP_PORT))
    sock.settimeout(1.0)

    log(f"Listening  arc:±{FORWARD_ARC_DEG}°  zone:{proximity_m}m  motion:{MOTION_THRESHOLD_M}m")
    dashboard.push_event("System started", "info")

    zone_was_occupied = False
    last_trigger_dist = None
    last_capture_t    = 0.0

    while _running:
        try:
            msg, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break

        try:
            points = json.loads(msg.decode()).get("data", [])
        except json.JSONDecodeError:
            continue

        cur_dist      = get_forward_distance(points, proximity_m)
        zone_occupied = cur_dist is not None
        dashboard.update_zone(zone_occupied, cur_dist)

        now   = time.monotonic()
        armed = (now - last_capture_t) >= COOLDOWN_S

        should_capture, reason = False, ""
        if zone_occupied and armed:
            if not zone_was_occupied:
                should_capture, reason = True, f"entry at {cur_dist:.2f} m"
            elif last_trigger_dist is not None:
                delta = abs(cur_dist - last_trigger_dist)
                if delta >= MOTION_THRESHOLD_M:
                    should_capture = True
                    reason = f"moved {delta:.2f} m ({last_trigger_dist:.2f}→{cur_dist:.2f} m)"

        if not zone_occupied:
            if zone_was_occupied:
                dashboard.push_event("Zone cleared", "info")
            last_trigger_dist = None

        if should_capture:
            log(f"Trigger — {reason}")
            dashboard.push_event(f"Trigger: {reason}", "info")
            t_trigger = time.perf_counter()
            photo_path = capture_photo()
            cam_ms = (time.perf_counter() - t_trigger) * 1000
            log(f"Camera capture:   {cam_ms:.0f} ms", "PERF")
            handle_capture(photo_path, known_encodings, known_names, config)
            log(f"End-to-end:       {(time.perf_counter()-t_trigger)*1000:.0f} ms total", "PERF")
            last_capture_t, last_trigger_dist, zone_was_occupied = time.monotonic(), cur_dist, True
            log(f"Cooldown {COOLDOWN_S:.0f}s...")
            time.sleep(COOLDOWN_S)
            drain_socket(sock)
            dashboard.push_event("Re-armed", "info")
        else:
            zone_was_occupied = zone_occupied

    sock.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    global _running

    parser = argparse.ArgumentParser(description="M3 — LiDAR + face recognition + Telegram + dashboard")
    parser.add_argument("--sensor",    choices=["rplidar", "hokuyo"], required=True)
    parser.add_argument("--proximity", type=float, default=PROXIMITY_M)
    parser.add_argument("--port",      type=int,   default=5000,
                        help="Web dashboard port (default: 5000)")
    args = parser.parse_args()

    # Populate dashboard config so the UI shows correct values
    dashboard.sensor      = args.sensor
    dashboard.proximity_m = args.proximity
    dashboard.arc_deg     = FORWARD_ARC_DEG
    dashboard.motion_m    = MOTION_THRESHOLD_M
    dashboard.cooldown_s  = COOLDOWN_S
    dashboard.min_points  = MIN_POINTS

    config                       = load_config()
    known_encodings, known_names = load_face_encodings()
    lidar_proc                   = None

    def shutdown(*_):
        global _running
        _running = False
        kill_proc(lidar_proc)
        subprocess.call(["pkill", "-KILL", "-f", "lidar_server.py"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log("Done.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    start_dashboard(args.port)

    tg_send_message(config,
        f"🟢 <b>Inspection Station M3 started</b>\n"
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"👥 Enrolled: {sorted(set(known_names)) or 'none'}\n"
        f"📡 {args.sensor}  |  Zone: {args.proximity} m\n"
        f"🌐 Dashboard: http://&lt;pi-ip&gt;:{args.port}")

    log(f"Starting LiDAR server ({args.sensor})...")
    lidar_proc = start_lidar_server(args.sensor)
    time.sleep(2)

    if lidar_proc.poll() is not None:
        log("LiDAR server exited — check hardware.", "ERR")
        return 1

    proximity_gate(args.proximity, known_encodings, known_names, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
