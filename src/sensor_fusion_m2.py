#!/usr/bin/env python3
"""
sensor_fusion_m2.py — Milestone 2: LiDAR proximity gate + camera snapshot

Trigger conditions (both require cooldown to have elapsed):
  ENTRY  — something enters the 1-m forward zone from outside
            (distance was > threshold → now within threshold).
  MOTION — something already in the zone moves ≥ MOTION_THRESHOLD_M
            from the position recorded at the last capture.
  IDLE   — zone is empty, or occupant is stationary → no trigger.

Usage:
    python sensor_fusion_m2.py --sensor hokuyo
    python sensor_fusion_m2.py --sensor rplidar --proximity 1.5
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE           = os.path.dirname(os.path.abspath(__file__))
USER_HOME      = os.path.expanduser("~")
HAILO_APPS_DIR = os.path.join(USER_HOME, "hailo-apps")
LIDAR_SERVER   = os.path.join(HERE, "lidar_server.py")
CAPTURES_DIR   = os.path.join(HERE, "captures")

# ── Tuneable parameters ───────────────────────────────────────────────────────
UDP_PORT           = 5001
PROXIMITY_M        = 1.0    # zone radius (metres)
FORWARD_ARC_DEG    = 45.0   # half-width of forward zone (degrees either side of 0°)
MIN_POINTS         = 3      # minimum LiDAR rays required to count as a hit
COOLDOWN_S         = 8.0    # minimum gap between captures
CAMERA_WARMUP_S    = 0.5
MOTION_THRESHOLD_M = 0.5    # minimum distance change to re-trigger inside zone

_running = True


def log(msg, tag="M2"):
    print(f"[{tag}] {msg}", flush=True)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def in_forward_arc(angle_deg: float) -> bool:
    a = angle_deg % 360.0
    return a <= FORWARD_ARC_DEG or a >= (360.0 - FORWARD_ARC_DEG)


def get_forward_distance(points: list, proximity_m: float):
    """
    Mean distance of in-range, forward-arc LiDAR points.
    Returns None if fewer than MIN_POINTS qualify (zone considered empty).
    """
    close = [
        p[1] for p in points
        if isinstance(p, (list, tuple)) and len(p) >= 2
        and p[1] > 0
        and p[1] <= proximity_m
        and in_forward_arc(p[0])
    ]
    if len(close) < MIN_POINTS:
        return None
    return sum(close) / len(close)


# ── Socket helper ─────────────────────────────────────────────────────────────

def drain_socket(sock: socket.socket):
    """Discard packets that queued up during the cooldown sleep."""
    saved = sock.gettimeout()
    sock.settimeout(0.0)
    while True:
        try:
            sock.recvfrom(65535)
        except OSError:
            break
    sock.settimeout(saved)


# ── Subprocess helpers ────────────────────────────────────────────────────────

def start_lidar_server(sensor: str) -> subprocess.Popen:
    setup_env = os.path.join(HAILO_APPS_DIR, "setup_env.sh")
    if os.path.exists(setup_env):
        cmd = (f"cd {HAILO_APPS_DIR} && source setup_env.sh && "
               f"cd {HERE} && python {LIDAR_SERVER} --lidar {sensor}")
    else:
        cmd = f"cd {HERE} && python {LIDAR_SERVER} --lidar {sensor}"
    return subprocess.Popen(["bash", "-c", cmd], preexec_fn=os.setsid)


def kill_proc(proc: subprocess.Popen, grace: float = 3.0):
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# ── Camera capture ────────────────────────────────────────────────────────────

def capture_photo() -> str | None:
    try:
        from picamera2 import Picamera2
    except ImportError:
        log("picamera2 not available — skipping capture", "WARN")
        return None

    Path(CAPTURES_DIR).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath  = os.path.join(CAPTURES_DIR, f"capture_{timestamp}.jpg")

    cam = None
    try:
        cam = Picamera2()
        cfg = cam.create_still_configuration(main={"size": (1280, 720)})
        cam.configure(cfg)
        cam.start()
        time.sleep(CAMERA_WARMUP_S)
        cam.capture_file(filepath)
        log(f"Photo saved → {filepath}")
        return filepath
    except Exception as e:
        log(f"Capture failed: {e}", "ERR")
        return None
    finally:
        if cam is not None:
            try:
                cam.stop()
                cam.close()
            except Exception:
                pass


# ── Main gate loop ────────────────────────────────────────────────────────────

def proximity_gate(proximity_m: float):
    """
    Listen to LiDAR UDP and trigger capture on:
      1. Entry into the forward zone from outside.
      2. Significant movement (≥ MOTION_THRESHOLD_M) within the zone.
    Stationary occupants never re-trigger.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", UDP_PORT))
    sock.settimeout(1.0)

    log(f"UDP :{UDP_PORT}  arc: ±{FORWARD_ARC_DEG}°  "
        f"zone: {proximity_m} m  min pts: {MIN_POINTS}  "
        f"motion threshold: {MOTION_THRESHOLD_M} m")
    log("State: IDLE — waiting for entry")

    zone_was_occupied = False
    last_trigger_dist = None   # distance recorded at the last capture
    last_capture_t    = 0.0

    while _running:
        try:
            msg, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break

        try:
            data = json.loads(msg.decode())
        except json.JSONDecodeError:
            continue

        points        = data.get("data", [])
        cur_dist      = get_forward_distance(points, proximity_m)
        zone_occupied = cur_dist is not None

        now   = time.monotonic()
        armed = (now - last_capture_t) >= COOLDOWN_S

        # ── Decide whether to capture ─────────────────────────────────────────
        should_capture = False
        reason         = ""

        if zone_occupied and armed:
            if not zone_was_occupied:
                # Something entered from outside the zone
                should_capture = True
                reason = f"entry at {cur_dist:.2f} m"
            elif last_trigger_dist is not None:
                # Already in zone — only re-trigger on notable movement
                delta = abs(cur_dist - last_trigger_dist)
                if delta >= MOTION_THRESHOLD_M:
                    should_capture = True
                    reason = (f"moved {delta:.2f} m  "
                              f"({last_trigger_dist:.2f} → {cur_dist:.2f} m)")

        # Reset reference when zone clears so next entry is treated as fresh
        if not zone_occupied:
            last_trigger_dist = None

        # ── Act ───────────────────────────────────────────────────────────────
        if should_capture:
            log(f"Capture triggered — {reason}")
            capture_photo()
            last_capture_t    = time.monotonic()
            last_trigger_dist = cur_dist
            # Keep zone_was_occupied = True so re-entry guard works correctly
            zone_was_occupied = True
            log(f"Cooldown ({COOLDOWN_S:.0f} s)...")
            time.sleep(COOLDOWN_S)
            drain_socket(sock)   # drop packets buffered during sleep
            log("State: IDLE — re-armed")
        else:
            zone_was_occupied = zone_occupied
            if zone_occupied:
                if not armed:
                    remaining = COOLDOWN_S - (now - last_capture_t)
                    log(f"  zone: OCCUPIED at {cur_dist:.2f} m "
                        f"[cooldown {remaining:.1f} s left]", tag="DBG")
                else:
                    log(f"  zone: OCCUPIED at {cur_dist:.2f} m "
                        f"[stationary — no trigger]", tag="DBG")

    try:
        sock.close()
    except Exception:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    global _running

    parser = argparse.ArgumentParser(
        description="M2 — LiDAR proximity gate with camera snapshot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python sensor_fusion_m2.py --sensor hokuyo\n"
            "  python sensor_fusion_m2.py --sensor rplidar --proximity 1.5\n"
        ),
    )
    parser.add_argument("--sensor", choices=["rplidar", "hokuyo"], required=True)
    parser.add_argument("--proximity", type=float, default=PROXIMITY_M,
                        help=f"Trigger distance in metres (default: {PROXIMITY_M})")
    args = parser.parse_args()

    lidar_proc = None

    def shutdown(*_):
        global _running
        _running = False
        log("Shutting down...", "WARN")
        kill_proc(lidar_proc)
        subprocess.call(["pkill", "-KILL", "-f", "lidar_server.py"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log("Done.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log(f"Starting LiDAR server ({args.sensor})...")
    lidar_proc = start_lidar_server(args.sensor)
    time.sleep(2)

    if lidar_proc.poll() is not None:
        log("LiDAR server exited immediately — check hardware/cable.", "ERR")
        return 1

    proximity_gate(args.proximity)
    return 0


if __name__ == "__main__":
    sys.exit(main())
