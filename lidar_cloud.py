#!/usr/bin/env python3
"""
lidar_cloud.py 
Features:
  • Meter labels on grid rings
  • Dark theme
  • Camera FOV cone overlay
  • Color gradient: close=red → mid=yellow → far=cyan
  • Point trail fade (last 3 scans)
  • FPS counter + point count
  • Zoom with +/- keys
  • "Waiting for data" indicator when lidar_server isn't running
"""

import cv2
import numpy as np
import socket
import json
import threading
import signal
import sys
import time
from collections import deque

# ─── Configuration ────────────────────────────────────────────────────────────
UDP_PORT      = 5001            # must match lidar_server.py
WIN_SIZE      = 1000
MAX_RANGE_M   = 12.0
GRID_RINGS    = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
TRAIL_SCANS   = 3
INITIAL_SCALE = 40
CAM_FOV_DEG   = 66 ##66.3
ANGLE_OFFSET  = 90              # rotate so LiDAR 0° = up (FWD)

# ─── Theme colours (BGR) ─────────────────────────────────────────────────────
BG_COLOR      = (20, 18, 12)
GRID_COLOR    = (50, 45, 35)
GRID_TEXT     = (120, 115, 100)
AXIS_COLOR    = (60, 55, 45)
CENTER_COLOR  = (0, 220, 255)
FOV_COLOR     = (40, 80, 40)
FOV_EDGE      = (60, 180, 60)
FPS_COLOR     = (0, 200, 0)
WARN_COLOR    = (0, 100, 255)   # orange for "waiting" text
LABEL_FONT    = cv2.FONT_HERSHEY_SIMPLEX

# ─── Shared state (UDP listener thread → main draw thread) ───────────────────
_scan_lock = threading.Lock()
_latest_scan = []
_last_udp_time = 0.0
_UDP_STALE_SEC = 2.0            # show warning if no data for 2s

scale = INITIAL_SCALE
center = WIN_SIZE // 2


def _signal_handler(_sig, _frame):
    print("\n[lidar_cloud] Shutting down...")
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ─── UDP listener thread ─────────────────────────────────────────────────────

def _udp_listener():
    """Receive LiDAR scans from lidar_server.py over UDP."""
    global _latest_scan, _last_udp_time

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', UDP_PORT))
    s.settimeout(1.0)  # so we can check for shutdown periodically

    print(f"[lidar_cloud] Listening for LiDAR data on UDP port {UDP_PORT}...")

    while True:
        try:
            msg, _ = s.recvfrom(65535)
            data = json.loads(msg.decode())
            points = data.get("data", [])

            scan = []
            for pt in points:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2 and pt[1] > 0:
                    scan.append((pt[0], pt[1]))  # (angle, dist_m)

            with _scan_lock:
                _latest_scan = scan
                _last_udp_time = time.monotonic()

        except socket.timeout:
            continue
        except json.JSONDecodeError:
            continue
        except Exception as e:
            print(f"[lidar_cloud] UDP error: {e}")
            time.sleep(0.5)


# ─── Drawing ──────────────────────────────────────────────────────────────────

def _dist_to_color(dist_m):
    t = min(1.0, dist_m / MAX_RANGE_M)
    if t < 0.3:
        f = t / 0.3
        r, g, b = 255, int(255 * f), 0
    elif t < 0.6:
        f = (t - 0.3) / 0.3
        r, g, b = int(255 * (1 - f)), 255, 0
    else:
        f = (t - 0.6) / 0.4
        r, g, b = 0, 255, int(255 * f)
    return (b, g, r)  # BGR


def _draw_frame(scan_history, data_stale):
    img = np.full((WIN_SIZE, WIN_SIZE, 3), BG_COLOR, dtype=np.uint8)

    # ── Camera FOV cone ───────────────────────────────────────────────
    fov_half = CAM_FOV_DEG / 2.0
    fov_radius = int(MAX_RANGE_M * scale)
    angle_start = 90 - fov_half
    angle_end   = 90 + fov_half

    axes = (fov_radius, fov_radius)
    cv2.ellipse(img, (center, center), axes,
                0, -angle_end, -angle_start, FOV_COLOR, -1)
    for ang in [angle_start, angle_end]:
        theta = np.deg2rad(ang)
        ex = int(center + fov_radius * np.cos(theta))
        ey = int(center - fov_radius * np.sin(theta))
        cv2.line(img, (center, center), (ex, ey), FOV_EDGE, 1, cv2.LINE_AA)

    '''
    cv2.putText(img, f"CAM {CAM_FOV_DEG}\xb0",
                (center - 30, center - int(2.5 * scale) - 8),
                LABEL_FONT, 0.4, FOV_EDGE, 1, cv2.LINE_AA)
    '''
    # ── Grid rings with meter labels ──────────────────────────────────
    for r_m in GRID_RINGS:
        r_px = int(r_m * scale)
        if r_px > WIN_SIZE:
            continue
        cv2.circle(img, (center, center), r_px, GRID_COLOR, 1, cv2.LINE_AA)
        lx = center + int(r_px * 0.707) + 4
        ly = center - int(r_px * 0.707) - 4
        if 0 < lx < WIN_SIZE - 30 and 0 < ly < WIN_SIZE:
            cv2.putText(img, f"{r_m}m", (lx, ly),
                        LABEL_FONT, 0.4, GRID_TEXT, 1, cv2.LINE_AA)

    # ── Axis cross + direction labels ─────────────────────────────────
    cv2.line(img, (center, 0), (center, WIN_SIZE), AXIS_COLOR, 1)
    cv2.line(img, (0, center), (WIN_SIZE, center), AXIS_COLOR, 1)
    cv2.putText(img, "FWD",  (center - 15, 20),          LABEL_FONT, 0.45, GRID_TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, "REAR", (center - 18, WIN_SIZE - 10),LABEL_FONT, 0.45, GRID_TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, "L",    (8, center + 5),             LABEL_FONT, 0.45, GRID_TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, "R",    (WIN_SIZE - 18, center + 5), LABEL_FONT, 0.45, GRID_TEXT, 1, cv2.LINE_AA)

    # ── Draw points (older scans dimmer) ──────────────────────────────
    n_scans = len(scan_history)
    for scan_idx, scan_data in enumerate(scan_history):
        age_factor = (scan_idx + 1) / max(n_scans, 1)
        point_size = 2 if scan_idx == n_scans - 1 else 1

        for angle_deg, dist_m in scan_data:
            if dist_m <= 0 or dist_m > MAX_RANGE_M * 1.5:
                continue
            theta = np.deg2rad(angle_deg + ANGLE_OFFSET)
            px = int(center + dist_m * np.cos(theta) * scale)
            py = int(center - dist_m * np.sin(theta) * scale)
            if 0 <= px < WIN_SIZE and 0 <= py < WIN_SIZE:
                base_color = _dist_to_color(dist_m)
                color = tuple(int(c * age_factor) for c in base_color)
                cv2.circle(img, (px, py), point_size, color, -1, cv2.LINE_AA)

    # ── Center dot ────────────────────────────────────────────────────
    cv2.circle(img, (center, center), 4, CENTER_COLOR, -1, cv2.LINE_AA)
    cv2.circle(img, (center, center), 6, CENTER_COLOR, 1, cv2.LINE_AA)

    # ── Stale data warning ────────────────────────────────────────────
    if data_stale:
        cv2.putText(img, "WAITING FOR LIDAR DATA...",
                    (center - 160, center + 40),
                    LABEL_FONT, 0.6, WARN_COLOR, 2, cv2.LINE_AA)
        cv2.putText(img, "Ensure lidar_server.py is running",
                    (center - 170, center + 70),
                    LABEL_FONT, 0.45, WARN_COLOR, 1, cv2.LINE_AA)

    return img


# ─── Main loop ────────────────────────────────────────────────────────────────

def lidar_viewer():
    global scale

    # Start UDP listener in background
    threading.Thread(target=_udp_listener, daemon=True).start()

    cv2.namedWindow("LiDAR View", cv2.WINDOW_NORMAL)

    scan_history = deque(maxlen=TRAIL_SCANS)
    fps_counter = deque(maxlen=30)
    prev_time = time.time()

    while True:
        # Grab latest scan from UDP thread
        with _scan_lock:
            scan = list(_latest_scan)
            data_age = time.monotonic() - _last_udp_time

        data_stale = data_age > _UDP_STALE_SEC

        if scan:
            scan_history.append(scan)

        # Draw
        img = _draw_frame(scan_history, data_stale)

        # FPS + status
        now = time.time()
        fps_counter.append(1.0 / max(now - prev_time, 1e-9))
        prev_time = now
        fps = np.mean(fps_counter)
        n_pts = len(scan) if scan else 0
        status = f"FPS: {fps:.0f}  Pts: {n_pts}  Zoom: {scale}px/m  [UDP:{UDP_PORT}]"
        cv2.putText(img, status, (10, WIN_SIZE - 12),
                    LABEL_FONT, 0.45, FPS_COLOR, 1, cv2.LINE_AA)

        cv2.imshow("LiDAR View", img)

        key = cv2.waitKey(30) & 0xFF  # 30ms = ~33 FPS max draw rate
        if key == 27 or key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            scale = min(200, scale + 10)
            print(f"[zoom] {scale} px/m")
        elif key == ord('-'):
            scale = max(20, scale - 10)
            print(f"[zoom] {scale} px/m")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("[lidar_cloud] Starting LiDAR visualizer (UDP mode)...")
    print("[lidar_cloud] Requires lidar_server.py to be running.")
    lidar_viewer()