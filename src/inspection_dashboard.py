"""
inspection_dashboard.py

Two purposes:
  1. LidarMapRenderer class — imported by detection_with_lidar.py to render the
     polar 2D LiDAR map pane of the composite dashboard.
  2. Standalone viewer — when run directly, shows a live LiDAR cloud (diagnostic).

Usage (standalone):
    python inspection_dashboard.py [--config config.yaml]
"""

import argparse
import json
import signal
import socket
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np


# ─── Colour palette (BGR) ────────────────────────────────────────────────────
_BG         = (20, 18, 12)
_GRID       = (50, 45, 35)
_GRID_TEXT  = (120, 115, 100)
_AXIS       = (60, 55, 45)
_CENTER     = (0, 220, 255)
_FOV_FILL   = (30, 60, 30)
_FOV_EDGE   = (60, 180, 60)
_SAFE_RING  = (0, 80, 180)    # dark red ring = safety distance
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_ANGLE_OFFSET = 90            # LiDAR 0° (forward) → up on map


# ─── LidarMapRenderer ─────────────────────────────────────────────────────────

class LidarMapRenderer:
    """
    Renders a top-down polar LiDAR map as a square BGR image.

    Parameters
    ----------
    size          : pixel width/height of the output image
    scale         : pixels per metre
    max_range_m   : maximum display range in metres
    cam_fov_deg   : camera horizontal FOV (used to draw the FOV cone)
    trail_scans   : number of consecutive scans to trail-fade
    """

    def __init__(
        self,
        size: int = 640,
        scale: int = 100,
        max_range_m: float = 4.0,
        cam_fov_deg: float = 66.3,
        trail_scans: int = 3,
    ):
        self.size = size
        self.scale = scale
        self.max_range_m = max_range_m
        self.cam_fov_deg = cam_fov_deg
        self.center = size // 2
        self._trail: deque = deque(maxlen=trail_scans)

        # Grid rings: every 0.5 m up to max_range_m
        step = 0.5
        self._rings = []
        r = step
        while r <= max_range_m + 0.01:
            self._rings.append(round(r, 1))
            r += step

    # ── helpers ────────────────────────────────────────────────────────────────

    def _dist_color(self, dist_m: float):
        t = min(1.0, dist_m / self.max_range_m)
        if t < 0.3:
            f = t / 0.3
            r, g, b = 255, int(255 * f), 0
        elif t < 0.6:
            f = (t - 0.3) / 0.3
            r, g, b = int(255 * (1 - f)), 255, 0
        else:
            f = (t - 0.6) / 0.4
            r, g, b = 0, 255, int(255 * f)
        return (b, g, r)

    def _polar_to_xy(self, angle_deg: float, dist_m: float):
        theta = np.deg2rad(angle_deg + _ANGLE_OFFSET)
        px = int(self.center + dist_m * np.cos(theta) * self.scale)
        py = int(self.center - dist_m * np.sin(theta) * self.scale)
        return px, py

    # ── public API ────────────────────────────────────────────────────────────

    def render(
        self,
        scan_dict: dict,
        obj_markers=None,
        safety_dist_m: float = 1.5,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        scan_dict    : {angle_deg: dist_m, ...}  current LiDAR scan snapshot
        obj_markers  : list of (angle_deg, dist_m, label, color_bgr)
                       angle_deg derived from bbox centre-x + H_FOV
                       pass [] or None if no objects
        safety_dist_m: radius of safety zone ring (metres)

        Returns
        -------
        BGR uint8 numpy array of shape (size, size, 3)
        """
        img = np.full((self.size, self.size, 3), _BG, dtype=np.uint8)
        c = self.center

        # ── Camera FOV cone ───────────────────────────────────────────────────
        fov_half = self.cam_fov_deg / 2.0
        fov_r_px = min(int(self.max_range_m * self.scale), self.size // 2 - 2)
        a_start = 90 - fov_half
        a_end   = 90 + fov_half
        cv2.ellipse(img, (c, c), (fov_r_px, fov_r_px),
                    0, -a_end, -a_start, _FOV_FILL, -1)
        for ang in [a_start, a_end]:
            th = np.deg2rad(ang)
            ex = int(c + fov_r_px * np.cos(th))
            ey = int(c - fov_r_px * np.sin(th))
            cv2.line(img, (c, c), (ex, ey), _FOV_EDGE, 1, cv2.LINE_AA)

        # ── Safety zone ring ──────────────────────────────────────────────────
        sz_px = int(safety_dist_m * self.scale)
        if 0 < sz_px < self.size // 2:
            cv2.circle(img, (c, c), sz_px, _SAFE_RING, 1, cv2.LINE_AA)

        # ── Grid rings with metre labels ──────────────────────────────────────
        for r_m in self._rings:
            r_px = int(r_m * self.scale)
            if r_px >= self.size // 2:
                continue
            cv2.circle(img, (c, c), r_px, _GRID, 1, cv2.LINE_AA)
            lx = c + int(r_px * 0.707) + 4
            ly = c - int(r_px * 0.707) - 4
            if 0 < lx < self.size - 40 and 0 < ly < self.size:
                cv2.putText(img, f"{r_m:.1f}m",
                            (lx, ly), _FONT, 0.32, _GRID_TEXT, 1, cv2.LINE_AA)

        # ── Axis cross + direction labels ─────────────────────────────────────
        cv2.line(img, (c, 0), (c, self.size), _AXIS, 1)
        cv2.line(img, (0, c), (self.size, c), _AXIS, 1)
        cv2.putText(img, "FWD",  (c - 15, 16),               _FONT, 0.4, _GRID_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, "REAR", (c - 18, self.size - 6),     _FONT, 0.4, _GRID_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, "L",    (6,  c + 5),                 _FONT, 0.4, _GRID_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, "R",    (self.size - 16, c + 5),     _FONT, 0.4, _GRID_TEXT, 1, cv2.LINE_AA)

        # ── LiDAR point cloud (with trail fade) ───────────────────────────────
        scan_list = list(scan_dict.items()) if scan_dict else []
        if scan_list:
            self._trail.append(scan_list)

        n = len(self._trail)
        for idx, scan_pts in enumerate(self._trail):
            age = (idx + 1) / max(n, 1)
            pt_size = 2 if idx == n - 1 else 1
            for angle_deg, dist_m in scan_pts:
                if dist_m <= 0 or dist_m > self.max_range_m * 1.5:
                    continue
                px, py = self._polar_to_xy(angle_deg, dist_m)
                if 0 <= px < self.size and 0 <= py < self.size:
                    base = self._dist_color(dist_m)
                    color = tuple(int(ch * age) for ch in base)
                    cv2.circle(img, (px, py), pt_size, color, -1, cv2.LINE_AA)

        # ── Detected object markers ───────────────────────────────────────────
        if obj_markers:
            for angle_deg, dist_m, label, color in obj_markers:
                if dist_m is None or dist_m <= 0:
                    continue
                px, py = self._polar_to_xy(angle_deg, dist_m)
                if 0 <= px < self.size and 0 <= py < self.size:
                    cv2.circle(img, (px, py), 9, color, -1, cv2.LINE_AA)
                    cv2.circle(img, (px, py), 9, (255, 255, 255), 1, cv2.LINE_AA)
                    lbl_str = f"{label} {dist_m:.1f}m"
                    tx = min(max(px + 12, 2), self.size - 90)
                    ty = max(py - 4, 14)
                    cv2.putText(img, lbl_str,
                                (tx, ty), _FONT, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Centre dot ────────────────────────────────────────────────────────
        cv2.circle(img, (c, c), 4, _CENTER, -1, cv2.LINE_AA)
        cv2.circle(img, (c, c), 6, _CENTER, 1,  cv2.LINE_AA)

        return img


# ─── Standalone diagnostic viewer ────────────────────────────────────────────

def _standalone_viewer(udp_port: int, cfg: dict):
    """Live LiDAR cloud viewer (no camera, diagnostic only)."""
    from collections import deque as _deque
    import queue as _queue

    scan_lock = threading.Lock()
    latest_scan: list = []
    last_update = [0.0]

    def _udp_thread():
        nonlocal latest_scan
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", udp_port))
        s.settimeout(1.0)
        print(f"[dashboard] Listening UDP :{udp_port} ...")
        while True:
            try:
                msg, _ = s.recvfrom(65535)
                d = json.loads(msg.decode())
                pts = [(p[0], p[1]) for p in d.get("data", [])
                       if isinstance(p, (list, tuple)) and len(p) >= 2 and p[1] > 0]
                with scan_lock:
                    latest_scan = pts
                    last_update[0] = time.monotonic()
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[dashboard] UDP error: {e}")
                time.sleep(0.2)

    threading.Thread(target=_udp_thread, daemon=True).start()

    renderer = LidarMapRenderer(
        size=cfg["display"]["map_size_px"],
        scale=cfg["display"]["map_scale_px_per_m"],
        max_range_m=cfg["display"]["max_range_m"],
        cam_fov_deg=cfg["camera"]["hfov_deg"],
    )
    safety_m = cfg["safety"]["distance_threshold_m"]

    cv2.namedWindow("LiDAR Diagnostic", cv2.WINDOW_NORMAL)
    while True:
        with scan_lock:
            scan = dict(latest_scan)
            stale = (time.monotonic() - last_update[0]) > 2.0

        img = renderer.render(scan, safety_dist_m=safety_m)
        if stale:
            cv2.putText(img, "WAITING FOR LIDAR DATA ...",
                        (renderer.center - 150, renderer.center + 40),
                        _FONT, 0.6, (0, 100, 255), 2, cv2.LINE_AA)
        cv2.imshow("LiDAR Diagnostic", img)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


def _load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _default_cfg() -> dict:
    return {
        "display":  {"map_size_px": 640, "map_scale_px_per_m": 100, "max_range_m": 4.0},
        "camera":   {"hfov_deg": 66.3},
        "lidar":    {"udp_port": 5001},
        "safety":   {"distance_threshold_m": 1.5},
    }


if __name__ == "__main__":
    signal.signal(signal.SIGINT,  lambda *_: (cv2.destroyAllWindows(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (cv2.destroyAllWindows(), sys.exit(0)))

    ap = argparse.ArgumentParser(description="LiDAR diagnostic viewer")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    try:
        cfg = _load_config(args.config)
    except FileNotFoundError:
        cfg = _default_cfg()

    _standalone_viewer(cfg["lidar"]["udp_port"], cfg)
