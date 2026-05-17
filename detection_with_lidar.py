# region imports
import gi
gi.require_version("Gst", "1.0")
import cv2
import numpy as np
import socket, json, threading, time
import hailo
from gi.repository import Gst
from hailo_apps.python.pipeline_apps.detection.detection_pipeline_with_lidar import GStreamerDetectionApp
from hailo_apps.python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
hailo_logger = get_logger(__name__)
# endregion imports

# ===============================================================================
# LiDAR Integration with Thread Safety
# ===============================================================================
lidar_scan = {}
lidar_lock = threading.Lock()
lidar_last_update = 0.0          # monotonic timestamp of last valid scan
LIDAR_STALE_SEC   = 0.5          # ignore data older than this (seconds)

def lidar_listener():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('', 5001))
    hailo_logger.info("Listening for LiDAR data on UDP port 5001...")
    while True:
        try:
            msg, _ = s.recvfrom(65535)
            lidar_data = json.loads(msg.decode())
            data_points = lidar_data.get("data", [])
            with lidar_lock:
                lidar_scan.clear()
                for pt in data_points:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2 and pt[1] > 0:
                        angle, distance = pt[0], pt[1]
                        lidar_scan[angle] = distance  # already in meters
                global lidar_last_update
                lidar_last_update = time.monotonic()
        except json.JSONDecodeError:
            hailo_logger.warning("Error decoding LiDAR JSON data.")
        except Exception as e:
            hailo_logger.error(f"LiDAR listener error: {e}")

threading.Thread(target=lidar_listener, daemon=True).start()
# ===============================================================================

# ===============================================================================
# LiDAR band calibration (640x480 resolution, LiDAR ~3 to 3.5 inches above camera)
# The LiDAR scan plane intersects the frame around y=220-360 for 1-5m range.
# Using wider margin (180-350) for safety — tighten after real-world calibration.
# To calibrate: place object at known distance, print y1/y2 of its bbox,
# adjust LIDAR_BAND_MIN/MAX until distance reads correctly.
# ===============================================================================
LIDAR_BAND_MIN = 250  # pixels from top
LIDAR_BAND_MAX = 290  # pixels from top

H_FOV = 66.3  # Pi camera horizontal FOV in degrees

# ===============================================================================
# Temporal smoothing (EMA) — keyed by label + horizontal position bucket
# No full tracker needed: we quantise the bbox centre-x into N_BUCKETS bins
# and maintain an exponential moving average per (label, bucket) pair.
# ===============================================================================
EMA_ALPHA      = 0.4    # weight of new measurement (0→full smooth, 1→no smooth)
EMA_EXPIRE_SEC = 1.0    # drop entries not seen for this long
N_BUCKETS      = 20     # horizontal position bins (640px / 20 = 32px each)

# { (label, bucket_idx): (smoothed_dist, last_seen_monotonic) }
_ema_tracks: dict[tuple[str, int], tuple[float, float]] = {}


def _ema_update(label: str, cx_norm: float, raw_dist: float) -> float:
    """Return smoothed distance for this (label, position) pair."""
    bucket = int(cx_norm * N_BUCKETS)
    bucket = max(0, min(bucket, N_BUCKETS - 1))
    key = (label, bucket)
    now = time.monotonic()

    prev = _ema_tracks.get(key)
    if prev is not None and (now - prev[1]) < EMA_EXPIRE_SEC:
        smoothed = EMA_ALPHA * raw_dist + (1.0 - EMA_ALPHA) * prev[0]
    else:
        smoothed = raw_dist  # first observation or expired — no smoothing

    _ema_tracks[key] = (smoothed, now)
    return smoothed


def _ema_cleanup():
    """Remove stale entries (call once per frame)."""
    now = time.monotonic()
    stale = [k for k, (_, t) in _ema_tracks.items() if now - t > EMA_EXPIRE_SEC * 3]
    for k in stale:
        del _ema_tracks[k]


# ===============================================================================
# Nearest-cluster distance selection
# ===============================================================================
CLUSTER_BIN_M    = 0.3   # bin width in metres
CLUSTER_MIN_FRAC = 0.15  # a bin needs ≥15% of total points to count (if enough data)

def _nearest_cluster_dist(dists: np.ndarray) -> float:
    """
    Given an array of LiDAR distances within a bbox's angular range,
    return the median of the NEAREST dense cluster — ignoring background.

    Algorithm:
      1. If ≤3 points: not enough for clustering, return 25th percentile
         (biases toward the closer surface, filters single-ray noise)
      2. Bin distances into CLUSTER_BIN_M (0.3m) wide buckets
      3. Walk bins from nearest to farthest
      4. First bin with enough points (≥15% of total, or ≥2) → take median of
         that bin + the next adjacent bin (catches object spread across bin edge)
      5. Fallback: 25th percentile of all data

    Example:
      LiDAR rays: [1.5, 1.6, 1.5, 4.0, 4.1, 3.9, 4.2] (laptop + wall)
      Bins:  1.5m bin → [1.5, 1.6, 1.5]   ← nearest dense cluster → returns ~1.53
             4.0m bin → [4.0, 4.1, 3.9, 4.2]  ← background, ignored
    """
    n = len(dists)

    # Too few points — just bias toward closer readings
    if n <= 3:
        return float(np.percentile(dists, 25))

    d_min = dists.min()
    d_max = dists.max()

    # If all readings are close together, no clustering needed
    if (d_max - d_min) < CLUSTER_BIN_M * 1.5:
        return float(np.median(dists))

    # Bin edges
    n_bins = max(1, int(np.ceil((d_max - d_min) / CLUSTER_BIN_M)))
    bin_edges = np.linspace(d_min, d_max + 0.001, n_bins + 1)

    min_count = max(2, int(n * CLUSTER_MIN_FRAC))

    # Walk bins from nearest to farthest
    for i in range(n_bins):
        lo = bin_edges[i]
        # Include adjacent bin to handle objects on bin boundary
        hi = bin_edges[min(i + 2, n_bins)]
        cluster = dists[(dists >= lo) & (dists < hi)]

        if len(cluster) >= min_count:
            return float(np.median(cluster))

    # Fallback: 25th percentile (bias toward foreground)
    return float(np.percentile(dists, 25))


class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()


def app_callback(element, buffer, user_data):
    if buffer is None:
        hailo_logger.warning("Received None buffer.")
        return

    pad = element.get_static_pad("src")
    format, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Snapshot LiDAR data thread-safely + staleness check
    with lidar_lock:
        data_age = time.monotonic() - lidar_last_update
        if data_age <= LIDAR_STALE_SEC:
            current_scan = lidar_scan.copy()
        else:
            current_scan = {}

    if len(current_scan) > 0:
        angles    = np.array(list(current_scan.keys()),   dtype=float)
        distances = np.array(list(current_scan.values()), dtype=float)
    else:
        angles    = np.array([])
        distances = np.array([])

    # Sort detections smallest bbox first → foreground objects get priority
    detections_sorted = sorted(
        detections,
        key=lambda d: (d.get_bbox().xmax() - d.get_bbox().xmin()) *
                      (d.get_bbox().ymax() - d.get_bbox().ymin())
    )

    used_angles = np.zeros_like(angles, dtype=bool)
    detection_distances = {}  # id(detection) → distance in meters or None

    for detection in detections_sorted:
        bbox = detection.get_bbox()

        # ── 1. Vertical band check ───────────────────────────────────────────
        bbox_y1 = bbox.ymin() * height
        bbox_y2 = bbox.ymax() * height
        in_lidar_band = not (bbox_y2 < LIDAR_BAND_MIN or bbox_y1 > LIDAR_BAND_MAX)
        
        

        if not in_lidar_band:
            detection_distances[id(detection)] = None
            continue

        # ── 2. No LiDAR data (or stale) ──────────────────────────────────────
        if len(angles) == 0:
            detection_distances[id(detection)] = None
            continue

        # ── 3. Map bbox horizontal extent → LiDAR angles ────────────────────
        angle_left  = (bbox.xmin() - 0.5) * H_FOV
        angle_right = (bbox.xmax() - 0.5) * H_FOV

        lidar_angle_min = (-angle_right) % 360
        lidar_angle_max = (-angle_left)  % 360

        # ── 4. Build angular mask ────────────────────────────────────────────
        if lidar_angle_min <= lidar_angle_max:
            mask = (angles >= lidar_angle_min) & (angles <= lidar_angle_max)
        else:
            mask = (angles >= lidar_angle_min) | (angles <= lidar_angle_max)

        mask = mask & (~used_angles)

        # ── 5. Compute distance (NEAREST CLUSTER + EMA smoothing) ─────────
        #
        # Problem: bbox angular range often extends beyond the physical object
        # (e.g. a laptop bbox is wider than the laptop), so some LiDAR rays
        # hit the wall behind → bimodal distances (object + background).
        # Median picks the middle, which can be background.
        #
        # Solution: bin distances into 0.3m buckets, find the closest bucket
        # with readings, and take the median of that cluster only.
        #
        if np.any(mask):
            valid_dist = distances[mask]
            valid_dist = valid_dist[np.isfinite(valid_dist) & (valid_dist > 0)]
            if len(valid_dist) > 0:
                raw_dist = float(_nearest_cluster_dist(valid_dist))
                cx_norm  = (bbox.xmin() + bbox.xmax()) * 0.5  # 0-1 normalised
                smoothed = _ema_update(detection.get_label(), cx_norm, raw_dist)
                detection_distances[id(detection)] = smoothed
                used_angles |= mask
            else:
                detection_distances[id(detection)] = None
        else:
            detection_distances[id(detection)] = None

    # ── Housekeeping ────────────────────────────────────────────────────────
    _ema_cleanup()

    # ── Draw ─────────────────────────────────────────────────────────────────
    detection_count = len(detections)

    if user_data.use_frame and frame is not None:

        cv2.line(frame, (0, LIDAR_BAND_MIN), (width, LIDAR_BAND_MIN), (255, 100, 0), 1)
        cv2.line(frame, (0, LIDAR_BAND_MAX), (width, LIDAR_BAND_MAX), (255, 100, 0), 1)

        for detection in detections:
            label      = detection.get_label()
            bbox       = detection.get_bbox()
            confidence = detection.get_confidence()

            x1 = int(bbox.xmin() * width)
            y1 = int(bbox.ymin() * height)
            x2 = int(bbox.xmax() * width)
            y2 = int(bbox.ymax() * height)
            center_x = (x1 + x2) // 2

            dist = detection_distances.get(id(detection), None)
            

            if dist is not None:
                dist_text = f"{dist:.2f}m"
                box_color = (0, 255, 0)
            else:
                dist_text = "--"
                box_color = (0, 165, 255)

            text = f"{label} {confidence:.2f} | {dist_text}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            tx = max(0, min(center_x - tw // 2, width - tw - 4))
            ty = max(th + 5, y1 - 10)
            cv2.rectangle(frame, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 4), box_color, cv2.FILLED)
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

        cv2.putText(frame, f"Total: {detection_count}", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    hailo_logger.debug(f"Frame: {user_data.get_count()} | Detections: {detection_count}")
    return


def main():
    hailo_logger.info("Starting Detection + LiDAR Fusion App.")
    user_data = user_app_callback_class()
    user_data.use_frame = True
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()

if __name__ == "__main__":
    main()