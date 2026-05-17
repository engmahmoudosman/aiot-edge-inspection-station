"""
distance_accuracy_test.py

Interactive LiDAR distance accuracy measurement.
Run AFTER lidar_server.py is already running.

Place a flat object (book, cardboard) perpendicular to the LiDAR beam.
The script collects 100 distance readings at the forward angle (0°) and
prints mean, median, std-dev, and error vs. the ground-truth you enter.

Usage (on RPi 5):
    # Terminal 1 — start LiDAR server
    python3 lidar_server.py --lidar hokuyo

    # Terminal 2 — run accuracy test
    python3 distance_accuracy_test.py --distance 1.0
    python3 distance_accuracy_test.py --distance 1.5
    python3 distance_accuracy_test.py --distance 2.0
    # etc.

Output is a ready-to-paste row for Table IV in the report.
"""

import argparse
import json
import socket
import time
import numpy as np


UDP_PORT = 5001
N_SAMPLES = 100       # samples to collect per run
WINDOW_DEG = 5.0      # angular window around 0° (forward)


def collect_samples(n: int, window_deg: float) -> list[float]:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", UDP_PORT))
    s.settimeout(2.0)

    samples = []
    print(f"Collecting {n} samples (forward ±{window_deg/2}°) ...", flush=True)

    while len(samples) < n:
        try:
            msg, _ = s.recvfrom(65535)
            data = json.loads(msg.decode()).get("data", [])
            for angle, dist in data:
                # Accept rays near 0° forward (handles wrap-around at 360°)
                a = angle if angle <= 180 else angle - 360
                if abs(a) <= window_deg / 2 and 0.05 < dist < 5.0:
                    samples.append(dist)
                    if len(samples) >= n:
                        break
        except socket.timeout:
            print("  Waiting for LiDAR data ... (is lidar_server.py running?)")
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(0.1)

    s.close()
    return samples


def main():
    ap = argparse.ArgumentParser(description="LiDAR distance accuracy test")
    ap.add_argument("--distance", type=float, required=True,
                    help="Ground-truth distance to target in metres")
    ap.add_argument("--samples", type=int, default=N_SAMPLES)
    ap.add_argument("--window", type=float, default=WINDOW_DEG,
                    help="Angular window around forward (degrees)")
    args = ap.parse_args()

    gt = args.distance
    samples = collect_samples(args.samples, args.window)

    arr = np.array(samples)
    mean   = arr.mean()
    median = np.median(arr)
    std    = arr.std()
    err    = median - gt
    pct    = abs(err) / gt * 100

    print(f"\n{'─'*50}")
    print(f"  Ground truth    : {gt:.3f} m")
    print(f"  Samples         : {len(arr)}")
    print(f"  Mean            : {mean:.4f} m")
    print(f"  Median          : {median:.4f} m")
    print(f"  Std-dev         : {std*1000:.1f} mm")
    print(f"  Error (median)  : {err*1000:+.1f} mm  ({pct:.2f}%)")
    print(f"{'─'*50}")
    print(f"\nTable row (copy into report.tex):")
    print(f"  {gt:.1f} & {median:.3f} & {err*1000:+.1f}~mm & {pct:.2f}\\% \\\\")


if __name__ == "__main__":
    main()
