#!/usr/bin/env python3
"""
monitor_system.py
Run this in a SECOND terminal while sensor_fusion_m3.py is running.
It samples CPU and RAM every second and prints a live table.
At the end (Ctrl+C) it prints a summary ready for the report table.

Usage:
    python3 monitor_system.py
    python3 monitor_system.py --duration 60   # auto-stop after 60s
"""

import argparse
import signal
import sys
import time
from datetime import datetime

try:
    import psutil
except ImportError:
    print("Install psutil first:  pip install psutil")
    sys.exit(1)

_running = True

def main():
    global _running

    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=0,
                    help="Stop after N seconds (0 = run until Ctrl+C)")
    args = ap.parse_args()

    def stop(*_):
        global _running
        _running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    cpu_samples = []
    ram_samples = []
    t_start = time.time()
    t_end   = t_start + args.duration if args.duration > 0 else float("inf")

    print("=" * 60)
    print("  SYSTEM MONITOR  —  run while sensor_fusion_m3.py is live")
    print("  Press Ctrl+C to stop and see summary")
    print("=" * 60)
    print(f"{'Time':>8}  {'CPU %':>7}  {'RAM MB':>8}  {'Cores busy':>10}")
    print("-" * 60)

    while _running and time.time() < t_end:
        cpu  = psutil.cpu_percent(interval=1.0)
        mem  = psutil.virtual_memory()
        ram  = mem.used / 1e6
        per_core = psutil.cpu_percent(percpu=True)
        busy = sum(1 for c in per_core if c > 20)

        cpu_samples.append(cpu)
        ram_samples.append(ram)

        elapsed = int(time.time() - t_start)
        h, r = divmod(elapsed, 3600)
        m, s = divmod(r, 60)
        print(f"{h:02d}:{m:02d}:{s:02d}  {cpu:>6.1f}%  {ram:>8.0f}   {busy}/4 cores")

    if not cpu_samples:
        return

    import statistics
    cpu_avg  = statistics.mean(cpu_samples)
    cpu_peak = max(cpu_samples)
    ram_avg  = statistics.mean(ram_samples)
    ram_peak = max(ram_samples)

    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Duration         : {int(time.time()-t_start)} s  ({len(cpu_samples)} samples)")
    print(f"  CPU avg          : {cpu_avg:.1f} %")
    print(f"  CPU peak         : {cpu_peak:.1f} %")
    print(f"  RAM avg          : {ram_avg:.0f} MB")
    print(f"  RAM peak         : {ram_peak:.0f} MB")
    print()
    print("  ── Paste into report Table III ──")
    print(f"  Sensor Fusion M3 (idle)  & --  & {cpu_avg:.0f} & {ram_avg:.0f} \\\\")
    print()
    print("  (Trigger a capture during the test to capture peak values)")
    print("=" * 60)

    # Save to file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"monitor_results_{ts}.txt"
    with open(out, "w") as f:
        f.write(f"Duration: {int(time.time()-t_start)}s, {len(cpu_samples)} samples\n")
        f.write(f"CPU avg: {cpu_avg:.1f}%,  peak: {cpu_peak:.1f}%\n")
        f.write(f"RAM avg: {ram_avg:.0f} MB,  peak: {ram_peak:.0f} MB\n")
        f.write("\nRaw samples (CPU%, RAM_MB):\n")
        for c, r in zip(cpu_samples, ram_samples):
            f.write(f"{c:.1f},{r:.0f}\n")
    print(f"  Saved → {out}")

if __name__ == "__main__":
    main()
