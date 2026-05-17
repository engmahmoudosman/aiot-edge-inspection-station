"""
benchmark_fps.py

Measures and compares object-detection FPS and CPU utilisation:
  A) Hailo-8L accelerated pipeline  (reads from live dashboard frame counter)
  B) CPU-only YOLOv6n ONNX Runtime  (single-process, no Hailo)

Run this AFTER the full pipeline is working.

Usage:
    # Hailo benchmark — run while sensor_fusion.py is active:
    python3 benchmark_fps.py --mode hailo --duration 30

    # CPU-only benchmark (standalone, no Hailo, no camera required):
    python3 benchmark_fps.py --mode cpu --duration 30

Dependencies:
    pip3 install onnxruntime psutil

ONNX model: download yolov6n.onnx from the Hailo model zoo or export with:
    yolo export model=yolov6n.pt format=onnx imgsz=640
Place it in the same directory as this script, or pass --onnx-path.

Output is ready-to-paste rows for Table III in the report.
"""

import argparse
import subprocess
import sys
import time

import numpy as np
import psutil


def _hailo_benchmark(duration: int, fps_override: float | None):
    """
    Measure system load while Hailo pipeline is running.
    FPS is taken from --hailo-fps if supplied, otherwise read from the
    detection_with_lidar.py terminal output or display window title.
    """
    print(f"[hailo] Sampling system for {duration}s while pipeline is running ...")
    print("        Make sure 'python3 sensor_fusion.py' is running in another terminal.\n")

    cpu_samples = []
    ram_samples = []
    t_end = time.time() + duration

    while time.time() < t_end:
        cpu_samples.append(psutil.cpu_percent(interval=1.0))
        ram_samples.append(psutil.virtual_memory().used / 1e6)

    # Try hailortcli first (may not be on PATH inside venv)
    fps_hailo = fps_override
    if fps_hailo is None:
        try:
            result = subprocess.run(
                ["hailortcli", "monitor", "--seconds", "5"],
                capture_output=True, text=True, timeout=10,
            )
            import re
            for line in result.stdout.splitlines():
                if "fps" in line.lower():
                    nums = re.findall(r"[\d.]+", line)
                    if nums:
                        fps_hailo = float(nums[0])
                        break
        except Exception:
            pass

    cpu_avg = np.mean(cpu_samples)
    ram_avg = np.mean(ram_samples)

    print("═" * 55)
    print("  MODE: Hailo-8L accelerated pipeline")
    print("─" * 55)
    if fps_hailo is not None:
        print(f"  FPS (Hailo)     : {fps_hailo:.1f}")
    else:
        print("  FPS (Hailo)     : NOT CAPTURED automatically.")
        print("  ► Read the FPS from one of these sources:")
        print("    1. The OpenCV display window title bar")
        print("    2. The detection_with_lidar.py terminal  (look for 'FPS:' lines)")
        print("    3. Re-run with:  --hailo-fps <value>")
    print(f"  CPU utilisation : {cpu_avg:.1f} %")
    print(f"  RAM used        : {ram_avg:.0f} MB")
    print("═" * 55)

    if fps_hailo is not None:
        fps_str = f"{fps_hailo:.0f}"
    else:
        fps_str = "??"
        print("\n  ⚠  Fill in FPS manually, then use the table row below.")
    print(f"\nTable row (paste into report):")
    print(f"  YOLOv6n, Hailo-8L & {fps_str} & {cpu_avg:.0f} & {ram_avg:.0f} \\\\")


def _cpu_benchmark(duration: int, onnx_path: str, input_size: int):
    """Run YOLOv6n with ONNX Runtime on CPU and measure FPS."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] onnxruntime not installed. Run: pip3 install onnxruntime")
        sys.exit(1)

    import os
    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX model not found: {onnx_path}")
        print("  Download or export yolov6n.onnx and place it in the project directory.")
        print("  Or pass --onnx-path /path/to/model.onnx")
        sys.exit(1)

    print(f"[cpu] Loading {onnx_path} ...")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4   # use all 4 RPi 5 cores
    sess = ort.InferenceSession(onnx_path, sess_opts,
                                 providers=["CPUExecutionProvider"])
    inp_name  = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape  # [1, 3, H, W]
    h = inp_shape[2] if isinstance(inp_shape[2], int) else input_size
    w = inp_shape[3] if isinstance(inp_shape[3], int) else input_size

    # Synthetic random frame (same as real camera input)
    dummy = np.random.rand(1, 3, h, w).astype(np.float32)

    # Warm-up
    print(f"[cpu] Warming up (input {w}×{h}) ...")
    for _ in range(5):
        sess.run(None, {inp_name: dummy})

    print(f"[cpu] Benchmarking for {duration}s ...")
    cpu_samples = []
    frame_count = 0
    t_start = time.time()
    t_end   = t_start + duration

    while time.time() < t_end:
        t0 = time.perf_counter()
        sess.run(None, {inp_name: dummy})
        elapsed = time.perf_counter() - t0
        frame_count += 1
        cpu_samples.append(psutil.cpu_percent(interval=None))

    total_sec = time.time() - t_start
    fps       = frame_count / total_sec
    cpu_avg   = np.mean(cpu_samples)
    ram_mb    = psutil.virtual_memory().used / 1e6

    print("═" * 55)
    print("  MODE: CPU-only (ONNX Runtime, 4 threads)")
    print("─" * 55)
    print(f"  FPS             : {fps:.1f}")
    print(f"  CPU utilisation : {cpu_avg:.1f} %")
    print(f"  RAM used        : {ram_mb:.0f} MB")
    print(f"  Frames run      : {frame_count}")
    print("═" * 55)
    print(f"\nTable row:\n  YOLOv6n, CPU only & {fps:.0f} & {cpu_avg:.0f} & {ram_mb:.0f} \\\\")


def main():
    ap = argparse.ArgumentParser(description="FPS benchmark: Hailo vs CPU")
    ap.add_argument("--mode",      choices=["hailo", "cpu"], required=True)
    ap.add_argument("--duration",  type=int,   default=30,
                    help="Benchmark duration in seconds")
    ap.add_argument("--hailo-fps", type=float, default=None,
                    help="Manually supply Hailo FPS (read from window title or terminal)")
    ap.add_argument("--onnx-path", default="yolov6n.onnx",
                    help="Path to YOLOv6n ONNX model (CPU mode only)")
    ap.add_argument("--input-size", type=int, default=640)
    args = ap.parse_args()

    if args.mode == "hailo":
        _hailo_benchmark(args.duration, args.hailo_fps)
    else:
        _cpu_benchmark(args.duration, args.onnx_path, args.input_size)


if __name__ == "__main__":
    main()
