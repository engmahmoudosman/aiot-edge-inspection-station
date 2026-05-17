#!/usr/bin/env python3
"""
Sensor Fusion Launcher

Usage:
    python sensor_fusion.py --lidar    --sensor rplidar
    python sensor_fusion.py --lidar    --sensor hokuyo
    python sensor_fusion.py --geometry --sensor rplidar
    python sensor_fusion.py --geometry --sensor hokuyo
"""
#pip install rplidar-roboticia pyserial opencv-python numpy psutil
import argparse
import os
import signal
import subprocess
import sys
import time

# === Paths ====================================================================
USER_HOME       = os.path.expanduser("~")
HAILO_APPS_DIR  = os.path.join(USER_HOME, "hailo-apps")
HERE            = os.path.dirname(os.path.abspath(__file__))
LIDAR_SERVER    = os.path.join(HERE, "lidar_server.py")
LIDAR_CLOUD     = os.path.join(HERE, "lidar_cloud.py")
DETECTION_LIDAR = os.path.join(HERE, "detection_with_lidar.py")
HEF_MODEL       = os.path.join(HAILO_APPS_DIR, "resources", "models",
                               "hailo8l", "yolov6n.hef")

# Scripts that may linger and need force-killing at shutdown
DEMO_SCRIPTS = ("lidar_server.py", "lidar_cloud.py", "detection_with_lidar.py")


# === Helpers ==================================================================

def log(msg, tag="INFO"):
    print(f"[{tag}] {msg}", flush=True)


def run_bash(cmd, use_setup_env=True):
    """Launch a bash command in its own process group so we can kill the whole tree."""
    env = os.environ.copy()
    env["HAILO_MONITOR"] = "1"
    if use_setup_env:
        full = f"cd {HAILO_APPS_DIR} && source setup_env.sh && {cmd}"
    else:
        full = cmd
    return subprocess.Popen(
        ["bash", "-c", full],
        env=env,
        preexec_fn=os.setsid,
    )


def kill_process_group(proc, grace=3.0):
    """Terminate a process group, escalating to SIGKILL if needed."""
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def pkill_demo_scripts():
    """Belt-and-braces cleanup - pkill anything still hanging around."""
    for name in DEMO_SCRIPTS:
        subprocess.call(
            ["pkill", "-KILL", "-f", name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

# === Demo commands ============================================================

def lidar_cmd():
    return f"cd {HERE} && python {LIDAR_CLOUD}"


def geometry_cmd():
    return (
        f"cd {HERE} && python {DETECTION_LIDAR} "
        f"--hef-path {HEF_MODEL} --use-frame --input rpi"
    )


def lidar_server_cmd(sensor):
    return f"cd {HERE} && python {LIDAR_SERVER} --lidar {sensor}"


# === Main =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sensor Fusion demo launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python sensor_fusion.py --lidar    --sensor rplidar\n"
            "  python sensor_fusion.py --lidar    --sensor hokuyo\n"
            "  python sensor_fusion.py --geometry --sensor rplidar\n"
            "  python sensor_fusion.py --geometry --sensor hokuyo\n"
        ),
    )

    demo = parser.add_mutually_exclusive_group(required=True)
    demo.add_argument("--lidar", action="store_true",
                      help="Run the LiDAR 360 scan demo")
    demo.add_argument("--geometry", action="store_true",
                      help="Run the geometry fusion demo (YOLO + LiDAR)")

    parser.add_argument("--sensor", choices=["rplidar", "hokuyo"], required=True,
                        help="Which LiDAR hardware is connected")

    args = parser.parse_args()

    demo_name, demo_command = (
        ("LiDAR 360 Scan", lidar_cmd()) if args.lidar
        else ("Geometry Fusion", geometry_cmd())
    )

    lidar_server_proc = None
    demo_proc = None

    try:
        # 1. Start the LiDAR server with the chosen driver
        log(f"Starting LiDAR server (driver: {args.sensor})...")
        lidar_server_proc = run_bash(lidar_server_cmd(args.sensor))
        time.sleep(2)  # give it a moment to connect and bind its socket

        if lidar_server_proc.poll() is not None:
            log("LiDAR server exited immediately -- check driver/hardware.", "ERR")
            return 1

        # 2. Launch the requested demo (blocks until demo exits or Ctrl+C)
        log(f"Launching '{demo_name}'...")
        demo_proc = run_bash(demo_command)
        demo_proc.wait()
        log(f"'{demo_name}' exited.", "OK")

    except KeyboardInterrupt:
        log("Interrupted - shutting down...", "WARN")

    finally:
        # 3. Shutdown everything
        log("Stopping demo...")
        kill_process_group(demo_proc)
        log("Stopping LiDAR server...")
        kill_process_group(lidar_server_proc)
        pkill_demo_scripts()
        log("Shutdown complete.", "OK")

    return 0


if __name__ == "__main__":
    sys.exit(main())