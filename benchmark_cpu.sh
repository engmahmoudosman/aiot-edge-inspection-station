#!/usr/bin/env bash
# benchmark_cpu.sh
# Step-by-step CPU benchmark following the course lab instructions.
# Run this on the Raspberry Pi 5.
#
# What this does:
#   1. Starts the camera streaming on a local TCP port
#   2. Runs YOLOv8s inference entirely on the CPU
#   3. Prints inference time per frame (expect ~900-1018 ms → ~1 FPS)
#   4. Open htop in another terminal to watch CPU usage
#
# Usage:
#   chmod +x benchmark_cpu.sh
#   ./benchmark_cpu.sh

set -e

echo "============================================"
echo " CPU BENCHMARK — YOLOv8s on Raspberry Pi 5"
echo "============================================"
echo ""
echo "STEP 1: Starting camera stream on tcp://127.0.0.1:8888 ..."
echo "       (runs in background, Ctrl+C this script to stop everything)"
echo ""

# Start camera stream in background
rpicam-vid -n -t 0 --inline --listen -o tcp://127.0.0.1:8888 &
CAM_PID=$!
sleep 2   # give camera time to start

echo "STEP 2: Setting up Python environment for Ultralytics ..."
echo ""

# Create venv if it doesn't exist
if [ ! -d "$HOME/yolo_cpu_test/.venv" ]; then
    mkdir -p ~/yolo_cpu_test
    cd ~/yolo_cpu_test
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip wheel -q
    pip install -U ultralytics -q
    echo "  ✓ Ultralytics installed"
    cd -
else
    source ~/yolo_cpu_test/.venv/bin/activate
    echo "  ✓ Using existing environment"
fi

echo ""
echo "STEP 3: Running YOLOv8s on CPU ..."
echo "        Watch the inference time printed after each frame."
echo "        Expected: ~900-1018 ms/frame  (~1 FPS)"
echo ""
echo "        ► In another terminal, run:  htop"
echo "          to see CPU load during inference."
echo ""
echo "        Press Ctrl+C to stop when done."
echo ""

# Trap Ctrl+C to clean up camera process
trap "kill $CAM_PID 2>/dev/null; echo ''; echo 'Stopped.'; exit 0" INT TERM

# Run inference — output shows speed per frame automatically
yolo predict model=yolov8s.pt source=tcp://127.0.0.1:8888 imgsz=640 device=cpu show=True

# Cleanup
kill $CAM_PID 2>/dev/null
echo ""
echo "============================================"
echo " DONE — write down the ms/frame value above"
echo " FPS = 1000 / ms_per_frame"
echo " e.g. 1000ms → 1 FPS,  500ms → 2 FPS"
echo "============================================"
