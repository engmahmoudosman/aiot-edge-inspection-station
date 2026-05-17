#!/usr/bin/env bash
# benchmark_npu.sh
# Step-by-step NPU benchmark following the course lab instructions.
# Run this on the Raspberry Pi 5.
#
# What this does:
#   Runs hailo-detect with --show-fps so the FPS is printed in the video window.
#   Uses HAILO_MONITOR=1 so hailortcli monitor can show NPU utilisation.
#
# Usage:
#   chmod +x benchmark_npu.sh
#   ./benchmark_npu.sh
#
# You will need TWO extra terminals open before running this:
#   Terminal A:  hailortcli monitor          (watch NPU usage)
#   Terminal B:  htop                        (watch CPU usage)

set -e

echo "============================================"
echo " NPU BENCHMARK — Hailo-8L on Raspberry Pi 5"
echo "============================================"
echo ""
echo "Before continuing, open these in two separate terminals:"
echo "   Terminal A:  hailortcli monitor"
echo "   Terminal B:  htop"
echo ""
read -p "Press ENTER when ready ..."

echo ""
echo "Sourcing Hailo environment ..."
cd ~/hailo-apps
source setup_env.sh

echo ""
echo "Enabling Hailo monitor export ..."
export HAILO_MONITOR=1

echo ""
echo "Starting Hailo detection with FPS display ..."
echo "  ► Watch the video window title / overlay for FPS"
echo "  ► Watch Terminal A (hailortcli monitor) for NPU usage"
echo "  ► Watch Terminal B (htop) for CPU usage"
echo "  ► Press Ctrl+C here to stop"
echo ""

hailo-detect --input rpi --show-fps

echo ""
echo "============================================"
echo " DONE — write down the values you observed:"
echo "   FPS        : ___"
echo "   CPU load   : ___  %"
echo "   NPU usage  : ___  %"
echo "============================================"
