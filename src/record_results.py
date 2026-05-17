#!/usr/bin/env python3
"""
record_results.py
After running benchmark_cpu.sh and benchmark_npu.sh, run this script.
It asks you to type in the numbers you observed and prints the
ready-to-paste LaTeX table rows for Table III in the report.

Usage:
    python3 record_results.py
"""

print("=" * 55)
print("  BENCHMARK RESULTS RECORDER")
print("  Fill in the values you observed.")
print("=" * 55)
print()

# CPU results
print("── CPU mode (from benchmark_cpu.sh) ──")
cpu_ms  = float(input("  Inference time per frame (ms)  : "))
cpu_cpu = float(input("  CPU load from htop (%)         : "))
cpu_ram = float(input("  RAM used from htop (MB)        : "))
cpu_fps = round(1000 / cpu_ms, 1)
print(f"  → Calculated FPS: {cpu_fps}")

print()
print("── NPU mode (from benchmark_npu.sh) ──")
npu_fps = float(input("  FPS shown in video window      : "))
npu_cpu = float(input("  CPU load from htop (%)         : "))
npu_npu = float(input("  NPU usage from hailortcli (%)  : "))
npu_ram = float(input("  RAM used from htop (MB)        : "))

print()
print("=" * 55)
print("  TABLE III — Paste into report.tex")
print("=" * 55)
print()
print(r"\begin{table}[t]")
print(r"\caption{Inference Performance: CPU vs.\ Hailo-8L NPU}")
print(r"\label{tab:benchmark}")
print(r"\centering")
print(r"\begin{tabular}{lccc}")
print(r"\hline")
print(r"Mode & FPS & CPU Load (\%) & RAM (MB) \\")
print(r"\hline")
print(f"YOLOv8s — CPU only  & {cpu_fps:.0f} & {cpu_cpu:.0f} & {cpu_ram:.0f} \\\\")
print(f"YOLOv8s — Hailo-8L  & {npu_fps:.0f} & {npu_cpu:.0f} & {npu_ram:.0f} \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")

speedup = npu_fps / cpu_fps if cpu_fps > 0 else 0
print()
print(f"  Speed-up:  {speedup:.0f}× faster with Hailo-8L")
print()

# Save to file
with open("benchmark_results.txt", "w") as f:
    f.write(f"CPU  — FPS: {cpu_fps}, CPU: {cpu_cpu}%, RAM: {cpu_ram} MB\n")
    f.write(f"NPU  — FPS: {npu_fps}, CPU: {npu_cpu}%, NPU: {npu_npu}%, RAM: {npu_ram} MB\n")
    f.write(f"Speed-up: {speedup:.0f}x\n")
print("  Results saved to benchmark_results.txt")
