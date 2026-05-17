"""
inspection_logger.py

Thread-safe CSV event logger for the inspection station.
Imported and used directly by detection_with_lidar.py.

CSV columns:
    timestamp_iso, class_label, confidence, distance_m, alert_triggered
"""

import csv
import os
import threading
from datetime import datetime, date


class InspectionLogger:
    def __init__(self, output_dir: str = "./logs", enabled: bool = True):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._output_dir = output_dir
        self._current_date = None
        self._file = None
        self._writer = None
        if enabled:
            os.makedirs(output_dir, exist_ok=True)
            self._open_file()

    def _log_path(self) -> str:
        today = date.today().isoformat()
        return os.path.join(self._output_dir, f"inspection_{today}.csv")

    def _open_file(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
        self._current_date = date.today()
        path = self._log_path()
        new_file = not os.path.exists(path)
        self._file = open(path, "a", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        if new_file:
            self._writer.writerow(
                ["timestamp_iso", "class_label", "confidence", "distance_m", "alert_triggered"]
            )

    def log_event(
        self,
        label: str,
        confidence: float,
        distance_m,          # float or None
        alert_triggered: bool,
    ):
        if not self.enabled:
            return
        with self._lock:
            # Rotate file on day change
            if date.today() != self._current_date:
                self._open_file()
            dist_str = f"{distance_m:.3f}" if distance_m is not None else ""
            self._writer.writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                label,
                f"{confidence:.3f}",
                dist_str,
                str(alert_triggered),
            ])

    def close(self):
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except Exception:
                    pass
                self._file = None
