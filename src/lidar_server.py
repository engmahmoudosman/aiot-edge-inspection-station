"""
LiDAR Server

Usage:
    python lidar_server.py --lidar rplidar
    python lidar_server.py --lidar hokuyo
"""

import argparse
import glob
import json
import os
import signal
import socket
import subprocess
import sys
import time
                                                                                                                                           

# === Network config ==========================================================
UDP_HOST = "127.0.0.1"
UDP_PORT = 5001


# === RPLidar config ==========================================================
RPLIDAR_BAUD           = 115200
RPLIDAR_PREFERRED_PORT = "/dev/ttyUSB0"
RPLIDAR_MAX_BUF_MEAS   = 5000
RPLIDAR_SCAN_INTERVAL  = 0.01


# === Hokuyo URG-04LX-UG01 config =============================================
# USB CDC: baud rate is virtual, but we follow the proven init sequence.
HOKUYO_BAUD_INITIAL    = 19200          # sensor default on power-up
HOKUYO_BAUD_FAST       = 115200
HOKUYO_PREFERRED_PORT  = "/dev/ttyACM0"
HOKUYO_ARES            = 1024           # steps per full rotation
HOKUYO_AMIN            = 44             # first valid step
HOKUYO_AMAX            = 725            # last  valid step
HOKUYO_AFRT            = 384            # step that points forward (0 deg)
HOKUYO_DMIN_MM         = 20             # minimum valid distance
HOKUYO_DMAX_MM         = 4000           # maximum valid distance
HOKUYO_STEP_DEG        = 360.0 / HOKUYO_ARES   # ~0.3516 deg/step


# === Globals =================================================================
_driver   = None
_sock     = None
_shutdown = False


# === Signal handling =========================================================

def _cleanup():
    global _driver, _sock
    if _driver is not None:
        try: _driver.close()
        except Exception: pass
        _driver = None
    if _sock is not None:
        try: _sock.close()
        except Exception: pass
        _sock = None


def _signal_handler(_sig, _frame):
    global _shutdown
    print("\n[lidar_server] Shutting down...")
    _shutdown = True
    _cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# === Shared port helper ======================================================

def _release_port(port):
    """Force-release a serial port if another process is holding it."""
    if not os.path.exists(port):
        return
    try:
        r = subprocess.run(["fuser", port], capture_output=True, text=True, timeout=2)
        pids = r.stdout.strip()
        if pids:
            print(f"[lidar_server] Port {port} held by PIDs: {pids} -- releasing...")
            subprocess.call(["fuser", "-k", port],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[lidar_server] Port release failed: {e}")


# =============================================================================
# Driver interface
#
# Each driver implements:
#   connect()    -> opens hardware, raises on failure
#   iter_scans() -> generator yielding lists of (angle_deg, dist_m) per scan,
#                   with angle normalized to [0, 360) and 0 deg = forward
#   close()      -> clean shutdown (motor off, laser off, port closed)
# =============================================================================


# ----------------------------------------------------------------------------
# RPLidar driver
# ----------------------------------------------------------------------------
class RPLidarDriver:
    def __init__(self):
        from rplidar import RPLidar as _RPLidar
        self._RPLidar = _RPLidar
        self._lidar = None
        self._port  = None

    def _find_port(self):
        # 1. by-id symlinks (survives USB port swaps)
        for path in glob.glob("/dev/serial/by-id/*"):
            name = os.path.basename(path).lower()
            if any(tag in name for tag in ("rplidar", "cp210x", "slamtec")):
                real = os.path.realpath(path)
                print(f"[rplidar] Found via by-id: {path} -> {real}")
                return real
        # 2. preferred port
        if os.path.exists(RPLIDAR_PREFERRED_PORT):
            return RPLIDAR_PREFERRED_PORT
        # 3. probe every ttyUSB
        for port in sorted(glob.glob("/dev/ttyUSB*")):
            try:
                test = self._RPLidar(port, baudrate=RPLIDAR_BAUD)
                test.get_info()
                test.stop(); test.disconnect()
                return port
            except Exception:
                continue
        return None

    def connect(self):
        port = self._find_port()
        if port is None:
            raise RuntimeError("No RPLidar found on /dev/ttyUSB* or /dev/serial/by-id/")
        _release_port(port)
        self._lidar = self._RPLidar(port, baudrate=RPLIDAR_BAUD)
        info = self._lidar.get_info()
        self._port = port
        print(f"[rplidar] Connected on {port} -- "
              f"model {info.get('model', '?')}, firmware {info.get('firmware', '?')}")

    def iter_scans(self):
        for scan in self._lidar.iter_scans(scan_type='express',
                                           max_buf_meas=RPLIDAR_MAX_BUF_MEAS):
            if _shutdown:
                break
            points = []
            for item in scan:
                # rplidar returns (quality, angle, dist_mm) or
                # (new_scan, quality, angle, dist_mm) depending on mode
                if len(item) == 3:
                    angle, dist_mm = item[1], item[2]
                elif len(item) == 4:
                    angle, dist_mm = item[2], item[3]
                else:
                    continue
                if dist_mm > 0:
                    points.append((angle % 360.0, dist_mm / 1000.0))
            yield points
            time.sleep(RPLIDAR_SCAN_INTERVAL)

    def close(self):
        if self._lidar is None:
            return
        for fn in (self._lidar.stop, self._lidar.stop_motor, self._lidar.disconnect):
            try: fn()
            except Exception: pass
        self._lidar = None


# ----------------------------------------------------------------------------
# Hokuyo URG-04LX-UG01 driver (SCIP2.0)
# ----------------------------------------------------------------------------
class HokuyoDriver:
    def __init__(self):
        import serial as _serial
        self._serial = _serial
        self._ser  = None
        self._port = None

    def _find_port(self):
        # 1. by-id symlinks
        for path in glob.glob("/dev/serial/by-id/*"):
            name = os.path.basename(path).lower()
            if any(tag in name for tag in ("hokuyo", "urg")):
                real = os.path.realpath(path)
                print(f"[hokuyo] Found via by-id: {path} -> {real}")
                return real
        # 2. preferred port
        if os.path.exists(HOKUYO_PREFERRED_PORT):
            return HOKUYO_PREFERRED_PORT
        # 3. first ttyACM
        acms = sorted(glob.glob("/dev/ttyACM*"))
        return acms[0] if acms else None

    def _send(self, cmd):
        self._ser.write((cmd + "\n").encode())

    @staticmethod
    def _decode_3char(raw):
        """SCIP2.0 3-char (18-bit) distance decoding."""
        distances = []
        for i in range(0, len(raw) - 2, 3):
            b0 = raw[i]   - 0x30
            b1 = raw[i+1] - 0x30
            b2 = raw[i+2] - 0x30
            distances.append((b0 << 12) | (b1 << 6) | b2)
        return distances

    def connect(self):
        port = self._find_port()
        if port is None:
            raise RuntimeError("No Hokuyo found on /dev/ttyACM* or /dev/serial/by-id/")
        _release_port(port)

        # Open at initial baud (USB CDC ignores the value, but the init
        # sequence works regardless and matches the proven standalone script).
        ser = self._serial.Serial(
            port,
            baudrate=HOKUYO_BAUD_INITIAL,
            timeout=2.0,
            bytesize=self._serial.EIGHTBITS,
            parity=self._serial.PARITY_NONE,
            stopbits=self._serial.STOPBITS_ONE,
        )
        time.sleep(0.5)
        self._ser  = ser
        self._port = port

        # Ensure SCIP2.0 mode (idempotent)
        self._send("SCIP2.0")
        time.sleep(0.1)
        ser.flushInput()

        # Switch to high baud (harmless on USB CDC)
        self._send(f"SS{HOKUYO_BAUD_FAST:06d}")
        time.sleep(0.2)
        ser.baudrate = HOKUYO_BAUD_FAST
        ser.flushInput()
        time.sleep(0.3)

        # Laser on
        self._send("BM")
        time.sleep(0.1)
        ser.flushInput()

        print(f"[hokuyo] Connected on {port}")

    def _one_scan(self):
        """Issue a GD command and return list of (angle_deg, dist_m)."""
        cmd = f"GD{HOKUYO_AMIN:04d}{HOKUYO_AMAX:04d}01"
        self._ser.write((cmd + "\n").encode())

        # Response lines: <echo> <status+checksum> <timestamp> <data...> <blank>
        self._ser.readline()                         # echo
        status = self._ser.readline().strip()        # status
        if not status.startswith(b"00"):
            raise RuntimeError(f"Hokuyo GD failed, status={status!r}")
        self._ser.readline()                         # timestamp

        raw = bytearray()
        while True:
            line = self._ser.readline()
            stripped = line.strip()
            if stripped == b"":
                break
            # each data line has a trailing checksum byte -> drop it
            raw.extend(stripped[:-1])

        distances_mm = self._decode_3char(bytes(raw))

        points = []
        for i, dist_mm in enumerate(distances_mm):
            if not (HOKUYO_DMIN_MM <= dist_mm <= HOKUYO_DMAX_MM):
                continue
            step = HOKUYO_AMIN + i
            # Hokuyo: step 384 = forward, +left / -right (CCW positive)
            angle_signed = (step - HOKUYO_AFRT) * HOKUYO_STEP_DEG
            # Normalize to [0, 360). If on-screen left/right appears mirrored
            # versus reality, change the line below to:  (-angle_signed) % 360.0
            angle_deg = angle_signed % 360.0
            points.append((angle_deg, dist_mm / 1000.0))
        return points

    def iter_scans(self):
        while not _shutdown:
            yield self._one_scan()

    def close(self):
        if self._ser is None:
            return
        try:
            self._ser.write(b"QT\n")   # laser off
            time.sleep(0.05)
        except Exception: pass
        try: self._ser.close()
        except Exception: pass
        self._ser = None


# =============================================================================
# Main sender loop
# =============================================================================

def _make_driver(kind):
    if kind == "rplidar":
        return RPLidarDriver()
    if kind == "hokuyo":
        return HokuyoDriver()
    raise ValueError(f"Unknown lidar kind: {kind}")


def _connect_with_retry(kind):
    global _driver
    backoff = 1
    while not _shutdown:
        drv = _make_driver(kind)
        try:
            drv.connect()
            _driver = drv
            return drv
        except Exception as e:
            print(f"[lidar_server] Connect failed ({e}); retrying in {backoff}s...")
            try: drv.close()
            except Exception: pass
            time.sleep(backoff)
            backoff = min(backoff * 2, 10)
    sys.exit(0)


def lidar_sender(kind):
    global _sock
    _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest  = (UDP_HOST, UDP_PORT)

    drv = _connect_with_retry(kind)
    print(f"[lidar_server] Streaming '{kind}' data on UDP {UDP_HOST}:{UDP_PORT}")

    while not _shutdown:
        try:
            for points in drv.iter_scans():
                if _shutdown:
                    break
                if points:
                    payload = json.dumps(
                        {"count": len(points), "data": points},
                        separators=(",", ":"),
                    )
                    _sock.sendto(payload.encode(), dest)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[lidar_server] Scan error ({e}); reconnecting...")
            try: drv.close()
            except Exception: pass
            time.sleep(1)
            if not _shutdown:
                drv = _connect_with_retry(kind)
                print(f"[lidar_server] Reconnected ({kind})")

    _cleanup()


def main():
    ap = argparse.ArgumentParser(
        description="LiDAR UDP server (supports RPLidar and Hokuyo URG-04LX)"
    )
    ap.add_argument(
        "--lidar",
        choices=["rplidar", "hokuyo"],
        required=True,
        help="Which LiDAR hardware is connected",
    )
    args = ap.parse_args()
    print(f"[lidar_server] Starting with --lidar {args.lidar}")
    lidar_sender(args.lidar)


if __name__ == "__main__":
    main()