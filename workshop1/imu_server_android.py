#!/usr/bin/env python3
"""
----------------------------------
▪ First download the Sensor Logger app (red recording icon) from Google Play store
▪ Enable the Accelerometer and Gyroscope options
▪ Go to Settings->Data Streaming and enable HTTP Push. 
▪ Match the Push URL to this server (run the server once and see what URL it says)
----------------------------------
This script:
▪ Receives HTTP POSTs from Sensor Logger
▪ Keeps only       gyro_x/y/z and accel_x/y/z
▪ Converts times   (ns → ms), rounds to the nearest ms,
                   then subtracts the first time so logs start at 0 ms
▪ If several samples share the same millisecond it averages them
▪ Forward-fills any missing IMU values from the previous row
▪ Appends rows to  processed_<OUT_CSV>.csv  *incrementally*

Run:
    pip install flask pandas
    python imu_server_android.py
"""

# ───────────────────────── CONFIGURATION ──────────────────────────────────── #
HOST          = "0.0.0.0"        # network interface to bind
PORT          = 5000             # listening port
OUT_CSV       = "workshop1/data/imu_cal.csv"

DESIRED_COLS  = [                # six IMU channels to keep
    "gyro_uncal_x", "gyro_uncal_y", "gyro_uncal_z",
    "accel_uncal_x", "accel_uncal_y", "accel_uncal_z",
]

SENSOR_MAP = {                   # incoming name  → output prefix
    "gyroscope":     "gyro",
    "accelerometer": "accel",
    "gyroscopeuncalibrated":  "gyro_uncal",
    "accelerometeruncalibrated": "accel_uncal",
    "totalacceleration":      "totalaccel",
}

# ────────────────────────── IMPLEMENTATION ───────────────────────────────── #
import csv
import json
from pathlib import Path
from threading import Lock

import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

CSV_PATH   = Path(OUT_CSV)
CSV_LOCK   = Lock()                      # file & state protection
HEADER     = ["time_ms"] + DESIRED_COLS  # final CSV header

_first_time_ms: int | None = None        # baseline for 0 ms
_last_values          = {c: None for c in DESIRED_COLS}  # for ffill

# ──────────────────────────── utilities ──────────────────────────────────── #
def _write_header_once() -> None:
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="") as fh:
            csv.writer(fh).writerow(HEADER)

def _append_rows(rows: list[dict]) -> None:
    """Thread-safe CSV append."""
    with CSV_LOCK:
        _write_header_once()
        with CSV_PATH.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=HEADER)
            writer.writerows(rows)

# ────────────────────────── request handler ──────────────────────────────── #
@app.route("/", methods=["POST"])
def receive_post() -> tuple:
    raw = request.get_data(as_text=False)

    # 1) decode JSON
    try:
        data    = json.loads(raw)
        payload = data.get("payload", [])
    except (json.JSONDecodeError, AttributeError):
        return jsonify(error="Body must be JSON with a 'payload' list"), 400

    # 2) collect rows -> DataFrame (may contain duplicates per ms)
    rows = []
    for item in payload:
        name = item.get("name")
        if name not in SENSOR_MAP:
            continue                                   # ignore other sensors

        t_ns      = item.get("time")
        if t_ns is None:
            continue

        time_ms   = int(round(t_ns / 1e6))            # ns → ms, rounded
        prefix    = SENSOR_MAP[name]
        values    = item.get("values", {})

        row = {"time_ms": time_ms}
        for axis in "xyz":
            row[f"{prefix}_{axis}"] = values.get(axis)
        rows.append(row)

    if not rows:                                      # nothing useful
        return jsonify(status="ok",
                       rows_written=0,
                       bytes_received=len(raw)), 200

    df = pd.DataFrame(rows)                           # DataFrame build
    df = df.groupby("time_ms", as_index=False).mean() # average dup ms
    df.sort_values("time_ms", inplace=True)

    # 3) normalise time-base (0 ms at first ever sample)
    global _first_time_ms
    if _first_time_ms is None:
        _first_time_ms = int(df["time_ms"].iloc[0])

    df["time_ms"] = df["time_ms"] - _first_time_ms

    # 4) forward-fill from previous row & build list of dicts for CSV
    global _last_values
    out_rows: list[dict] = []

    for _, r in df.iterrows():
        row_dict = {"time_ms": int(r["time_ms"])}
        for col in DESIRED_COLS:
            val = r.get(col)
            if pd.isna(val):
                val = _last_values[col]               # carry previous
            row_dict[col] = val

        # update last-known values
        for col in DESIRED_COLS:
            if row_dict[col] is not None:
                _last_values[col] = row_dict[col]

        out_rows.append(row_dict)

    # 5) write to CSV and return
    _append_rows(out_rows)
    return jsonify(status="ok",
                   rows_written=len(out_rows),
                   bytes_received=len(raw)), 200

# ───────────────────────────────── MAIN ──────────────────────────────────── #
if __name__ == "__main__":
    print(f"Starting IMU logger on http://{HOST}:{PORT}/ → {CSV_PATH}")
    app.run(host=HOST, port=PORT)
