"""Load and clean the LoRaWAN soil-moisture sensor log.

The raw CSV has two quirks that a plain ``pd.read_csv`` silently gets wrong:

1. Runs of NUL bytes where the logger was interrupted mid-write.
2. Two row layouts under a single header. The first rows (layout A) match the
   header. From 2026-07-16 23:50 UTC on (layout B) an always-empty field sits
   after ``probe_voltage`` and ``mqtt_topic`` is gone, so ``rssi``, ``snr``,
   ``frame_counter`` and ``application_port`` are each shifted one column right.

This module never modifies the raw file; it only reads it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

N_FIELDS = 10

COLUMNS = [
    "received_at_utc",
    "device_name",
    "dev_eui",
    "adc_raw",
    "probe_voltage",
    "rssi",
    "snr",
    "frame_counter",
    "application_port",
    "layout",
    "line_no",
]


def load_config(path: str | Path = "config/sensors.yaml") -> dict:
    """Read the sensor config YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def read_lines(path: str | Path) -> list[str]:
    """Return the CSV's lines with NUL bytes stripped, header included."""
    raw = Path(path).read_bytes().replace(b"\x00", b"")
    return raw.decode("utf-8").splitlines()


def parse_row(fields: list[str], line_no: int) -> dict:
    """Map one split CSV line to canonical fields, detecting its layout.

    Raises ValueError for a line matching neither known layout, so a new
    logger format fails loudly instead of being mis-parsed.
    """
    if len(fields) != N_FIELDS:
        raise ValueError(f"line {line_no}: expected {N_FIELDS} fields, got {len(fields)}")
    head = fields[:5]
    if fields[9].startswith("application/"):
        layout, tail = "A", fields[5:9]
    elif fields[5] == "":
        layout, tail = "B", fields[6:10]
    else:
        raise ValueError(f"line {line_no}: unrecognised row layout: {','.join(fields)}")
    return dict(zip(COLUMNS, head + tail + [layout, line_no]))


def load_sensor_csv(path: str | Path) -> pd.DataFrame:
    """Parse the sensor CSV into a typed DataFrame sorted by time.

    Timestamps are tz-aware UTC. Empty payload fields (e.g. port-0 uplinks)
    become NaN; nothing is dropped here, use ``flag_invalid`` for that.
    """
    lines = read_lines(path)
    rows = [parse_row(line.split(","), i) for i, line in enumerate(lines[1:], start=2) if line]
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["received_at_utc"] = pd.to_datetime(df["received_at_utc"], format="ISO8601", utc=True)
    for col in ["adc_raw", "probe_voltage", "snr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["rssi", "frame_counter", "application_port"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df.sort_values("received_at_utc", kind="stable").reset_index(drop=True)


def flag_invalid(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add ``valid`` (bool) and ``invalid_reason`` columns based on ``cfg['adc']``."""
    adc_cfg = cfg["adc"]
    out = df.copy()
    reason = pd.Series("", index=out.index, dtype=object)
    reason[out["adc_raw"].isna()] = "no_payload"
    reason[(reason == "") & out["adc_raw"].isin(adc_cfg["error_values"])] = "adc_error"
    reason[(reason == "") & (out["adc_raw"] < adc_cfg["min_valid"])] = "probe_off"
    expected_v = out["adc_raw"] * adc_cfg["volts_per_count"]
    mismatch = ~np.isclose(out["probe_voltage"], expected_v, atol=1e-5)
    reason[(reason == "") & mismatch] = "voltage_mismatch"
    out["invalid_reason"] = reason
    out["valid"] = reason == ""
    return out


def assign_sessions(df: pd.DataFrame, gap_minutes: float) -> pd.DataFrame:
    """Add ``session_id``: a new session starts after a gap > ``gap_minutes`` across all nodes."""
    out = df.sort_values("received_at_utc", kind="stable").copy()
    new = out["received_at_utc"].diff() > pd.Timedelta(minutes=gap_minutes)
    out["session_id"] = new.cumsum().astype(int)
    return out
