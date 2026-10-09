"""Tests for the sensor-log parser: both row layouts, NUL bytes, and invalid flags."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_io import assign_sessions, flag_invalid, load_sensor_csv  # noqa: E402

HEADER = ("received_at_utc,device_name,dev_eui,adc_raw,probe_voltage,rssi,snr,"
          "frame_counter,application_port,mqtt_topic\r\n")
ROW_A = ("2026-07-16T21:44:55.356323+00:00,n3,a7b2,11696,2.193000,-73,14.0,57,2,"
         "application/x/device/a7b2/event/up\r\n")
ROW_B = "2026-07-16T23:50:51.488898+00:00,n3,a7b2,11696,2.193000,,-72,9.0,0,2\r\n"
ROW_OFF = "2026-07-17T00:10:00+00:00,n1,9fb2,7,0.001313,,-81,13.2,7,2\r\n"
ROW_ERR = "2026-07-17T00:10:05+00:00,n2,88cf,-32768,-6.144000,,-90,4.2,3,2\r\n"
ROW_EMPTY = "2026-07-17T00:10:10+00:00,n1,9fb2,,,,-93,13.5,0,0\r\n"
ROW_LATE = "2026-07-17T03:00:00+00:00,n3,a7b2,9000,1.687500,,-70,10.0,1,2\r\n"

CFG = {"adc": {"volts_per_count": 0.0001875, "min_valid": 200, "error_values": [-32768]}}


def write(tmp_path, *rows, nul=b""):
    """Write a CSV made of HEADER + rows, optionally with a NUL run before the last row."""
    p = tmp_path / "s.csv"
    body = (HEADER + "".join(rows[:-1])).encode() + nul + rows[-1].encode()
    p.write_bytes(body)
    return p


def test_both_layouts_map_to_same_fields(tmp_path):
    df = load_sensor_csv(write(tmp_path, ROW_A, ROW_B))
    assert list(df["layout"]) == ["A", "B"]
    assert list(df["rssi"]) == [-73, -72]
    assert list(df["snr"]) == [14.0, 9.0]
    assert list(df["frame_counter"]) == [57, 0]
    assert list(df["application_port"]) == [2, 2]
    assert str(df["received_at_utc"].dt.tz) == "UTC"


def test_nul_bytes_are_stripped(tmp_path):
    df = load_sensor_csv(write(tmp_path, ROW_A, ROW_B, nul=b"\x00" * 87))
    assert len(df) == 2
    assert df["dev_eui"].iloc[1] == "a7b2"


def test_unknown_layout_raises(tmp_path):
    bad = "2026-07-17T00:00:00+00:00,n,e,1,2,3,4,5,6,7\r\n"
    with pytest.raises(ValueError, match="unrecognised row layout"):
        load_sensor_csv(write(tmp_path, ROW_A, bad))


def test_flag_invalid_reasons(tmp_path):
    df = flag_invalid(load_sensor_csv(write(tmp_path, ROW_B, ROW_OFF, ROW_ERR, ROW_EMPTY)), CFG)
    assert list(df["invalid_reason"]) == ["", "probe_off", "adc_error", "no_payload"]
    assert list(df["valid"]) == [True, False, False, False]


def test_sessions_split_on_gap(tmp_path):
    df = assign_sessions(load_sensor_csv(write(tmp_path, ROW_B, ROW_OFF, ROW_LATE)), gap_minutes=120)
    assert list(df["session_id"]) == [0, 0, 1]
