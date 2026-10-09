"""Stage 1: clean the ChirpStack sensor export and build per-video label candidates.

Usage: python -I src/sensors.py data/raw/sensors/soil_moisture_1.csv data/processed

What it fixes / produces:
- From 2026-07-16 23:50 UTC on, the exporter inserted an empty field after
  probe_voltage and dropped mqtt_topic, shifting rssi/snr/frame_counter/port
  one column right. Those rows are realigned.
- Plateaus: each probe placement shows up as a flat run of readings. A new
  plateau starts when the voltage jumps by more than JUMP_V between readings.
- For each video, every node's plateau that covers the whole video window
  becomes a label candidate; nodes that changed mid-video are flagged.
"""
import sys
from pathlib import Path

import pandas as pd

JUMP_V = 0.06          # volts; step between consecutive readings that marks a move
MIN_PLATEAU_N = 3      # readings needed to trust a plateau
AIR_VWC = 1.5          # calibrated VWC below this = probe out of soil (air reads about -3 to 1%)

# Per-node linear calibration, VWC % = a * ADC + b (from the team's calibration sheet).
# The sheet's first row had no visible label; neha-test-3 is the only node not listed,
# so it is assigned by elimination. Confirm with the team.
CALIB = {
    "neha-test-3":   (-0.00321390, 37.96116),
    "node-1":        (-0.00260556, 35.19296),
    "neha-1":        (-0.00263474, 35.25458),
    "neha-test-2":   (-0.00225662, 28.34305),
    "alyssa-test-3": (-0.00199915, 28.19189),
}

# The drone/camera clock runs behind the gateway clock. Sweeping the offset, every video
# lands inside a stable in-soil plateau, one placement per video, only for offsets of
# about +405 to +480 s (44 of 45 node-videos vs 32 at zero offset). Mid-range is used.
CAMERA_CLOCK_OFFSET_S = 445

# The drone flies the row from the alyssa-test-3 end to node-1 (node-1 is the small
# square box and is always the last sighting).
FLIGHT_ORDER = ["alyssa-test-3", "neha-test-2", "neha-1", "neha-test-3", "node-1"]

# Order the probes were moved in, read off the plateau start times (9/18 every row,
# 9/17 mostly). Taken to be their physical order along each row.
ROW_ORDER = ["node-1", "neha-test-3", "neha-1", "neha-test-2", "alyssa-test-3"]


def vwc(device, adc):
    a, b = CALIB[device]
    return a * adc + b

# Video start times (UTC) from the MP4 creation_time minutes/seconds; the
# camera stamps a 12-hour clock, the filenames give the hour. Durations from ffprobe.
VIDEOS = [
    ("9.17_23.07", "2026-09-17 23:06:58", 49.0),
    ("9.17_23.15", "2026-09-17 23:14:32", 53.5),
    ("9.17_23.23", "2026-09-17 23:22:54", 49.2),
    ("9.17_23.32", "2026-09-17 23:31:29", 47.8),
    ("9.18_21.46", "2026-09-18 21:46:05", 46.6),
    ("9.18_21.55", "2026-09-18 21:54:58", 44.8),
    ("9.18_22.07", "2026-09-18 22:06:43", 44.0),
    ("9.18_22.17", "2026-09-18 22:17:06", 52.5),
    ("9.18_22.45_WET_ROW_WITH_SHADOWS", "2026-09-18 22:44:41", 46.6),
]


def load(path):
    df = pd.read_csv(path, dtype=str)
    shifted = df["rssi"].isna() & df["mqtt_topic"].str.len().lt(5)
    cols = ["rssi", "snr", "frame_counter", "application_port"]
    df.loc[shifted, cols] = df.loc[shifted, ["snr", "frame_counter", "application_port", "mqtt_topic"]].values
    df.loc[shifted, "mqtt_topic"] = None
    df["t"] = pd.to_datetime(df["received_at_utc"], format="ISO8601", utc=True)
    for c in ["adc_raw", "probe_voltage", "rssi", "snr"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["realigned"] = shifted
    return df.sort_values("t").reset_index(drop=True)


def plateaus(df):
    out = []
    for dev, g in df.groupby("device_name"):
        g = g.reset_index(drop=True)
        seg = (g["probe_voltage"].diff().abs() > JUMP_V).cumsum()
        s = g.groupby(seg).agg(start=("t", "min"), end=("t", "max"), n=("t", "size"),
                               v_median=("probe_voltage", "median"), v_std=("probe_voltage", "std"),
                               adc_median=("adc_raw", "median"))
        s["vwc_pct"] = vwc(dev, s["adc_median"])
        s["device_name"] = dev
        out.append(s[s["n"] >= MIN_PLATEAU_N])
    p = pd.concat(out, ignore_index=True)
    p["looks_like_air"] = p["vwc_pct"] < AIR_VWC
    return p


def label_candidates(p):
    rows = []
    for vid, t0, dur in VIDEOS:
        t0 = pd.Timestamp(t0, tz="UTC") + pd.Timedelta(seconds=CAMERA_CLOCK_OFFSET_S)
        t1 = t0 + pd.Timedelta(seconds=dur)
        for dev, g in p.groupby("device_name"):
            cover = g[(g["start"] <= t0) & (g["end"] >= t1)]
            overlap = g[(g["end"] >= t0) & (g["start"] <= t1)]
            if len(cover):
                r = cover.iloc[0]
                status = "air?" if r["looks_like_air"] else "ok"
            elif len(overlap):
                r = overlap.iloc[-1]
                status = "changed_mid_video"
            else:
                nxt = g[g["start"] > t1].head(1)
                if len(nxt) and (nxt.iloc[0]["start"] - t1) < pd.Timedelta("8min"):
                    r = nxt.iloc[0]
                    status = "after_gap"   # gateway gap; next plateau after the video
                else:
                    rows.append(dict(video=vid, device_name=dev, status="no_data"))
                    continue
            rows.append(dict(video=vid, device_name=dev, status=status, voltage=round(r["v_median"], 3), vwc_pct=round(r["vwc_pct"], 2),
                             plateau_start=r["start"], plateau_end=r["end"], n=r["n"]))
    return pd.DataFrame(rows)


def sighting_labels(sightings, p):
    """Attach a device and calibrated VWC to each drone sighting (k-th box seen = FLIGHT_ORDER[k]).
    Uses the plateau covering the sighting's own time span, not the whole video."""
    starts = {v: pd.Timestamp(t, tz="UTC") + pd.Timedelta(seconds=CAMERA_CLOCK_OFFSET_S) for v, t, _ in VIDEOS}
    rows = []
    for r in sightings.itertuples():
        n_seen = (sightings["video"] == r.video).sum()
        dev = FLIGHT_ORDER[r.sighting] if n_seen == len(FLIGHT_ORDER) else None
        rec = dict(video=r.video, sighting=r.sighting, t_start=r.t_start, t_end=r.t_end, frame=r.frame,
                   cx=r.cx, cy=r.cy, device_name=dev, status="unassigned_incomplete_flight", vwc_pct=None)
        if dev:
            a = starts[r.video] + pd.Timedelta(seconds=r.t_start)
            b = starts[r.video] + pd.Timedelta(seconds=r.t_end)
            g = p[(p["device_name"] == dev) & (p["start"] <= a + pd.Timedelta("60s")) & (p["end"] >= b - pd.Timedelta("60s"))]
            if len(g) == 1 and not g.iloc[0]["looks_like_air"]:
                rec.update(status="ok", vwc_pct=round(g.iloc[0]["vwc_pct"], 2), voltage=round(g.iloc[0]["v_median"], 3),
                           plateau_id=int(g.index[0]))
            else:
                rec["status"] = "air" if len(g) == 1 else ("ambiguous" if len(g) > 1 else "no_plateau")
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    src, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    df = load(src)
    p = plateaus(df)
    lab = label_candidates(p)
    df.drop(columns=["t"]).to_csv(outdir / "sensor_clean.csv", index=False)
    p.to_csv(outdir / "plateaus.csv", index=False)
    lab.to_csv(outdir / "label_candidates.csv", index=False)
    sp = outdir / "sightings.csv"
    if sp.exists():
        sl = sighting_labels(pd.read_csv(sp), p)
        sl.to_csv(outdir / "sighting_labels.csv", index=False)
        print(sl[["video", "sighting", "device_name", "status", "vwc_pct"]].to_string(index=False))
        print("usable sightings:", (sl["status"] == "ok").sum())
    print(lab.pivot(index="video", columns="device_name", values="vwc_pct")[FLIGHT_ORDER].to_string())
    print()
    print(lab.pivot(index="video", columns="device_name", values="status")[FLIGHT_ORDER].to_string())
