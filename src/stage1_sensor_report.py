"""Stage 1: summarise the sensor log without assuming anything about it.

Writes:
  outputs/metrics/stage1_sensor_nodes.csv     one row per node (dev_eui)
  outputs/metrics/stage1_sensor_sessions.csv  one row per field session x node
  outputs/figures/stage1/sensor_sessions.png  raw ADC vs time, one panel per session

Usage: python src/stage1_sensor_report.py [--config config/sensors.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sensor_io import assign_sessions, flag_invalid, load_config, load_sensor_csv

# Categorical slots in fixed order (validated for CVD separation).
NODE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MIN_ROWS_TO_PLOT = 100


def node_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-node counts, time span, invalid breakdown and valid-ADC percentiles."""
    valid = df[df["valid"]]
    base = df.groupby("dev_eui").agg(
        device_name=("device_name", lambda s: "/".join(sorted(s.unique()))),
        n_rows=("received_at_utc", "size"),
        first_utc=("received_at_utc", "min"),
        last_utc=("received_at_utc", "max"),
        n_sessions=("session_id", "nunique"),
        frame_counter_resets=("frame_counter", lambda s: int((s.diff() < 0).sum())),
    )
    reasons = df[~df["valid"]].groupby(["dev_eui", "invalid_reason"]).size().unstack(fill_value=0)
    pct = valid.groupby("dev_eui")["adc_raw"].quantile([0.01, 0.5, 0.99]).unstack()
    pct.columns = ["adc_p01", "adc_p50", "adc_p99"]
    out = base.join(reasons.add_prefix("invalid_")).join(pct).fillna(0)
    out["n_valid"] = valid.groupby("dev_eui").size()
    return out.reset_index()


def session_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per session x node: time span, valid count and ADC spread (watering shows up as spread)."""
    valid = df[df["valid"]]
    g = valid.groupby(["session_id", "dev_eui"])["adc_raw"]
    out = g.agg(n_valid="size", adc_min="min", adc_median="median", adc_max="max")
    out["adc_iqr"] = g.quantile(0.75) - g.quantile(0.25)
    span = df.groupby("session_id")["received_at_utc"].agg(session_start_utc="min", session_end_utc="max")
    return out.reset_index().merge(span.reset_index(), on="session_id")


def plot_sessions(df: pd.DataFrame, path: Path) -> None:
    """Small multiples: one panel per session, valid raw ADC vs UTC time, one color per node."""
    sessions = [s for s, g in df.groupby("session_id") if len(g) >= MIN_ROWS_TO_PLOT]
    nodes = sorted(df["dev_eui"].unique())
    labels = df.groupby("dev_eui")["device_name"].first()
    colors = dict(zip(nodes, NODE_COLORS))
    fig, axes = plt.subplots(len(sessions), 1, figsize=(12, 2.0 * len(sessions)), sharey=True)
    for ax, sid in zip(axes, sessions):
        g = df[(df["session_id"] == sid) & df["valid"]]
        for eui in nodes:
            gg = g[g["dev_eui"] == eui]
            ax.plot(gg["received_at_utc"], gg["adc_raw"], "o", ms=2.5, color=colors[eui],
                    label=f"{labels[eui]} ({eui[-4:]})")
        start = df.loc[df["session_id"] == sid, "received_at_utc"].min()
        ax.set_title(f"session {sid}: {start:%Y-%m-%d %H:%M} UTC", fontsize=9, loc="left", color="#444")
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.tick_params(labelsize=7, colors="#666")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=7, ncol=len(nodes), markerscale=2, frameon=False, loc="lower left",
                   bbox_to_anchor=(0, 1.15))
    fig.supylabel("raw ADC counts (valid readings only)", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    """Load, flag, sessionise, summarise and plot the sensor log."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/sensors.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    df = assign_sessions(flag_invalid(load_sensor_csv(cfg["sensor_csv"]), cfg), cfg["session_gap_minutes"])

    metrics = Path("outputs/metrics")
    metrics.mkdir(parents=True, exist_ok=True)
    nodes = node_summary(df)
    nodes.to_csv(metrics / "stage1_sensor_nodes.csv", index=False)
    session_summary(df).to_csv(metrics / "stage1_sensor_sessions.csv", index=False)
    plot_sessions(df, Path("outputs/figures/stage1/sensor_sessions.png"))

    print(f"rows: {len(df)}  layouts: {df['layout'].value_counts().to_dict()}")
    print(f"time range (UTC): {df['received_at_utc'].min()} -> {df['received_at_utc'].max()}")
    print(f"invalid: {df.loc[~df['valid'], 'invalid_reason'].value_counts().to_dict()}")
    print(f"sessions: {df['session_id'].nunique()}")
    print(nodes.to_string(index=False))


if __name__ == "__main__":
    main()
