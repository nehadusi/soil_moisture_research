"""Stage 6: heatmaps.

1. measured_grid_<day>.png: sensor VWC at every node position for every grid row
   (ground truth, straight from the calibrated probes).
2. predicted_<row>.png: the random forest applied to a grid of cells over one frame of
   each row. Cells are coloured on the same scale as the measured map, and the node in
   frame is marked with its measured value so the two can be compared directly.
   predicted_cells.csv holds every cell's prediction.

Usage: python -I src/heatmaps.py <frames_root> <sightings.csv> <sighting_labels.csv> <dataset.csv> <out_dir>
"""
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, str(Path(__file__).parent))
import features as F
import train as T

# sequential blue ramp, light (dry) -> dark (wet)
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("vwc", BLUES)
NORM = Normalize(vmin=0, vmax=20)
INK, MUTED = "#1f1f1d", "#6b6a66"
CELL = 240          # px at full res
GRID_ROWS = {
    "9.17_23.07": "row 1", "9.17_23.15": "row 2", "9.17_23.23": "row 3", "9.17_23.32": "row 4",
    "9.18_21.46": "row 1", "9.18_21.55": "row 2", "9.18_22.07": "row 3", "9.18_22.17": "row 4",
    "9.18_22.45_WET_ROW_WITH_SHADOWS": "wet row",
}
ORDER = ["alyssa-test-3", "neha-test-2", "neha-1", "neha-test-3", "node-1"]


def ink_for(v):
    return "white" if v >= 10 else INK


def node_tile(crops, root, video, sighting, size=300, view=700, sight=None, froot=None):
    """Close-up of one node from the middle frame of its sighting, `view` px square at 4K.
    Falls back to the sighting's best full frame when the node had no crop (flag near the edge)."""
    g = crops[(crops["video"] == video) & (crops["sighting"] == sighting)].sort_values("t")
    if len(g):
        r = g.iloc[len(g) // 2]
        img, fx, fy = cv2.imread(str(root / r["crop"])), r["flag_x"], r["flag_y"]
    elif sight is not None and froot is not None:
        r = sight[(sight["video"] == video) & (sight["sighting"] == sighting)].iloc[0]
        fp = froot / video / r["frame"]
        if not fp.exists():
            return None
        img, fx, fy = cv2.imread(str(fp)), r["cx"], r["cy"]
    else:
        return None
    h = view // 2
    x0 = int(np.clip(fx - h, 0, img.shape[1] - view)); y0 = int(np.clip(fy - h, 0, img.shape[0] - view))
    tile = cv2.resize(img[y0:y0 + view, x0:x0 + view], (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def measured_grid(lab, out, crops=None, root=None, sight=None, froot=None):
    """Measured VWC for every grid row x node. Each cell shows the drone close-up of that probe,
    washed and framed in the moisture colour, with the reading on it. One PNG per day."""
    lab = lab[lab["status"] == "ok"].copy()
    lab["day"] = lab["video"].str[:4]
    S, BORDER, GAP = 300, 12, 10
    for day, g in lab.groupby("day"):
        vids = [v for v in GRID_ROWS if v.startswith(day) and v in set(g["video"])]
        nr, nc = len(vids), len(ORDER)
        mosaic = np.ones((nr * (S + GAP) - GAP, nc * (S + GAP) - GAP, 3), np.float32)
        vals = np.full((nr, nc), np.nan)
        for i, v in enumerate(vids):
            for j, n in enumerate(ORDER):
                x = g[(g["video"] == v) & (g["device_name"] == n)]
                if not len(x):
                    continue
                val = float(x["vwc_pct"].iloc[0]); vals[i, j] = val
                col = np.array(CMAP(NORM(val))[:3], np.float32)
                tile = node_tile(crops, root, v, int(x["sighting"].iloc[0]), S, sight=sight, froot=froot) if crops is not None else None
                if tile is None:
                    tile = np.ones((S, S, 3), np.float32) * col
                else:
                    tile = tile * 0.62 + col * 0.38                      # wash in the moisture colour
                tile[:BORDER], tile[-BORDER:], tile[:, :BORDER], tile[:, -BORDER:] = col, col, col, col
                y0, x0 = i * (S + GAP), j * (S + GAP)
                mosaic[y0:y0 + S, x0:x0 + S] = tile

        fig_w = 1.55 * nc + 1.9
        fig, ax = plt.subplots(figsize=(fig_w, 1.55 * nr + 1.3))
        ax.imshow(mosaic, extent=(-0.5, nc - 0.5, nr - 0.5, -0.5))
        for i in range(nr):
            for j in range(nc):
                if np.isnan(vals[i, j]):
                    continue
                c = CMAP(NORM(vals[i, j]))
                ax.text(j, i + 0.33, f"{vals[i, j]:.1f}%", ha="center", va="center", fontsize=12, weight="bold",
                        color=ink_for(vals[i, j]), bbox=dict(boxstyle="round,pad=0.25", fc=c, ec="none"))
        ax.set_xticks(range(nc), [n.replace("-test", "\ntest") for n in ORDER], fontsize=9, color=INK)
        ax.set_yticks(range(nr), [GRID_ROWS[v] for v in vids], fontsize=10, color=INK)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("node, in the order the drone passes them", fontsize=9, color=MUTED)
        ax.set_title(f"Measured soil moisture (VWC %), {day.replace('.', '/')}", fontsize=13, color=INK, loc="left",
                     weight="bold", pad=10)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("VWC %  (drier \u2192 wetter)", color=MUTED); cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED, length=0)
        fig.tight_layout()
        fig.savefig(out / f"measured_grid_{day.replace('.', '')}.png", dpi=150, facecolor="white")
        plt.close(fig)


def predict_frame(img_full, flag_xy_full, model, cols):
    img = cv2.resize(img_full, None, fx=F.WORK, fy=F.WORK, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    med = tuple(np.median(lab, axis=0))
    fl = (flag_xy_full[0] * F.WORK, flag_xy_full[1] * F.WORK)
    m = F.masks(img, fl, med[0], use_shadow=True)        # flag/box near the node, shadows everywhere
    c = int(CELL * F.WORK)
    H, W = img.shape[:2]
    rows = []
    for y in range(0, H - c + 1, c):
        for x in range(0, W - c + 1, c):
            roi = np.zeros((H, W), bool); roi[y:y + c, x:x + c] = True
            valid = roi & ~m["red"] & ~m["box"] & ~m["shadow"]   # m["valid"] is limited to the node disk
            cell = dict(roi=roi, valid=valid, shadow=m["shadow"])
            vf = valid.sum() / roi.sum()
            if vf < 0.3:
                rows.append(dict(x=x, y=y, valid_frac=vf, pred=np.nan)); continue
            f = F.features(img, cell, med)
            X = pd.DataFrame([f])[cols]
            rows.append(dict(x=x, y=y, valid_frac=vf, pred=float(model.predict(X)[0])))
    return img, m, pd.DataFrame(rows), c


def two_panel_figure(img, P, c, node_xy, node, truth, title, path):
    """Frame on the left; on the right the frame faded with the predicted-VWC cells on top."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    axes[0].imshow(rgb); axes[0].set_title("frame", fontsize=10, color=MUTED, loc="left")
    axes[1].imshow(rgb, alpha=0.45)
    axes[1].imshow(P, cmap=CMAP, norm=NORM, alpha=0.8, extent=(0, P.shape[1] * c, P.shape[0] * c, 0),
                   interpolation="nearest")
    fx, fy = node_xy
    for ax in axes:
        ax.add_patch(plt.Circle((fx, fy), F.ROI_R * F.WORK, fill=False, ec="white", lw=1.6))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[1].annotate(f"{node}\nsensor {truth:.1f}%", (fx, fy), xytext=(8, -8), textcoords="offset points",
                     color="white", fontsize=9, fontweight="bold", va="top")
    axes[1].set_title(f"predicted VWC per {CELL}px cell (blank = masked)",
                      fontsize=10, color=MUTED, loc="left")
    sm = plt.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01); cb.set_label("VWC %", color=MUTED); cb.outline.set_visible(False)
    fig.suptitle(f"{title}: model map (not validated: held-out error is no better than predicting the mean)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def overlay_figure(img, P, c, valid_px, node_xy, node, truth, node_pred, title, path):
    """The frame with the prediction grid laid over it: each cell tinted by predicted VWC
    (masked pixels such as shadows, the flag and the box are left untinted), thin grid lines,
    the cell value printed in each cell, and the node in frame marked with its sensor value."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    H, W = rgb.shape[:2]
    tint = np.zeros((H, W, 4), np.float32)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if np.isnan(P[i, j]):
                continue
            tint[i * c:(i + 1) * c, j * c:(j + 1) * c] = CMAP(NORM(P[i, j]))
    alpha = np.where(valid_px, 0.45, 0.0)[..., None] * (tint[..., 3:] > 0)
    comp = rgb * (1 - alpha) + tint[..., :3] * alpha

    fig = plt.figure(figsize=(12, 7.6))
    ax = fig.add_axes([0.02, 0.13, 0.96, 0.76])
    ax.imshow(comp)
    for k in range(0, W + 1, c):
        ax.axvline(k - 0.5, color="white", lw=0.6, alpha=0.55)
    for k in range(0, H + 1, c):
        ax.axhline(k - 0.5, color="white", lw=0.6, alpha=0.55)
    stroke = [pe.withStroke(linewidth=2.2, foreground=(0, 0, 0, 0.55))]
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if not np.isnan(P[i, j]):
                ax.text(j * c + c / 2, i * c + c / 2, f"{P[i, j]:.0f}", ha="center", va="center",
                        fontsize=7.5, color="white", path_effects=stroke)
    fx, fy = node_xy
    ax.add_patch(plt.Circle((fx, fy), F.ROI_R * F.WORK, fill=False, ec="white", lw=2))
    ax.annotate(f"{node}\nsensor {truth:.1f}%  ·  model {node_pred:.1f}% (avg of cells in ring)", (fx + F.ROI_R * F.WORK * 0.72, fy - F.ROI_R * F.WORK * 0.72),
                xytext=(14, -14), textcoords="offset points", fontsize=10, color=INK, va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="none", alpha=0.92),
                arrowprops=dict(arrowstyle="-", color="white", lw=1.4))
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5); ax.axis("off")
    fig.text(0.02, 0.955, f"{title}: predicted soil moisture", fontsize=14, color=INK, weight="bold")
    fig.text(0.02, 0.918, f"Random forest on {CELL} px cells (numbers = predicted VWC %). Untinted pixels are masked: "
             "shadows, flag, box. Not validated: held-out error matches predicting the average.",
             fontsize=9.5, color=MUTED)
    cax = fig.add_axes([0.30, 0.065, 0.40, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8.5, colors=MUTED, length=0)
    cb.set_label("VWC %", fontsize=9, color=MUTED)
    fig.text(0.295, 0.072, "drier", ha="right", fontsize=9, color=MUTED)
    fig.text(0.705, 0.072, "wetter", ha="left", fontsize=9, color=MUTED)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    froot, sight, labp, dsp, out = map(Path, sys.argv[1:6])
    crops_csv = Path(sys.argv[6]) if len(sys.argv) > 6 else None        # optional: node close-ups
    crops_root = Path(sys.argv[7]) if len(sys.argv) > 7 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(labp); s = pd.read_csv(sight)
    measured_grid(lab, out, pd.read_csv(crops_csv) if crops_csv else None, crops_root, s, froot)

    ds = pd.read_csv(dsp); ds = ds[(ds["status"] == "ok") & ds["vwc_pct"].notna()]
    cols = [c for c in T.FEATURE_SETS["all"] if c in ds.columns]
    model = T.models()["random_forest"].fit(ds[cols], ds["vwc_pct"])

    all_cells = []
    for v, row_name in GRID_ROWS.items():
        r = s[(s["video"] == v) & (s["sighting"] == 2)].iloc[0]
        fp = froot / v / r["frame"]
        if not fp.exists():
            print("missing", fp); continue
        img, m, cells, c = predict_frame(cv2.imread(str(fp)), (r["cx"], r["cy"]), model, cols)
        cells["video"] = v; cells["grid_row"] = row_name; all_cells.append(cells)
        truth = lab[(lab["video"] == v) & (lab["sighting"] == 2)]["vwc_pct"].iloc[0]
        node = lab[(lab["video"] == v) & (lab["sighting"] == 2)]["device_name"].iloc[0]

        P = np.full((img.shape[0] // c, img.shape[1] // c), np.nan)
        for q in cells.itertuples():
            P[q.y // c, q.x // c] = q.pred
        valid_px = ~m["red"] & ~m["box"] & ~m["shadow"]
        # model value at the node: average of the unmasked cells that overlap the probe disk
        fxw, fyw, rad = r["cx"] * F.WORK, r["cy"] * F.WORK, F.ROI_R * F.WORK
        near = [P[i, j] for i in range(P.shape[0]) for j in range(P.shape[1])
                if not np.isnan(P[i, j]) and np.hypot(np.clip(fxw, j * c, (j + 1) * c) - fxw,
                                                      np.clip(fyw, i * c, (i + 1) * c) - fyw) < rad]
        node_cell = float(np.mean(near)) if near else np.nan
        two_panel_figure(img, P, c, (r["cx"] * F.WORK, r["cy"] * F.WORK), node, truth,
                         f"{v[:4].replace('.', '/')} {row_name}",
                         out / f"predicted_{v[:4].replace('.', '')}_{row_name.replace(' ', '')}.png")
        print(v, "cells", cells["pred"].notna().sum(), "pred range", np.nanmin(P).round(1), np.nanmax(P).round(1), "truth", truth)
    pd.concat(all_cells).to_csv(out / "predicted_cells.csv", index=False)
