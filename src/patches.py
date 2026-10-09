"""Stage 5: export a patch-image dataset, split by grid row (not by patch).

For every frame where a node is detected, saves a square patch centred on the node's flag
plus its valid-pixel mask (flag, box and shadow removed), named with the node and its
calibrated VWC. Every patch from one grid row lands in the same split, so no row ever
appears in both train and test.

Needs only OpenCV + pandas (runs next to the frames).
Usage: python src/patches.py <crops.csv> <crops root> <sighting_labels.csv> <out_dir>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PATCH = 700  # px at full 4K resolution (about the area the probe sits in)

# grid row ids (day + row number in the order flown) and the split each row goes to
GRID = {
    "9.17_23.07": ("0917_row1", "train"),
    "9.17_23.15": ("0917_row2", "val"),
    "9.17_23.23": ("0917_row3", "test"),
    "9.17_23.32": ("0917_row4", "train"),
    "9.18_21.46": ("0918_row1", "train"),
    "9.18_21.55": ("0918_row2", "val"),
    "9.18_22.07": ("0918_row3", "test"),
    "9.18_22.17": ("0918_row4", "train"),
    "9.18_22.45_WET_ROW_WITH_SHADOWS": ("0918_wetrow", "train"),
}

# alyssa-test-3 drops packets; Alyssa logged its dry-run readings by hand (VWC %, ADC).
# Kept as a cross-check column, not used to overwrite the logged plateau values.
ALYSSA_MANUAL = {"0918_row1": (3.8, 10630), "0918_row2": (9.4, 9408),
                 "0918_row3": (5.5, 11358), "0918_row4": (11.6, 8284)}


def valid_mask(img, fx, fy):
    """Same logic as features.masks, simplified so it runs without scikit-image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    red = (((H <= 12) | (H >= 168)) & (S >= 150) & (V >= 60)).astype(np.uint8)
    red = cv2.dilate(red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))
    white = ((S <= 45) & (V >= 165)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # drop grass glints
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    n, lbl, st, cen = cv2.connectedComponentsWithStats(white)
    box = np.zeros_like(white)
    for i in range(1, n):
        if 1200 < st[i, cv2.CC_STAT_AREA] < 40000 and np.hypot(cen[i][0] - fx, cen[i][1] - fy) < 340:
            pts = np.column_stack(np.nonzero(lbl == i))[:, ::-1].astype(np.float32)
            (_, _), (w, h), _ = cv2.minAreaRect(pts)
            if st[i, cv2.CC_STAT_AREA] / max(w * h, 1) > 0.6:
                box[lbl == i] = 1
    hot = ((V >= 245) & (S <= 35)).astype(np.uint8)          # overexposed box core
    n, lbl, st, cen = cv2.connectedComponentsWithStats(hot)
    for i in range(1, n):                                     # ignore specular glints on grass
        if st[i, cv2.CC_STAT_AREA] >= 120 and np.hypot(cen[i][0] - fx, cen[i][1] - fy) < 340:
            box[lbl == i] = 1
    box = cv2.dilate(box, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    f = img.astype(np.float32) + 1
    br = f[..., 0] / f[..., 2]
    hard = (L < 0.62 * cv2.blur(L, (301, 301))) & (br > cv2.blur(br, (301, 301)) * 1.03)
    Lb, brb = cv2.blur(L, (61, 61)), cv2.blur(br, (61, 61))
    t, _ = cv2.threshold(Lb.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = Lb < t
    broad = np.zeros_like(dark)
    if 0.05 < dark.mean() < 0.95 and Lb[dark].mean() / Lb[~dark].mean() < 0.75 \
            and brb[dark].mean() - brb[~dark].mean() > 0.12:
        broad = dark & (brb > (brb[dark].mean() + brb[~dark].mean()) / 2)
    shadow = hard | broad
    return ((red == 0) & (box == 0) & ~shadow).astype(np.uint8) * 255


if __name__ == "__main__":
    crops = pd.read_csv(sys.argv[1]); root = Path(sys.argv[2])
    lab = pd.read_csv(sys.argv[3]); out = Path(sys.argv[4])
    lab = lab[lab["status"] == "ok"]
    m = crops.merge(lab[["video", "sighting", "device_name", "vwc_pct", "plateau_id"]], on=["video", "sighting"])
    rows = []
    for r in m.itertuples():
        grid_id, split = GRID[r.video]
        img = cv2.imread(str(root / r.crop))
        h = PATCH // 2
        x0 = int(np.clip(r.flag_x - h, 0, img.shape[1] - PATCH)); y0 = int(np.clip(r.flag_y - h, 0, img.shape[0] - PATCH))
        p = img[y0:y0 + PATCH, x0:x0 + PATCH]
        mask = valid_mask(p, r.flag_x - x0, r.flag_y - y0)
        d = out / split / grid_id; d.mkdir(parents=True, exist_ok=True)
        stem = f"{grid_id}_{r.device_name}_vwc{r.vwc_pct:05.2f}_t{r.t:04.1f}s"
        cv2.imwrite(str(d / f"{stem}.jpg"), p, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(d / f"{stem}_mask.png"), mask)
        manual = ALYSSA_MANUAL.get(grid_id) if r.device_name == "alyssa-test-3" else None
        rows.append(dict(patch=f"{split}/{grid_id}/{stem}.jpg", mask=f"{split}/{grid_id}/{stem}_mask.png",
                         split=split, grid_id=grid_id, video=r.video, node=r.device_name, t_s=r.t,
                         vwc_pct=round(r.vwc_pct, 2), placement_id=int(r.plateau_id),
                         valid_frac=round((mask > 0).mean(), 3),
                         alyssa_manual_vwc=manual[0] if manual else None))
    man = pd.DataFrame(rows).sort_values(["split", "grid_id", "node", "t_s"])
    man.to_csv(out / "manifest.csv", index=False)
    print(man.groupby(["split", "grid_id"]).agg(patches=("patch", "size"), nodes=("node", "nunique"),
                                                vwc_mean=("vwc_pct", "mean")).round(2).to_string())
