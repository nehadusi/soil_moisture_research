"""Stage 3: masks (flag, box, shadow) + per-sighting image features.

Input: crops (1200x1200 at full 4K res, centred on each detected flag) + crops.csv.
For each crop we look at a disk of radius ROI_R around the flag (where the probe sits),
drop pixels that are flag, node box, or shadow, and compute colour / vegetation /
texture features on what is left. Features are then aggregated per sighting (median
over that sighting's frames), giving one row per sensor placement observation.

Usage: python -I src/features.py <crops_dir_root> <crops.csv> <sighting_labels.csv> <out_dir> [--no-shadow-mask]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

WORK = 0.5          # process crops at half res (600x600)
ROI_R = 350         # px at full res around the flag (about the probe footprint + margin)
FLAG_DILATE = 25    # px (full res) grown around the red flag
BOX_DILATE = 40     # px (full res) grown around the white box (edges, cable, colour fringing)
SHADOW_LOCAL = 0.62  # pixel L / local-median L below this -> hard shadow (flag/box/people)
BROAD_L_RATIO = 0.75   # tree shadow: dark class mean L / bright class mean L below this ...
BROAD_BLUE_SHIFT = 0.12  # ... and dark class at least this much bluer (B/R) than the bright class
EXG_VEG = 0.05      # chromatic excess-green above this = vegetation pixel


def near_point(mask, xy, radius, min_area=0):
    """Keep only connected components of `mask` that come within `radius` px of xy."""
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    keep = np.zeros(mask.shape, np.uint8)
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    near = ((xx - xy[0]) ** 2 + (yy - xy[1]) ** 2) <= radius ** 2
    for i in range(1, n):
        comp = lbl == i
        if st[i, cv2.CC_STAT_AREA] >= min_area and (comp & near).any():
            keep[comp] = 1
    return keep


def box_component(white, xy, radius=170, min_area=300, min_rect=0.6):
    """The node box: the most rectangular white blob near the flag. Bright dry-soil patches
    are white-ish too, but irregular, so they fail the rectangularity test."""
    n, lbl, st, _ = cv2.connectedComponentsWithStats(white.astype(np.uint8))
    yy, xx = np.mgrid[0:white.shape[0], 0:white.shape[1]]
    near = ((xx - xy[0]) ** 2 + (yy - xy[1]) ** 2) <= radius ** 2
    best, best_score = None, 0
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < min_area:
            continue
        comp = (lbl == i)
        if not (comp & near).any():
            continue
        pts = np.column_stack(np.nonzero(comp))[:, ::-1].astype(np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
        rect = a / max(w * h, 1)
        if rect >= min_rect and a * rect > best_score:
            best, best_score = i, a * rect
    keep = np.zeros(white.shape, np.uint8)
    if best is not None:
        keep[lbl == best] = 1
    return keep


def masks(img, flag_xy, frame_L_med, use_shadow=True):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)

    red = (((H <= 12) | (H >= 168)) & (S >= 150) & (V >= 60)).astype(np.uint8)
    red = near_point(red, flag_xy, 90)   # reddish soil specks elsewhere are not the flag
    k = int(FLAG_DILATE * WORK) * 2 + 1
    red = cv2.dilate(red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    white = ((S <= 45) & (V >= 165)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # the lid is clear, so the box reads as a white frame around dark electronics: close it
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    keep = box_component(white, flag_xy)   # the rectangular white blob beside this flag
    # fallback when the box merges with bright dry soil and fails the shape test: its
    # overexposed white core (V~255, near-zero saturation) is still easy to find.
    hot = near_point(((V >= 245) & (S <= 35)).astype(np.uint8), flag_xy, 170, min_area=30)
    keep = keep | cv2.dilate(hot, np.ones((9, 9), np.uint8))
    k = int(BOX_DILATE * WORK) * 2 + 1
    box = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    # shadows: compare each pixel to its neighbourhood (hard cast shadows) and to the
    # whole frame (broad tree shadow). Shadows are lit by blue skylight, so they also
    # shift blue relative to red; wet soil darkens without that shift.
    local = cv2.blur(L, (151, 151))
    bgr = img.astype(np.float32) + 1
    blue_ratio = bgr[..., 0] / bgr[..., 2]
    local_br = cv2.blur(blue_ratio, (151, 151))
    hard = (L < SHADOW_LOCAL * local) & (blue_ratio > local_br * 1.03)
    # broad shadow (trees): split the smoothed brightness into dark/bright with Otsu and
    # call the dark class shadow only if it is much darker AND clearly bluer (skylight).
    Lb = cv2.blur(L, (31, 31)); brb = cv2.blur(blue_ratio, (31, 31))
    t, _ = cv2.threshold(Lb.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = Lb < t
    broad = np.zeros_like(dark)
    if 0.05 < dark.mean() < 0.95:
        if (Lb[dark].mean() / Lb[~dark].mean() < BROAD_L_RATIO) and (brb[dark].mean() - brb[~dark].mean() > BROAD_BLUE_SHIFT):
            broad = dark & (brb > (brb[dark].mean() + brb[~dark].mean()) / 2)
    shadow = (hard | broad).astype(np.uint8)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    yy, xx = np.mgrid[0:h, 0:w]
    roi = ((xx - flag_xy[0]) ** 2 + (yy - flag_xy[1]) ** 2) <= (ROI_R * WORK) ** 2
    valid = roi & (red == 0) & (box == 0)
    if use_shadow:
        valid &= shadow == 0
    return dict(roi=roi, red=red > 0, box=box > 0, shadow=shadow > 0, valid=valid)


def glcm_feats(gray, valid, levels=32):
    q = (gray.astype(np.float32) / 256 * (levels - 1)).astype(np.uint8) + 1   # 1..levels
    q[~valid] = 0                                                            # 0 = masked
    g = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 2], levels=levels + 1, symmetric=True)
    g = g[1:, 1:].astype(np.float64)        # drop pairs touching masked pixels
    g /= g.sum(axis=(0, 1), keepdims=True) + 1e-12
    out = {}
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        out[f"glcm_{prop}"] = float(graycoprops(g, prop).mean())
    return out


def features(img, m, frame_lab_med):
    v = m["valid"]
    px = img[v].astype(np.float32)
    b, g, r = px[:, 0], px[:, 1], px[:, 2]
    s = r + g + b + 1e-6
    rn, gn, bn = r / s, g / s, b / s
    exg = 2 * gn - rn - bn
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[v].astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[v].astype(np.float32)
    veg = exg > EXG_VEG
    f = {
        "valid_frac": v.sum() / m["roi"].sum(),
        "shadow_frac": (m["shadow"] & m["roi"]).sum() / m["roi"].sum(),
        "r_chroma": np.median(rn), "g_chroma": np.median(gn), "b_chroma": np.median(bn),
        "exg": np.median(exg), "exg_std": exg.std(),
        "vari": np.mean((g - r) / (g + r - b + 1e-3).clip(1, None)),
        "gli": np.mean((2 * g - r - b) / (2 * g + r + b + 1e-3)),
        "veg_frac": veg.mean(),
        "L": np.median(lab[:, 0]), "a": np.median(lab[:, 1]), "b": np.median(lab[:, 2]), "L_std": lab[:, 0].std(),
        "hue": np.median(hsv[:, 0]), "sat": np.median(hsv[:, 1]), "val": np.median(hsv[:, 2]),
        # exposure-normalised: brightness/colour relative to the whole frame
        "L_rel": np.median(lab[:, 0]) / frame_lab_med[0],
        "a_rel": np.median(lab[:, 1]) - frame_lab_med[1],
        "b_rel": np.median(lab[:, 2]) - frame_lab_med[2],
    }
    for name, sel in [("soil", ~veg), ("veg", veg)]:
        if sel.sum() > 200:
            f[f"{name}_L_rel"] = np.median(lab[sel, 0]) / frame_lab_med[0]
            f[f"{name}_a"] = np.median(lab[sel, 1])
            f[f"{name}_b"] = np.median(lab[sel, 2])
        else:
            f[f"{name}_L_rel"] = f[f"{name}_a"] = f[f"{name}_b"] = np.nan
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f.update(glcm_feats(gray, v))
    return f


def overlay(img, m):
    o = img.copy()
    o[m["shadow"]] = (0.5 * o[m["shadow"]] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    o[m["red"]] = (255, 0, 255)
    o[m["box"]] = (255, 255, 0)
    cnt = np.zeros(m["roi"].shape, np.uint8); cnt[m["roi"]] = 255
    cs, _ = cv2.findContours(cnt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, cs, -1, (255, 255, 255), 2)
    return np.hstack([img, o])


if __name__ == "__main__":
    root, crops_csv, labels_csv, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
    use_shadow = "--no-shadow-mask" not in sys.argv
    out.mkdir(parents=True, exist_ok=True); (out / "mask_overlays").mkdir(exist_ok=True)
    crops = pd.read_csv(crops_csv)
    rows = []
    for i, c in enumerate(crops.itertuples()):
        img = cv2.imread(str(root / c.crop))
        img = cv2.resize(img, None, fx=WORK, fy=WORK, interpolation=cv2.INTER_AREA)
        fl = (c.flag_x * WORK, c.flag_y * WORK)
        med = (c.frame_L_med, c.frame_a_med, c.frame_b_med)
        m = masks(img, fl, c.frame_L_med, use_shadow)
        if m["valid"].sum() < 2000:
            continue
        f = features(img, m, med)
        f.update(video=c.video, sighting=c.sighting, t=c.t)
        rows.append(f)
        if use_shadow and i % 6 == 0:
            cv2.imwrite(str(out / "mask_overlays" / f"{Path(c.crop).stem}.jpg"), overlay(img, m), [cv2.IMWRITE_JPEG_QUALITY, 80])
    frames = pd.DataFrame(rows)
    feat_cols = [k for k in frames.columns if k not in ("video", "sighting", "t")]
    per_sighting = frames.groupby(["video", "sighting"])[feat_cols].median().reset_index()
    per_sighting["n_frames"] = frames.groupby(["video", "sighting"]).size().values
    lab = pd.read_csv(labels_csv)
    ds = per_sighting.merge(lab[["video", "sighting", "device_name", "status", "vwc_pct", "plateau_id"]],
                            on=["video", "sighting"], how="left")
    ds["day"] = ds["video"].str[:4]
    tag = "" if use_shadow else "_noshadowmask"
    frames.to_csv(out / f"frame_features{tag}.csv", index=False)
    ds.to_csv(out / f"dataset{tag}.csv", index=False)
    print(f"{len(frames)} frames -> {len(ds)} sightings, {ds['vwc_pct'].notna().sum()} labelled")
