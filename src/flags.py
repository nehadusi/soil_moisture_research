"""Stage 2a: find the red/orange marker flags (and the white node box next to each).
Writes detections.csv (one row per flag blob per frame) and mask overlays."""
import cv2, numpy as np, glob, os, csv, sys
SCALE = 0.25                     # work at 960x540
os.makedirs('outputs/masks', exist_ok=True)
rows = []
for d in sorted(glob.glob('frames/*/')):
    vid = os.path.basename(d.rstrip('/'))
    for fp in sorted(glob.glob(d + '*_t*.jpg')):
        t = int(fp.rsplit('_t', 1)[1][:6]) / 1000
        im = cv2.imread(fp); sm = cv2.resize(im, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        red = (((h <= 12) | (h >= 168)) & (s >= 110) & (v >= 55)).astype(np.uint8)
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        white = ((s <= 50) & (v >= 150)).astype(np.uint8)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(red)
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA]
            if a < 6: continue
            cx, cy = cen[i]
            # white box within ~40 px (160 px at 4K) of the flag
            y0, y1 = int(max(0, cy - 40)), int(min(sm.shape[0], cy + 40)); x0, x1 = int(max(0, cx - 40)), int(min(sm.shape[1], cx + 40))
            wb = int(white[y0:y1, x0:x1].sum())
            rows.append(dict(video=vid, t=t, frame=os.path.basename(fp), cx=round(cx / SCALE), cy=round(cy / SCALE), area=int(a / SCALE**2), white_px=int(wb / SCALE**2)))
        if int(t) % 10 == 0 and t == int(t):
            ov = sm.copy(); ov[red > 0] = (255, 0, 255); ov[white > 0] = (255, 255, 0)
            cv2.imwrite(f'outputs/masks/{vid}_t{int(t):02d}.jpg', np.hstack([sm, ov]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(vid, sum(r['video'] == vid for r in rows), flush=True)
with open('outputs/detections.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
