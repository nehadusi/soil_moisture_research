"""Stage 3 prep: for every sighting, cut a 1200x1200 full-res crop centred on the flag from
each frame where the flag is detected, and record whole-frame colour stats (for exposure
normalisation). Output: crops/<video>/<frame>_crop.jpg, outputs/crops.csv"""
import pandas as pd, cv2, numpy as np, os
H = 600
d = pd.read_csv('outputs/detections.csv'); d = d[d.white_px > 1200]
d = d.loc[d.groupby(['video', 't']).white_px.idxmax()]
s = pd.read_csv('outputs/sightings.csv')
import sys
only = sys.argv[1] if len(sys.argv) > 1 else None
rows = []
for sg in s.itertuples():
    if only and sg.video != only: continue
    g = d[(d.video == sg.video) & (d.t >= sg.t_start) & (d.t <= sg.t_end)]
    for r in g.itertuples():
        # keep flags at least 300 px from the frame edge so the ring is mostly in view
        if not (300 <= r.cx <= 3840 - 300 and 300 <= r.cy <= 2160 - 300): continue
        im = cv2.imread(f'frames/{r.video}/{r.frame}')
        x0, y0 = int(np.clip(r.cx - H, 0, 3840 - 2 * H)), int(np.clip(r.cy - H, 0, 2160 - 2 * H))
        c = im[y0:y0 + 2 * H, x0:x0 + 2 * H]
        os.makedirs(f'crops/{r.video}', exist_ok=True)
        fn = f'crops/{r.video}/{r.frame[:-4]}_crop.jpg'; cv2.imwrite(fn, c, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lab = cv2.cvtColor(cv2.resize(im, (960, 540)), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        rows.append(dict(video=r.video, sighting=sg.sighting, t=r.t, crop=fn, flag_x=r.cx - x0, flag_y=r.cy - y0,
                         frame_L_med=np.median(lab[:, 0]), frame_a_med=np.median(lab[:, 1]), frame_b_med=np.median(lab[:, 2])))
os.makedirs('outputs/crops_csv', exist_ok=True)
pd.DataFrame(rows).to_csv(f'outputs/crops_csv/{only or "all"}.csv', index=False)
print(only, len(rows))
