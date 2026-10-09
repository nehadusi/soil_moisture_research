"""Stage 2b: group flag detections into node sightings. The drone hovers over the
first node, then flies along the row, so each node sweeps top->bottom through the frame.
A new sighting starts when the flag jumps back up the frame or after a >3 s gap."""
import pandas as pd, cv2, numpy as np
d = pd.read_csv('outputs/detections.csv'); d = d[d.white_px > 1200].sort_values(['video', 't', 'cy'])
# keep one detection per frame: the one with most box pixels nearby
d = d.loc[d.groupby(['video', 't']).white_px.idxmax()].sort_values(['video', 't'])
out = []
for v, g in d.groupby('video'):
    k = 0; prev = None
    for r in g.itertuples():
        if prev is not None and (r.t - prev.t > 8 or r.cy < prev.cy - 600): k += 1
        out.append(dict(video=v, sighting=k, t=r.t, frame=r.frame, cx=r.cx, cy=r.cy)); prev = r
s = pd.DataFrame(out)
s['dist_c'] = ((s.cx - 1920)**2 + (s.cy - 1080)**2)**0.5
best = s.loc[s.groupby(['video', 'sighting']).dist_c.idxmin()]
summ = s.groupby(['video', 'sighting']).agg(t_start=('t', 'min'), t_end=('t', 'max'), n=('t', 'size')).reset_index()
summ = summ.merge(best[['video', 'sighting', 'frame', 'cx', 'cy']], on=['video', 'sighting'])
summ.to_csv('outputs/sightings.csv', index=False)
print(summ.to_string(index=False))
# contact sheet: one 720x405 tile per sighting, cropped around the flag
tiles = []
for v, g in summ.groupby('video'):
    row = []
    for r in g.itertuples():
        im = cv2.imread(f'frames/{v}/{r.frame}')
        x0 = int(np.clip(r.cx - 960, 0, 3840 - 1920)); y0 = int(np.clip(r.cy - 540, 0, 2160 - 1080))
        c = cv2.resize(im[y0:y0 + 1080, x0:x0 + 1920], (480, 270))
        cv2.putText(c, f'{v[:10]} #{r.sighting} t={r.t_start:.0f}-{r.t_end:.0f}s', (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        row.append(c)
    while len(row) < 6: row.append(np.zeros((270, 480, 3), np.uint8))
    tiles.append(np.hstack(row[:6]))
cv2.imwrite('outputs/sightings_sheet.jpg', np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 82])
