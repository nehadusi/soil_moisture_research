"""Fixed train / val / test evaluation, split by grid row (same split as the patch folder).
Pick the model on val, report it once on test. Usage: python -I src/eval_split.py <dataset.csv> <out_csv>"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import train as T
from patches import GRID

ds = pd.read_csv(sys.argv[1]); ds = ds[(ds["status"] == "ok") & ds["vwc_pct"].notna()].reset_index(drop=True)
ds["split"] = ds["video"].map(lambda v: GRID[v][1])
tr, va, te = (ds[ds.split == k] for k in ("train", "val", "test"))
rows = []
for fs, cols in T.FEATURE_SETS.items():
    cols = [c for c in cols if c in ds.columns]
    for name, model in T.models().items():
        if model is None:
            pv, pt = np.full(len(va), tr.vwc_pct.mean()), np.full(len(te), tr.vwc_pct.mean())
            fs_name = "-"
        else:
            m = T.clone(model).fit(tr[cols], tr.vwc_pct)
            pv, pt = m.predict(va[cols]), m.predict(te[cols]); fs_name = fs
        rows.append(dict(features=fs_name, model=name, val_MAE=np.abs(pv - va.vwc_pct).mean(),
                         test_MAE=np.abs(pt - te.vwc_pct).mean()))
r = pd.DataFrame(rows).drop_duplicates(["features", "model"]).round(2).sort_values("val_MAE")
r.to_csv(sys.argv[2], index=False)
print(f"train {len(tr)} / val {len(va)} / test {len(te)} sightings")
print(r.to_string(index=False))
best = r[r.model != "mean_baseline"].iloc[0]; base = r[r.model == "mean_baseline"].iloc[0]
print(f"\nchosen on val: {best.model} ({best.features}) -> test MAE {best.test_MAE} vs mean baseline {base.test_MAE}")
