"""Stage 4: train and evaluate soil-moisture models on per-sighting image features.

One row = one probe placement seen by the drone (median features over its frames),
label = calibrated VWC % from that probe's stable plateau.

Evaluation is always grouped so the model is never tested on a placement it trained on:
  - leave-one-placement-out (LOPO): hold out one plateau at a time
  - leave-one-row-out (LORO): hold out a whole drone flight / grid row (9 folds)
  - leave-one-day-out (LODO): train on 9/17, test on 9/18 and vice versa (2 folds)
Any training row that shares a plateau with the test fold is dropped (two placements were
filmed twice).

Usage: python -I src/train.py data/processed/features outputs
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
COLOR = ["r_chroma", "g_chroma", "b_chroma", "L", "a", "b", "L_std", "hue", "sat", "val",
         "L_rel", "a_rel", "b_rel", "soil_L_rel", "soil_a", "soil_b", "veg_L_rel", "veg_a", "veg_b"]
VEG = ["exg", "exg_std", "vari", "gli", "veg_frac"]
TEXTURE = ["glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation"]
QC = ["shadow_frac", "valid_frac"]
FEATURE_SETS = {
    "exposure_normalised": ["L_rel", "a_rel", "b_rel", "soil_L_rel", "veg_L_rel", "r_chroma", "g_chroma", "b_chroma"] + VEG,
    "color+veg": COLOR + VEG,
    "all": COLOR + VEG + TEXTURE + QC,
}


def models():
    return {
        "mean_baseline": None,
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               RidgeCV(alphas=np.logspace(-1, 3, 20))),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestRegressor(n_estimators=500, max_depth=4, min_samples_leaf=3,
                                                             max_features=0.4, random_state=SEED)),
    }


def folds(df, scheme):
    key = {"LOPO": "plateau_id", "LORO": "video", "LODO": "day"}[scheme]
    for g in df[key].unique():
        test = df[key] == g
        train = ~test & ~df["plateau_id"].isin(df.loc[test, "plateau_id"])
        yield np.where(train)[0], np.where(test)[0]


def evaluate(df, cols, model, scheme):
    y = df["vwc_pct"].values
    pred = np.full(len(df), np.nan)
    for tr, te in folds(df, scheme):
        if model is None:
            pred[te] = y[tr].mean()
        else:
            m = clone(model).fit(df.iloc[tr][cols], y[tr])
            pred[te] = m.predict(df.iloc[te][cols])
    err = pred - y
    rho = spearmanr(pred, y).correlation if np.std(pred) > 0 else np.nan
    return dict(MAE=np.abs(err).mean(), RMSE=np.sqrt((err ** 2).mean()),
                R2=1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum(), spearman=rho), pred


def clone(m):
    from sklearn.base import clone as c
    return c(m)


def run(ds, tag, out):
    rows, preds = [], {}
    for fs_name, cols in FEATURE_SETS.items():
        cols = [c for c in cols if c in ds.columns]
        for mname, model in models().items():
            if model is None and fs_name != "all":
                continue
            for scheme in ["LOPO", "LORO", "LODO"]:
                met, pred = evaluate(ds, cols, model, scheme)
                rows.append(dict(masking=tag, features=fs_name if model is not None else "-", model=mname,
                                 cv=scheme, n=len(ds), **{k: round(v, 3) for k, v in met.items()}))
                preds[(fs_name, mname, scheme)] = pred
    return pd.DataFrame(rows), preds


if __name__ == "__main__":
    fdir, out = Path(sys.argv[1]), Path(sys.argv[2])
    (out / "metrics").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    results, all_preds = [], {}
    for tag, fn in [("flag+box+shadow", "dataset.csv"), ("flag+box only", "dataset_noshadowmask.csv")]:
        ds = pd.read_csv(fdir / fn)
        ds = ds[(ds["status"] == "ok") & ds["vwc_pct"].notna()].reset_index(drop=True)
        ds["day"] = ds["video"].str[:4]          # keep as text ('9.17'), not a float
        r, p = run(ds, tag, out)
        results.append(r); all_preds[tag] = (ds, p)
    res = pd.concat(results, ignore_index=True)
    res.to_csv(out / "metrics" / "cv_results.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.sort_values(["cv", "MAE"]).to_string(index=False))

    # predicted vs actual for the main configuration
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds, p = all_preds["flag+box+shadow"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    for ax, scheme in zip(axes, ["LOPO", "LORO", "LODO"]):
        pr = p[("all", "random_forest", scheme)]
        for day, mk in [("9.17", "o"), ("9.18", "s")]:
            sel = ds["day"] == day
            ax.scatter(ds.loc[sel, "vwc_pct"], pr[sel], marker=mk, s=36, alpha=0.8, label=day)
        lo, hi = 0, 20
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=1, ls="--")
        mae = np.abs(pr - ds["vwc_pct"]).mean()
        ax.set_title(f"{scheme}  (MAE {mae:.2f} pts)"); ax.set_xlabel("sensor VWC %")
    axes[0].set_ylabel("predicted VWC %"); axes[0].legend(title="day", frameon=False)
    fig.suptitle("Random forest, all features, shadow-masked: held-out predictions")
    fig.tight_layout(); fig.savefig(out / "figures" / "pred_vs_actual.png", dpi=130)

    # final model on all labelled data + permutation importance (on LORO-held-out folds)
    cols = [c for c in FEATURE_SETS["all"] if c in ds.columns]
    final = models()["random_forest"].fit(ds[cols], ds["vwc_pct"])
    imp = np.zeros(len(cols)); n = 0
    for tr, te in folds(ds, "LORO"):
        m = clone(models()["random_forest"]).fit(ds.iloc[tr][cols], ds["vwc_pct"].iloc[tr])
        if len(te) >= 4:
            pi = permutation_importance(m, ds.iloc[te][cols], ds["vwc_pct"].iloc[te], n_repeats=20,
                                        random_state=SEED, scoring="neg_mean_absolute_error")
            imp += pi.importances_mean; n += 1
    imp = pd.Series(imp / n, index=cols).sort_values(ascending=False)
    imp.to_csv(out / "metrics" / "permutation_importance_LORO.csv", header=["mae_increase"])
    print("\nTop permutation importances (MAE increase when shuffled, held-out rows):")
    print(imp.head(10).round(3).to_string())
    import joblib
    joblib.dump(dict(model=final, features=cols), out / "models" / "rf_all_features.joblib")
    json.dump(dict(n_samples=len(ds), features=cols, seed=SEED), open(out / "models" / "rf_all_features.json", "w"), indent=2)
