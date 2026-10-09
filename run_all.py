"""Run the whole soil-moisture pipeline, start to finish.

    python run_all.py                       # everything
    python run_all.py --from features       # start at a later step (earlier outputs must exist)
    python run_all.py --only heatmaps       # one step

Run it from anywhere; it works inside the folder this file sits in.
By default it looks for the sensor CSV and the .MP4 videos in this folder (or a videos/ subfolder),
and falls back to the folder one level up (e.g. Downloads). Override with --videos / --sensor-csv.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def sh(*args):
    print("  $", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True, cwd=ROOT)


def step_frames(a):      sh(PY, "src/extract_frames.py", a.videos)
def step_detect(a):      sh(PY, "src/flags.py")
def step_sightings(a):   sh(PY, "src/sightings.py")
def step_labels(a):      sh(PY, "-I", "src/sensors.py", a.sensor_csv, "outputs")


def step_crops(a):
    vids = sorted(pd.read_csv(ROOT / "outputs/sightings.csv")["video"].unique())
    for v in vids:
        sh(PY, "src/crops.py", v)
    parts = [pd.read_csv(ROOT / "outputs/crops_csv" / f"{v}.csv") for v in vids]
    pd.concat(parts).to_csv(ROOT / "outputs/crops.csv", index=False)


def step_features(a):
    for extra in ([], ["--no-shadow-mask"]):
        sh(PY, "-I", "src/features.py", ".", "outputs/crops.csv", "outputs/sighting_labels.csv", "features", *extra)


def step_train(a):
    sh(PY, "-I", "src/train.py", "features", "outputs")
    sh(PY, "-I", "src/eval_split.py", "features/dataset.csv", "outputs/metrics/fixed_split_results.csv")


def step_patches(a):     sh(PY, "src/patches.py", "outputs/crops.csv", ".", "outputs/sighting_labels.csv", "patch_dataset")
def step_heatmaps(a):    sh(PY, "-I", "src/heatmaps.py", "frames", "outputs/sightings.csv", "outputs/sighting_labels.csv",
                            "features/dataset.csv", "heatmaps", "outputs/crops.csv", ".")


STEPS = [("frames", step_frames), ("detect", step_detect), ("sightings", step_sightings), ("labels", step_labels),
         ("crops", step_crops), ("features", step_features), ("train", step_train), ("patches", step_patches),
         ("heatmaps", step_heatmaps)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    def first(*cands):
        return next((str(c) for c in cands if c.exists()), str(cands[-1]))
    ap.add_argument("--videos", default=first(ROOT / "videos", ROOT.parent), help="folder with the .MP4 files")
    ap.add_argument("--sensor-csv", default=first(ROOT / "soil_moisture 1.csv", ROOT.parent / "soil_moisture 1.csv"))
    ap.add_argument("--from", dest="start", choices=[s for s, _ in STEPS])
    ap.add_argument("--only", choices=[s for s, _ in STEPS])
    a = ap.parse_args()
    os.chdir(ROOT)
    names = [s for s, _ in STEPS]
    todo = [a.only] if a.only else names[names.index(a.start):] if a.start else names
    for name, fn in STEPS:
        if name in todo:
            print(f"\n== {name} ==", flush=True)
            fn(a)
    print("\ndone. See heatmaps/, patch_dataset/, outputs/metrics/, outputs/figures/")
