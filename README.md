# soil_moisture_research

Predicting soil moisture from drone RGB video, with LoRaWAN soil-moisture probes as ground truth.

## Pipeline and first results (Oct 8, 2026)

## Data
- 9 drone videos (9/17 and 9/18, 4K at 59.94 fps, about 50 s each). Each video is one grid row: the drone hovers on the first box, then flies down the row past all 5 nodes.
- `soil_moisture_1.csv`: ChirpStack export, 5 nodes, about one reading every 20 s.

## Stages and scripts
| Stage | Script | What it does |
|---|---|---|
| 1 | `src/sensors.py` | Realigns the shifted exporter columns (rows from 7/16 23:50 UTC on), applies the per-node calibration (VWC % = a*ADC + b), finds stable plateaus (one per probe placement), and labels every drone sighting. |
| 0 | `src/extract_frames.py` | One frame per second from each video into `frames/` (ffmpeg). |
| 2a | `src/flags.py` | Red-flag and white-box detection on frames sampled at 1 fps. |
| 2b | `src/sightings.py` | Groups detections into 5 sightings per video (the k-th box seen is node FLIGHT_ORDER[k]). |
| 2c | `src/crops.py` | Takes 1200 px crops around each flag, plus whole-frame colour medians for exposure normalisation. |
| 3 | `src/features.py` | Masks the flag, the box, and shadows (hard cast shadows by local darkness plus blue shift; tree shadow by an Otsu dark/bright split that must also be bluer). Computes colour, vegetation index, and GLCM texture features in a 350 px disk around the flag, then takes the median per sighting. |
| 5 | `src/patches.py` | Patch dataset: a 700 px patch around each node per frame, plus a valid-pixel mask, named `<gridrow>_<node>_vwc<value>_t<sec>.jpg`, in `patch_dataset/{train,val,test}/<gridrow>/`, with `manifest.csv`. **Splits are by grid row, never by patch.** |
| 6 | `src/heatmaps.py` | `heatmaps/measured_grid_917.png` and `measured_grid_918.png`: sensor VWC for every row × node, each cell showing the drone close-up of that probe. `heatmaps/predicted_*.png`: the model applied to 240 px cells over one frame per row, plus `predicted_cells.csv`. |
| 7 | `src/eval_split.py` | Fixed train/val/test by grid row: pick the model on val, report it once on test. |
| 4 | `src/train.py` | Mean baseline, ridge, and random forest; leave-one-placement-out, leave-one-row-out, and leave-one-day-out CV; shadow-mask ablation; permutation importance. |

## Alignment decisions (none of them use image appearance)
- **Camera clock offset +445 s.** With zero offset, two videos land on the same probe placement. Sweeping the offset, every video lands inside its own stable in-soil plateau only for offsets of +405 to +480 s.
- **Flight order alyssa-test-3 → neha-test-2 → neha-1 → neha-test-3 → node-1.** The move order comes from the plateau timestamps (node-1 is moved first). node-1 is the small square box, which the drone always sees last.
- **Calibration row 1 = neha-test-3**, by elimination.
- **Independent check:** Alyssa's hand-logged alyssa-test-3 readings for the 9/18 dry rows match the aligned labels for rows 2–4 (9.4 / 5.5 / 11.6 vs 9.41 / 5.76 / 11.63%). Row 1 does not match: she logged 3.8% next to ADC 10630, which calibrates to 6.9%, and the gateway plateau reads 5.9%. She also noticed the drone clock running about 5 min off.

## Grid-row split
| split | rows |
|---|---|
| train | 9/17 rows 1 & 4; 9/18 rows 1 & 4; 9/18 wet row |
| val | 9/17 row 2; 9/18 row 2 |
| test | 9/17 row 3; 9/18 row 3 |

Fixed-split result: the model picked on val (ridge) gets **test MAE 4.64, the same as predicting the training mean (4.64)**.

## Results (43 labelled sightings, 42 distinct placements)
MAE in VWC percentage points:

| | leave-one-placement-out | leave-one-row-out | leave-one-day-out |
|---|---|---|---|
| Predict the training mean | 3.20 | 3.26 | 3.63 |
| Random forest, colour + veg | 3.25 | 3.26 | 3.68 |
| Random forest, all features | 3.37 | 3.48 | 3.69 |
| Ridge, all features | 3.29 | 3.93 | 3.76 |

No model beats the mean baseline. The strongest single feature correlation (|ρ| = 0.38) is below what chance gives across 30 features (0.44). The predicted-vs-actual plot shows the forest only learning the day-level difference (9/17 was greener and slightly wetter).

## Likely reasons
1. **The camera mostly sees grass, not soil.** On 9/17, 98–100% of the pixels around the flags are vegetation, and grass colour doesn't change in the minutes between placements.
2. **The probe reads several cm down.** Surface appearance and root-zone moisture are weakly coupled on turf.
3. **Small spread compared to the sensor error.** Most labels fall between 5% and 15%. Per-probe calibration error is probably a few points.
4. **Auto exposure and white balance.** These shift colour between frames. The exposure-normalised features didn't help.
5. **Sample size.** n is about 43, from 9 rows.

## What would most likely move the needle
- Controlled wet/dry contrasts on **bare soil patches** (the wet-row clip shows visible dark rings, which is the kind of signal RGB can see).
- Locked exposure and white balance, plus a grey card in frame.
- More placements per flight.
- A thermal or NIR band if one is available.

## How to run it yourself
Put the drone `.MP4` files in a `videos/` folder inside this repo (they are too big for git and are ignored), or one folder up. `soil_moisture 1.csv` is already in the repo.

One-time setup (Windows PowerShell, in this folder):
```
winget install Gyan.FFmpeg          # only needed for the frame-extraction step
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Run everything:
```
python run_all.py
python run_all.py --from labels        # skip frame extraction and detection if frames/ and outputs/sightings.csv exist
python run_all.py --only heatmaps      # just redraw the maps (needs features/dataset.csv from a full run)
```
Steps in order: `frames → detect → sightings → labels → crops → features → train → patches → heatmaps`.

**Adding new flights:** put the new MP4s next to the old ones, add each video's start time and duration to `VIDEOS` in `src/sensors.py`, add it to `GRID` in `src/patches.py` (grid-row name + split) and to `GRID_ROWS` in `src/heatmaps.py`, then `python run_all.py`.

## What's in the repo
- Code: `run_all.py`, `src/`, `requirements.txt`.
- Results from the Sept 17–18 flights: `heatmaps/`, `outputs/metrics/`, `outputs/figures/pred_vs_actual.png`.
- Not committed (regenerated by `run_all.py`, or too large): videos, `frames/`, `crops/`, `features/`, `patch_dataset/`, intermediate CSVs, trained model files.
