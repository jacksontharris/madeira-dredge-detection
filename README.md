# Classifying gold-mining dredges in Sentinel-1 on the Madeira River

TECH 27 final project, Jackson Harris.

Two USGS detectors flag bright spots on the Madeira River in seven Sentinel-1 radar scenes (June 4th to July 10th, 2019), and a lot of those spots aren't dredges. This project trains models to decide which flagged spots sit on a dredge, scored against the analysts' labeled dredges the same way Alessi, Chirico and Millones (2023) scored their detectors. On 34 held-out sites, my final model (logistic regression averaged with a small CNN) gets an F1 of 0.93, vs. 0.75 for the Local method's detections. Across the whole river it gets 0.84, vs. 0.70.

## Contents

| Path | What it is |
|---|---|
| `notebooks/` | The analysis, in run order (below). Each notebook explains its method and what its outputs mean. |
| `notebooks/helpers.py` | The few things every notebook uses: paths, loading the USGS points, reading chips, and the paper-style scoring. |
| `scripts/candidate_peaks_test.py` | A feasibility test of my own candidate detector, which the models don't use. It returns about 25,000 peaks, and 158 of the 204 labels have a peak within 30m on the same date. The USGS candidates reach 178 with 292 points. |
| `outputs/` | The figures the notebooks save for the report. |
| `requirements.txt` | Pinned package versions. |

## Running it

Use Python 3.12:

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the notebooks from `notebooks/`, in this order. Each one reads files the one before it writes.

| Order | Notebook | What it does | Run time |
|---|---|---|---|
| 1 | `01_data_inventory.ipynb` | Downloads the USGS data release (labels and seven scenes, about 4 GB, doi:10.5066/P9OML7YH), reproduces the paper's scores, and checks that the labels sit on bright objects. | under 1 min after download |
| 2 | `02_1_preprocessing.ipynb` | Rebuilds the corrupt June 10th scene from raw Sentinel-1 data on Microsoft's Planetary Computer (no login needed). | about 7 min |
| 3 | `02_candidates_features.ipynb` | Builds the 292-candidate table with 21 features. | under 1 min |
| 4 | `02_2_sentinel2_check.ipynb` | Checks the candidates by eye in Sentinel-2 true color. The models don't use it. | about 1 min |
| 5 | `03_models.ipynb` | Baselines, nine classifiers, nested tuning, a CNN, feature ablation, ship pretraining, and the one-time test. | about 37 min on CPU |

Notebook 03's pretraining section needs LS-SSDD-v1.0 (Zhang et al. 2020, *Remote Sensing* 12(18): 2997), a Sentinel-1 ship dataset of about 2.7 GB. Download it from the links at https://github.com/TianwenZhang0825/LS-SSDD-v1.0-OPEN and unpack it into `data/raw/ls_ssdd/`. The peak script runs from the project root: `.venv/bin/python scripts/candidate_peaks_test.py`.

## Data

`data/` isn't included because it's about 11 GB. The notebooks recreate it:

- `data/raw/labels/`, `data/raw/sar/`: the USGS release, downloaded by notebook 01
- `data/interim/sigma0_radar/`: notebook 02_1's download cache
- `data/processed/sar_mine/`: my rebuilt June 10th scene, written by notebook 02_1
- `data/processed/candidates.gpkg`: the candidate table, written by notebook 02
- `data/raw/ls_ssdd/`: downloaded by hand (see above)
