# the few things every notebook needs: paths, the usgs points, reading chips, and the paper-style scoring
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.windows import Window
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

RAW, PROC, OUT = Path("../data/raw"), Path("../data/processed"), Path("../outputs")
UTM = "EPSG:32720"  # utm zone 20s, so distances come out in meters
TOL = 30  # scoring distance, picked in notebook 01
ON_DREDGE = 60  # a candidate this close to a labeled dredge sits on it, picked in notebook 02
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def load_points(method, crs=UTM, keep_invalid=False):
    """one usgs point layer (Visual, SUMO, Local or Global) with its scene date."""
    g = gpd.read_file(RAW / "labels" / f"Madeira_Dredges_{method}_Results.shp")
    g["date"] = pd.to_datetime(g["SAR_date"], format="%m/%d/%Y")
    # one visual point sits at a placeholder coordinate far off the globe (notebook 01)
    valid = (g.geometry.x.abs() <= 180) & (g.geometry.y.abs() <= 90)
    return g.loc[valid | keep_invalid, ["date", "geometry"]].to_crs(crs)


def scene(date):
    """the sar file for a date; my rebuilt june 10 image (notebook 02_1) stands in for the corrupt usgs one."""
    mine = PROC / "sar_mine" / f"S1B_IW_GRDH_1SDV_{date:%Y%m%d}.tif"
    return mine if mine.exists() else next((RAW / "sar").glob(f"*_{date:%Y%m%d}.tif"))


def chip(src, x, y, size, band=None):
    """a size x size window centered on (x, y), in the file's crs; anything past the edge comes back as 0."""
    row, col = src.index(x, y)
    return src.read(band, window=Window(col - size // 2, row - size // 2, size, size), boundless=True, fill_value=0)


def db(x, floor=1e-5):
    return 10 * np.log10(np.maximum(x, floor))


def pairs(a, b, tol=TOL):
    """one-to-one pairs (row positions i in a, j in b) within tol meters on the same date, closest overall."""
    out = []
    for date in a["date"].unique():
        ia, ib = np.flatnonzero(a["date"].values == date), np.flatnonzero(b["date"].values == date)
        if len(ia) and len(ib):
            dist = cdist(np.c_[a.geometry.x.values[ia], a.geometry.y.values[ia]],
                         np.c_[b.geometry.x.values[ib], b.geometry.y.values[ib]])
            # pairs past tol get a huge cost, so the assignment only uses them when it has to (and they're dropped)
            r, c = linear_sum_assignment(np.where(dist <= tol, dist, 1e9))
            keep = dist[r, c] <= tol
            out += list(zip(ia[r[keep]], ib[c[keep]]))
    return out


def prf(tp, fp, fn):
    return {"TP": tp, "FP": fp, "FN": fn, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "F1": 2 * tp / max(2 * tp + fp + fn, 1)}


def score(detections, truth, tol=TOL):
    """the paper's scoring: each labeled dredge can be claimed by one detection within tol on the same date."""
    tp = len(pairs(detections, truth, tol))
    return prf(tp, len(detections) - tp, len(truth) - tp)
