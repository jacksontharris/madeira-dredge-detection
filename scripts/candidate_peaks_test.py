# verdict: my own peak detector returned about 25,000 peaks (24,987 on the 2026-11-19 run), and
# 158 of the 204 labels had a same-date peak within 30m. the 292 usgs candidates reach 178 of 204
# with about 85 times fewer points, so the models use the usgs candidates and this stays future work.
#
# the test (2026-11-07): candidates = bright local peaks on water, outside permanent land, on each
# of the seven dates. coverage is a nearest-peak check at 30m, not the one-to-one scoring rule.
# run from the project root: .venv/bin/python scripts/candidate_peaks_test.py
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from scipy.spatial import cKDTree

RAW, PROC = Path("data/raw"), Path("data/processed")
OUT = Path("data/interim/candidate_peaks.csv")
dates = pd.to_datetime(["2019-06-04", "2019-06-10", "2019-06-16", "2019-06-22", "2019-06-28", "2019-07-04", "2019-07-10"])
# water thresholds (vv db) for each date, from notebook 02
thresh = dict(zip(dates, [-14.3, -15.5, -14.9, -15.4, -14.9, -15.3, -15.2]))


def scene(date):
    # my own june 10 image from notebook 02_1 replaces the corrupt usgs file
    mine = PROC / "sar_mine" / f"S1B_IW_GRDH_1SDV_{date:%Y%m%d}.tif"
    return mine if mine.exists() else next((RAW / "sar").glob(f"*_{date:%Y%m%d}.tif"))


srcs = {d: rasterio.open(scene(d)) for d in dates}
ref = srcs[dates[0]]
H, W = ref.height, ref.width
# each file's grid is shifted by up to a pixel; offset reads so all dates line up with the reference
shift = {d: (round((ref.transform.f - s.transform.f) / s.transform.e), round((ref.transform.c - s.transform.c) / s.transform.a))
         for d, s in srcs.items()}

STRIP, PAD = 2048, 16  # read in strips, with overlap so peaks at strip edges aren't lost
rows, t0 = [], time.time()
for r0 in range(0, H, STRIP):
    w = Window(0, max(r0 - PAD, 0), W, min(STRIP + 2 * PAD, H - max(r0 - PAD, 0)))
    off = r0 - max(r0 - PAD, 0)
    stack = {}
    for d, s in srcs.items():
        dr, dc = shift[d]
        try:
            a = s.read(2, window=Window(w.col_off + dc, w.row_off + dr, w.width, w.height), boundless=True, fill_value=0).astype("float32")
            a[a == 0] = np.nan
        except Exception:
            a = np.full((w.height, w.width), np.nan, "float32")
        a[a > 1e30] = np.nan
        stack[d] = 10 * np.log10(np.maximum(a, 1e-5))
    med = np.nanmedian(np.stack(list(stack.values())), axis=0)
    # permanent land: bright in the seven-date median and wider than 70m (same rule as notebook 02), grown by a pixel
    land = ndimage.binary_opening(med >= np.median(list(thresh.values())), structure=np.ones((7, 7)))
    land = ndimage.binary_dilation(land, iterations=1)
    for d, v in stack.items():
        valid = np.isfinite(v)
        vv = np.where(valid, v, -99)
        # peak = brightest pixel within 30m (7x7) and at least 6 db above the water cutoff
        peak = (vv == ndimage.maximum_filter(vv, size=7)) & (vv >= thresh[d] + 6) & ~land & valid
        peak[:off] = False
        peak[off + STRIP:] = False
        for r_, c_ in zip(*np.nonzero(peak)):
            x, y = ref.xy(w.row_off + r_, c_)
            rows.append((d, x, y, vv[r_, c_]))
    print(f"strip {r0}: {len(rows)} peaks, {time.time() - t0:.0f}s", flush=True)

peaks = pd.DataFrame(rows, columns=["date", "x", "y", "vv"])
peaks.to_csv(OUT, index=False)
print(peaks.groupby("date").size().rename("peaks"))

# coverage: how many labeled dredges have a peak within 30m on their date
p = gpd.GeoDataFrame(peaks, geometry=gpd.points_from_xy(peaks.x, peaks.y), crs="EPSG:4326").to_crs("EPSG:32720")
v = gpd.read_file(RAW / "labels" / "Madeira_Dredges_Visual_Results.shp").to_crs("EPSG:32720")
v = v[np.isfinite(v.geometry.x)]
v["date"] = pd.to_datetime(v["SAR_date"], format="%m/%d/%Y")
covered = 0
for d, g in v.groupby("date"):
    pp = p[p["date"] == d]
    dist = cKDTree(np.c_[pp.geometry.x, pp.geometry.y]).query(np.c_[g.geometry.x, g.geometry.y])[0]
    covered += int((dist <= 30).sum())
print(f"{covered} of {len(v)} labeled dredges within 30m of a peak; {len(peaks)} peaks in total")
