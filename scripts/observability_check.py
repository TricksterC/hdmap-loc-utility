from pathlib import Path

import numpy as np

from hdloc.data.schema import CLASSES
from hdloc.experiments.common import load_scenes
from hdloc.maps.observe import observe
from hdloc.metrics.observability import class_information

REPO = Path(__file__).resolve().parents[1]
scenes = [s for s in load_scenes(REPO / "data" / "nuscenes" / "v1.0-mini") if s.path_len > 10]
n_frames = sum(len(s.poses) for s in scenes)

for radius in (30.0, 50.0):
    tot = {c: np.zeros(2) for c in CLASSES}
    for s in scenes:
        for pose in s.poses:
            for c, (lon, lat) in class_information(observe(s.vmap, pose, radius)).items():
                tot[c] += (lon, lat)
    print(f"\nobservation radius {radius:.0f} m: mean constraint per frame (m of line length)")
    print(f"{'class':14s}{'longitudinal':>14s}{'lateral':>10s}{'lon share':>11s}")
    for c in CLASSES:
        lon, lat = tot[c] / n_frames
        share = lon / (lon + lat) if lon + lat > 0 else float("nan")
        print(f"{c:14s}{lon:14.1f}{lat:10.1f}{share:11.2f}")
