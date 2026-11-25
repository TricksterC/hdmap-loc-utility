from __future__ import annotations

import numpy as np

from hdloc.data.schema import CLASSES, TRANSVERSE, VectorMap
from hdloc.maps.observe import observe


def class_information(vmap: VectorMap) -> dict[str, tuple[float, float]]:
    """Geometric constraint each class gives, from line orientation (ego frame, x forward).

    A segment of length L with unit normal n adds L*n_x^2 to the longitudinal total and
    L*n_y^2 to the lateral total (metres of line length). A line along the road
    constrains only lateral position; a line across the road only longitudinal position.
    """
    out = {}
    for cls in CLASSES:
        lon = lat = 0.0
        for e in vmap.of_class(cls):
            t = np.diff(e.points, axis=0)
            length = np.linalg.norm(t, axis=1)
            ok = length > 1e-9
            if not ok.any():
                continue
            nx = -t[ok, 1] / length[ok]
            ny = t[ok, 0] / length[ok]
            lon += float((length[ok] * nx**2).sum())
            lat += float((length[ok] * ny**2).sum())
        out[cls] = (lon, lat)
    return out


def route_coverage(vmap: VectorMap, poses, radius: float, thresholds=(1.0, 5.0, 10.0)) -> dict:
    """Share of frames along a route with each class in range, and with enough constraint.

    `<cls>_present`: fraction of frames with any element of that class within `radius`.
    `lon_<group>_ge_<t>m`: fraction of frames with at least t metres of longitudinally
    constraining line length, for the groups all / divider / transverse.
    """
    groups = {"all": CLASSES, "divider": ("divider", "boundary"), "transverse": TRANSVERSE}
    n = len(poses)
    present = {c: 0 for c in CLASSES}
    trans_any = 0
    lon = {g: [] for g in groups}
    for pose in poses:
        seen = observe(vmap, pose, radius)
        info = class_information(seen)
        for c in CLASSES:
            present[c] += bool(seen.of_class(c))
        trans_any += any(seen.of_class(c) for c in TRANSVERSE)
        for g, cs in groups.items():
            lon[g].append(sum(info[c][0] for c in cs))
    out = {f"{c}_present": present[c] / n for c in CLASSES}
    out["transverse_present"] = trans_any / n
    for g, vals in lon.items():
        arr = np.array(vals)
        for t in thresholds:
            out[f"lon_{g}_ge_{t:g}m"] = float((arr >= t).mean())
    return out
