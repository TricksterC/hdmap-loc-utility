from __future__ import annotations

import numpy as np

from hdloc.data.schema import CLASSES, VectorMap


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
