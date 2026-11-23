from __future__ import annotations

import numpy as np

from hdloc.data.schema import Element, VectorMap
from hdloc.maps.crop import crop_to_range
from hdloc.maps.transforms import global_to_ego


def resample_polyline(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Points spaced ~`spacing` metres apart along a polyline."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total == 0:
        return pts[:1]
    n = max(int(np.floor(total / spacing)) + 1, 2)
    t = np.linspace(0.0, total, n)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)


def densify_map(vmap: VectorMap, spacing: float = 1.0) -> VectorMap:
    """Resample every element so cropping by point distance is safe. Do this once."""
    return VectorMap(
        [Element(e.cls, resample_polyline(e.points, spacing)) for e in vmap.elements],
        vmap.frame,
    )


def observe(vmap: VectorMap, pose, radius: float) -> VectorMap:
    """What a perfect sensor at `pose` (x, y, yaw) sees: ego-frame map within `radius`."""
    x, y, yaw = pose
    ego = vmap.transformed(lambda p: global_to_ego(p, x, y, yaw), frame="ego")
    return crop_to_range(ego, radius)
