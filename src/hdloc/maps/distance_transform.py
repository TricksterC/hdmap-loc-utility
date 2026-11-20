from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from hdloc.maps.rasterize import Grid


def distance_transform(raster: np.ndarray, resolution: float, max_dist: float = 10.0) -> np.ndarray:
    """Metres from each pixel to the nearest line pixel, capped at max_dist."""
    if not raster.any():
        return np.full(raster.shape, max_dist, dtype=np.float32)
    dt = distance_transform_edt(raster == 0) * resolution
    return np.minimum(dt, max_dist).astype(np.float32)


def lookup_distance(dt: np.ndarray, grid: Grid, pts_world, fill: float) -> np.ndarray:
    """Bilinear lookup of dt at world points. Points outside the grid get `fill`."""
    uv = grid.world_to_pix(pts_world)
    u, v = uv[:, 0], uv[:, 1]
    inside = (u >= 0) & (u <= grid.width - 1) & (v >= 0) & (v <= grid.height - 1)
    out = np.full(len(uv), fill, dtype=np.float32)
    if inside.any():
        coords = np.stack([v[inside], u[inside]])  # (row, col) order
        out[inside] = map_coordinates(dt, coords, order=1, mode="nearest")
    return out
