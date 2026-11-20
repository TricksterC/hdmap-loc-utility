from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hdloc.data.schema import CLASSES, VectorMap
from hdloc.maps.distance_transform import distance_transform, lookup_distance
from hdloc.maps.rasterize import Grid, rasterize_class


@dataclass
class PriorMap:
    """Per-class distance transforms of the prior map, in the global frame."""

    grid: Grid
    dts: dict[str, np.ndarray]
    max_dist: float

    def distance(self, cls: str, pts_world) -> np.ndarray:
        n = len(np.asarray(pts_world))
        dt = self.dts.get(cls)
        if dt is None:  # class absent from the prior map
            return np.full(n, self.max_dist, dtype=np.float32)
        return lookup_distance(dt, self.grid, pts_world, self.max_dist)


def grid_for_poses(poses: np.ndarray, margin: float = 80.0, resolution: float = 0.2) -> Grid:
    """A grid covering the trajectory (rows of [x, y, yaw]) plus a margin."""
    xs, ys = poses[:, 0], poses[:, 1]
    return Grid.from_bounds(
        xs.min() - margin, ys.min() - margin, xs.max() + margin, ys.max() + margin, resolution
    )


def build_prior(
    vmap: VectorMap, grid: Grid, max_dist: float = 10.0, classes=CLASSES
) -> PriorMap:
    dts = {}
    for cls in classes:
        raster = rasterize_class(vmap, cls, grid)
        if raster.any():
            dts[cls] = distance_transform(raster, grid.resolution, max_dist)
    return PriorMap(grid, dts, max_dist)
