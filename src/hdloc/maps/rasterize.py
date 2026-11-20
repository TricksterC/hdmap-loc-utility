from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from hdloc.data.schema import VectorMap


@dataclass(frozen=True)
class Grid:
    """A pixel grid over the world. Pixel (row, col) has its centre at
    (xmin + (col + 0.5) * res, ymin + (row + 0.5) * res). Row 0 is the lowest y."""

    xmin: float
    ymin: float
    resolution: float
    width: int
    height: int

    @classmethod
    def from_bounds(cls, xmin, ymin, xmax, ymax, resolution) -> Grid:
        width = int(np.ceil((xmax - xmin) / resolution))
        height = int(np.ceil((ymax - ymin) / resolution))
        return cls(float(xmin), float(ymin), float(resolution), width, height)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) for matplotlib imshow(origin='lower')."""
        return (
            self.xmin,
            self.xmin + self.width * self.resolution,
            self.ymin,
            self.ymin + self.height * self.resolution,
        )

    def world_to_pix(self, pts) -> np.ndarray:
        """World (x, y) -> continuous pixel coordinates (u = column, v = row)."""
        p = np.asarray(pts, dtype=float)
        u = (p[:, 0] - self.xmin) / self.resolution - 0.5
        v = (p[:, 1] - self.ymin) / self.resolution - 0.5
        return np.stack([u, v], axis=1)


def rasterize_class(vmap: VectorMap, cls: str, grid: Grid) -> np.ndarray:
    """Binary image (uint8, 1 = line pixel) of all elements of one class."""
    img = np.zeros(grid.shape, dtype=np.uint8)
    for e in vmap.of_class(cls):
        uv = np.round(grid.world_to_pix(e.points))
        uv = np.clip(uv, -100_000, 100_000).astype(np.int32)
        cv2.polylines(img, [uv.reshape(-1, 1, 2)], isClosed=False, color=1, thickness=1)
    return img
