from __future__ import annotations

import numpy as np

from hdloc.data.schema import VectorMap


def shift_map(vmap: VectorMap, dx: float = 0.0, dy: float = 0.0) -> VectorMap:
    """Shift every point in the ego frame (dx forward, dy left, metres)."""
    off = np.array([dx, dy], dtype=float)
    return vmap.transformed(lambda p: p + off, vmap.frame)


def rotate_map(vmap: VectorMap, angle: float) -> VectorMap:
    """Rotate the map about the ego origin by `angle` radians (counter-clockwise)."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return vmap.transformed(lambda p: p @ rot.T, vmap.frame)


def jitter_map(vmap: VectorMap, sigma: float, rng: np.random.Generator) -> VectorMap:
    """Independent Gaussian noise (std `sigma` metres) on every point."""
    return vmap.transformed(lambda p: p + rng.normal(0.0, sigma, p.shape), vmap.frame)
