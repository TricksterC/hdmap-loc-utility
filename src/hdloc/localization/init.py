from __future__ import annotations

import numpy as np

from hdloc.maps.transforms import wrap_angle


def init_particles(
    pose, n: int, sigma_xy: float, sigma_yaw: float, rng: np.random.Generator
) -> np.ndarray:
    """Particles scattered around a rough starting pose (x, y, yaw)."""
    p = np.empty((n, 3))
    p[:, :2] = np.asarray(pose[:2]) + rng.normal(0.0, sigma_xy, (n, 2))
    p[:, 2] = wrap_angle(pose[2] + rng.normal(0.0, sigma_yaw, n))
    return p
