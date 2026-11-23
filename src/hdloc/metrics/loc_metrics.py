from __future__ import annotations

import numpy as np

from hdloc.maps.transforms import wrap_angle


def pose_errors(est: np.ndarray, gt: np.ndarray) -> dict[str, np.ndarray]:
    """Signed errors with longitudinal/lateral axes taken from the *ground-truth* heading."""
    d = est[:, :2] - gt[:, :2]
    c, s = np.cos(gt[:, 2]), np.sin(gt[:, 2])
    return {
        "lon": c * d[:, 0] + s * d[:, 1],
        "lat": -s * d[:, 0] + c * d[:, 1],
        "yaw": wrap_angle(est[:, 2] - gt[:, 2]),
    }
