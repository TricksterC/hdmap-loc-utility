from __future__ import annotations

import numpy as np

from hdloc.localization.motion_model import compose_pose


def odometry_only(init_pose, incs: np.ndarray) -> np.ndarray:
    """Dead reckoning: integrate the (drifting) increments, no map correction."""
    poses = [np.asarray(init_pose, dtype=float)]
    for inc in incs:
        poses.append(compose_pose(poses[-1], inc))
    return np.array(poses)


def gnss_only(poses: np.ndarray, rng: np.random.Generator, sigma_xy: float = 3.0) -> np.ndarray:
    """Noisy, drift-free positions (T, 2)."""
    return poses[:, :2] + rng.normal(0.0, sigma_xy, (len(poses), 2))
