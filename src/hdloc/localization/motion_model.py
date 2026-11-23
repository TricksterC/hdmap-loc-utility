from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hdloc.maps.transforms import global_to_ego, wrap_angle


@dataclass
class DriftModel:
    scale_error: float = 0.0  # fractional error on travelled distance (0.03 = 3 %)
    yaw_bias: float = 0.0  # rad added to every step
    trans_noise: float = 0.0  # m, per-step random noise
    yaw_noise: float = 0.0  # rad, per-step random noise


def relative_increments(poses: np.ndarray) -> np.ndarray:
    """(T, 3) global poses -> (T-1, 3) increments (dx, dy, dyaw) in the previous body frame."""
    incs = np.zeros((len(poses) - 1, 3))
    for t in range(len(poses) - 1):
        x, y, yaw = poses[t]
        d = global_to_ego(poses[t + 1 : t + 2, :2], x, y, yaw)[0]
        incs[t] = (d[0], d[1], wrap_angle(poses[t + 1, 2] - yaw))
    return incs


def inject_drift(incs: np.ndarray, drift: DriftModel, rng: np.random.Generator) -> np.ndarray:
    """Corrupt clean increments the way a drifting odometer would."""
    out = incs.copy()
    out[:, :2] *= 1.0 + drift.scale_error
    out[:, :2] += rng.normal(0.0, drift.trans_noise, out[:, :2].shape)
    out[:, 2] += drift.yaw_bias + rng.normal(0.0, drift.yaw_noise, len(out))
    return out


def compose_pose(pose: np.ndarray, inc: np.ndarray) -> np.ndarray:
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [x + c * inc[0] - s * inc[1], y + s * inc[0] + c * inc[1], wrap_angle(yaw + inc[2])]
    )


def propagate(
    particles: np.ndarray,
    inc: np.ndarray,
    rng: np.random.Generator,
    trans_sigma: float,
    yaw_sigma: float,
) -> np.ndarray:
    """Move all particles (N, 3) by one odometry increment plus process noise."""
    n = len(particles)
    dx = inc[0] + rng.normal(0.0, trans_sigma, n)
    dy = inc[1] + rng.normal(0.0, trans_sigma, n)
    dyaw = inc[2] + rng.normal(0.0, yaw_sigma, n)
    c, s = np.cos(particles[:, 2]), np.sin(particles[:, 2])
    out = np.empty_like(particles)
    out[:, 0] = particles[:, 0] + c * dx - s * dy
    out[:, 1] = particles[:, 1] + s * dx + c * dy
    out[:, 2] = wrap_angle(particles[:, 2] + dyaw)
    return out
