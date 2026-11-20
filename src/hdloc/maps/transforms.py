import numpy as np


def quat_to_yaw(q) -> float:
    """nuScenes quaternion (w, x, y, z) to heading angle in radians."""
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def global_to_ego(points, x, y, yaw):
    """Express global points in the ego frame (x forward, y left)."""
    d = np.asarray(points, dtype=float) - np.array([x, y])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)


def ego_to_global(points, x, y, yaw):
    p = np.asarray(points, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * p[:, 0] - s * p[:, 1] + x, s * p[:, 0] + c * p[:, 1] + y], axis=1)
