from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from hdloc.data.schema import VectorMap
from hdloc.maps.observe import resample_polyline

# Metres. Turns the rotation unknown into a length so all three unknowns are on one scale
# (it only conditions the solve; it does not change the answer).
LEVER = 10.0


def _rot(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


class TruthIndex:
    """Nearest-line lookup over the true map: sampled points, unit normals, KD-tree."""

    def __init__(self, truth: VectorMap, spacing: float = 0.5):
        pts, nrm = [], []
        for e in truth.elements:
            p = resample_polyline(e.points, spacing)
            if len(p) < 2:
                continue
            t = np.diff(p, axis=0)
            t = np.vstack([t, t[-1:]])
            length = np.linalg.norm(t, axis=1)
            ok = length > 1e-9
            t = t[ok] / length[ok, None]
            pts.append(p[ok])
            nrm.append(np.stack([-t[:, 1], t[:, 0]], axis=1))
        self.n = sum(len(p) for p in pts)
        if self.n:
            self.points = np.concatenate(pts)
            self.normals = np.concatenate(nrm)
            self.tree = cKDTree(self.points)


def prediction_points(pred: VectorMap, spacing: float = 1.0) -> np.ndarray:
    """Predicted elements as evenly spaced points (what the localizer observes)."""
    chunks = [resample_polyline(e.points, spacing) for e in pred.elements]
    chunks = [c for c in chunks if len(c)]
    return np.concatenate(chunks) if chunks else np.zeros((0, 2))


@dataclass
class Alignment:
    offset: np.ndarray  # (forward m, sideways m, rotation rad) of the prediction vs the truth
    n_inliers: int
    n_points: int
    info: np.ndarray  # constraint per unknown (forward, sideways, rotation in metre units)


def align_offset(
    index: TruthIndex,
    pred_pts: np.ndarray,
    trunc: float = 3.0,
    huber: float = 0.5,
    iters: int = 25,
    damping: float = 1e-2,
    tol: float = 1e-5,
) -> Alignment | None:
    """Rigid transform (x, y, rotation about the ego origin) that best puts the predicted
    points onto the true lines, by point-to-line Gauss-Newton with Huber weights.

    This is the localizer's own cost (distance of observed points to the prior map,
    truncated at `trunc`), minimised directly: no odometry, no noise, no drift.
    A damping term keeps directions the map does not constrain at zero, which is the
    correct answer for them: an offset nothing can observe is not inherited by a localizer.
    Returns None when there is nothing to align (empty prediction or too few inliers).
    """
    if index.n == 0 or len(pred_pts) == 0:
        return None
    theta, t = 0.0, np.zeros(2)
    info = np.zeros(3)
    n_in = 0
    for _ in range(iters):
        q = pred_pts @ _rot(theta).T + t
        d, idx = index.tree.query(q)
        inl = d < trunc
        n_in = int(inl.sum())
        if n_in < 3:
            return None
        qi, nr = q[inl], index.normals[idx[inl]]
        r = ((qi - index.points[idx[inl]]) * nr).sum(axis=1)
        w = np.minimum(1.0, huber / np.maximum(np.abs(r), 1e-9))
        perp = np.stack([-qi[:, 1], qi[:, 0]], axis=1)
        a = np.column_stack([nr[:, 0], nr[:, 1], (nr * perp).sum(axis=1) / LEVER])
        h = a.T @ (a * w[:, None])
        delta = np.linalg.solve(h + damping * n_in * np.eye(3), -(a.T @ (w * r)))
        dtheta = delta[2] / LEVER
        t = _rot(dtheta) @ t + delta[:2]
        theta += dtheta
        info = np.diag(h).copy()
        if np.abs(delta).max() < tol:
            break
    # the transform above maps prediction -> truth; report the prediction's offset (inverse)
    off_t = -_rot(-theta) @ t
    return Alignment(np.array([off_t[0], off_t[1], -theta]), n_in, len(pred_pts), info)


class AlignmentAccumulator:
    """Per-frame alignment offsets over one run, summarised like the pose errors are."""

    def __init__(self, spacing: float = 1.0, **kwargs):
        self.spacing, self.kwargs = spacing, kwargs
        self.rows: list = []
        self.n_frames = 0

    def add(self, truth: VectorMap, pred: VectorMap) -> None:
        self.n_frames += 1
        a = align_offset(TruthIndex(truth), prediction_points(pred, self.spacing), **self.kwargs)
        if a is not None:
            self.rows.append([*a.offset, *a.info, a.n_inliers / a.n_points])

    def result(self) -> dict:
        keys = ("align_lon_med", "align_lat_med", "align_yaw_med_deg", "align_lon_bias",
                "align_lat_bias", "align_info_lon", "align_info_lat", "align_inlier_frac")
        out = {k: np.nan for k in keys}
        out["align_valid"] = len(self.rows) / self.n_frames if self.n_frames else 0.0
        if not self.rows:
            return out
        r = np.array(self.rows)
        out["align_lon_med"] = float(np.median(np.abs(r[:, 0])))
        out["align_lat_med"] = float(np.median(np.abs(r[:, 1])))
        out["align_yaw_med_deg"] = float(np.degrees(np.median(np.abs(r[:, 2]))))
        out["align_lon_bias"] = float(np.median(r[:, 0]))
        out["align_lat_bias"] = float(np.median(r[:, 1]))
        out["align_info_lon"] = float(np.median(r[:, 3]))
        out["align_info_lat"] = float(np.median(r[:, 4]))
        out["align_inlier_frac"] = float(np.mean(r[:, 6]))
        return out
