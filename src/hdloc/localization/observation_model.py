from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hdloc.data.schema import CLASSES, VectorMap
from hdloc.maps.observe import resample_polyline
from hdloc.maps.prior_map import PriorMap


def _default_weights() -> dict:
    return {c: 1.0 for c in CLASSES}


@dataclass
class ObservationConfig:
    spacing: float = 1.0  # m between sampled observation points
    sigma: float = 0.5  # m, likelihood width
    trunc: float = 3.0  # m, distances beyond this count as `trunc`
    class_weights: dict = field(default_factory=_default_weights)


def sample_points(vmap: VectorMap, spacing: float = 1.0) -> dict[str, np.ndarray]:
    """Observed map -> {class: (M, 2) points, evenly spaced along each element}."""
    out = {}
    for cls in CLASSES:
        chunks = [resample_polyline(e.points, spacing) for e in vmap.of_class(cls)]
        if chunks:
            out[cls] = np.concatenate(chunks)
    return out


def log_likelihood(
    particles: np.ndarray, obs_pts: dict, prior: PriorMap, cfg: ObservationConfig
) -> np.ndarray:
    """Per-particle log-likelihood: class-weighted mean squared truncated distance."""
    n = len(particles)
    ll = np.zeros(n)
    c = np.cos(particles[:, 2])[:, None]
    s = np.sin(particles[:, 2])[:, None]
    for cls, pts in obs_pts.items():
        w = cfg.class_weights.get(cls, 0.0)
        if w == 0.0 or len(pts) == 0:
            continue
        gx = particles[:, 0:1] + c * pts[None, :, 0] - s * pts[None, :, 1]
        gy = particles[:, 1:2] + s * pts[None, :, 0] + c * pts[None, :, 1]
        d = prior.distance(cls, np.stack([gx.ravel(), gy.ravel()], axis=1)).reshape(n, -1)
        d = np.minimum(d, cfg.trunc)
        ll -= w * (d**2).mean(axis=1) / (2.0 * cfg.sigma**2)
    return ll
