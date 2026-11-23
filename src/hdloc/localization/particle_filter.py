from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hdloc.localization.init import init_particles
from hdloc.localization.motion_model import propagate
from hdloc.localization.observation_model import ObservationConfig, log_likelihood
from hdloc.localization.resampling import effective_sample_size, systematic_resample
from hdloc.maps.prior_map import PriorMap
from hdloc.maps.transforms import wrap_angle


@dataclass
class FilterConfig:
    n_particles: int = 500
    trans_noise: float = 0.15  # m per step; must cover odometry drift per step
    yaw_noise: float = 0.01  # rad per step
    ess_threshold: float = 0.5  # resample when ESS < threshold * N
    roughen_xy: float = 0.03  # m, jitter after resampling
    roughen_yaw: float = 0.002  # rad
    init_xy: float = 1.0  # m, initial spread
    init_yaw: float = 0.05  # rad
    obs: ObservationConfig = field(default_factory=ObservationConfig)

    @classmethod
    def from_dict(cls, d: dict) -> FilterConfig:
        d = dict(d)
        obs = ObservationConfig(**d.pop("obs", {}))
        return cls(obs=obs, **d)


class ParticleFilter:
    def __init__(self, prior: PriorMap, cfg: FilterConfig, init_pose, rng: np.random.Generator):
        self.prior, self.cfg, self.rng = prior, cfg, rng
        n = cfg.n_particles
        self.particles = init_particles(init_pose, n, cfg.init_xy, cfg.init_yaw, rng)
        self.weights = np.full(n, 1.0 / n)

    def predict(self, inc) -> None:
        self.particles = propagate(
            self.particles, inc, self.rng, self.cfg.trans_noise, self.cfg.yaw_noise
        )

    def update(self, obs_pts: dict | None) -> None:
        if not obs_pts:
            return
        ll = log_likelihood(self.particles, obs_pts, self.prior, self.cfg.obs)
        logw = np.log(self.weights + 1e-300) + ll
        logw -= logw.max()
        w = np.exp(logw)
        self.weights = w / w.sum()

    def estimate(self) -> tuple[np.ndarray, float]:
        """Weighted pose estimate and positional spread (m)."""
        w, p = self.weights, self.particles
        xy = (w[:, None] * p[:, :2]).sum(axis=0)
        yaw = np.arctan2((w * np.sin(p[:, 2])).sum(), (w * np.cos(p[:, 2])).sum())
        spread = float(np.sqrt((w * ((p[:, :2] - xy) ** 2).sum(axis=1)).sum()))
        return np.array([xy[0], xy[1], yaw]), spread

    def maybe_resample(self) -> None:
        n = self.cfg.n_particles
        if effective_sample_size(self.weights) >= self.cfg.ess_threshold * n:
            return
        idx = systematic_resample(self.weights, self.rng)
        self.particles = self.particles[idx].copy()
        self.particles[:, :2] += self.rng.normal(0.0, self.cfg.roughen_xy, (n, 2))
        self.particles[:, 2] = wrap_angle(
            self.particles[:, 2] + self.rng.normal(0.0, self.cfg.roughen_yaw, n)
        )
        self.weights = np.full(n, 1.0 / n)


def run_filter(prior, init_pose, incs, obs_list, cfg, rng):
    """Run over a whole sequence.

    incs: (T-1, 3) odometry increments; obs_list: T observation dicts (one per pose).
    Returns estimated poses (T, 3) and spread (T,).
    """
    pf = ParticleFilter(prior, cfg, init_pose, rng)
    est = np.zeros((len(obs_list), 3))
    spread = np.zeros(len(obs_list))
    for t, obs in enumerate(obs_list):
        if t > 0:
            pf.predict(incs[t - 1])
        pf.update(obs)
        est[t], spread[t] = pf.estimate()
        pf.maybe_resample()
    return est, spread
