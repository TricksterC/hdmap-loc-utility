import numpy as np

from hdloc.data.schema import Element, VectorMap
from hdloc.localization.baselines import odometry_only
from hdloc.localization.motion_model import DriftModel, inject_drift, relative_increments
from hdloc.localization.observation_model import (
    ObservationConfig,
    log_likelihood,
    sample_points,
)
from hdloc.localization.particle_filter import FilterConfig, run_filter
from hdloc.maps.observe import densify_map, observe
from hdloc.maps.prior_map import build_prior
from hdloc.maps.rasterize import Grid
from hdloc.utils.seed import set_seed


def make_road(length=300.0):
    """Straight road along +x: two dividers and a stop line every 25 m."""
    els = [Element("divider", np.array([[0.0, y], [length, y]])) for y in (-1.75, 1.75)]
    for x in np.arange(25.0, length, 25.0):
        els.append(Element("stop_line", np.array([[x, -3.5], [x, 3.5]])))
    return densify_map(VectorMap(els), 1.0)


GRID = Grid.from_bounds(-20, -30, 340, 30, 0.2)


def test_true_pose_scores_higher_than_shifted_poses():
    vmap = make_road()
    prior = build_prior(vmap, GRID)
    pose = np.array([60.0, 0.0, 0.0])
    obs = sample_points(observe(vmap, pose, 30.0))
    particles = np.array([pose, pose + [0.0, 1.5, 0.0], pose + [3.0, 0.0, 0.0]])
    ll = log_likelihood(particles, obs, prior, ObservationConfig())
    assert ll[0] > ll[1] and ll[0] > ll[2]


def test_filter_beats_odometry_on_synthetic_road():
    rng = set_seed(0)
    vmap = make_road()
    prior = build_prior(vmap, GRID)
    poses = np.array([[5.0 + 2.0 * t, 0.0, 0.0] for t in range(60)])
    drift = DriftModel(scale_error=0.05, yaw_bias=0.004, trans_noise=0.02, yaw_noise=0.002)
    incs = inject_drift(relative_increments(poses), drift, rng)
    dead = odometry_only(poses[0], incs)

    obs = [sample_points(observe(vmap, p, 30.0)) for p in poses]
    est, _ = run_filter(prior, poses[0], incs, obs, FilterConfig(n_particles=500), rng)

    err_dead = np.linalg.norm(dead[:, :2] - poses[:, :2], axis=1)
    err_pf = np.linalg.norm(est[:, :2] - poses[:, :2], axis=1)
    print(f"median err: odometry {np.median(err_dead):.2f} m, filter {np.median(err_pf):.2f} m")
    assert np.median(err_pf) < 0.5 * np.median(err_dead)
    assert err_pf[-10:].mean() < 2.0
