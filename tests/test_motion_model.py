import numpy as np

from hdloc.localization.baselines import odometry_only
from hdloc.localization.motion_model import DriftModel, inject_drift, relative_increments
from hdloc.maps.transforms import wrap_angle


def test_increments_reproduce_trajectory():
    rng = np.random.default_rng(1)
    poses = np.cumsum(rng.normal(size=(20, 3)) * [1.0, 1.0, 0.1], axis=0)
    poses[:, 2] = wrap_angle(poses[:, 2])
    rec = odometry_only(poses[0], relative_increments(poses))
    assert np.allclose(rec[:, :2], poses[:, :2], atol=1e-9)
    assert np.allclose(wrap_angle(rec[:, 2] - poses[:, 2]), 0.0, atol=1e-9)


def test_scale_error_stretches_distance():
    incs = np.array([[2.0, 0.0, 0.0]] * 10)
    out = inject_drift(incs, DriftModel(scale_error=0.1), np.random.default_rng(0))
    assert np.allclose(out[:, 0], 2.2)
