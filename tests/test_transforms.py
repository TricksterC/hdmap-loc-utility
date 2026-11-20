import numpy as np

from hdloc.maps.transforms import ego_to_global, global_to_ego, quat_to_yaw, wrap_angle


def test_round_trip():
    pts = np.random.default_rng(0).normal(size=(20, 2)) * 30
    ego = global_to_ego(pts, 5.0, -3.0, 1.2)
    back = ego_to_global(ego, 5.0, -3.0, 1.2)
    assert np.allclose(pts, back)


def test_point_ahead_is_positive_x():
    # car at origin facing +y (yaw = 90 deg); a point at (0, 10) is 10 m ahead
    ego = global_to_ego(np.array([[0.0, 10.0]]), 0.0, 0.0, np.pi / 2)
    assert np.allclose(ego, [[10.0, 0.0]], atol=1e-9)


def test_yaw_and_wrap():
    assert abs(quat_to_yaw((1, 0, 0, 0))) < 1e-12
    assert abs(wrap_angle(3 * np.pi) - np.pi) < 1e-9 or abs(wrap_angle(3 * np.pi) + np.pi) < 1e-9
