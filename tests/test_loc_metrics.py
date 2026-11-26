import numpy as np

from hdloc.metrics.loc_metrics import pose_errors


def test_longitudinal_is_along_the_true_heading():
    gt = np.array([[0.0, 0.0, np.pi / 2]])  # facing +y
    e = pose_errors(np.array([[0.0, 2.0, np.pi / 2]]), gt)
    assert abs(e["lon"][0] - 2.0) < 1e-9 and abs(e["lat"][0]) < 1e-9


def test_left_of_the_car_is_positive_lateral():
    gt = np.array([[0.0, 0.0, np.pi / 2]])
    e = pose_errors(np.array([[-1.0, 0.0, np.pi / 2]]), gt)  # left of a north-facing car
    assert abs(e["lat"][0] - 1.0) < 1e-9 and abs(e["lon"][0]) < 1e-9
