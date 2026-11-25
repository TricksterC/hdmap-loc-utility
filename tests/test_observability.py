import numpy as np

from hdloc.data.schema import Element, VectorMap
from hdloc.metrics.observability import class_information


def _info(cls, pts):
    return class_information(VectorMap([Element(cls, np.array(pts, float))], "ego"))[cls]


def test_line_along_road_constrains_lateral_only():
    lon, lat = _info("divider", [[0, 0], [10, 0]])
    assert abs(lon) < 1e-9 and abs(lat - 10.0) < 1e-9


def test_line_across_road_constrains_longitudinal_only():
    lon, lat = _info("stop_line", [[5, -3], [5, 3]])
    assert abs(lon - 6.0) < 1e-9 and abs(lat) < 1e-9


def test_closed_rectangle_constrains_both():
    lon, lat = _info("stop_line", [[0, 0], [0, 6], [1, 6], [1, 0], [0, 0]])
    assert abs(lon - 12.0) < 1e-9 and abs(lat - 2.0) < 1e-9
