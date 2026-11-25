import numpy as np
import pytest

from hdloc.data.schema import Element, VectorMap
from hdloc.degradation.orientation import split_by_orientation
from hdloc.metrics.observability import class_information


def _map():
    along = Element("divider", np.array([[0.0, 0.0], [10.0, 0.0]]))
    across = Element("stop_line", np.array([[5.0, -3.0], [5.0, 3.0]]))
    ring = Element("ped_crossing", np.array([[0, 0], [0, 6], [1, 6], [1, 0], [0, 0]], float))
    return VectorMap([along, across, ring], "ego")


def _length(vmap):
    return sum(np.linalg.norm(np.diff(e.points, axis=0), axis=1).sum() for e in vmap.elements)


def test_keep_along_has_no_longitudinal_constraint():
    info = class_information(split_by_orientation(_map(), "along"))
    assert all(abs(v[0]) < 1e-9 for v in info.values())
    assert sum(v[1] for v in info.values()) > 0


def test_keep_across_has_no_lateral_constraint():
    info = class_information(split_by_orientation(_map(), "across"))
    assert all(abs(v[1]) < 1e-9 for v in info.values())
    assert sum(v[0] for v in info.values()) > 0


def test_the_two_halves_partition_the_map():
    m = _map()
    parts = split_by_orientation(m, "along"), split_by_orientation(m, "across")
    assert abs(_length(parts[0]) + _length(parts[1]) - _length(m)) < 1e-9


def test_bad_keep_raises():
    with pytest.raises(ValueError):
        split_by_orientation(_map(), "diagonal")
