import numpy as np

from hdloc.data.schema import Element, VectorMap
from hdloc.maps.crop import crop_to_range


def test_crop_keeps_only_points_in_range():
    line = Element("divider", np.array([[0.0, 0.0], [10.0, 0.0], [60.0, 0.0]]))
    out = crop_to_range(VectorMap([line], frame="ego"), 50.0)
    assert len(out.elements) == 1
    assert out.elements[0].points.shape == (2, 2)


def test_crop_drops_far_elements():
    far = Element("stop_line", np.array([[80.0, 0.0], [90.0, 0.0]]))
    assert crop_to_range(VectorMap([far], frame="ego"), 50.0).elements == []
