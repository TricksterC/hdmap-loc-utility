import numpy as np
import pytest

from hdloc.data.schema import Element, VectorMap


def test_element_validates():
    with pytest.raises(ValueError):
        Element("bogus", np.zeros((3, 2)))
    with pytest.raises(ValueError):
        Element("divider", np.zeros((3, 3)))


def test_transformed_returns_new_map():
    m = VectorMap([Element("divider", np.zeros((2, 2)))])
    m2 = m.transformed(lambda p: p + 1.0, frame="ego")
    assert m2.frame == "ego" and m.frame == "global"
    assert np.allclose(m2.elements[0].points, 1.0)
    assert np.allclose(m.elements[0].points, 0.0)
