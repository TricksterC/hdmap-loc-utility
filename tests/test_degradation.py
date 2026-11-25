import numpy as np
import pytest

from hdloc.data.schema import Element, VectorMap
from hdloc.degradation.dropout import add_false_positives, drop_classes, drop_elements
from hdloc.degradation.geometric import jitter_map, rotate_map, shift_map
from hdloc.degradation.pipeline import degrade


def _map():
    rng = np.random.default_rng(0)
    els = [Element("divider", rng.normal(size=(5, 2)) * 10) for _ in range(200)]
    els += [Element("stop_line", rng.normal(size=(2, 2)) * 10) for _ in range(20)]
    return VectorMap(els, frame="ego")


def test_shift_moves_every_point_exactly_half_a_metre():
    m = _map()
    out = shift_map(m, dy=0.5)
    for a, b in zip(m.elements, out.elements):
        assert np.allclose(np.linalg.norm(b.points - a.points, axis=1), 0.5)
        assert np.allclose(b.points - a.points, [0.0, 0.5])


def test_rotation_preserves_distances_and_turns_by_angle():
    m = VectorMap([Element("divider", np.array([[1.0, 0.0], [2.0, 0.0]]))], frame="ego")
    out = rotate_map(m, np.pi / 2)
    assert np.allclose(out.elements[0].points, [[0.0, 1.0], [0.0, 2.0]], atol=1e-12)


def test_jitter_has_requested_std_and_is_seeded():
    m = _map()
    a = jitter_map(m, 0.3, np.random.default_rng(1))
    b = jitter_map(m, 0.3, np.random.default_rng(1))
    diff = np.concatenate([(x.points - y.points).ravel() for x, y in zip(a.elements, m.elements)])
    assert abs(diff.std() - 0.3) < 0.03
    assert all(np.array_equal(x.points, y.points) for x, y in zip(a.elements, b.elements))


def test_element_dropout_removes_about_the_requested_fraction():
    out = drop_elements(_map(), 0.3, np.random.default_rng(0))
    removed = 1 - len(out.elements) / 220
    assert 0.2 < removed < 0.4


def test_class_dropout_removes_only_that_class():
    out = drop_classes(_map(), ("stop_line",))
    assert len(out.of_class("stop_line")) == 0
    assert len(out.of_class("divider")) == 200


def test_false_positives_are_added_inside_range():
    out = add_false_positives(_map(), 5, np.random.default_rng(0), 30.0, "stop_line")
    assert len(out.elements) == 225
    new = out.elements[-5:]
    assert all(e.cls == "stop_line" for e in new)
    assert all(np.linalg.norm(e.points, axis=1).max() <= 30.0 + 5.0 + 1e-6 for e in new)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        degrade(_map(), "nonsense", 1.0, np.random.default_rng(0), 50.0)


def test_new_ablation_kinds():
    m = _map()
    rng = np.random.default_rng(0)
    assert degrade(m, "drop_all", 1, rng, 50.0).elements == []
    out = degrade(m, "drop_dividers", 1, rng, 50.0)
    assert len(out.of_class("divider")) == 0 and len(out.of_class("stop_line")) == 20
    assert len(degrade(m, "drop_dividers", 0, rng, 50.0).elements) == 220
