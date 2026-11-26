import numpy as np

from hdloc.data.schema import Element, VectorMap
from hdloc.maps.observe import densify_map
from hdloc.metrics.alignment import (
    AlignmentAccumulator,
    TruthIndex,
    align_offset,
    prediction_points,
)


def _road(with_across=True):
    els = [Element("divider", np.array([[-50.0, y], [50.0, y]])) for y in (-1.75, 1.75)]
    if with_across:
        for x in (-30.0, -10.0, 10.0, 30.0):
            els.append(Element("stop_line", np.array([[x, -3.5], [x, 3.5]])))
    return densify_map(VectorMap(els, "ego"), 1.0)


def _shifted(vmap, dx=0.0, dy=0.0):
    return vmap.transformed(lambda p: p + np.array([dx, dy]), "ego")


def _rotated(vmap, deg):
    a = np.radians(deg)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return vmap.transformed(lambda p: p @ r.T, "ego")


def _align(truth, pred):
    return align_offset(TruthIndex(truth), prediction_points(pred))


def test_identical_maps_have_zero_offset():
    m = _road()
    a = _align(m, m)
    assert np.abs(a.offset).max() < 1e-6


def test_lateral_shift_is_recovered():
    m = _road()
    a = _align(m, _shifted(m, dy=0.5))
    assert abs(a.offset[1] - 0.5) < 0.03 and abs(a.offset[0]) < 0.03


def test_longitudinal_shift_is_recovered_when_lines_cross_the_road():
    m = _road(with_across=True)
    a = _align(m, _shifted(m, dx=0.4))
    assert abs(a.offset[0] - 0.4) < 0.03


def test_unconstrained_direction_stays_at_zero():
    m = _road(with_across=False)  # dividers only: nothing can see a forward shift
    a = _align(m, _shifted(m, dx=0.4))
    assert abs(a.offset[0]) < 0.05


def test_rotation_is_recovered():
    m = _road()
    a = _align(m, _rotated(m, 2.0))
    assert abs(np.degrees(a.offset[2]) - 2.0) < 0.1


def test_random_jitter_gives_a_small_offset():
    m = _road()
    rng = np.random.default_rng(0)
    noisy = m.transformed(lambda p: p + rng.normal(0.0, 0.3, p.shape), "ego")
    assert np.abs(_align(m, noisy).offset[:2]).max() < 0.1


def test_distant_false_elements_do_not_drag_the_offset():
    m = _road()
    fake = Element("divider", np.array([[-20.0, 9.0], [20.0, 9.0]]))
    pred = VectorMap(list(m.elements) + [fake], "ego")
    assert np.abs(_align(m, pred).offset[:2]).max() < 0.1


def test_empty_prediction_is_invalid_not_zero():
    m = _road()
    assert _align(m, VectorMap([], "ego")) is None
    acc = AlignmentAccumulator()
    acc.add(m, VectorMap([], "ego"))
    r = acc.result()
    assert r["align_valid"] == 0.0 and np.isnan(r["align_lat_med"])


def test_accumulator_reports_median_absolute_offsets():
    m = _road()
    acc = AlignmentAccumulator()
    for _ in range(3):
        acc.add(m, _shifted(m, dy=0.25))
    r = acc.result()
    assert abs(r["align_lat_med"] - 0.25) < 0.03 and r["align_valid"] == 1.0
