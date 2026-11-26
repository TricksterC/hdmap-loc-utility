import numpy as np

from hdloc.data.schema import Element, VectorMap
from hdloc.metrics.map_metrics import MapMetricAccumulator


def _line(y, cls="divider"):
    return Element(cls, np.array([[0.0, y], [40.0, y]]))


def _score(truth, pred, radius=50.0):
    acc = MapMetricAccumulator(radius)
    acc.add(truth, pred)
    return acc.result()


def _map(*lines):
    return VectorMap(list(lines), "ego")


def test_identical_maps_score_one():
    m = _map(_line(0.0), _line(4.0))
    r = _score(m, m)
    assert r["map_ap"] == 1.0 and abs(r["map_miou"] - 1.0) < 1e-9


def test_shift_crosses_the_distance_thresholds():
    truth = _map(_line(0.0))
    small = _score(truth, _map(_line(0.25)))
    mid = _score(truth, _map(_line(0.75)))
    big = _score(truth, _map(_line(2.0)))
    assert small["map_ap"] == 1.0
    assert abs(mid["map_ap"] - 2 / 3) < 1e-9 and mid["map_ap_t05"] == 0.0
    assert big["map_ap"] == 0.0


def test_missing_and_extra_elements():
    four = _map(*[_line(y) for y in (-6.0, -2.0, 2.0, 6.0)])
    half = _map(_line(-6.0), _line(-2.0))
    extra = _map(*[_line(y) for y in (-6.0, -2.0, 2.0, 6.0, 20.0, -20.0)])
    assert abs(_score(four, half)["map_ap"] - 0.5) < 1e-9
    assert abs(_score(four, extra)["map_ap"] - 4 / 6) < 1e-9


def test_iou_of_shifted_thick_lines():
    r = _score(_map(_line(0.0)), _map(_line(0.5)))
    assert 0.2 < r["map_miou"] < 0.5  # two 1 m wide bands offset by 0.5 m: about 1/3


def test_class_without_ground_truth_is_skipped_for_ap_but_costs_iou():
    r = _score(_map(_line(0.0)), _map(_line(0.0), _line(5.0, "stop_line")))
    assert r["map_ap"] == 1.0 and r["map_miou"] < 1.0


def test_empty_maps_score_one():
    r = _score(_map(), _map())
    assert r["map_ap"] == 1.0 and r["map_miou"] == 1.0
