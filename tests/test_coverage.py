import numpy as np

from hdloc.data.schema import Element, VectorMap
from hdloc.maps.observe import densify_map
from hdloc.metrics.observability import route_coverage


def test_coverage_shrinks_with_radius():
    els = [Element("divider", np.array([[0.0, y], [300.0, y]])) for y in (-1.75, 1.75)]
    for x in np.arange(25.0, 300.0, 25.0):
        els.append(Element("stop_line", np.array([[x, -3.5], [x, 3.5]])))
    vmap = densify_map(VectorMap(els), 1.0)
    poses = np.array([[5.0 + 2.0 * t, 0.0, 0.0] for t in range(60)])
    big, small = route_coverage(vmap, poses, 50.0), route_coverage(vmap, poses, 5.0)
    assert big["stop_line_present"] == 1.0 and big["divider_present"] == 1.0
    assert small["stop_line_present"] < 1.0
    assert big["lon_transverse_ge_5m"] >= small["lon_transverse_ge_5m"]
