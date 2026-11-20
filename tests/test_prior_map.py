import numpy as np

from hdloc.data.schema import Element, VectorMap
from hdloc.maps.distance_transform import distance_transform
from hdloc.maps.prior_map import build_prior
from hdloc.maps.rasterize import Grid, rasterize_class


def _horizontal_line_map():
    # a divider along y = 5.05 from x = 1 to x = 9
    return VectorMap([Element("divider", np.array([[1.0, 5.05], [9.0, 5.05]]))])


def test_world_to_pix_pixel_centres():
    g = Grid.from_bounds(0, 0, 10, 10, 0.1)
    assert g.shape == (100, 100)
    # centre of pixel (row 50, col 30) is at world (3.05, 5.05)
    uv = g.world_to_pix(np.array([[3.05, 5.05]]))
    assert np.allclose(uv, [[30.0, 50.0]], atol=1e-6)


def test_rasterize_draws_a_thin_line():
    g = Grid.from_bounds(0, 0, 10, 10, 0.1)
    img = rasterize_class(_horizontal_line_map(), "divider", g)
    rows, cols = np.nonzero(img)
    assert len(np.unique(rows)) == 1          # a single row
    assert abs(len(cols) - 80) <= 2           # about 8 m / 0.1 m
    assert rasterize_class(_horizontal_line_map(), "stop_line", g).sum() == 0


def test_distance_grows_linearly_away_from_line():
    g = Grid.from_bounds(0, 0, 10, 10, 0.1)
    prior = build_prior(_horizontal_line_map(), g, max_dist=20.0)
    pts = np.array([[5.0, 5.05], [5.0, 6.05], [5.0, 7.05], [5.0, 8.05]])
    d = prior.distance("divider", pts)
    assert np.allclose(d, [0.0, 1.0, 2.0, 3.0], atol=0.15)


def test_distance_is_capped_and_outside_grid_is_filled():
    g = Grid.from_bounds(0, 0, 10, 10, 0.1)
    prior = build_prior(_horizontal_line_map(), g, max_dist=2.0)
    far = prior.distance("divider", np.array([[5.0, 9.5]]))
    outside = prior.distance("divider", np.array([[50.0, 50.0]]))
    assert far[0] == 2.0 and outside[0] == 2.0


def test_missing_class_returns_max_dist():
    g = Grid.from_bounds(0, 0, 10, 10, 0.1)
    prior = build_prior(_horizontal_line_map(), g, max_dist=7.0)
    assert np.allclose(prior.distance("stop_line", np.array([[5.0, 5.0]])), 7.0)


def test_empty_raster():
    dt = distance_transform(np.zeros((5, 5), dtype=np.uint8), 0.1, max_dist=3.0)
    assert (dt == 3.0).all()
