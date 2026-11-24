import numpy as np

from hdloc.data.schema import Element, VectorMap
from hdloc.experiments.common import SceneData, evaluate
from hdloc.localization.motion_model import DriftModel
from hdloc.localization.particle_filter import FilterConfig
from hdloc.maps.observe import densify_map
from hdloc.maps.prior_map import build_prior
from hdloc.maps.rasterize import Grid


def _scene():
    els = [Element("divider", np.array([[0.0, y], [300.0, y]])) for y in (-1.75, 1.75)]
    for x in np.arange(25.0, 300.0, 25.0):
        els.append(Element("stop_line", np.array([[x, -3.5], [x, 3.5]])))
    vmap = densify_map(VectorMap(els), 1.0)
    grid = Grid.from_bounds(-20, -30, 340, 30, 0.2)
    poses = np.array([[5.0 + 2.0 * t, 0.0, 0.0] for t in range(60)])
    return SceneData("road", "synthetic", poses, vmap, build_prior(vmap, grid))


def test_transverse_elements_bound_longitudinal_error():
    scene = _scene()
    drift = DriftModel(scale_error=0.05, yaw_bias=0.004, trans_noise=0.02, yaw_noise=0.002)
    cfg = FilterConfig(n_particles=500)
    full = evaluate(scene, "drop_transverse", 0, 0, cfg, drift)
    cut = evaluate(scene, "drop_transverse", 1, 0, cfg, drift)
    print(f"lon median: with {full['pf_lon_med']:.2f} m, without {cut['pf_lon_med']:.2f} m")
    assert full["transverse_frac"] == 1.0 and cut["transverse_frac"] == 0.0
    assert cut["pf_lon_med"] > 2 * full["pf_lon_med"]
    assert cut["pf_lat_med"] < 0.5  # dividers still pin the lateral position
