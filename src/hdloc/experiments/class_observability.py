from __future__ import annotations

import pandas as pd

from hdloc.experiments.common import SceneData, evaluate
from hdloc.localization.motion_model import DriftModel
from hdloc.localization.particle_filter import FilterConfig
from hdloc.metrics.observability import route_coverage

# (label, degradation kind, value). With the boundary class empty, "no transverse" equals
# "dividers_only" and "no dividers" equals "transverse_only", so each pair is run once.
ABLATIONS = (
    ("all", "drop_transverse", 0),
    ("dividers_only", "drop_transverse", 1),
    ("transverse_only", "drop_dividers", 1),
    ("along_only", "drop_across", 1),
    ("across_only", "drop_along", 1),
    ("none", "drop_all", 1),
)


def coverage_table(scenes: list[SceneData], radii=(30.0, 50.0)) -> pd.DataFrame:
    rows = []
    for s in scenes:
        for r in radii:
            cov = route_coverage(s.vmap, s.poses, r)
            rows.append({"scene": s.name, "radius": r, "path_len": s.path_len, **cov})
    return pd.DataFrame(rows)


def run_ablation(
    scenes: list[SceneData],
    filter_cfg: FilterConfig,
    drift_cfg: dict,
    scale_errors,
    seeds,
    radius: float = 50.0,
) -> pd.DataFrame:
    rows = []
    for scene in scenes:
        for se in scale_errors:
            drift = DriftModel(**{**drift_cfg, "scale_error": se})
            for label, kind, value in ABLATIONS:
                for seed in seeds:
                    res = evaluate(scene, kind, value, seed, filter_cfg, drift, radius)
                    rows.append({"scene": scene.name, "path_len": scene.path_len,
                                 "scale_error": se, "condition": label, "seed": seed, **res})
        print("finished", scene.name, flush=True)
    return pd.DataFrame(rows)
