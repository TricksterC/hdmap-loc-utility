from __future__ import annotations

import numpy as np

from hdloc.data.schema import TRANSVERSE, VectorMap
from hdloc.degradation.dropout import add_false_positives, drop_classes, drop_elements
from hdloc.degradation.geometric import jitter_map, rotate_map, shift_map

KINDS = (
    "lateral_shift",
    "longitudinal_shift",
    "rotation_deg",
    "jitter_sigma",
    "element_dropout",
    "drop_transverse",
    "false_dividers",
    "false_stop_lines",
    "range_m",
    "drop_dividers",
    "drop_all",
    "drift_scale_error",
    "drift_yaw_bias",
)


def degrade(
    vmap: VectorMap, kind: str, value: float, rng: np.random.Generator, radius: float
) -> VectorMap:
    """Apply one named degradation of size `value` to an ego-frame observed map."""
    if kind == "lateral_shift":
        return shift_map(vmap, dy=value)
    if kind == "longitudinal_shift":
        return shift_map(vmap, dx=value)
    if kind == "rotation_deg":
        return rotate_map(vmap, np.radians(value))
    if kind == "jitter_sigma":
        return jitter_map(vmap, value, rng)
    if kind == "element_dropout":
        return drop_elements(vmap, value, rng)
    if kind == "drop_transverse":
        return drop_classes(vmap, TRANSVERSE) if value else vmap
    if kind == "false_dividers":
        return add_false_positives(vmap, value, rng, radius, "divider")
    if kind == "false_stop_lines":
        return add_false_positives(vmap, value, rng, radius, "stop_line")
    if kind == "range_m":
        return vmap  # the range cut itself is applied by observe(); see evaluate()
    if kind == "drop_dividers":
        return drop_classes(vmap, ("divider", "boundary")) if value else vmap
    if kind == "drop_all":
        return VectorMap([], vmap.frame) if value else vmap
    if kind in ("drift_scale_error", "drift_yaw_bias"):
        return vmap  # these change the odometry, not the map; see evaluate()
    raise ValueError(f"unknown degradation: {kind}")
