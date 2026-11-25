from __future__ import annotations

import numpy as np

from hdloc.data.schema import Element, VectorMap


def split_by_orientation(vmap: VectorMap, keep: str, max_angle_deg: float = 45.0) -> VectorMap:
    """Keep only segments by direction relative to the ego x axis (all classes).

    keep="along":  segments within max_angle of the road direction (constrain lateral position).
    keep="across": segments within max_angle of the cross-road direction (constrain longitudinal).
    Every kept segment becomes its own two-point element.
    """
    if keep not in ("along", "across"):
        raise ValueError("keep must be 'along' or 'across'")
    out = []
    for e in vmap.elements:
        p = e.points
        d = np.diff(p, axis=0)
        ang = np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0])))
        mask = ang <= max_angle_deg if keep == "along" else ang >= 90.0 - max_angle_deg
        mask &= np.linalg.norm(d, axis=1) > 1e-9
        for i in np.nonzero(mask)[0]:
            out.append(Element(e.cls, p[i : i + 2]))
    return VectorMap(out, vmap.frame)
