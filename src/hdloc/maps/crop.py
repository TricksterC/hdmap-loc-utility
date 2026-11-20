import numpy as np

from hdloc.data.schema import Element, VectorMap


def crop_to_range(vmap: VectorMap, radius: float) -> VectorMap:
    """Keep only points within `radius` metres of the origin (ego frame)."""
    out = []
    for e in vmap.elements:
        keep = np.linalg.norm(e.points, axis=1) <= radius
        if keep.sum() >= 2:
            out.append(Element(e.cls, e.points[keep]))
    return VectorMap(out, vmap.frame)
