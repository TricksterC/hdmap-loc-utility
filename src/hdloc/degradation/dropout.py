from __future__ import annotations

import numpy as np

from hdloc.data.schema import TRANSVERSE, Element, VectorMap
from hdloc.maps.observe import resample_polyline


def drop_elements(vmap: VectorMap, frac: float, rng: np.random.Generator) -> VectorMap:
    """Remove each element independently with probability `frac` (missed detections)."""
    keep = rng.random(len(vmap.elements)) >= frac
    return VectorMap([e for e, k in zip(vmap.elements, keep) if k], vmap.frame)


def drop_classes(vmap: VectorMap, classes) -> VectorMap:
    """Remove every element of the given classes (e.g. all crosswalks and stop lines)."""
    drop = set(classes)
    return VectorMap([e for e in vmap.elements if e.cls not in drop], vmap.frame)


def add_false_positives(
    vmap: VectorMap,
    n: int,
    rng: np.random.Generator,
    radius: float,
    cls: str = "divider",
    length: float | None = None,
) -> VectorMap:
    """Add `n` hallucinated straight segments at random places within `radius`."""
    if length is None:
        length = 7.0 if cls in TRANSVERSE else 10.0
    elements = list(vmap.elements)
    for _ in range(int(n)):
        r = radius * np.sqrt(rng.random())
        theta = rng.uniform(-np.pi, np.pi)
        centre = np.array([r * np.cos(theta), r * np.sin(theta)])
        ang = rng.uniform(0.0, np.pi)
        half = 0.5 * length * np.array([np.cos(ang), np.sin(ang)])
        pts = resample_polyline(np.stack([centre - half, centre + half]), 1.0)
        elements.append(Element(cls, pts))
    return VectorMap(elements, vmap.frame)
