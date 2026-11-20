from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CLASSES = ("divider", "boundary", "ped_crossing", "stop_line")
TRANSVERSE = ("ped_crossing", "stop_line")


@dataclass
class Element:
    """One map element: a class label and an (N, 2) array of points."""

    cls: str
    points: np.ndarray

    def __post_init__(self):
        if self.cls not in CLASSES:
            raise ValueError(f"unknown class: {self.cls}")
        self.points = np.asarray(self.points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")


@dataclass
class VectorMap:
    """A set of elements expressed in one frame ('global' or 'ego')."""

    elements: list[Element] = field(default_factory=list)
    frame: str = "global"

    def of_class(self, cls: str) -> list[Element]:
        return [e for e in self.elements if e.cls == cls]

    def transformed(self, fn, frame: str) -> VectorMap:
        """Return a new map with fn(points) applied to every element."""
        return VectorMap([Element(e.cls, fn(e.points)) for e in self.elements], frame)
