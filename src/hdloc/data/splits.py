"""Choosing held-out scenes. Pure functions, so the rule can be tested without nuScenes.

Nothing in this project is trained, so a scene-disjoint split keeps the held-out run honest.
The dev scenes already cover all four nuScenes locations, so a location-disjoint split is not
possible. We sample held-out scenes evenly across locations instead.
"""

from __future__ import annotations

import numpy as np


def select_heldout(
    rows: list[dict],
    exclude: set[str],
    per_location: int,
    seed: int,
    min_len: float = 10.0,
    min_samples: int = 30,
) -> tuple[list[str], dict[str, int]]:
    """Pick up to `per_location` moving scenes from each location, never one in `exclude`.

    rows: dicts with name, location, path_len, n_samples. Deterministic for a given seed.
    Returns (scene names grouped by location, number of candidates per location).
    """
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    available: dict[str, int] = {}
    for loc in sorted({r["location"] for r in rows}):
        cands = sorted(
            r["name"]
            for r in rows
            if r["location"] == loc
            and r["name"] not in exclude
            and r["path_len"] > min_len
            and r["n_samples"] >= min_samples
        )
        available[loc] = len(cands)
        k = min(per_location, len(cands))
        pick = rng.choice(len(cands), size=k, replace=False) if k else []
        chosen += sorted(cands[i] for i in pick)
    return chosen, available
