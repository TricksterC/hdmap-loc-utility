"""When does a filter run count as lost? One rule, fixed before the held-out run.

A run is flagged `pf_diverged` when its position error (sideways and forward combined) stays
above LOST_M metres for LOST_FRAMES keyframes in a row (about 2.5 s at nuScenes' 2 Hz keyframes).
LOST_M is 5 m: more than twice the largest rigid offset we inject (2 m), so a map that is merely
shifted counts as "biased", not "lost". The rule only adds columns; no run is dropped from a CSV.
"""

from __future__ import annotations

import numpy as np

LOST_M = 5.0
LOST_FRAMES = 5
FINAL_FRAMES = 5


def longest_run(mask) -> int:
    """Length of the longest stretch of consecutive True values."""
    best = cur = 0
    for v in np.asarray(mask, bool):
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def convergence_summary(
    err: dict, threshold: float = LOST_M, min_frames: int = LOST_FRAMES
) -> dict:
    """Columns describing whether a run got lost. `err` is the output of pose_errors."""
    d = np.hypot(np.asarray(err["lat"], float), np.asarray(err["lon"], float))
    over = d > threshold
    return {
        "pf_diverged": bool(longest_run(over) >= min_frames),
        "pf_over_frac": float(over.mean()),
        "pf_final_err": float(np.median(d[-FINAL_FRAMES:])),
    }
