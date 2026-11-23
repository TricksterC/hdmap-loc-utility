from __future__ import annotations

import numpy as np


def effective_sample_size(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(weights**2))


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)
