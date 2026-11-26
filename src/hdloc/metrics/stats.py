from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def spearman_ci(x, y, groups=None, n_boot=1000, seed=0, alpha=0.05):
    """Spearman rho with a cluster-bootstrap interval (resamples whole groups, e.g. scenes)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho = float(spearmanr(x, y)[0])
    g = np.asarray(groups)[ok] if groups is not None else np.arange(len(x))
    members = [np.nonzero(g == lab)[0] for lab in np.unique(g)]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        r = spearmanr(x[idx], y[idx])[0]
        if np.isfinite(r):
            boots.append(r)
    if not boots:
        return rho, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return rho, float(lo), float(hi)
