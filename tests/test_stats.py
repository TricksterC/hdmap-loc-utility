import numpy as np

from hdloc.metrics.stats import spearman_ci


def test_spearman_ci_on_monotone_data():
    x = np.arange(40.0)
    rho, lo, hi = spearman_ci(x, x**2, groups=x % 8)
    assert abs(rho - 1.0) < 1e-12 and lo > 0.9 and hi <= 1.0 + 1e-12
