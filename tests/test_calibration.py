import numpy as np
import pandas as pd

from hdloc.metrics.calibration import loso_mae, pav


def test_pav_fits_a_monotone_series_and_never_decreases():
    x = np.arange(10.0)
    y = x**2
    assert np.allclose(pav(x, y)(x), y)
    noisy = y.copy()
    noisy[4], noisy[5] = noisy[5], noisy[4]
    assert np.all(np.diff(pav(x, noisy)(x)) >= -1e-12)


def test_loso_mae_perfect_alignment_has_zero_error():
    e = np.random.default_rng(0).uniform(0, 2, 60)
    df = pd.DataFrame({"scene": np.repeat(list("abcdef"), 10), "map_ap": 1 - e / 2,
                       "map_miou": 1 - e / 2, "off": e, "err": e})
    out = loso_mae(df, "off", "err")
    assert out["alignment"] < 1e-12
    assert out["predict 0"] > 0.5
    assert out["AP"] < 0.1
