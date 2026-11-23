import numpy as np

from hdloc.localization.resampling import effective_sample_size, systematic_resample


def test_ess_of_uniform_weights_is_n():
    assert abs(effective_sample_size(np.full(100, 0.01)) - 100.0) < 1e-9


def test_resampling_favours_heavy_particle():
    w = np.array([0.97, 0.01, 0.01, 0.01])
    idx = systematic_resample(w, np.random.default_rng(0))
    assert (idx == 0).sum() >= 3
