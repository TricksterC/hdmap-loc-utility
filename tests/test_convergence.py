import numpy as np

from hdloc.metrics.convergence import convergence_summary, longest_run


def _err(d):
    d = np.asarray(d, float)
    return {"lat": d, "lon": np.zeros_like(d)}


def test_longest_run():
    assert longest_run([0, 1, 1, 0, 1, 1, 1, 0]) == 3
    assert longest_run([0, 0]) == 0
    assert longest_run([]) == 0


def test_small_error_is_not_lost():
    out = convergence_summary(_err(np.full(40, 0.3)))
    assert not out["pf_diverged"]
    assert out["pf_over_frac"] == 0.0
    assert abs(out["pf_final_err"] - 0.3) < 1e-9


def test_a_constant_two_metre_bias_is_not_lost():
    assert not convergence_summary(_err(np.full(40, 2.0)))["pf_diverged"]


def test_long_excursion_is_lost():
    d = np.full(40, 0.3)
    d[10:16] = 8.0
    out = convergence_summary(_err(d))
    assert out["pf_diverged"]
    assert abs(out["pf_over_frac"] - 6 / 40) < 1e-9


def test_short_spike_is_not_lost():
    d = np.full(40, 0.3)
    d[10:14] = 8.0  # four frames, one short of the rule
    assert not convergence_summary(_err(d))["pf_diverged"]


def test_uses_both_axes():
    err = {"lat": np.full(10, 4.0), "lon": np.full(10, 4.0)}  # norm 5.66 m
    assert convergence_summary(err)["pf_diverged"]
