import numpy as np
import pytest

from lessPy import simCLT
from lessPy.simCLT import SimCLTResults


def test_population_formulas():
    r = simCLT(ns=500, n=20, p1=100, p2=15, dist="normal", seed=1)
    assert (r.mu, r.sigma) == (100, 15)
    u = simCLT(ns=500, n=20, p1=0, p2=10, dist="uniform", seed=1)
    assert u.mu == 5 and abs(u.sigma - 10 / np.sqrt(12)) < 1e-9
    ln = simCLT(ns=500, n=20, p1=0, p2=1, dist="lognormal", seed=1)
    assert abs(ln.mu - np.exp(0.5)) < 1e-9
    assert abs(ln.median - 1.0) < 1e-9 and ln.skew > 0
    an = simCLT(ns=500, n=20, p1=0, p2=10, dist="antinormal",
                seed=1)
    assert an.mu == 5 and an.sigma is None


def test_shape_and_plots():
    r = simCLT(ns=300, n=25, dist="normal", seed=7)
    assert isinstance(r, SimCLTResults)
    assert r.ymean.size == 300 and r.ysd.size == 300
    assert set(r.plots) == {"population", "sampling"}
    for f in r.plots.values():
        assert len(f.data) >= 1


def test_clt_convergence_and_seed():
    r = simCLT(ns=4000, n=40, p1=0, p2=1, dist="uniform", seed=3)
    # sd of the sample means ~ sigma / sqrt(n)
    assert abs(r.sd_of_means - r.sigma / np.sqrt(40)) < 0.02
    r2 = simCLT(ns=4000, n=40, p1=0, p2=1, dist="uniform", seed=3)
    assert np.array_equal(r.ymean, r2.ymean)     # seed reproducible


def test_errors():
    with pytest.raises(ValueError, match="ns"):
        simCLT(n=10, dist="normal")
    with pytest.raises(ValueError, match="size"):
        simCLT(ns=10, dist="normal")
    with pytest.raises(ValueError, match="dist"):
        simCLT(ns=10, n=10, dist="cauchy")
    with pytest.raises(ValueError, match="must be 0"):
        simCLT(ns=10, n=10, p1=2, p2=10, dist="antinormal")
