import numpy as np
import pytest
from scipy.stats import t as t_dist

from lessPy import simCImean
from lessPy.simCImean import SimCImeanResults


def test_basic_and_interval_math():
    r = simCImean(ns=40, n=30, mu=100, sigma=15, seed=1)
    assert isinstance(r, SimCImeanResults)
    assert r.tcut == t_dist.ppf(0.975, 29)
    # interval is symmetric about the mean and hit/miss consistent
    assert np.allclose(r.ub - r.ymean, r.ymean - r.lb)
    exp_hit = (r.lb < 100) & (r.ub > 100)
    assert np.array_equal(r.hit, exp_hit)
    assert r.n_miss == int((~r.hit).sum())
    assert r.hit.sum() + r.n_miss == 40
    assert set(r.plots) == {"ci"}


def test_coverage_and_seed():
    r = simCImean(ns=20000, n=25, mu=0, sigma=1, cl=0.95, seed=7)
    assert abs((100 - r.miss_rate) - 95) < 1.5      # ~95% coverage
    r2 = simCImean(ns=20000, n=25, mu=0, sigma=1, cl=0.95, seed=7)
    assert np.array_equal(r.ymean, r2.ymean)         # reproducible
    # wider level -> wider intervals -> fewer misses
    lo = simCImean(ns=8000, n=20, cl=0.80, seed=3).miss_rate
    hi = simCImean(ns=8000, n=20, cl=0.99, seed=3).miss_rate
    assert hi < lo


def test_errors():
    with pytest.raises(ValueError, match="ns"):
        simCImean(n=10)
    with pytest.raises(ValueError, match="size"):
        simCImean(ns=10)
    with pytest.raises(ValueError, match="sigma"):
        simCImean(ns=10, n=10, sigma=-1)
    with pytest.raises(ValueError, match="cl"):
        simCImean(ns=10, n=10, cl=1.2)
