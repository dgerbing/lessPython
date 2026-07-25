import numpy as np
import pytest

from lessPy import simMeans
from lessPy.simMeans import SimMeansResults


def test_basic_shape_and_plot():
    r = simMeans(ns=50, n=20, mu=10, sigma=2, seed=1)
    assert isinstance(r, SimMeansResults)
    assert (r.mu, r.sigma, r.n_samples, r.n) == (10, 2, 50, 20)
    assert r.ymean.size == 50 and r.ysd.size == 50
    assert set(r.plots) == {"means"}
    assert len(r.plots["means"].data) == 1        # scatter trace


def test_sort_and_order():
    r = simMeans(ns=30, n=15, seed=5)             # sort default
    assert np.all(np.diff(r.ymean) >= 0)          # ascending means
    assert sorted(r.order.tolist()) == list(range(30))
    u = simMeans(ns=30, n=15, seed=5, sort=False)
    assert list(u.order) == list(range(30))       # original order


def test_se_and_seed():
    r = simMeans(ns=4000, n=25, mu=0, sigma=5, seed=8)
    # empirical SE ~ sigma / sqrt(n)
    assert abs(r.se - 5 / np.sqrt(25)) < 0.05
    assert abs(r.mean_of_means) < 0.1
    r2 = simMeans(ns=4000, n=25, mu=0, sigma=5, seed=8)
    assert np.array_equal(r.ymean, r2.ymean)      # reproducible


def test_set_mu_and_errors():
    r = simMeans(ns=10, n=10, seed=2, set_mu=True)
    assert 0 <= r.mu <= 100 and 1 <= r.sigma <= 25
    with pytest.raises(ValueError, match="ns"):
        simMeans(n=10)
    with pytest.raises(ValueError, match="size"):
        simMeans(ns=10)
    with pytest.raises(ValueError, match="sigma"):
        simMeans(ns=10, n=10, sigma=-1)
