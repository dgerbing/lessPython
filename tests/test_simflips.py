import numpy as np
import pytest

from lessPy import simFlips
from lessPy.simFlips import SimFlipsResults


def test_basic_and_plot():
    r = simFlips(n=200, prob=0.5, seed=1)
    assert isinstance(r, SimFlipsResults)
    assert r.flips.size == 200 and set(np.unique(r.flips)) <= {0, 1}
    assert r.n_heads + r.n_tails == 200
    assert r.n_heads == int(r.flips.sum())
    assert set(r.plots) == {"flips"}


def test_running_mean_and_lln():
    r = simFlips(n=5000, prob=0.3, seed=2)
    # running mean is the cumulative proportion of heads
    exp = np.cumsum(r.flips) / np.arange(1, 5001)
    assert np.allclose(r.running_mean, exp)
    assert r.running_mean[0] == r.flips[0]
    assert abs(r.final_mean - 0.3) < 0.03          # converges
    r2 = simFlips(n=5000, prob=0.3, seed=2)
    assert np.array_equal(r.flips, r2.flips)        # reproducible


def test_errors():
    with pytest.raises(ValueError, match="number of flips"):
        simFlips()
    with pytest.raises(ValueError, match="prob"):
        simFlips(n=10, prob=1.5)
