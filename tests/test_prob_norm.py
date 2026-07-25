import math

import pytest
from scipy.stats import norm

from lessPy import prob_norm
from lessPy.prob_norm import ProbNormResults


def test_probabilities_match_theory():
    r = prob_norm(lo=-1, hi=1)
    assert isinstance(r, ProbNormResults)
    assert math.isclose(r.prob, 0.6826894921, abs_tol=1e-9)
    assert math.isclose(prob_norm(lo=100, hi=115, mu=100,
                                  sigma=15).prob, 0.3413447461,
                        abs_tol=1e-9)
    assert set(r.plots) == {"norm"}


def test_open_bounds():
    # hi only -> lower tail up to hi
    assert math.isclose(prob_norm(hi=1.96).prob, norm.cdf(1.96),
                        abs_tol=1e-9)
    # lo only -> upper tail; lo at mu -> 0.5
    assert math.isclose(prob_norm(lo=0).prob, 0.5, abs_tol=1e-9)
    # no bounds -> essentially the whole area
    assert prob_norm().prob > 0.999999


def test_z_axis_annotations():
    # non-standard normal -> z-score annotations (-4..4)
    fig = prob_norm(lo=0, hi=1, mu=5, sigma=2).plots["norm"]
    zs = sorted(int(a.text) for a in fig.layout.annotations)
    assert zs == list(range(-4, 5))
    # standard normal -> z suppressed
    fig0 = prob_norm(lo=-1, hi=1).plots["norm"]
    assert len(fig0.layout.annotations) == 0


def test_errors():
    with pytest.raises(ValueError, match="cannot be larger"):
        prob_norm(lo=5, hi=1)
    with pytest.raises(ValueError, match="sigma"):
        prob_norm(lo=0, hi=1, sigma=0)
