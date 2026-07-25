import math

import pytest
from scipy.stats import t as t_dist

from lessPy import prob_tcut
from lessPy.prob_tcut import ProbTcutResults


def test_cutoff_matches_qt():
    r = prob_tcut(df=30)
    assert isinstance(r, ProbTcutResults)
    # upper two-tailed t critical value = qt(0.975, 30)
    assert math.isclose(r.cutoff, 2.042272456, abs_tol=1e-8)
    assert math.isclose(r.cutoff, t_dist.ppf(0.975, 30),
                        abs_tol=1e-12)
    assert set(r.plots) == {"tcut"}


def test_alpha_widens_cutoff():
    c05 = prob_tcut(df=10).cutoff
    c01 = prob_tcut(df=10, alpha=0.01).cutoff
    assert c01 > c05
    assert math.isclose(c01, t_dist.ppf(0.995, 10), abs_tol=1e-12)


def test_figure_structure():
    fig = prob_tcut(df=30).plots["tcut"]
    # central fill + two tail fills + t curve + normal curve
    assert len(fig.data) == 5
    named = {tr.name for tr in fig.data if tr.name}
    assert named == {"Normal", "t, df=30"}
    # four cutoff lines (t lo/hi, normal lo/hi)
    assert len(fig.layout.shapes) == 4
    # y range fixed to [0, .42]
    assert tuple(fig.layout.yaxis.range) == (0, 0.42)


def test_confidence_label():
    fig = prob_tcut(df=30, alpha=0.05).plots["tcut"]
    texts = [a.text for a in fig.layout.annotations]
    assert "95%" in texts
    fig99 = prob_tcut(df=30, alpha=0.01).plots["tcut"]
    assert "99%" in [a.text for a in fig99.layout.annotations]


def test_errors():
    with pytest.raises(ValueError, match="df must be 2"):
        prob_tcut(df=1)
