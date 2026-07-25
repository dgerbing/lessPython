import io
import contextlib

import numpy as np
import pytest

from lessPy import Prop_test, read_data

# verified against lessR Prop_test() (counts + Employee data)


def _q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return Prop_test(*a, **k)


def test_one_proportion_counts():
    r = _q(n_succ=35, n_tot=50)
    assert r.kind == "one-proportion"
    assert r.proportion == 0.7
    assert round(r.p_value, 4) == 0.0066
    assert tuple(round(c, 4) for c in r.conf_int) == (0.5539,
                                                      0.8214)


def test_many_proportions_counts():
    r = _q(n_succ=[20, 30], n_tot=[50, 60])
    assert r.kind == "many-proportions"
    assert round(r.chi2, 4) == 1.1
    assert r.df == 1
    assert round(r.p_value, 4) == 0.2943
    assert list(np.round(r.proportions, 4)) == [0.4, 0.5]


def test_goodness_of_fit_counts():
    r = _q(n_tot=[10, 20, 30, 40])
    assert r.kind == "goodness-of-fit"
    assert round(r.chi2, 4) == 20.0
    assert r.df == 3
    assert list(np.round(r.stdres, 4)) == [-3.4641, -1.1547,
                                           1.1547, 3.4641]


def test_data_modes_match_R():
    d = read_data("Employee")
    p1 = _q("Gender", success="M", data=d)
    assert round(p1.proportion, 4) == 0.4865
    assert round(p1.p_value, 4) == 1.0
    gf = _q("Dept", data=d)
    assert round(gf.chi2, 4) == 10.9444 and gf.df == 4
    ct = _q("Dept", by="Gender", data=d)
    assert ct.kind == "independence"
    assert round(ct.chi2, 4) == 6.2 and ct.df == 4
    assert round(ct.p_value, 4) == 0.1847


def test_n_fail_and_errors():
    r = _q(n_succ=35, n_fail=15)             # n_tot derived
    assert r.n_tot == 50 and r.proportion == 0.7
    with pytest.raises(ValueError, match="alternative"):
        Prop_test(n_succ=1, n_tot=2, alternative="x")
