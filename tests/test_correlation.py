import io
import contextlib

import numpy as np
import pytest

from lessPy import Correlation, read_data

# verified against lessR Correlation() on the Employee data


@pytest.fixture
def d():
    return read_data("Employee")


def _q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return Correlation(*a, **k)


def test_two_var_pearson_match_R(d):
    r = _q("Years", "Salary", data=d, heat_map=False)
    assert r.method == "pearson"
    assert r.r == 0.852
    assert r.tvalue == 9.501
    assert r.df == 34
    assert r.n == 36                       # 1 missing Years deleted
    assert r.lb == 0.727 and r.ub == 0.923


def test_two_var_ranks_match_R(d):
    s = _q("Years", "Salary", data=d, method="spearman",
           heat_map=False)
    assert s.r == 0.8 and s.lb is None     # no CI for ranks
    k = _q("Years", "Salary", data=d, method="kendall",
           heat_map=False)
    assert k.r == 0.635 and k.df is None


def test_matrix_match_R(d):
    m = _q(d[["Years", "Salary", "Pre", "Post"]])
    assert list(m.columns) == ["Years", "Salary", "Pre", "Post"]
    assert m.loc["Years", "Salary"] == 0.85
    assert m.loc["Pre", "Post"] == 0.91
    assert np.allclose(np.diag(m.to_numpy()), 1.0)
    assert "heatmap" in m.attrs["plots"]


def test_matrix_list_and_show_missing(d):
    m = _q(["Years", "Salary"], data=d)
    assert m.shape == (2, 2)
    mm = _q(d[["Years", "Salary"]], show="missing")
    n = mm.attrs["missing_n"]
    assert n.loc["Salary", "Salary"] == 37    # Salary complete
    assert n.loc["Years", "Years"] == 36      # 1 missing


def test_errors(d):
    with pytest.raises(ValueError, match="miss:"):
        Correlation(d[["Years", "Salary"]], miss="x")
    with pytest.raises(ValueError, match="method:"):
        Correlation("Years", "Salary", data=d, method="x")
