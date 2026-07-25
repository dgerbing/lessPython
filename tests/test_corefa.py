import io
import contextlib

import numpy as np
import pytest

from lessPy import corEFA, read_data

# loadings/SS verified against lessR corEFA() on Mach4 item corrs


@pytest.fixture
def R6():
    d = read_data("Mach4")
    return d[[f"m{i:02d}" for i in range(1, 7)]].corr()


def _run(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return corEFA(*a, **k)


def test_promax_2factor_match_R(R6):
    r = _run(R6, n_factors=2)
    assert r.converged
    # sorted loadings: m03 loads ~1.0 on F1, first row
    assert r.loadings.index[0] == "m03"
    assert round(r.loadings.loc["m03", "Factor1"], 3) == 1.002
    assert round(r.loadings.loc["m05", "Factor2"], 3) == 0.417
    # SS table
    assert round(r.ss.loc["SS loadings", "Factor1"], 3) == 1.096
    assert round(r.ss.loc["Proportion Var", "Factor2"], 3) \
        == 0.093
    assert round(r.ss.loc["Cumulative Var", "Factor2"], 3) \
        == 0.276
    assert r.model.splitlines()[0] == "  F1 =~ m03"


def test_varimax_3factor_ss_match_R():
    d = read_data("Mach4")
    R = d[[f"m{i:02d}" for i in range(1, 11)]].corr()
    r = _run(R, n_factors=3, rotate="varimax")
    ss = r.ss.round(3)
    assert list(ss.loc["SS loadings"]) == [1.661, 1.032, 0.816]
    assert list(ss.loc["Cumulative Var"]) == [0.166, 0.269,
                                              0.351]
    assert r.model.splitlines()[0] == \
        "  F1 =~ m07 + m06 + m10 + m09 + m03"


def test_none_rotation_and_raw_data():
    d = read_data("Mach4")
    items = [f"m{i:02d}" for i in range(1, 7)]
    # raw data is correlated internally -> same as passing corr
    r1 = _run(d[items], n_factors=2, rotate="none")
    r2 = _run(d[items].corr(), n_factors=2, rotate="none")
    assert np.allclose(r1.loadings.to_numpy(),
                       r2.loadings.to_numpy())


def test_errors(R6):
    with pytest.raises(ValueError, match="rotate"):
        corEFA(R6, n_factors=2, rotate="oblimin")
