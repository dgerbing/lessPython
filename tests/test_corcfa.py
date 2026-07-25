import io
import contextlib

import numpy as np
import pytest

from lessPy import corCFA, read_data

# alpha/omega/loadings/residuals verified vs lessR corCFA()


@pytest.fixture
def R6():
    d = read_data("Mach4")
    return d[[f"m{i:02d}" for i in range(1, 7)]].corr()


def _run(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return corCFA(*a, **k)


def test_mimm_match_R(R6):
    r = _run(R6, F1=["m01", "m02", "m03"], F2=["m04", "m05",
                                               "m06"])
    assert np.allclose(r.alpha, [0.1954, -0.1202], atol=1e-4)
    assert np.allclose(r.omega, [0.4468, 0.5735], atol=1e-4)
    # within-factor sort by loading
    assert list(r.if_cor.index) == ["m01", "m03", "m02",
                                    "m04", "m06", "m05"]
    assert round(r.if_cor.loc["m01", "F1"], 3) == 1.036
    assert round(r.if_cor.loc["m04", "F2"], 3) == 0.599
    assert round(r.ff_cor.loc["F1", "F2"], 4) == 0.1364


def test_residuals_match_R(R6, capsys):
    r = _run(R6, F1=["m01", "m02", "m03"], F2=["m04", "m05",
                                               "m06"])
    # residual matrix and its summary
    assert r.resid.loc["m01", "m03"] == pytest.approx(0.0054,
                                                      abs=1e-4)
    ss = float((r.resid.to_numpy() ** 2).sum())
    assert round(ss, 3) == 1.072


def test_model_string_equiv(R6):
    r1 = _run(R6, F1=["m01", "m02", "m03"], F2=["m04", "m05",
                                                "m06"])
    r2 = _run(R6, model="F1 =~ m01 + m02 + m03\n"
                        "F2 =~ m04 + m05 + m06")
    assert np.allclose(r1.if_cor.to_numpy(), r2.if_cor.to_numpy())
    assert "heatmap" in r1.plots


def test_errors(R6):
    with pytest.raises(ValueError, match="measurement model"):
        corCFA(R6)
    with pytest.raises(ValueError, match="not in the correlation"):
        corCFA(R6, F1=["m01", "zz"])
