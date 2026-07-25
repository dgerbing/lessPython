import numpy as np
import pytest

from lessPy import corProp, read_data

# proportionalities verified against lessR corProp() on Mach4


@pytest.fixture
def R6():
    d = read_data("Mach4")
    return d[[f"m{i:02d}" for i in range(1, 7)]].corr()


def test_match_R(R6):
    P = corProp(R6)
    assert list(P.columns) == list(R6.columns)
    assert np.allclose(np.diag(P.to_numpy()), 1.0)
    assert P.loc["m01", "m05"] == 0.85
    assert P.loc["m02", "m04"] == -0.84
    assert P.loc["m03", "m06"] == -0.51
    # symmetric
    assert np.allclose(P.to_numpy(), P.to_numpy().T)
    assert "heatmap" in P.attrs["plots"]


def test_raw_data(R6):
    d = read_data("Mach4")[[f"m{i:02d}" for i in range(1, 7)]]
    assert np.allclose(corProp(d).to_numpy(),
                       corProp(R6).to_numpy())


def test_no_heatmap(R6):
    P = corProp(R6, heat_map=False)
    assert P.attrs["plots"] == {}
