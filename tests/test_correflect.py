import numpy as np
import pytest

from lessPy import corReflect, read_data

# verified against lessR corReflect() on Mach4 correlations


@pytest.fixture
def R6():
    d = read_data("Mach4")
    return d[[f"m{i:02d}" for i in range(1, 7)]].corr()


def test_reflect_match_R(R6):
    out = corReflect(R6, ["m03", "m05"])
    # a reflected var's correlations with unreflected vars flip
    assert out.loc["m03", "m01"] == pytest.approx(
        -R6.loc["m03", "m01"], abs=1e-9)
    assert out.loc["m05", "m02"] == pytest.approx(
        -R6.loc["m05", "m02"], abs=1e-9)
    # two reflected vars: their mutual correlation is unchanged
    assert out.loc["m03", "m05"] == pytest.approx(
        R6.loc["m03", "m05"], abs=1e-9)
    # unreflected pair unchanged
    assert out.loc["m01", "m02"] == pytest.approx(
        R6.loc["m01", "m02"], abs=1e-9)
    assert np.allclose(np.diag(out.to_numpy()), 1.0)
    assert np.allclose(out.to_numpy(), out.to_numpy().T)
    assert "heatmap" in out.attrs["plots"]


def test_single_var_and_raw_data(R6):
    o1 = corReflect(R6, "m01")             # bare string
    assert o1.loc["m01", "m02"] == pytest.approx(
        -R6.loc["m01", "m02"], abs=1e-9)
    d = read_data("Mach4")[[f"m{i:02d}" for i in range(1, 7)]]
    assert np.allclose(corReflect(d, ["m03"]).to_numpy(),
                       corReflect(R6, ["m03"]).to_numpy())


def test_errors(R6):
    with pytest.raises(ValueError, match="not in the matrix"):
        corReflect(R6, ["m03", "zz"])
