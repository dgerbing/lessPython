import io
import contextlib

import numpy as np
import pytest

from lessPy import corScree, read_data

# eigenvalues verified against lessR corScree() on Mach4 corrs


@pytest.fixture
def R10():
    d = read_data("Mach4")
    return d[[f"m{i:02d}" for i in range(1, 11)]].corr()


def test_eigenvalues_match_R(R10):
    with contextlib.redirect_stdout(io.StringIO()):
        r = corScree(R10)
    assert len(r.eigenvalues) == 10
    assert r.eigenvalues[0] == pytest.approx(2.438, abs=1e-3)
    assert r.eigenvalues[-1] == pytest.approx(0.464, abs=1e-3)
    # descending
    assert np.all(np.diff(r.eigenvalues) <= 0)
    # differences = successive drops
    assert r.differences[0] == pytest.approx(0.995, abs=1e-3)
    assert len(r.differences) == 9
    assert np.allclose(r.differences,
                       r.eigenvalues[:-1] - r.eigenvalues[1:])
    assert set(r.plots) == {"scree", "differences"}


def test_prints(R10, capsys):
    corScree(R10)
    out = capsys.readouterr().out
    assert "Eigenvalues" in out
    assert "Differences of Successive Eigenvalues" in out


def test_raw_data_correlated(R10):
    d = read_data("Mach4")[[f"m{i:02d}" for i in range(1, 11)]]
    with contextlib.redirect_stdout(io.StringIO()):
        r1 = corScree(d)
        r2 = corScree(R10)
    assert np.allclose(r1.eigenvalues, r2.eigenvalues)
