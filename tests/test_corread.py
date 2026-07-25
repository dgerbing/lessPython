import numpy as np
import pytest

from lessPy import corEFA, corRead, read_data


def _write(tmp_path, arr, name="R.txt", fmt="%.12f"):
    p = tmp_path / name
    np.savetxt(p, arr, fmt=fmt)
    return str(p)


def test_read_and_names(tmp_path):
    d = read_data("Mach4")
    R = d[[f"m{i:02d}" for i in range(1, 5)]].corr().to_numpy()
    f = _write(tmp_path, R)
    m = corRead(f)
    assert list(m.columns) == ["X1", "X2", "X3", "X4"]
    assert list(m.index) == list(m.columns)
    assert np.allclose(m.to_numpy(), R)
    m2 = corRead(f, var_names=["A", "B", "C", "D"])
    assert list(m2.columns) == ["A", "B", "C", "D"]


def test_feeds_factor_family(tmp_path):
    d = read_data("Mach4")
    R = d[[f"m{i:02d}" for i in range(1, 7)]].corr().to_numpy()
    m = corRead(_write(tmp_path, R))
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        r = corEFA(m, n_factors=2)
    assert r.loadings.shape[0] == 6


def test_errors(tmp_path):
    with pytest.raises(ValueError, match="file= is required"):
        corRead()
    # non-square
    f = _write(tmp_path, np.ones((3, 4)), "ns.txt")
    with pytest.raises(ValueError, match="square"):
        corRead(f)
    # var_names length mismatch
    R = np.eye(3)
    with pytest.raises(ValueError, match="var_names length"):
        corRead(_write(tmp_path, R, "id.txt"), var_names=["A"])
