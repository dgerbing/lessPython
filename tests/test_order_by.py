import numpy as np
import pandas as pd
import pytest

from lessPy import order_by


def _df():
    return pd.DataFrame({
        "g": ["b", "a", "b", "a", "b"],
        "x": [2, 1, 2, 3, 1],
        "y": [10, 20, 30, 40, 50],
    }, index=["r2", "r5", "r1", "r4", "r3"])


def test_single_and_default_direction():
    d = _df()
    up = order_by(d, "x", quiet=True)
    assert up["x"].tolist() == [1, 1, 2, 2, 3]
    # default direction is ascending, ties stable (orig order kept)
    assert up["y"].tolist() == [20, 50, 10, 30, 40]
    dn = order_by(d, "x", direction="-", quiet=True)
    assert dn["x"].tolist() == [3, 2, 2, 1, 1]


def test_multi_column_mixed():
    d = _df()
    r = order_by(d, ["g", "x"], direction=["+", "-"], quiet=True)
    assert list(zip(r["g"], r["x"])) == [
        ("a", 3), ("a", 1), ("b", 2), ("b", 2), ("b", 1)]


def test_row_names_and_random():
    d = _df()
    rn = order_by(d, "row.names", quiet=True)
    assert list(rn.index) == ["r1", "r2", "r3", "r4", "r5"]
    rd = order_by(d, "row.names", direction="-", quiet=True)
    assert list(rd.index) == ["r5", "r4", "r3", "r2", "r1"]
    a = order_by(d, "random", seed=7, quiet=True)
    b = order_by(d, "random", seed=7, quiet=True)
    assert list(a.index) == list(b.index)             # reproducible
    assert sorted(a.index) == sorted(d.index)         # same rows


def test_returns_copy_and_errors():
    d = _df()
    order_by(d, "x", quiet=True)
    assert d["x"].tolist() == [2, 1, 2, 3, 1]         # unchanged
    with pytest.raises(ValueError, match="not found"):
        order_by(d, "nope", quiet=True)
    with pytest.raises(ValueError, match="direction value"):
        order_by(d, "x", direction="up", quiet=True)
    with pytest.raises(ValueError, match="must equal"):
        order_by(d, ["g", "x"], direction=["+"], quiet=True)
    with pytest.raises(ValueError, match="row.names"):
        order_by(d, "row.names", direction=["+", "-"], quiet=True)
