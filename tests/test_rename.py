import io
import contextlib

import pytest

from lessPy import read_data, rename


def _q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        out = rename(*a, **k)
    return out, buf.getvalue()


def test_single_and_multi():
    d = read_data("Employee")
    d2, _ = _q(d, "Salary", "Pay")
    assert "Pay" in d2.columns and "Salary" not in d2.columns
    assert "Salary" in d.columns                 # original intact
    d3, out = _q(d, ["Salary", "Pre"], ["Pay", "Before"])
    assert list(d3.columns) == ["Years", "Gender", "Dept",
                                "Pay", "JobSat", "Plan",
                                "Before", "Post"]
    assert "Salary --> Pay" in out
    assert "Pre --> Before" in out


def test_errors():
    d = read_data("Employee")
    with pytest.raises(ValueError, match="must match"):
        rename(d, ["Salary", "Pre"], ["Pay"])
    with pytest.raises(ValueError, match="not columns"):
        rename(d, "Nope", "X")
    with pytest.raises(TypeError, match="DataFrame"):
        rename([1, 2], "a", "b")
