import io
import contextlib

import numpy as np
import pytest

from lessPy import corPrint, read_data


@pytest.fixture
def R6():
    d = read_data("Mach4")
    return d[[f"m{i:02d}" for i in range(1, 7)]].corr()


def test_compact_format_match_R(R6):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        text = corPrint(R6)
    assert buf.getvalue().rstrip("\n") == text
    lines = text.splitlines()
    # header + 6 rows
    assert len(lines) == 7
    assert lines[0].split() == ["m01", "m02", "m03", "m04",
                                "m05", "m06"]
    # diagonal 1.00 -> "100"; no leading "0."
    assert lines[1].split() == ["m01", "100", "07", "16", "-08",
                                "17", "-11"]
    assert "0." not in text                 # leading zero stripped


def test_min_value_blanks(R6):
    with contextlib.redirect_stdout(io.StringIO()):
        text = corPrint(R6, min_value=0.1)
    # m01 row: m02 (.07) and m04 (-.08) blanked; m03 (.16) kept
    row = text.splitlines()[1]
    assert row.split() == ["m01", "100", "16", "17", "-11"]


def test_raw_data(R6):
    d = read_data("Mach4")[[f"m{i:02d}" for i in range(1, 7)]]
    with contextlib.redirect_stdout(io.StringIO()):
        t1 = corPrint(d)
        t2 = corPrint(R6)
    assert t1 == t2
