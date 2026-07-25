import pytest

from lessPy import showColors
from lessPy.showColors import _NAMED


def test_all_and_filter():
    f = showColors()
    assert len(f.data[0].x) == 148           # standard CSS colors
    assert list(f.data[0].text) == sorted(_NAMED)
    fb = showColors("blue")
    names = list(fb.data[0].text)
    assert all("blue" in n for n in names)
    assert "dodgerblue" in names and "navy" not in names


def test_known_hex_values():
    assert _NAMED["red"] == "#FF0000"
    assert _NAMED["dodgerblue"] == "#1E90FF"
    assert _NAMED["rebeccapurple"] == "#663399"
    # swatch marker colors are the hex values
    f = showColors("red")
    i = list(f.data[0].text).index("red")
    assert f.data[0].marker.color[i] == "#FF0000"


def test_no_match():
    with pytest.raises(ValueError, match="no named color"):
        showColors("zzznope")
