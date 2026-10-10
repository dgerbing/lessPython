# Group colors of X() and XY() under each color theme, golden against
# lessR 4.5.7: .color_range(.get_fill(theme), n) for 18 themes and
# 2, 3, 5 groups (two grays set apart as gray60, gray30).

import re

import pytest

import lessPy as lp
from lessPy.plotly_utils import by_colors

GOLDEN = [
    ("colors", 2, ['#4398D0', '#B28B2A']),
    ("colors", 3, ['#4398D0', '#B28B2A', '#5FA140']),
    ("colors", 5, ['#4398D0', '#B28B2A', '#5FA140', '#D57388', '#9A84D6']),
    ("lightbronze", 2, ['#7899B9', '#004F93']),
    ("lightbronze", 3, ['#86A7C7', '#2474A5', '#004A8F']),
    ("lightbronze", 5, ['#A2C2E2', '#6E9CC5', '#2E79A9', '#005992', '#00408D']),
    ("dodgerblue", 2, ['#7899B9', '#004F93']),
    ("dodgerblue", 3, ['#86A7C7', '#2474A5', '#004A8F']),
    ("dodgerblue", 5, ['#A2C2E2', '#6E9CC5', '#2E79A9', '#005992', '#00408D']),
    ("darkred", 2, ['#BF8791', '#890C3A']),
    ("darkred", 3, ['#CD949F', '#A85467', '#810032']),
    ("darkred", 5, ['#E8AFBA', '#CB8492', '#AD5A6C', '#8F2C48', '#730024']),
    ("gray", 2, ['#999999', '#4D4D4D']),
    ("gray", 3, ['#A3A3A3', '#6E6E6E', '#3E3E3E']),
    ("gray", 5, ['#BEBEBE', '#989898', '#737373', '#515151', '#303030']),
    ("gold", 2, ['#AA936C', '#643E00']),
    ("gold", 3, ['#B8A079', '#8A6A05', '#5C3800']),
    ("gold", 5, ['#D3BB95', '#B1945F', '#8F6E15', '#6E4B00', '#4E2A00']),
    ("darkgreen", 2, ['#7C9F71', '#005400']),
    ("darkgreen", 3, ['#8AAD7F', '#437C25', '#004D00']),
    ("darkgreen", 5, ['#A5C89A', '#77A467', '#48812D', '#0D5F00', '#003F00']),
    ("blue", 2, ['#7899B9', '#004F93']),
    ("blue", 3, ['#86A7C7', '#2474A5', '#004A8F']),
    ("blue", 5, ['#A2C2E2', '#6E9CC5', '#2E79A9', '#005992', '#00408D']),
    ("red", 2, ['#BF8791', '#890C3A']),
    ("red", 3, ['#CD949F', '#A85467', '#810032']),
    ("red", 5, ['#E8AFBA', '#CB8492', '#AD5A6C', '#8F2C48', '#730024']),
    ("rose", 2, ['#BF8791', '#890C3A']),
    ("rose", 3, ['#CD949F', '#A85467', '#810032']),
    ("rose", 5, ['#E8AFBA', '#CB8492', '#AD5A6C', '#8F2C48', '#730024']),
    ("green", 2, ['#7C9F71', '#005400']),
    ("green", 3, ['#8AAD7F', '#437C25', '#004D00']),
    ("green", 5, ['#A5C89A', '#77A467', '#48812D', '#0D5F00', '#003F00']),
    ("purple", 2, ['#AE8AB6', '#7E008D']),
    ("purple", 3, ['#BB97C3', '#9457A0', '#7A0089']),
    ("purple", 5, ['#D6B2DE', '#B787C0', '#995CA5', '#802A8D', '#760086']),
    ("sienna", 2, ['#AA936C', '#643E00']),
    ("sienna", 3, ['#B8A079', '#8A6A05', '#5C3800']),
    ("sienna", 5, ['#D3BB95', '#B1945F', '#8F6E15', '#6E4B00', '#4E2A00']),
    ("brown", 2, ['#AA936C', '#643E00']),
    ("brown", 3, ['#B8A079', '#8A6A05', '#5C3800']),
    ("brown", 5, ['#D3BB95', '#B1945F', '#8F6E15', '#6E4B00', '#4E2A00']),
    ("orange", 2, ['#B98C7C', '#7A2D00']),
    ("orange", 3, ['#C69989', '#9E5E40', '#722500']),
    ("orange", 5, ['#E2B4A4', '#C38B76', '#A36345', '#833D00', '#621500']),
    ("white", 2, ['#999999', '#4D4D4D']),
    ("white", 3, ['#A3A3A3', '#6E6E6E', '#3E3E3E']),
    ("white", 5, ['#BEBEBE', '#989898', '#737373', '#515151', '#303030']),
    ("light", 2, ['#7899B9', '#004F93']),
    ("light", 3, ['#86A7C7', '#2474A5', '#004A8F']),
    ("light", 5, ['#A2C2E2', '#6E9CC5', '#2E79A9', '#005992', '#00408D']),
    ("slatered", 2, ['#BF8791', '#890C3A']),
    ("slatered", 3, ['#CD949F', '#A85467', '#810032']),
    ("slatered", 5, ['#E8AFBA', '#CB8492', '#AD5A6C', '#8F2C48', '#730024']),
]


@pytest.mark.parametrize("theme, n, want", GOLDEN)
def test_by_colors_match_lessR(theme, n, want):
    assert [c.upper() for c in by_colors(n, theme)] == want


def _hex(c):
    m = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)", c)
    return ("#%02X%02X%02X" % tuple(int(x) for x in m.groups())
            if m else c.upper()[:7])


@pytest.mark.parametrize("theme", ["colors", "darkred", "gray"])
def test_x_and_xy_by_follow_theme(theme):
    d = lp.read_data("Employee")
    want = [c.upper() for c in by_colors(2, theme)]
    lp.style(theme)
    try:
        hist = lp.X("Salary", by="Gender", data=d, quiet=True)
        assert [_hex(t.marker.color) for t in hist.data[:2]] == want
        xy = lp.XY("Years", "Salary", by="Gender", data=d, quiet=True)
        pts = [t for t in xy.data if t.mode == "markers" and t.name]
        assert [_hex(t.marker.color) for t in pts[:2]] == want
        vbs = lp.X("Salary", by="Gender", data=d, form="vbs", quiet=True)
        strip = [t for t in vbs.data if t.mode == "markers" and t.name]
        assert [_hex(t.marker.color) for t in strip[:2]] == want
    finally:
        lp.style()
