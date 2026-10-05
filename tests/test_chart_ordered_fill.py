# An ordered category is ordinal: Chart()'s default bar fill is the
# theme's sequential palette, light to dark in level order. Fill
# encodes x without by and by with it. Reference colors from lessR
# 4.5.7 Chart() on the Employee data, JobSat as low < med < high.

import pandas as pd
import pytest

import lessPy as lp

LV = ["low", "med", "high"]
HUES = ["#4398D0", "#B28B2A", "#5FA140"]
BLUES = ["#86A7C7", "#2474A5", "#004A8F"]
REDS = ["#CD949F", "#A85467", "#810032"]


@pytest.fixture
def d():
    d = lp.read_data("Employee")
    d["J0"] = d["JobSat"].astype(pd.CategoricalDtype(LV))
    d["J1"] = d["JobSat"].astype(pd.CategoricalDtype(LV, ordered=True))
    return d


def _fills(fig):
    out = []
    for t in fig.data:
        c = t.marker.color
        out += [str(x)[:7] for x in (c if isinstance(c, (list, tuple))
                                     else [c])]
    return out


@pytest.mark.parametrize("args, kw, want", [
    (("J0",), {}, HUES),
    (("J1",), {}, BLUES),
    (("J1",), {"fill": "hues"}, HUES),
    (("J1",), {"theme": "darkred"}, REDS),
    (("Gender",), {"by": "J1"}, BLUES),
    (("Gender",), {"by": "J0"}, HUES),
    (("J1",), {"by": "Gender"}, HUES[:2]),
])
def test_ordered_fill_matches_r(d, args, kw, want):
    fig = lp.Chart(*args, data=d, quiet=True, **kw)
    assert _fills(fig) == want


def test_ordered_fill_style_theme(d):
    lp.style("darkred")
    assert _fills(lp.Chart("J1", data=d, quiet=True)) == REDS
