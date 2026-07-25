import numpy as np
import pandas as pd
import pytest

from lessPy import Chart, pie_plotly
from lessPy.plotly_utils import auto_text_color, make_trans


@pytest.fixture
def d():
    rng = np.random.default_rng(7)
    n = 60
    return pd.DataFrame({
        "Dept": rng.choice(["ACCT", "ADMN", "FINC", "MKTG"], n),
        "Gender": rng.choice(["F", "M"], n),
        "Salary": rng.normal(60000, 12000, n).round(2),
    })


def test_pie_1d(d):
    fig = Chart("Dept", data=d, form="pie")
    assert len(fig.data) == 1
    assert fig.data[0].type == "pie"
    assert list(fig.data[0].labels) == sorted(d["Dept"].unique())
    assert (list(fig.data[0].values) ==
            d["Dept"].value_counts().sort_index().tolist())
    # counts default to % labels, 0 decimals
    assert fig.data[0].text[0].endswith("%")
    # auto-title, R analog .build_chart_title
    assert fig.layout.title.text == "Count of Dept"


def test_pie_hole_and_clamp():
    s = pd.Series([3, 4, 5], index=list("abc"))
    fig = pie_plotly(s, hole=0.8)             # inside labels default
    assert fig.data[0].hole == 0.62            # clamped
    fig2 = pie_plotly(s, hole=0.8, labels_position="out")
    assert fig2.data[0].hole == 0.8            # outside: no clamp


def test_pie_grid(d):
    # the pie GRID is R's treatment of facet= on a plain pie;
    # Chart(by=) routes to a sunburst instead (see test_hier).
    # Until facet= exists, the grid renders via pie_plotly directly.
    tbl = pd.crosstab(d["Gender"], d["Dept"])
    fig = pie_plotly(tbl)
    assert len(fig.data) == 2                  # one pie per level
    # non-overlapping domains
    doms = [tuple(t.domain.x) for t in fig.data]
    assert len(set(doms)) == 2
    # group name annotated in each hole
    texts = [a.text for a in fig.layout.annotations]
    assert texts == ["F", "M"]


def test_pie_stat(d):
    fig = Chart("Dept", y="Salary", stat="mean", data=d,
                form="pie")
    want = d.groupby("Dept")["Salary"].mean()
    assert list(fig.data[0].values) == pytest.approx(
        want.tolist())
    # aggregated data: input labels, not %
    assert "%" not in fig.data[0].text[0]
    assert fig.layout.title.text == "Mean of Salary by Dept"


def test_pie_labels_off():
    s = pd.Series([3, 4], index=["a", "b"])
    fig = pie_plotly(s, labels="off")
    assert list(fig.data[0].text) == ["a", "b"]  # names only


def test_pie_fill_dict():
    s = pd.Series([3, 4], index=["a", "b"])
    fig = pie_plotly(s, fill={"b": "#FF0000", "a": "#0000FF"},
                     opacity=1)
    assert list(fig.data[0].marker.colors) == \
        ["rgba(0,0,255,1.000)", "rgba(255,0,0,1.000)"]


def test_make_trans_and_autotext():
    assert make_trans("#000000", 0.5) == "rgba(0,0,0,0.500)"
    assert make_trans("off", 0.5) == "rgba(0,0,0,0)"
    assert auto_text_color("rgba(0,0,0,1.000)") == "white"
    assert auto_text_color("rgba(0,0,0,0)", bg="white") == "black"
    assert auto_text_color("#FFFF99") == "black"
