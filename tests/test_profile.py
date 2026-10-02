import numpy as np
import pandas as pd
import pytest

from lessPy import Chart, read_data


@pytest.fixture
def d():
    return read_data("Employee")


def _series(fig):
    return [t for t in fig.data if t.mode == "lines+markers"]


def test_profile_single(d):
    # one point per category, connected; R's marker size 1.5 * 7.25
    fig = Chart("Dept", y="Salary", stat="mean", form="profile",
                data=d, quiet=True)
    (s,) = _series(fig)
    assert list(s.x) == [1, 2, 3, 4, 5]
    want = d.groupby("Dept")["Salary"].mean()
    assert list(s.y) == pytest.approx(want.tolist())
    assert s.marker.size == pytest.approx(10.875)
    assert fig.layout.xaxis.ticktext == tuple(want.index)
    assert fig.layout.yaxis.title.text == "Mean of Salary"
    assert fig.layout.title.text is None


def test_profile_by_is_interaction_plot(d):
    fig = Chart("Dept", y="Salary", stat="mean", by="Gender",
                form="profile", data=d, quiet=True)
    s = _series(fig)
    assert [t.name for t in s] == ["M", "W"]
    assert s[0].marker.size == pytest.approx(9.75)
    m = d.groupby(["Gender", "Dept"])["Salary"].mean()
    assert list(s[1].y) == pytest.approx(m["W"].tolist())
    assert fig.layout.legend.title.text == "Gender"


def test_profile_counts(d):
    fig = Chart("Dept", form="profile", data=d, quiet=True)
    (s,) = _series(fig)
    assert list(s.y) == [5, 6, 4, 6, 15]
    assert fig.layout.yaxis.title.text == "Count"


def test_profile_origin_y(d):
    fig = Chart("Dept", y="Salary", stat="mean", form="profile",
                origin_y=0, data=d, quiet=True)
    assert fig.layout.yaxis.tickvals[0] == 0
    assert fig.layout.yaxis.range[0] < 0 < fig.layout.yaxis.range[1]


def test_profile_empty_cell_is_a_gap():
    # an interaction plot with an empty cell draws no point there,
    # rather than halting as a bar chart of the cells must
    df = pd.DataFrame({"x": list("aaaabbbbcc"),
                       "g": list("ppqqppqqpp"),
                       "y": [1.0, 1, 2, 2, 3, 3, 4, 4, 5, 6]})
    fig = Chart("x", y="y", stat="mean", by="g", form="profile",
                data=df, quiet=True)
    q = [t for t in _series(fig) if t.name == "q"][0]
    assert list(q.y)[:2] == [2, 4] and np.isnan(q.y[2])


def test_profile_facet(d):
    fig = Chart("Dept", y="Salary", stat="mean", by="Gender",
                facet="Plan", form="profile", data=d, quiet=True)
    s = _series(fig)
    assert len(s) == 6                         # 2 groups x 3 panels
    assert [t.showlegend for t in s] == [True, True] + [False] * 4
    # shared value scale: every panel has the same range
    rng = {tuple(fig.layout[f"yaxis{k}"].range) for k in ("", "2", "3")}
    assert len(rng) == 1
    # Plan 1, men: no MKTG case, so no point at position 4
    sub = d[(d.Plan == 1) & (d.Gender == "M")].dropna(
        subset=["Dept", "Salary"])
    want = sub.groupby("Dept")["Salary"].mean()
    m1 = s[0]
    assert m1.name == "M" and 4 not in list(m1.x)
    assert list(m1.y) == pytest.approx(want.tolist())
    assert fig.layout.yaxis.title.text == "Mean of Salary"
