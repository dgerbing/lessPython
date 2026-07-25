import numpy as np
import pandas as pd
import pytest

from lessPy import Chart


@pytest.fixture
def d():
    rng = np.random.default_rng(7)
    n = 60
    return pd.DataFrame({
        "Dept": rng.choice(["ACCT", "ADMN", "FINC", "MKTG"], n),
        "Gender": rng.choice(["F", "M"], n),
        "Salary": rng.normal(60000, 12000, n).round(2),
    })


def test_radar_counts(d):
    fig = Chart("Dept", data=d, form="radar")
    assert len(fig.data) == 1
    tr = fig.data[0]
    assert tr.type == "scatterpolar"
    # polygon closes: first point repeated at the end
    assert tr.r[0] == tr.r[-1]
    assert len(tr.r) == d["Dept"].nunique() + 1
    counts = d["Dept"].value_counts().sort_index()
    assert list(tr.r[:-1]) == counts.tolist()
    assert tr.showlegend is False
    assert "Count: " in tr.hovertemplate
    assert fig.layout.title.text == "Count of Dept"


def test_radar_by(d):
    fig = Chart("Dept", by="Gender", data=d, form="radar")
    assert len(fig.data) == 2
    assert [t.name for t in fig.data] == ["F", "M"]
    assert all(t.showlegend for t in fig.data)
    assert fig.layout.showlegend is True
    # by= defaults to translucent fills (trans 0.4 -> alpha 0.6)
    assert fig.data[0].fillcolor.endswith("0.600)")
    # radial range spans the data
    top = fig.layout.polar.radialaxis.range[1]
    assert top == pytest.approx(
        pd.crosstab(d["Gender"], d["Dept"]).to_numpy().max() * 1.08)


def test_radar_stat(d):
    fig = Chart("Dept", y="Salary", stat="mean", data=d,
                form="radar")
    want = d.groupby("Dept")["Salary"].mean().sort_index()
    assert list(fig.data[0].r[:-1]) == pytest.approx(want.tolist())
    assert "Mean Salary: " in fig.data[0].hovertemplate


def test_radar_axis_order_fixed(d):
    # sort= does not reorder a radar's axes
    fig = Chart("Dept", data=d, form="radar", sort="-")
    assert list(fig.layout.polar.angularaxis.ticktext) == \
        sorted(d["Dept"].unique())


def test_radar_errors(d):
    with pytest.raises(ValueError, match="3 levels"):
        Chart("Gender", data=d, form="radar")
    with pytest.raises(ValueError, match="2 levels"):
        Chart("Dept", by="Gender", data=d, form="radar",
              filter="Gender == 'F'")
    sparse = pd.DataFrame({
        "x": ["a", "b", "c", "a", "b"],     # no (g2, c) cell
        "g": ["g1", "g1", "g1", "g2", "g2"],
    })
    with pytest.raises(ValueError, match="empty"):
        Chart("x", by="g", data=sparse, form="radar")
