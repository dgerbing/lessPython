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
        "Salary": rng.normal(60000, 12000, n).round(2),
    })


@pytest.fixture
def paired():
    return pd.DataFrame({
        "Name": ["Ann", "Bob", "Cat", "Dee"],
        "Pre": [72.0, 65.0, 81.0, 58.0],
        "Post": [78.0, 71.0, 80.0, 66.0],
    })


def _marker_traces(fig):
    return [t for t in fig.data if t.mode == "markers"]


def _line_traces(fig):
    return [t for t in fig.data if t.mode == "lines"]


def test_dot_counts(d):
    fig = Chart("Dept", data=d, form="dot")
    mk = _marker_traces(fig)
    assert len(mk) == 1
    got = dict(zip(mk[0].x, mk[0].y))
    assert got == d["Dept"].value_counts().to_dict()
    # counts anchor stems at the origin, 0
    seg = _line_traces(fig)[0]
    assert min(v for v in seg.y if v is not None) == 0
    assert fig.layout.title.text == "Count of Dept"


def test_dot_stat_mean(d):
    fig = Chart("Dept", y="Salary", stat="mean", data=d,
                form="dot", sort="-")
    mk = _marker_traces(fig)[0]
    want = (d.groupby("Dept")["Salary"].mean()
            .sort_values(ascending=False))
    assert list(mk.x) == list(want.index)
    assert list(mk.y) == pytest.approx(want.tolist())
    # continuous data: origin one grid step below the first tick,
    # not zero — the point of a dot chart vs a bar chart
    assert fig.layout.yaxis.range[0] > 0
    assert fig.layout.title.text == "Mean of Salary by Dept"


def test_dot_horiz(d):
    fig = Chart("Dept", y="Salary", stat="mean", data=d,
                form="dot", horiz=True)
    mk = _marker_traces(fig)[0]
    assert set(mk.y) == set(d["Dept"].unique())  # cats on y-axis
    assert fig.layout.xaxis.autorange is False


def test_dot_segments_off(d):
    fig = Chart("Dept", data=d, form="dot", segments_y=False)
    assert len(_line_traces(fig)) == 0


def test_paired_dot(paired):
    fig = Chart("Name", y=["Pre", "Post"], data=paired, form="dot")
    mk = _marker_traces(fig)
    assert [t.name for t in mk] == ["Pre", "Post"]
    assert all(t.showlegend for t in mk)
    # horizontal: names on the y axis
    assert list(mk[0].y) == paired["Name"].tolist()
    assert list(mk[0].x) == paired["Pre"].tolist()
    assert fig.layout.title.text == "Pre & Post by Name"


def test_paired_sort_by_row_mean(paired):
    fig = Chart("Name", y=["Pre", "Post"], data=paired,
                form="dot", sort="-")
    mk = _marker_traces(fig)[0]
    means = paired.set_index("Name")[["Pre", "Post"]].mean(axis=1)
    assert list(mk.y) == list(
        means.sort_values(ascending=False).index)


def test_dot_errors(d, paired):
    with pytest.raises(ValueError, match="not meaningful"):
        Chart("Dept", by="Dept", data=d, form="dot")
    with pytest.raises(ValueError, match="numerical variable"):
        # unique categories, no y: nothing to count
        Chart("Name", data=paired, form="dot")
    with pytest.raises(ValueError, match="paired"):
        Chart("Name", y=["Pre", "Post"], data=paired, form="bar")
    with pytest.raises(ValueError, match="stat"):
        Chart("Name", y=["Pre", "Post"], data=paired, form="dot",
              stat="mean")


def test_dot_preaggregated(paired):
    fig = Chart("Name", y="Pre", data=paired, form="dot")
    mk = _marker_traces(fig)[0]
    assert list(mk.y) == paired.set_index("Name")["Pre"] \
        .sort_index().tolist()
    # pre-aggregated value label is the variable name itself
    assert fig.layout.yaxis.title.text == "Pre"
