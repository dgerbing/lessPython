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
    # Cleveland: categories on the vertical axis, no title, the value
    # axis named for the count and starting at zero
    fig = Chart("Dept", data=d, form="dot")
    mk = _marker_traces(fig)
    assert len(mk) == 1
    got = dict(zip(mk[0].y, mk[0].x))
    assert got == d["Dept"].value_counts().to_dict()
    seg = _line_traces(fig)[0]
    assert min(v for v in seg.x if v is not None) == 0
    assert fig.layout.title.text is None
    assert fig.layout.xaxis.title.text == "Count of Dept"
    assert fig.layout.xaxis.range[0] == 0


def test_dot_stat_mean(d):
    # sort="-" puts the largest at the top: a horizontal chart lists
    # its first category at the bottom, so the order is ascending
    fig = Chart("Dept", y="Salary", stat="mean", data=d,
                form="dot", sort="-")
    mk = _marker_traces(fig)[0]
    want = (d.groupby("Dept")["Salary"].mean()
            .sort_values(ascending=True))
    assert list(mk.y) == list(want.index)
    assert list(mk.x) == pytest.approx(want.tolist())
    # the length of the segment carries the value: origin at zero
    assert fig.layout.xaxis.range[0] == 0
    assert fig.layout.xaxis.title.text == "Mean of Salary"
    assert fig.layout.title.text is None
    fig = Chart("Dept", y="Salary", stat="mean", data=d, form="dot",
                main="Salaries")
    assert fig.layout.title.text == "Salaries"


def test_dot_vertical(d):
    fig = Chart("Dept", y="Salary", stat="mean", data=d,
                form="dot", horiz=False, sort="-")
    mk = _marker_traces(fig)[0]
    want = (d.groupby("Dept")["Salary"].mean()
            .sort_values(ascending=False))
    assert list(mk.x) == list(want.index)        # cats on x-axis
    assert fig.layout.yaxis.title.text == "Mean of Salary"


def test_dot_value_axis_headroom():
    # a step past the top tick only when that tick would clip the
    # largest value (R .dot_val_axis): 0..15 -> 20, 0..9 -> 10
    df = pd.DataFrame({"g": list("a" * 15 + "b" * 4)})
    assert Chart("g", data=df, form="dot").layout.xaxis.range[1] == 20
    df = pd.DataFrame({"g": list("a" * 9 + "b" * 4)})
    assert Chart("g", data=df, form="dot").layout.xaxis.range[1] == 10


def test_dot_segments_off(d, capsys):
    fig = Chart("Dept", data=d, form="dot", segments_x=False)
    assert len(_line_traces(fig)) == 0
    # the other axis's parameter is named, not silently dropped
    fig = Chart("Dept", data=d, form="dot", segments_y=False)
    assert len(_line_traces(fig)) == 1
    assert "set  segments_x" in capsys.readouterr().out


def test_paired_dot(paired):
    fig = Chart("Name", y=["Pre", "Post"], data=paired, form="dot",
                quiet=True)
    mk = _marker_traces(fig)
    assert [t.name for t in mk] == ["Pre", "Post"]
    assert all(t.showlegend for t in mk)
    # ordered by Post - Pre ascending, largest gain at the top:
    # Cat -1, Ann 6, Bob 6, Dee 8 (ties keep data order)
    assert list(mk[0].y) == ["Cat", "Ann", "Bob", "Dee"]
    assert list(mk[0].x) == [81.0, 72.0, 65.0, 58.0]
    assert fig.layout.title.text is None
    assert fig.layout.xaxis.title.text == "Pre & Post"
    assert fig.layout.xaxis.range[0] == 0
    # sort="0" keeps the data order
    fig = Chart("Name", y=["Pre", "Post"], data=paired, form="dot",
                sort="0", quiet=True)
    assert list(_marker_traces(fig)[0].y) == paired["Name"].tolist()


def test_paired_diff_listing(paired, capsys):
    Chart("Name", y=["Pre", "Post"], data=paired, form="dot")
    out = [ln.rstrip() for ln in capsys.readouterr().out.split("\n")]
    i = out.index("Post - Pre")
    assert out[i + 1:i + 7] == [
        "n  diff  Name",
        "------------",
        "1  8.0 Dee",
        "2  6.0 Bob",
        "3  6.0 Ann",
        "4 -1.0 Cat"]


def test_paired_sort_by_row_mean():
    # three series are ordered by their row mean when sort is given
    df = pd.DataFrame({"Name": ["a", "b", "c"], "T1": [1.0, 5, 3],
                       "T2": [2.0, 6, 3], "T3": [3.0, 7, 3]})
    fig = Chart("Name", y=["T1", "T2", "T3"], data=df, form="dot",
                sort="-")
    assert list(_marker_traces(fig)[0].y) == ["a", "c", "b"]


def test_paired_dot_stat_repeated_x(d):
    d = d.assign(Pre=d["Salary"] / 1000, Post=d["Salary"] / 900)
    with pytest.raises(ValueError, match="specify stat"):
        Chart("Dept", y=["Pre", "Post"], data=d, form="dot")
    fig = Chart("Dept", y=["Pre", "Post"], stat="mean", data=d,
                form="dot", quiet=True)
    mk = _marker_traces(fig)
    m = d.groupby("Dept")[["Pre", "Post"]].mean()
    got = dict(zip(mk[0].y, mk[0].x))
    assert got == pytest.approx(m["Pre"].to_dict())
    assert fig.layout.xaxis.title.text == "Mean of Pre & Post"


def test_dot_by(d):
    d = d.assign(Gender=np.where(np.arange(len(d)) % 3, "M", "W"))
    fig = Chart("Dept", y="Salary", stat="mean", by="Gender", data=d,
                form="dot", quiet=True)
    mk = _marker_traces(fig)
    assert [t.name for t in mk] == ["M", "W"]
    assert fig.layout.legend.title.text == "Gender"
    m = d.groupby(["Gender", "Dept"])["Salary"].mean().unstack()
    dif = (m.loc["W"] - m.loc["M"]).sort_values()
    assert list(mk[0].y) == list(dif.index)      # by the difference
    assert fig.layout.xaxis.title.text == "Mean of Salary"
    # three levels: one series each, no difference ordering
    d = d.assign(G3=np.arange(len(d)) % 3)
    fig = Chart("Dept", by="G3", data=d, form="dot", quiet=True)
    mk = _marker_traces(fig)
    assert len(mk) == 3
    assert list(mk[0].y) == sorted(d["Dept"].unique())
    assert fig.layout.xaxis.title.text == "Count"


def test_dot_by_summary_diff_row(capsys):
    from lessPy import read_data
    Chart("Dept", y="Salary", stat="mean", by="Gender",
          data=read_data("Employee"), form="dot")
    out = [ln.rstrip() for ln in capsys.readouterr().out.split("\n")]
    i = out.index("Summary Table for Mean of Salary")
    assert out[i + 2:i + 8] == [
        "          ACCT      ADMN      FINC      MKTG      SALE",
        "M     69626.20  90963.35  82967.60 109062.66  96150.97",
        "W     73237.16  91434.00  67139.90  74496.02  74188.25",
        "Diff   3610.97    470.66 -15827.70 -34566.64 -21962.72",
        "",
        "Diff: W-M"]


def test_dot_by_facet(d):
    d = d.assign(Gender=np.where(np.arange(len(d)) % 3, "M", "W"),
                 Site=np.where(np.arange(len(d)) % 2, "N", "S"))
    fig = Chart("Dept", y="Salary", stat="mean", by="Gender",
                facet="Site", data=d, form="dot", quiet=True)
    mk = _marker_traces(fig)
    assert len(mk) == 4                       # 2 series x 2 panels
    assert [t.showlegend for t in mk] == [True, True, False, False]
    assert fig.layout.legend.title.text == "Gender"
    assert fig.layout.xaxis.title.text == "Mean of Salary"


def test_dot_errors(d, paired):
    with pytest.raises(ValueError, match="numerical variable"):
        # unique categories, no y: nothing to count
        Chart("Name", data=paired, form="dot")
    with pytest.raises(ValueError, match="paired"):
        Chart("Name", y=["Pre", "Post"], data=paired, form="bar")
    with pytest.raises(ValueError, match="do not specify"):
        # x identifies the cases: the values display directly
        Chart("Name", y=["Pre", "Post"], data=paired, form="dot",
              stat="mean")


def test_dot_preaggregated(paired):
    fig = Chart("Name", y="Pre", data=paired, form="dot")
    mk = _marker_traces(fig)[0]
    assert list(mk.x) == paired.set_index("Name")["Pre"] \
        .sort_index().tolist()
    # pre-aggregated value label is the variable name itself
    assert fig.layout.xaxis.title.text == "Pre"


def test_paired_dot_segments_join_the_pair(paired):
    # lessR (2026-08) draws one connector per category joining the two
    #   values, not a stem from the origin to each: the gap between the
    #   values is what a paired dot plot is read for
    fig = Chart("Name", y=["Pre", "Post"], form="dot", data=paired,
                sort="0", quiet=True)
    segs = [t for t in fig.data if t.mode == "lines"]
    assert len(segs) == 1                      # one trace, not one per column

    xs = [v for v in segs[0].x if v is not None]
    assert 0 not in xs                         # no endpoint at the origin
    lo, hi = xs[0], xs[1]                      # first category, Ann
    assert min(lo, hi) == pytest.approx(72.0)  # Pre
    assert max(lo, hi) == pytest.approx(78.0)  # Post

    # segments_x=False removes them
    fig2 = Chart("Name", y=["Pre", "Post"], form="dot", data=paired,
                 segments_x=False, quiet=True)
    assert not [t for t in fig2.data if t.mode == "lines"]
