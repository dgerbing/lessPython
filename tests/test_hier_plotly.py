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


def test_treemap_counts(d):
    fig = Chart("Dept", data=d, form="treemap")
    tr = fig.data[0]
    assert tr.type == "treemap"
    counts = d["Dept"].value_counts().sort_index()
    assert list(tr.labels) == list(counts.index)
    assert list(tr.values) == counts.tolist()
    assert set(tr.parents) == {""}             # all top-level
    assert tr.branchvalues == "remainder"
    # counts default to % labels (Chart resolves labels before the
    # hier renderer, as in R), % of parent in hover
    assert tr.textinfo == "label+percent root"
    assert "% of parent" in tr.hovertemplate


def test_sunburst_from_pie_by(d):
    # by= on a pie nests inside the wedges as a sunburst, as in R
    fig = Chart("Dept", by="Gender", data=d, form="pie")
    tr = fig.data[0]
    assert tr.type == "sunburst"
    n_dept = d["Dept"].nunique()
    n_cells = (d.groupby(["Dept", "Gender"]).size() > 0).sum()
    assert len(tr.ids) == n_dept + n_cells
    # leaf ids nest under their parent wedge
    assert "I_ACCT_F" in tr.ids
    assert tr.parents[list(tr.ids).index("I_ACCT_F")] == "I_ACCT"
    # top-level nodes carry value 0 with branchvalues="remainder"
    top = [v for i, v in zip(tr.ids, tr.values)
           if tr.parents[list(tr.ids).index(i)] == ""]
    assert set(top) == {0}
    # node colors inherit the top-level wedge color
    ids = list(tr.ids)
    cols = list(tr.marker.colors)
    assert cols[ids.index("I_ACCT_F")] == cols[ids.index("I_ACCT")]
    assert fig.layout.title.text == "Count of Dept by Gender"


def test_sunburst_alias(d):
    fig = Chart("Dept", by="Gender", data=d, form="sunburst")
    assert fig.data[0].type == "sunburst"
    # without by, sunburst falls back to a plain donut, as in R
    fig2 = Chart("Dept", data=d, form="sunburst")
    assert fig2.data[0].type == "pie"


def test_icicle_stat_mean(d):
    fig = Chart("Dept", y="Salary", stat="mean", data=d,
                form="icicle")
    tr = fig.data[0]
    assert tr.type == "icicle"
    want = d.groupby("Dept")["Salary"].mean().sort_index()
    got = [c["stat"] for c in tr.customdata]
    assert got == pytest.approx(want.tolist())
    # non-additive: stat shown via texttemplate, no % of parent
    assert tr.texttemplate is not None
    assert "% of parent" not in tr.hovertemplate
    assert "Mean of Salary" in tr.hovertemplate


def test_hier_deviation_rejected(d):
    with pytest.raises(ValueError, match="deviation"):
        Chart("Dept", y="Salary", stat="deviation", data=d,
              form="treemap")
