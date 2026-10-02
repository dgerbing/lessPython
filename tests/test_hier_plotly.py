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


def test_hier_fill_sequential_palette():
    # a palette name selects the palette and each shade is read off
    # its ramp by magnitude (R pal_explicit); hex values are R's
    # Chart(..., form="treemap", fill="greens") on Employee
    from lessPy import read_data
    emp = read_data("Employee")
    tr = Chart("Dept", data=emp, form="treemap", fill="greens",
               quiet=True).data[0]
    assert dict(zip(tr.labels, tr.marker.colors)) == {
        "ACCT": "#B8DFAC", "ADMN": "#A2CC95", "FINC": "#CFF2C4",
        "MKTG": "#A2CC95", "SALE": "#003200"}
    tr = Chart("Dept", y="Salary", stat="mean", data=emp,
               form="icicle", fill="greens", quiet=True).data[0]
    assert dict(zip(tr.labels, tr.marker.colors)) == {
        "ACCT": "#CFF2C4", "ADMN": "#003200", "FINC": "#74A662",
        "MKTG": "#649950", "SALE": "#004700"}


def test_hier_nested_by_list(capsys):
    # each further by variable nests one level deeper; node table
    # identical to R's Chart(Dept, by=c(Gender, Plan), "treemap")
    from lessPy import read_data
    emp = read_data("Employee")
    tr = Chart("Dept", by=["Gender", "Plan"], data=emp,
               form="treemap").data[0]
    assert len(tr.ids) == 33 and sum(tr.values) == 36
    assert "I_SALE_M_1" in tr.ids
    i = list(tr.ids).index("I_SALE_M_1")
    assert tr.parents[i] == "I_SALE_M" and tr.values[i] == 6
    out = capsys.readouterr().out
    assert "Joint Frequencies of Dept, Gender, Plan" in out
    # pie with a list of by is a sunburst of the same nesting
    tr = Chart("Dept", by=["Gender", "Plan"], data=emp, form="pie",
               quiet=True).data[0]
    assert tr.type == "sunburst" and len(tr.ids) == 33
