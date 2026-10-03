import numpy as np
import pandas as pd
import pytest

from lessPy import Chart
from lessPy.getColors import getColors


@pytest.fixture
def lik():
    rng = np.random.default_rng(4)
    n = 120
    p = {"q1": [.4, .3, .2, .1, 0], "q2": [.1, .2, .3, .2, .2],
         "q3": [0, .1, .2, .3, .4]}
    d = {k: rng.choice([1, 2, 3, 4, 5], n, p=v) for k, v in p.items()}
    d["Grp"] = rng.choice(["A", "B"], n)
    return pd.DataFrame(d)


def test_items_order_by_mean_largest_on_top(lik, capsys):
    fig = Chart(["q1", "q2", "q3"], data=lik)
    order = list(fig.layout.yaxis.categoryarray)
    means = lik[["q1", "q2", "q3"]].mean()
    assert order == list(means.sort_values().index)    # bottom->top
    assert [t.name for t in fig.data] == ["1", "2", "3", "4", "5"]
    assert fig.layout.barmode == "stack"
    assert fig.layout.legend.title.text == "Responses"
    out = capsys.readouterr().out
    assert "Frequencies of Responses by Variable" in out
    # the table reads top of the chart first, with each mean
    rows = [ln.split()[0] for ln in out.splitlines()
            if ln.strip().startswith("q")]
    assert rows == order[::-1]
    assert f"{means['q3']:.3f}" in out


def test_items_percent_labels_within_item(lik):
    fig = Chart(["q1", "q2", "q3"], data=lik, quiet=True)
    order = list(fig.layout.yaxis.categoryarray)
    k = order.index("q2")
    cnt = lik["q2"].value_counts().reindex([1, 2, 3, 4, 5],
                                           fill_value=0)
    for t, c in zip(fig.data, cnt):
        share = c / cnt.sum()
        want = "" if share < 0.04 else f"{100 * share:.0f}%"
        assert t.text[k] == want


def test_items_divergent_palette_and_getcolors_fix():
    # a single ending name is the usual divergent call (the string
    # was indexed as "b" and silently ignored before)
    assert getColors("reds", "blues", n=5, quiet=True) == [
        "#7A2B40FF", "#9A6B74FF", "#C6C6C6FF", "#5E7B96FF",
        "#004D7AFF"]                                    # R's values


def test_items_facet(lik, capsys):
    fig = Chart(["q1", "q2", "q3"], facet="Grp", data=lik)
    out = capsys.readouterr().out
    assert "Frequencies of Responses by Variable, Grp: A" in out
    assert "Frequencies of Responses by Variable, Grp: B" in out
    # the panels keep the order given, first item at the top
    assert list(fig.layout.yaxis.categoryarray) == ["q3", "q2", "q1"]
    assert sum(t.showlegend for t in fig.data) == 5
    with pytest.raises(ValueError, match="labels"):
        Chart(["q1", "q2"], facet="Grp", data=lik, labels="%")


def test_items_refusals(lik):
    with pytest.raises(ValueError, match="facet"):
        Chart(["q1", "q2"], by="Grp", data=lik)
    # no shared response set: a chart per variable, as R; the
    # numeric item is passed over, the categorical one charted
    fig = Chart(["q1", "Grp"], data=lik, quiet=True)
    assert [a.text for a in fig.layout.annotations] == ["Grp"]
    with pytest.raises(ValueError, match="cannot be stacked|share"):
        Chart(["q1", "Grp"], one_plot=True, data=lik)
    with pytest.raises(ValueError, match="y and stat"):
        Chart(["q1", "q2"], y="q3", data=lik)


def test_items_vertical(lik):
    # horiz=False: the items left to right, ascending mean
    fig = Chart(["q1", "q2", "q3"], horiz=False, data=lik, quiet=True)
    means = lik[["q1", "q2", "q3"]].mean()
    assert list(fig.layout.xaxis.categoryarray) == \
        list(means.sort_values().index)
    assert all(t.orientation == "v" for t in fig.data)
    # faceted vertical: the items in the order given, left to right
    fig = Chart(["q1", "q2", "q3"], horiz=False, facet="Grp",
                data=lik, quiet=True)
    assert list(fig.data[0].x) == ["q1", "q2", "q3"]


def test_items_bubble_matrix(lik, capsys):
    # the bubble plot frequency matrix: rows are items (first at the
    # bottom), columns responses; counts at most power/2.5 * max are
    # left unlabeled, as R's .dpmat.main()
    fig = Chart(["q1", "q2", "q3"], form="bubble", data=lik)
    out = capsys.readouterr().out
    assert "Sum" in out and "Mean" in out
    rows = [t.y[0] for t in fig.data]
    assert rows == ["q3", "q2", "q1"]       # plotly lists top first
    cnt = lik["q1"].value_counts().reindex([1, 2, 3, 4, 5],
                                           fill_value=0)
    allmax = max(lik[q].value_counts().max() for q in ("q1", "q2", "q3"))
    q1 = [t for t in fig.data if t.y[0] == "q1"][0]
    for c, txt in zip(cnt, q1.text):
        assert txt == ("" if c <= 0.2 * allmax else str(c))


def test_one_plot_false_panels(capsys):
    # one bar chart per categorical variable, as panels of one figure;
    # numeric variables are passed over, as R's bc.data.frame() does
    from lessPy import read_data
    emp = read_data("Employee")
    fig = Chart(["Gender", "Dept", "Plan", "Salary"], one_plot=False,
                data=emp)
    assert [a.text for a in fig.layout.annotations] == ["Gender", "Dept"]
    out = capsys.readouterr().out
    assert "--- Gender ---" in out and "--- Dept ---" in out
    assert "Chisq = 10.944, df = 4, p-value = 0.027" in out   # R
    with pytest.raises(ValueError, match="No categorical"):
        Chart(["Plan", "Salary"], one_plot=False, data=emp)
    with pytest.raises(ValueError, match="no panels"):
        Chart(["Gender", "Dept"], one_plot=False, facet="Plan",
              data=emp)


def test_one_plot_default_from_shared_responses():
    # items without one shared response set are charted one by one
    from lessPy import read_data
    emp = read_data("Employee")
    fig = Chart(["Gender", "Dept"], data=emp, quiet=True)
    assert len(fig.layout.annotations) == 2
