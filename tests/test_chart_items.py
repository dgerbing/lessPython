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
    with pytest.raises(ValueError, match="share"):
        Chart(["q1", "Grp"], data=lik)
    with pytest.raises(ValueError, match="y and stat"):
        Chart(["q1", "q2"], y="q3", data=lik)
