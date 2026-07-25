import numpy as np
import pandas as pd
import pytest

from lessPy import Chart, bubble_plotly


@pytest.fixture
def d():
    rng = np.random.default_rng(7)
    n = 60
    return pd.DataFrame({
        "Dept": rng.choice(["ACCT", "ADMN", "FINC", "MKTG"], n),
        "Gender": rng.choice(["F", "M"], n),
        "Salary": rng.normal(60000, 12000, n).round(2),
    })


def test_bubble_1d(d):
    fig = Chart("Dept", data=d, form="bubble")
    assert len(fig.data) == 1
    tr = fig.data[0]
    assert list(tr.x) == sorted(d["Dept"].unique())
    assert set(tr.y) == {0}                    # bubbles on a line
    # diameters ordered like the counts (power is monotonic)
    counts = d["Dept"].value_counts().sort_index().to_numpy()
    sizes = np.array(tr.marker.size)
    assert (np.argsort(sizes).tolist() ==
            np.argsort(counts).tolist())
    # largest bubble gets the full 2*radius*dpi diameter
    assert sizes.max() == pytest.approx(2 * 0.50 * 96)
    # one stem per bubble
    assert len(fig.layout.shapes) == len(counts)
    assert fig.layout.title.text == "Count of Dept"


def test_bubble_size_formula():
    s = pd.Series([4, 16], index=["a", "b"])
    fig = bubble_plotly(s, power=0.5, radius=0.50)
    sizes = fig.data[0].marker.size
    # sqrt scaling: 4 -> 2, 16 -> 4; max = 96 px
    assert sizes[1] == pytest.approx(96)
    assert sizes[0] == pytest.approx(48)


def test_bubble_2d_matrix(d):
    fig = Chart("Dept", by="Gender", data=d, form="bubble")
    assert len(fig.data) == 2                  # one trace per group
    tr0 = fig.data[0]
    assert set(tr0.y) == {"F"}                 # row per by level
    want = pd.crosstab(d["Gender"], d["Dept"]).loc["F"]
    assert list(tr0.hovertext) == [f"{v:.0f}" for v in want]
    # first by level renders as the top row
    assert list(fig.layout.yaxis.categoryarray) == ["M", "F"]
    assert fig.layout.yaxis.title.text == "Gender"


def test_bubble_stat(d):
    fig = Chart("Dept", y="Salary", stat="mean", data=d,
                form="bubble")
    want = d.groupby("Dept")["Salary"].mean().sort_index()
    got = [float(t) for t in fig.data[0].hovertext]
    assert got == pytest.approx(want.round(2).tolist(), abs=0.01)
    # aggregated data: input labels, not %
    shown = [t for t in fig.data[0].text if t]
    assert all("%" not in t for t in shown)


def test_bubble_label_min_px():
    s = pd.Series([1, 100], index=["a", "b"])
    fig = bubble_plotly(s, labels="input")
    # tiny bubble suppresses its label; big one shows it
    assert fig.data[0].text[0] == ""
    assert fig.data[0].text[1] != ""
