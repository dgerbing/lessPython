# pytest analog of R CMD check tests; run from lessPy/:  pytest

import pandas as pd
import pytest

from lessPy import bc_plotly
from lessPy.plotly_utils import contrast_text_for_hex, to_hex
from lessPy.utils import pretty


@pytest.fixture
def counts_1d():
    return pd.Series([21, 22, 16, 11, 14, 9],
                     index=["Small", "Midsize", "Compact",
                            "Sporty", "Large", "Van"],
                     name="Count").rename_axis("Type")


@pytest.fixture
def counts_2d(counts_1d):
    return pd.DataFrame(
        [[7, 10, 9, 8, 7, 5], [14, 12, 7, 3, 7, 4]],
        index=pd.Index(["nonUSA", "USA"], name="Source"),
        columns=counts_1d.index)


def test_1d_basic(counts_1d):
    fig = bc_plotly(counts_1d)
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == list(counts_1d.index)
    assert list(fig.data[0].y) == list(counts_1d.values)
    # default labels mode is "input"
    assert fig.data[0].text[0] == "21.00"
    # category order preserved from the Series
    assert list(fig.layout.xaxis.categoryarray) == \
        list(counts_1d.index)


def test_1d_percent_labels(counts_1d):
    fig = bc_plotly(counts_1d, labels="%")
    assert fig.data[0].text[0] == "23%"      # 21/93


def test_1d_horiz(counts_1d):
    fig = bc_plotly(counts_1d, horiz=True)
    assert fig.data[0].orientation == "h"
    assert list(fig.data[0].y) == list(counts_1d.index)


def test_2d_stack_and_group(counts_2d):
    fig = bc_plotly(counts_2d)
    assert len(fig.data) == 2                # one trace per by level
    assert fig.layout.barmode == "stack"
    assert fig.data[0].name == "nonUSA"
    fig2 = bc_plotly(counts_2d, beside=True)
    assert fig2.layout.barmode == "group"


def test_2d_hover_has_both_percents(counts_2d):
    fig = bc_plotly(counts_2d)
    assert "% of total" in fig.data[0].hovertemplate
    assert "% of Type" in fig.data[0].hovertemplate


def test_labels_input_suppresses_pct_hover(counts_1d):
    fig = bc_plotly(counts_1d, labels="input")
    assert "% of total" not in fig.data[0].hovertemplate


def test_to_hex():
    assert to_hex("#abc") == "#aabbcc"
    assert to_hex("gray85") == "#D9D9D9"
    assert to_hex("rgb(255, 0, 0)") == "#FF0000"
    assert to_hex("off") == "#FFFFFF00"


def test_contrast():
    assert contrast_text_for_hex("#000000") == "#FFFFFF"
    assert contrast_text_for_hex("#FFFF99") == "#000000"


def test_pretty():
    assert pretty(0, 22) == [0, 5, 10, 15, 20, 25]
    assert 0.0 in pretty(0, 0.83)
