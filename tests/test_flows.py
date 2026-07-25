import pandas as pd
import pytest

from lessPy import Flows


@pytest.fixture
def df():
    return pd.DataFrame({
        "From": ["A", "A", "B", "B", "A", "B"],
        "Mid": ["X", "Y", "X", "Y", "X", "Y"],
        "To": ["P", "Q", "P", "Q", "Q", "P"],
        "n": [10, 5, 7, 8, 4, 6]})


def test_two_stage(df):
    f = Flows("n", "From", "Mid", data=df)
    sk = f.data[0]
    assert list(sk.node.label) == ["A", "B", "X", "Y"]
    assert len(sk.link.source) == 4          # A/B x X/Y
    # A->X aggregates the two A,X rows (10 + 4)
    idx = [i for i in range(4)
           if sk.link.source[i] == 0 and sk.link.target[i] == 2]
    assert sk.link.value[list(idx)[0]] == 14


def test_three_stage(df):
    f = Flows("n", "From", "Mid", "To", data=df, title="P")
    sk = f.data[0]
    assert list(sk.node.label) == ["A", "B", "X", "Y", "P", "Q"]
    assert len(sk.link.source) == 10         # 4 stage1-2 + 6 stage2-3
    assert "P" in f.layout.title.text        # custom title


def test_nodes_gray_and_colors(df):
    f = Flows("n", "From", "Mid", data=df, nodes_gray=True)
    colors = list(f.data[0].node.color)
    assert len(set(colors)) == 1             # all gray
    f2 = Flows("n", "From", "Mid", data=df)
    # sources colored, mids gray
    assert f2.data[0].node.color[0] != f2.data[0].node.color[2]


def test_errors(df):
    with pytest.raises(ValueError, match="data="):
        Flows("n", "From", "Mid")
