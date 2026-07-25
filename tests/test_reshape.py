import pandas as pd
import pytest

from lessPy import read_data, reshape_long, reshape_wide


def test_rect_defaults():
    d = read_data("Anova_rb")
    v = ["sup1", "sup2", "sup3", "sup4"]
    dl = reshape_long(d, v)
    assert list(dl.columns) == ["ID", "Person", "Group",
                                "Response"]
    assert len(dl) == 28
    assert dl["ID"].iloc[0] == "ID1"
    assert dl["Group"].iloc[0] == "sup1"
    assert dl["Response"].iloc[2] == 8


def test_rect_existing_id_no_prefix():
    d = read_data("Anova_rb")
    dl = reshape_long(d, ["sup1", "sup2", "sup3", "sup4"],
                      ID="Person", prefix=None)
    assert list(dl.columns) == ["Person", "Group", "Response"]
    # round-trips back to wide
    dw = reshape_wide(dl, widen="Group", response="Response",
                      ID="Person")
    assert list(dw.columns) == ["Person", "sup1", "sup2",
                                "sup3", "sup4"]
    assert dw.set_index("Person").loc["p3", "sup1"] == 8


def test_wide_numeric_widen_keeps_prefix():
    d = pd.DataFrame({"ID": ["a", "a", "b", "b"],
                      "Time": [1, 2, 1, 2],
                      "Score": [10, 20, 30, 40]})
    dw = reshape_wide(d, widen="Time", response="Score", ID="ID")
    assert list(dw.columns) == ["ID", "Score_1", "Score_2"]


def test_square(capsys):
    m = pd.DataFrame([[1, 2, 3], [4, 5, 6]],
                     index=["r1", "r2"], columns=["cA", "cB", "cC"])
    sq = reshape_long(m, shape="square")
    assert list(sq.columns) == ["Row", "Col", "Response"]
    # column-major: Row varies fastest within Col
    assert list(sq["Row"])[:2] == ["r1", "r2"]
    assert list(sq["Col"])[:2] == ["cA", "cA"]
    assert list(sq["Row"].cat.categories) == ["r2", "r1"]  # rev
    sq2 = reshape_long(m, shape="square", group="V",
                       reverse_y=False)
    assert list(sq2.columns) == ["V1", "V2", "Response"]
    assert list(sq2["Row" if False else "V1"].cat.categories) \
        == ["r1", "r2"]
