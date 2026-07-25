import os
import tempfile

import pandas as pd
import pytest

from lessPy import VariableLabels


def _df():
    return pd.DataFrame({"Salary": [1, 2], "Years": [3, 4]})


def test_set_from_dict_and_get():
    d = _df()
    f = VariableLabels(d, {"Salary": "Annual pay"}, quiet=True)
    # stored on the frame's attrs
    assert d.attrs["variable_labels"] == {"Salary": "Annual pay"}
    # returned labels frame
    assert list(f.index) == ["Salary"]
    assert f.loc["Salary", "label"] == "Annual pay"
    # get mode returns the same content
    g = VariableLabels(d, quiet=True)
    assert g.loc["Salary", "label"] == "Annual pay"


def test_units_add_column():
    d = _df()
    f = VariableLabels(d, {"Salary": "Pay", "Years": "Tenure"},
                       units={"Salary": "USD"}, quiet=True)
    assert "unit" in f.columns
    assert f.loc["Salary", "unit"] == "USD"
    assert f.loc["Years", "unit"] == ""          # missing -> blank
    assert d.attrs["variable_units"] == {"Salary": "USD"}


def test_update_merges():
    d = _df()
    VariableLabels(d, {"Salary": "Pay"}, quiet=True)
    VariableLabels(d, {"Years": "Tenure"}, quiet=True)
    assert d.attrs["variable_labels"] == {"Salary": "Pay",
                                          "Years": "Tenure"}


def test_from_csv_file():
    d = _df()
    p = os.path.join(tempfile.gettempdir(), "vl_test.csv")
    with open(p, "w") as fh:
        fh.write("Salary,Annual pay,USD\nYears,Years employed\n")
    VariableLabels(d, p, quiet=True)
    assert d.attrs["variable_labels"]["Years"] == "Years employed"
    assert d.attrs["variable_units"]["Salary"] == "USD"


def test_get_when_none():
    d = _df()
    f = VariableLabels(d, quiet=True)
    assert list(f.index) == []


def test_unknown_column_rejected():
    d = _df()
    with pytest.raises(ValueError, match="not column"):
        VariableLabels(d, {"Nope": "x"}, quiet=True)


def test_bad_types():
    with pytest.raises(TypeError, match="DataFrame"):
        VariableLabels([1, 2, 3])
    with pytest.raises(TypeError, match="dict"):
        VariableLabels(_df(), 42)
