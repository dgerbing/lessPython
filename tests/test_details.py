import numpy as np
import pandas as pd
import pytest

from lessPy import read_data, details
from lessPy.details import DetailsResults, _type_label


def test_employee_summary_matches_r():
    d = read_data("Employee")
    r = details(d)
    assert isinstance(r, DetailsResults)
    assert (r.n_var, r.n_obs) == (8, 37)
    # counts verified against R's details(d)
    assert r.n_miss == 4
    assert r.prop_miss == 0.014
    assert list(r.summary.loc["Years"]) == ["integer", 36, 1, 16]
    assert list(r.summary.loc["Salary"]) == ["double", 37, 0, 37]
    assert list(r.summary.loc["Gender"]) == ["character", 37, 0, 2]
    # three rows carry the four missing values
    assert r.missing_rows is not None
    assert len(r.missing_rows) == 3


def test_type_labels():
    df = pd.DataFrame({
        "i": pd.array([1, 2, 3], dtype="int64"),
        "x": [1.5, 2.5, 3.5],
        "whole_na": [1.0, np.nan, 3.0],   # int coerced to float by NA
        "s": ["a", "b", "c"],
        "b": [True, False, True],
        "cat": pd.Categorical(["lo", "hi", "lo"]),
        "ocat": pd.Categorical(["lo", "hi", "lo"], ordered=True),
        "dt": pd.to_datetime(["2024-01-01", "2024-02-01",
                              "2024-03-01"]),
    })
    assert _type_label(df["i"]) == "integer"
    assert _type_label(df["x"]) == "double"
    assert _type_label(df["whole_na"]) == "integer"
    assert _type_label(df["s"]) == "character"
    assert _type_label(df["b"]) == "logical"
    assert _type_label(df["cat"]) == "factor"
    assert _type_label(df["ocat"]) == "ordfactor"
    assert _type_label(df["dt"]) == "Date"


def test_id_column_detected():
    df = pd.DataFrame({"code": ["A1", "A2", "A3"],
                       "val": [10, 20, 30]})
    r = details(df)
    # every 'code' value is unique -> flagged as a possible ID
    assert r.maybe_id == "code"


def test_no_missing():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    r = details(df)
    assert r.n_miss == 0
    assert r.prop_miss == 0.0
    assert r.missing_rows is None


def test_variable_labels_from_attrs(capsys):
    df = pd.DataFrame({"Salary": [1, 2], "Years": [3, 4]})
    df.attrs["variable_labels"] = {"Salary": "Annual pay",
                                   "Years": "Years employed"}
    details(df)
    out = capsys.readouterr().out
    assert "Variable Labels" in out
    assert "Annual pay" in out


def test_errors():
    with pytest.raises(ValueError, match="specify a data frame"):
        details()
    with pytest.raises(ValueError, match="no rows or no columns"):
        details(pd.DataFrame())
