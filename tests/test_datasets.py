import pandas as pd
import pytest

from lessPy import datasets, read_data


def test_all_datasets_load():
    names = datasets()
    assert len(names) == 17
    for nm in names:
        d = read_data(nm)
        assert isinstance(d, pd.DataFrame)
        assert len(d) > 0


def test_employee_faithful():
    e = read_data("Employee")
    assert e.shape == (37, 8)
    assert e.index[0] == "Ritchie, Darnell"     # R row name
    assert int(e["Years"].isna().sum()) == 1
    assert round(e["Salary"].mean(), 2) == 83795.56
    assert e["Dept"].value_counts()["SALE"] == 15


def test_special_cases():
    assert read_data("Cars93").index[0] == "Integra"
    assert pd.api.types.is_datetime64_any_dtype(
        read_data("StockPrice")["Month"])
    lbl = read_data("Employee_lbl")
    assert list(lbl.columns) == ["label"]
    assert "Years" in lbl.index


def test_bad_name():
    with pytest.raises(ValueError, match="no bundled dataset"):
        read_data("Nope")
