import pandas as pd
import pytest

from lessPy import date_infer, format_date_labels

# verified against lessR date.infer() / .format_date_labels()


def _dates(v):
    return [str(d.date()) for d in date_infer(v)]


def test_numeric_formats_match_R():
    # order inferred from the component values
    assert _dates(["08/18/2024", "03/05/2024",
                   "11/30/2024"])[:2] == ["2024-08-18",
                                          "2024-03-05"]
    assert _dates(["18-08-2024", "05-03-2024",
                   "30-11-2024"])[0] == "2024-08-18"   # d-m-Y
    assert _dates(["2024-08-18", "2024-03-05"])[0] == \
        "2024-08-18"                                   # Y-m-d
    # 2-digit year
    assert _dates(["18-08-24", "05-03-24",
                   "30-11-24"])[0] == "2024-08-18"


def test_monthname_and_quarter():
    assert _dates(["2024Jan", "2024Feb", "2024Mar"]) == \
        ["2024-01-01", "2024-02-01", "2024-03-01"]
    assert _dates(["2024 Q1", "2024 Q2", "2024 Q3"]) == \
        ["2024-01-01", "2024-04-01", "2024-07-01"]


def test_non_date_returned_unchanged():
    x = ["apple", "banana"]
    out = date_infer(x)
    assert list(out) == x                   # too short / not a date


def test_format_date_labels_match_R():
    d = ["2024-01-15", "2024-08-18"]
    assert list(format_date_labels(d, "years")) == ["2024",
                                                    "2024"]
    assert list(format_date_labels(d, "quarters")) == \
        ["2024 Q1", "2024 Q3"]
    assert list(format_date_labels(d, "months")) == \
        ["Jan 2024", "Aug 2024"]
    assert list(format_date_labels(d, "days")) == \
        ["15 Jan 2024", "18 Aug 2024"]
    with pytest.raises(ValueError, match="ts_unit"):
        format_date_labels(d, "decades")


def test_xy_wires_date_infer():
    from lessPy import XY
    import io
    import contextlib
    d = pd.DataFrame({
        "Month": ["2024-01-15", "2024-02-15", "2024-03-15",
                  "2024-04-15", "2024-05-15"],
        "Price": [10.0, 12.5, 11.0, 13.2, 14.1]})
    with contextlib.redirect_stdout(io.StringIO()):
        f = XY("Month", "Price", data=d)
        # same as passing an actual datetime column
        f2 = XY("Month", "Price",
                data=d.assign(Month=pd.to_datetime(d.Month)))
    tr = f.data[0]
    assert "lines" in tr.mode                     # time series
    assert str(tr.x[0]).startswith("2024-01-15")  # parsed date
    assert list(tr.x) == list(f2.data[0].x)       # matches datetime


def test_xy_nondate_string_unaffected():
    from lessPy import XY
    d = pd.DataFrame({"City": ["London", "Berlin", "Madrid"],
                      "Pop": [9.0, 3.6, 3.2]})
    # a categorical x with numeric y is not a date -> XY's normal
    # redirect, not a datetime parse
    with pytest.raises(TypeError, match="numerical"):
        XY("City", "Pop", data=d)
