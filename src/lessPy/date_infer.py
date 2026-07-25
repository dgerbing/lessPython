# date_infer.py — analog of date.infer.R / .charToDate.
#
# date_infer(): infer the format of a column of date strings and
# convert to pandas datetimes. Handles the numeric delimited
# forms (Y/m/d, d/m/Y, m/d/Y with "/", "-", or "."), the
# month-name form "2024Jan", and the quarter form "2024 Q3" —
# the last two beyond pandas' own inference. The month/day/year
# order of a numeric date is inferred from the component values,
# as R's .charToDate.

import re

import numpy as np
import pandas as pd

_MONTHS = ("Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec")


def date_infer(x, quiet=True):
    """Infer the date format of the string column x (a list or
    Series) and return a datetime Series. Recognizes numeric
    dates delimited by / - or . (order inferred), "2024Jan"
    month-name dates, and "2024 Q3" quarter dates. Returns the
    input unchanged if no date format is recognized.
    R analog: date.infer()"""
    s = pd.Series(list(x) if not isinstance(x, pd.Series) else x)
    vals = s.dropna().astype(str)
    if vals.empty:
        return s
    first = vals.iloc[0]
    n_ch = len(first)
    if not 6 <= n_ch <= 10:
        return s

    if re.search(_MONTHS, first):          # "2024Jan"
        t = s.astype(str).str.replace(" ", "", regex=False)
        return pd.to_datetime(t.str[:4] + "-" + t.str[4:7]
                              + "-01", format="%Y-%b-%d")
    if re.search(r"Q[1-4]", first):        # "2024 Q3"
        t = s.astype(str).str.replace(r"\s+", "", regex=True)
        yr = t.str.split("Q").str[0].astype(int)
        q = t.str.split("Q").str[1].astype(int)
        month = 1 + (q - 1) * 3
        return pd.to_datetime(
            {"year": yr, "month": month, "day": 1})

    # numeric date: the delimiter is the punctuation that appears
    # exactly twice in the first value
    punct = next((p for p in ("/", "-", ".")
                  if first.count(p) == 2), None)
    if punct is None:
        return s
    return _char_to_date(s.astype(str), punct, quiet)


def _char_to_date(char, punct, quiet):
    parts = char.str.split(re.escape(punct), expand=True)
    c1 = pd.to_numeric(parts[0], errors="coerce")
    c2 = pd.to_numeric(parts[1], errors="coerce")
    c3 = pd.to_numeric(parts[2], errors="coerce")
    mx1, mx2, mx3 = c1.max(), c2.max(), c3.max()
    unq1, unq2 = c1.nunique(), c2.nunique()
    if np.isnan(mx1) or np.isnan(mx2):
        raise ValueError("at least one date has non-numeric "
                         "characters where a number is expected")

    fmt = None
    if mx1 > 31:
        fmt = f"%Y{punct}%m{punct}%d"
    elif mx1 > 12:
        fmt = f"%d{punct}%m{punct}%Y"
    elif mx2 > 12:
        fmt = f"%m{punct}%d{punct}%Y"
    elif unq2 <= 12 and unq2 in (2, 4, 12):
        fmt = f"%d{punct}%m{punct}%Y"
    elif unq1 <= 12 and unq1 in (2, 4, 12):
        fmt = f"%m{punct}%d{punct}%Y"
    if fmt is None:
        raise ValueError(
            "the date format could not be inferred; supply "
            "dates as one of Y-m-d, d-m-Y, or m-d-Y")

    # a 2-digit year in the 3rd position takes %y
    if fmt[1] != "Y" and mx3 <= 99:
        fmt = fmt.replace("Y", "y")
    if not quiet:
        print(f"Best guess for the date format: {fmt}")
    return pd.to_datetime(char, format=fmt)


def format_date_labels(dates, ts_unit):
    """Format a column of dates as axis labels for a time unit:
    years "2024", quarters "2024 Q3", months "Aug 2024", and
    weeks / days "18 Aug 2024". Returns a Series of strings.
    R analog: .format_date_labels()"""
    if ts_unit not in ("years", "quarters", "months", "weeks",
                        "days", "days7", "unknown"):
        raise ValueError(
            'ts_unit: "years", "quarters", "months", "weeks", '
            '"days", "days7"')
    d = pd.to_datetime(pd.Series(list(dates)
                                 if not isinstance(dates,
                                                   pd.Series)
                                 else dates))
    if ts_unit == "years":
        return d.dt.strftime("%Y")
    if ts_unit == "quarters":
        return (d.dt.strftime("%Y") + " Q"
                + d.dt.quarter.astype(str))
    if ts_unit == "months":
        return d.dt.strftime("%b %Y")
    return d.dt.strftime("%d %b %Y")       # weeks, days, days7
