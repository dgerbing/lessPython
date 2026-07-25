# details.py — analog of details.R.
#
# details(): a diagnostic report on a data frame. Prints the
# dimensions and row names, a legend of the data types present, a
# per-variable table (type, non-missing / missing / unique counts,
# and the first and last data values), a suggestion when a text
# column is a unique per-row ID, a note on numeric variables with
# few unique values, and a missing-data analysis. Returns a
# DetailsResults with the per-variable summary as a DataFrame.
#
# R's storage types map onto pandas dtypes: an unordered/ordered
# pandas Categorical is a factor/ordfactor, object/string is
# "character", and integer/float/datetime/bool map to integer/
# double/Date/logical. R reads variable labels and units from the
# data-frame attributes attr(data, "variable.labels"/"units"); the
# pandas analog is data.attrs["variable_labels"/"variable_units"]
# (each a name -> text mapping).

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from .utils import get_option

_TYPE_ORDER = ("factor", "ordfactor", "character", "integer",
               "Date", "double", "logical")
_TYPE_DESC = {
    "factor":    "Non-numeric categories, read as unordered "
                 "categories",
    "ordfactor": "Ordered, non-numeric categories",
    "character": "Non-numeric data values",
    "integer":   "Numeric data values, integers only",
    "Date":      "Date with year, month and day",
    "double":    "Numeric data values with decimal digits",
    "logical":   "Boolean True/False values",
}


class DetailsResults:
    """Result of details(): the dimensions, the total/proportion of
    missing values, the per-variable summary DataFrame, a detected
    ID column (or None), and the rows with missing data (or None)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return (f"<lessPy details: {self.n_var} variables, "
                f"{self.n_obs} rows>")


def _dash(n):
    print("-" * n)


def _type_label(s):
    dt = s.dtype
    if isinstance(dt, pd.CategoricalDtype):
        return "ordfactor" if dt.ordered else "factor"
    if pdt.is_bool_dtype(dt):
        return "logical"
    if pdt.is_datetime64_any_dtype(dt):
        return "Date"
    if pdt.is_integer_dtype(dt):
        return "integer"
    if pdt.is_float_dtype(dt):
        # pandas stores an integer column that has any missing value
        # as float64; if every non-missing value is whole, report it
        # as integer to match R's storage-based view (else double).
        v = s.dropna()
        v = v[pd.Series(np.isfinite(v), index=v.index)]
        if len(v) and (v % 1 == 0).all():
            return "integer"
        return "double"
    return "character"


def _cell(x):
    if pd.isna(x):
        return "NA"
    if isinstance(x, float) and np.isfinite(x) and x == int(x):
        return str(int(x))
    return str(x)


def _first_last(col, n_obs):
    """First up to three and last up to three values, joined with
    ' ... ', trimmed toward the ends as the line grows (the width
    rule of details.R)."""
    v = col.tolist()
    n1 = _cell(v[0])
    n2 = _cell(v[1]) if n_obs >= 2 else ""
    n3 = _cell(v[2]) if n_obs >= 3 else ""
    e3 = _cell(v[n_obs - 3]) if n_obs >= 5 else ""
    e2 = _cell(v[n_obs - 2]) if n_obs >= 4 else ""
    e1 = _cell(v[n_obs - 1]) if n_obs >= 2 else ""
    tot = len(" ".join(x for x in (n1, n2, n3, e3, e2, e1) if x))
    if tot > 34:
        n3 = e3 = ""
    if tot > 58:
        n2 = e2 = ""
    fp = " ".join(x for x in (n1, n2, n3) if x)
    lp = " ".join(x for x in (e3, e2, e1) if x)
    return f"{fp} ... {lp}" if lp else fp


def details(data=None, n_mcut=1, max_lines=30, miss_show=30,
            miss_zero=False, miss_matrix=False, var_labels=False,
            brief=None, n_cat=None):
    """Diagnostic report on a data frame: dimensions, data-type
    legend, a per-variable table of type / non-missing / missing /
    unique counts with first and last values, ID-column and
    numeric-category notes, and a missing-data analysis. Returns a
    DetailsResults with the per-variable summary. R analog:
    details()"""
    if data is None:
        raise ValueError("specify a data frame: data=")
    if brief is None:
        brief = get_option("brief", False)
    if n_cat is None:
        n_cat = get_option("n_cat", 0)

    n_var = data.shape[1]
    n_obs = data.shape[0]
    if n_obs == 0 or n_var == 0:
        raise ValueError("data has no rows or no columns")
    max_lines = min(max_lines, n_obs)
    n_miss_tot = int(data.isna().sum().sum())

    names = list(data.columns)
    types = [_type_label(data[c]) for c in names]
    n_val = [int(data[c].notna().sum()) for c in names]
    n_miss = [int(data[c].isna().sum()) for c in names]
    n_uniq = [int(data[c].dropna().nunique()) for c in names]

    _print_header(data, n_var, n_obs, brief)
    _print_types(types)
    first_last = [_first_last(data[c], n_obs) for c in names]
    maybe_id, id_col = _print_table(names, types, n_val, n_miss,
                                    n_uniq, first_last, n_var)

    if maybe_id is not None and not var_labels:
        _print_id_note(maybe_id, id_col)

    if not brief:
        _print_num_cat(names, types, n_uniq, n_cat)

    missing_rows = None
    if not brief:
        if n_miss_tot > 0:
            missing_rows = _print_missing(
                data, n_var, n_obs, n_miss_tot, n_mcut, miss_show,
                miss_zero, miss_matrix)
        else:
            print("No missing data\n")

    _print_meta(data, max_lines)

    summary = pd.DataFrame(
        {"Type": types, "Values": n_val, "Missing": n_miss,
         "Unique": n_uniq}, index=pd.Index(names, name="Variable"))
    prop_miss = round(n_miss_tot / (n_var * n_obs), 3)
    return DetailsResults(
        n_var=n_var, n_obs=n_obs, n_miss=n_miss_tot,
        prop_miss=prop_miss, summary=summary, maybe_id=maybe_id,
        missing_rows=missing_rows)


# --- report sections -----------------------------------------------

def _print_header(data, n_var, n_obs, brief):
    if brief:
        return
    print()
    _dash(58)
    print(f"Dimensions: {n_var} variables over {n_obs} rows of "
          "data")
    print()
    idx = data.index
    if n_obs >= 2:
        print(f"First two row names: {idx[0]}    {idx[1]}")
        print(f"Last two row names:  {idx[-2]}    {idx[-1]}")
    else:
        print(f"Only one row, row name: {idx[0]}")
    _dash(58)


def _print_types(types):
    present = [t for t in _TYPE_ORDER if t in types]
    print("Data Types")
    _dash(60)
    for t in present:
        print(f"{t}: {_TYPE_DESC[t]}")
    _dash(60)
    print()


def _print_table(names, types, n_val, n_miss, n_uniq, first_last,
                 n_var):
    w_num = max(2, len(str(n_var)))
    w_nam = max(len("Variable"), *(len(x) for x in names))
    w_typ = max(len("Type"), *(len(t) for t in types))
    w_val = max(len("Values"), *(len(str(x)) for x in n_val))
    w_mis = max(len("Missing"), *(len(str(x)) for x in n_miss))
    w_uni = max(len("Unique"), *(len(str(x)) for x in n_uniq))

    hdr = (f"{'':>{w_num}}  {'Variable':<{w_nam}}  "
           f"{'Type':<{w_typ}}  {'Values':>{w_val}}  "
           f"{'Missing':>{w_mis}}  {'Unique':>{w_uni}}  "
           "First and last values")
    _dash(len(hdr))
    print(hdr)
    _dash(len(hdr))

    maybe_id, id_col = None, 0
    for i in range(n_var):
        print(f"{i + 1:>{w_num}}  {names[i]:<{w_nam}}  "
              f"{types[i]:<{w_typ}}  {n_val[i]:>{w_val}}  "
              f"{n_miss[i]:>{w_mis}}  {n_uniq[i]:>{w_uni}}  "
              f"{first_last[i]}")
        if (n_uniq[i] == n_val[i]
                and types[i] in ("factor", "ordfactor",
                                 "character")):
            maybe_id, id_col = names[i], i + 1
    _dash(len(hdr))
    return maybe_id, id_col


def _print_id_note(maybe_id, id_col):
    print("\n")
    print(f"For the column {maybe_id}, each row of data is unique. "
          "Are these values")
    print("a unique ID for each row? To define as a row name, "
          "re-read the data")
    print(f"file with index_col={id_col} in your read_data() / "
          "pandas call.")


def _print_num_cat(names, types, n_uniq, n_cat):
    if n_cat <= 0:
        return
    flagged = [names[j] for j in range(len(names))
               if types[j] == "double" and n_uniq[j] <= n_cat]
    if not flagged:
        return
    print("\n")
    print(f"Each of these variables is numeric but has fewer than "
          f"{n_cat}")
    print("unique values. Perhaps they are categorical. Consider "
          "converting")
    print("each to a category, or set n_cat, e.g. "
          "set_option('n_cat', 4).")
    _dash(63)
    for nm in flagged:
        print(nm)
    _dash(63)
    print()


def _print_missing(data, n_var, n_obs, n_miss_tot, n_mcut,
                   miss_show, miss_zero, miss_matrix):
    print("Missing Data Analysis")
    _dash(60)
    mcut = 0 if miss_zero else n_mcut
    per_row = data.isna().sum(axis=1)
    bad = []
    for i in range(n_obs):
        if int(per_row.iloc[i]) >= mcut and len(bad) < miss_show:
            bad.append(i)
        if len(bad) == miss_show:
            break
    n_lines = len(bad)

    print(f"Number of cells in the data table: {n_var * n_obs}")
    print(f"Number of missing data values: {n_miss_tot}")
    print("Proportion of missing data values: "
          f"{round(n_miss_tot / (n_var * n_obs), 3)}")
    label = ("Number of rows of data listed: " if miss_zero
             else "Number of rows of data with missing values: ")
    print(f"{label}{n_lines}")
    _dash(60)
    print()
    rows = data.iloc[bad]
    print(rows.to_string())
    _dash(60)
    print()

    if miss_matrix:
        print("\nTable of Missing Values, 1 means missing")
        print(data.isna().astype(int).to_string())
    return rows


def _print_meta(data, max_lines):
    labels = data.attrs.get("variable_labels")
    units = data.attrs.get("variable_units")
    if labels:
        _print_named(labels, "Variable Labels", max_lines)
    if units:
        _print_named(units, "Variable Units", max_lines)
    print()


def _print_named(mapping, header, max_lines):
    items = list(mapping.items())[:max_lines]
    w = max(len("Variable Names"),
            *(len(str(k)) for k, _ in items)) + 1
    print(f"\n{'Variable Names':<{w}} {header}")
    width = max(len(f"{'Variable Names':<{w}} {header}"),
                *(len(f"{k:<{w}} {v}") for k, v in items))
    _dash(min(width + 1, 80))
    for k, v in items:
        print(f"{str(k):<{w}} {'' if pd.isna(v) else v}")
    _dash(min(width + 1, 80))
