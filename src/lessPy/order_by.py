# order_by.py — analog of order_by.R.
#
# order_by(): return a copy of a data frame with its rows sorted
# by one or more columns, each ascending ("+") or descending
# ("-"), or by the special keyword "row.names" (the index) or
# "random" (a shuffle). Prints the sort specification unless quiet.
#
# Column names are passed as strings, as elsewhere in lessPy.
# pandas sorts categoricals by their category order natively, so
# R's xtfrm() handling of factors needs no special case. The sort
# is stable (kind="stable"), matching R's stable order().

import numpy as np
import pandas as pd


def order_by(data, by, direction=None, seed=None, quiet=False):
    """Sort the rows of data and return the sorted copy. by is a
    column name or list of names, or the keyword "row.names" (sort
    by the index) or "random" (shuffle). direction is "+"
    (ascending, the default) or "-" (descending), one per sort
    column. seed makes a "random" sort reproducible.
    R analog: order_by()"""
    by_list = [by] if isinstance(by, str) else list(by)

    if not quiet:
        print("\nSort Specification")

    if by_list == ["row.names"]:
        d = _by_row_names(data, direction, quiet)
    elif by_list == ["random"]:
        d = _by_random(data, seed, quiet)
    else:
        d = _by_columns(data, by_list, direction, quiet)

    if not quiet:
        print()
    return d


def _dir_word(sign):
    return "ascending" if sign == "+" else "descending"


def _check_dirs(direction, n):
    if direction is None:
        return ["+"] * n
    dirs = [direction] if isinstance(direction, str) \
        else list(direction)
    if len(dirs) != n:
        raise ValueError(
            f"number of sort columns ({n}) must equal the number "
            f"of direction values ({len(dirs)})")
    for s in dirs:
        if s not in ("+", "-"):
            raise ValueError(
                f"direction value {s!r}: use '+' (ascending) or "
                "'-' (descending)")
    return dirs


def _by_row_names(data, direction, quiet):
    if direction is not None and (
            not isinstance(direction, str)
            and len(direction) != 1):
        raise ValueError(
            "sorting by row.names takes exactly one direction "
            "value ('+' or '-')")
    sign = _check_dirs(direction, 1)[0]
    if not quiet:
        print(f"  row.names --> {_dir_word(sign)}")
    return data.sort_index(ascending=(sign == "+"),
                           kind="stable")


def _by_random(data, seed, quiet):
    if not quiet:
        print("  random")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data))
    return data.iloc[order]


def _by_columns(data, by_list, direction, quiet):
    missing = [c for c in by_list if c not in data.columns]
    if missing:
        raise ValueError(f"column(s) not found: {missing}")
    dirs = _check_dirs(direction, len(by_list))
    if not quiet:
        for c, s in zip(by_list, dirs):
            print(f"  {c} --> {_dir_word(s)}")
    ascending = [s == "+" for s in dirs]
    return data.sort_values(by=by_list, ascending=ascending,
                            kind="stable")
