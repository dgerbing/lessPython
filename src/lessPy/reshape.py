# reshape.py — analog of reshape_long.R and reshape_wide.R.
#
# reshape_long(): wide -> long, either the "rect" melt of a set of
# columns into a group/response pair, or the "square" conversion
# of a matrix (rows x cols) into a Row/Col/value long form for a
# heat map. reshape_wide(): the inverse long -> wide pivot.
# Thin wrappers over pandas melt / pivot that reproduce lessR's
# column naming, ID handling, and (square) factor-level order.

import numpy as np
import pandas as pd

_DEFAULT = object()          # prefix sentinel: default to the ID


def reshape_long(data, transform=None, group=None,
                 response="Response", ID="ID", prefix=_DEFAULT,
                 sep="", shape="rect", reverse_y=None):
    """Reshape wide data to long. shape="rect" stacks the
    transform columns into a group column (their names) and a
    response column (their values), adding an ID for the original
    row. shape="square" turns a matrix (a correlation or
    frequency table) into Row/Col/value long form for a heat map.
    R analog: reshape_long()"""
    if shape not in ("rect", "square"):
        raise ValueError('shape: "rect" or "square"')
    if shape == "square":
        return _reshape_square(data, group, response, reverse_y)

    if transform is None:
        raise ValueError("reshape_long needs transform=, the "
                         "columns to stack into long form")
    transform = [transform] if isinstance(transform, str) \
        else list(transform)
    grp = "Group" if group is None else group
    id_vars = [c for c in data.columns if c not in transform]
    long = data.melt(id_vars=id_vars, value_vars=transform,
                     var_name=grp, value_name=response)

    if ID is not None:
        if prefix is _DEFAULT:
            prefix = ID
        if ID not in data.columns:        # synthetic row id 1..n
            long.insert(0, ID, np.tile(
                np.arange(1, len(data) + 1), len(transform)))
        if prefix is not None:
            long[ID] = [f"{prefix}{sep}{v}" for v in long[ID]]
            long = long[[ID] + [c for c in long.columns
                                if c != ID]]
    return long.reset_index(drop=True)


def _reshape_square(data, group, response, reverse_y):
    """Matrix -> long, column-major (Row varies fastest within
    Col), as R's as.data.frame(as.table()). ~ reshape_long square"""
    var1, var2 = (("Row", "Col") if group is None
                  else (f"{group}1", f"{group}2"))
    df = pd.DataFrame(data)
    rows = [str(r) for r in df.index]
    cols = [str(c) for c in df.columns]
    tmp = df.copy()
    tmp.index, tmp.columns = rows, cols
    tmp.index.name = var1
    long = tmp.reset_index().melt(id_vars=var1, var_name=var2,
                                  value_name=response)
    if reverse_y is None:
        reverse_y = True
        print('reshape_long: for shape="square", row levels are '
              "reversed by default so the first row appears at "
              'the top of a heat map y-axis.\n  Set '
              "reverse_y=False to keep the original row order.")
    long[var1] = pd.Categorical(
        long[var1], categories=rows[::-1] if reverse_y else rows)
    long[var2] = pd.Categorical(long[var2], categories=cols)
    return long


def reshape_wide(data, widen, response, ID, prefix=None,
                 sep="_"):
    """Reshape long data to wide: each value of widen becomes a
    column, filled from response, one row per ID. Extra columns
    are dropped. By default the response prefix on the new column
    names is kept only when widen is numeric. R analog:
    reshape_wide()"""
    wide = data[[ID, widen, response]].pivot(
        index=ID, columns=widen, values=response)
    if prefix is None:
        prefix = pd.api.types.is_numeric_dtype(data[widen])
    if prefix:
        wide.columns = [f"{response}{sep}{c}" for c in wide.columns]
    else:
        wide.columns = [str(c) for c in wide.columns]
    wide = wide.reset_index()
    wide.columns.name = None
    return wide
