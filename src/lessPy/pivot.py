# pivot.py — analog of pivot.R (the aggregation core).
#
# pivot(): aggregate a numeric variable over the categories of one
# or more `by` grouping variables, computing one or more summary
# statistics, or tabulate frequencies. The result is a long-form
# DataFrame with the by columns, an n (and na) count, and one
# {variable}_{stat} column per statistic — lessR's pivot output.
#
# Ported: the aggregation over by groups (sum, mean, median, min,
# max, sd, var, IQR, mad), the n/na counts and show_n, all group
# combinations (empty cells kept, as R's drop=FALSE), NA groups,
# sort= by the statistic, and the one- and two-way frequency
# table (compute="table"). Not ported: by_cols wide cross-tabs,
# table_prop row/col proportions, quantiles, and skew/kurtosis.
#
# The interface uses string names (compute="mean", variable=,
# by=), the lessPy convention.

import numpy as np
import pandas as pd

from .utils import get_column

# statistic name -> (column-name abbreviation, aggregator).
# pandas quantile is linear (R type 7); std/var use n-1; all skip
# NaN, so na_remove is the default. R analog: pivot.R fun.vec
_STAT = {
    "sum":    ("sum", lambda s: s.sum()),
    "mean":   ("mean", lambda s: s.mean()),
    "median": ("mdn", lambda s: s.median()),
    "min":    ("min", lambda s: s.min()),
    "max":    ("max", lambda s: s.max()),
    "sd":     ("sd", lambda s: s.std(ddof=1)),
    "var":    ("var", lambda s: s.var(ddof=1)),
    "IQR":    ("IQR", lambda s: s.quantile(0.75)
              - s.quantile(0.25)),
    "mad":    ("mad", lambda s:
               1.4826 * (s - s.median()).abs().median()),
}


def pivot(data, compute, variable=None, by=None, filter=None,
          show_n=True, na_remove=True, sort=None, digits_d=None,
          quiet=False):
    """Aggregate a numeric variable over by groups, or tabulate
    frequencies. compute is a statistic name or a list of names
    ("mean", ["mean","sd"], "table"); variable is the numeric
    column to aggregate; by is the grouping column(s). Returns a
    long-form DataFrame. R analog: pivot()"""
    if filter is not None:
        data = data.query(filter)
    computes = [compute] if isinstance(compute, str) \
        else list(compute)
    by = ([by] if isinstance(by, str)
          else list(by) if by is not None else [])
    if sort is not None and sort not in ("+", "-"):
        raise ValueError('sort: "+" or "-"')

    if "table" in computes:
        if len(computes) > 1:
            raise ValueError('compute="table" cannot be combined '
                             "with other statistics")
        if not by:
            raise ValueError('compute="table" needs by=')
        return _pivot_table(data, by, show_n)

    if variable is None:
        raise ValueError("variable= is required: the numeric "
                         "column to aggregate")
    unknown = [c for c in computes if c not in _STAT]
    if unknown:
        raise ValueError(
            f"unknown compute {unknown}; use "
            f"{', '.join(_STAT)}, or \"table\"")
    v = get_column(data, variable, "variable")
    if not pd.api.types.is_numeric_dtype(v):
        raise TypeError(
            f"the variable to aggregate '{variable}' is "
            f"{v.dtype}: it must be numeric. Put categorical "
            "variables in by=.")
    if not by:
        raise ValueError("by= is required: the grouping "
                         "column(s)")
    return _pivot_agg(data, variable, by, computes, show_n, sort)


def _cat_frame(data, by):
    """Copy of data with each by column made an ordered Categorical
    (sorted levels), so groupby keeps every level combination and
    the NA group, as R's factor()/drop=FALSE."""
    g = data.copy()
    for b in by:
        col = g[b]
        cats = sorted(pd.unique(col.dropna()),
                      key=lambda x: (str(type(x)), x))
        g[b] = pd.Categorical(col, categories=cats)
    return g


def _pivot_agg(data, variable, by, computes, show_n, sort):
    g = _cat_frame(data, by)
    grp = g.groupby(by, observed=False, dropna=False, sort=True)
    v = grp[variable]
    out = pd.DataFrame({
        "n": v.apply(lambda s: int(s.notna().sum())),
        "na": v.apply(lambda s: int(s.isna().sum()))})
    for c in computes:
        abbr, fn = _STAT[c]
        out[f"{variable}_{abbr}"] = v.apply(fn)
    out = out.reset_index()
    # R aggregate row order: the FIRST by var varies fastest, so
    # sort by the by columns in reverse listing order, NA last
    out = out.sort_values(by[::-1], na_position="last",
                          kind="stable").reset_index(drop=True)
    if sort is not None:
        stat_col = out.columns[-1]
        out = out.sort_values(
            stat_col, ascending=(sort == "+"),
            na_position="last", kind="stable"
        ).reset_index(drop=True)
    if not show_n:
        out = out.drop(columns=["n", "na"])
    return out


def _pivot_table(data, by, show_n):
    if len(by) == 1:
        b = by[0]
        col = data[b]
        cats = sorted(pd.unique(col.dropna()),
                      key=lambda x: (str(type(x)), x))
        cc = pd.Categorical(col, categories=cats)
        n = pd.Series(cc, name=b).value_counts(
            dropna=False, sort=False)
        # value_counts on Categorical excludes NaN; add it back
        n = n.reindex(cats)
        na_count = int(col.isna().sum())
        idx = list(cats)
        counts = [int(n[c]) for c in cats]
        if na_count:
            idx.append(np.nan)
            counts.append(na_count)
        total = sum(counts)
        out = pd.DataFrame({b: idx, "n": counts})
        out["Prop"] = np.round(np.array(counts) / total, 2)
        return out
    if len(by) > 2:
        raise ValueError('compute="table" supports one or two '
                         "by variables")
    # two-way: long-form counts, second by var first, as R
    b1, b2 = by
    g = _cat_frame(data, by)
    ct = (g.groupby([b1, b2], observed=False, dropna=False)
          .size().reset_index(name="n"))
    ct = ct[[b2, b1, "n"]]
    ct = ct.sort_values([b1, b2], na_position="last",
                        kind="stable").reset_index(drop=True)
    return ct
