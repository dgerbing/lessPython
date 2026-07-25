# rename.py — analog of rename.R.
#
# rename(): rename one or more columns of a data frame, reporting
# each old --> new change, and return the renamed data frame.
# R's from/to are `old` and `new` here (`from` is a Python
# keyword); each is a column name or a list of names.

import pandas as pd


def rename(data, old, new):
    """Rename columns of data: old (a name or list of names) to
    new (the same length). Prints each change and returns the
    renamed DataFrame (a copy). R analog: rename() (from/to)."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    old = [old] if isinstance(old, str) else list(old)
    new = [new] if isinstance(new, str) else list(new)
    if len(old) != len(new):
        raise ValueError(
            f"old has {len(old)} name(s) but new has {len(new)}; "
            "they must match")
    missing = [c for c in old if c not in data.columns]
    if missing:
        raise ValueError(
            f"not columns of data: {', '.join(map(str, missing))}")

    out = data.rename(columns=dict(zip(old, new)))
    print("Change the following variable names:\n")
    for o, n in zip(old, new):
        print(f"{o} --> {n}")
    print()
    return out
