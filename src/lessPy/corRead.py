# corRead.py — analog of corRead.R.
#
# corRead(): read a correlation matrix from a text file that
# holds only the coefficients — no header row, no row labels,
# whitespace- (or sep-) delimited. Validates that the values are
# numeric and the matrix is square, then labels the rows/columns
# from var_names, or X1..Xn by default. Returns the matrix as a
# DataFrame, ready for corEFA / corCFA / corScree, etc.

import numpy as np
import pandas as pd


def corRead(file=None, var_names=None, sep=None):
    """Read a square correlation matrix of coefficients (no
    header, no row labels) from a text file, whitespace-delimited
    unless sep= is given. Names the variables from var_names, or
    X1..Xn. Returns the matrix as a DataFrame. R analog: corRead()
    (its `from` argument is this `file`; there is no interactive
    file chooser)."""
    if file is None:
        raise ValueError(
            "file= is required: the path to a text file of "
            "correlation coefficients (Python has no interactive "
            "file chooser)")
    df = pd.read_csv(file, sep=r"\s+" if sep is None else sep,
                     header=None, engine="python")
    if not all(pd.api.types.is_numeric_dtype(df[c])
               for c in df.columns):
        raise ValueError(
            "the correlation matrix must be numeric; the file "
            "has non-numeric values. Check that it contains only "
            "coefficients, with no variable names or row labels.")
    myc = df.to_numpy(dtype=float)
    if myc.shape[0] != myc.shape[1]:
        raise ValueError(
            "the correlation matrix must be square; the read "
            f"matrix is {myc.shape[0]} x {myc.shape[1]}.")
    n = myc.shape[1]
    if var_names is not None:
        if len(var_names) != n:
            raise ValueError(
                f"var_names length ({len(var_names)}) must equal "
                f"the number of variables ({n}).")
        names = [str(v) for v in var_names]
    else:
        names = [f"X{i}" for i in range(1, n + 1)]
    return pd.DataFrame(myc, index=names, columns=names)
