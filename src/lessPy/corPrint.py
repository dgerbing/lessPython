# corPrint.py — analog of corPrint.R.
#
# corPrint(): print a correlation matrix in lessR's compact
# style — two decimals with the leading "0." stripped (0.85 ->
# "85", -0.07 -> "-07"), the diagonal 1.00 shown as "100", and
# any coefficient with |r| < min_value blanked. Prints the
# formatted matrix and returns it as a text string.

import numpy as np
import pandas as pd


def _compact(v):
    s = f"{v:.2f}".replace("0.", "")        # 0.85 -> "85"
    return s.replace("1.00", "100").replace("-1.00", "-100")


def corPrint(R, min_value=0):
    """Print a correlation matrix R (a DataFrame or array; raw
    data is correlated first) in the compact lessR style, blanking
    coefficients with |r| < min_value. Returns the formatted text.
    R analog: corPrint()"""
    Rm = pd.DataFrame(R)
    vals = Rm.to_numpy(dtype=float)
    if (Rm.shape[0] != Rm.shape[1]
            or not np.allclose(vals, vals.T, atol=1e-8)):
        Rm = Rm.select_dtypes("number").corr()
        vals = Rm.to_numpy(dtype=float)
    names = [str(c) for c in Rm.columns]
    n = len(names)
    lab_w = max(len(s) for s in names) + 2
    col_w = [max(4, len(nm) + 1) for nm in names]

    lines = [" " * lab_w + "".join(
        nm.rjust(col_w[j]) for j, nm in enumerate(names))]
    for i in range(n):
        row = (" " + names[i]).rjust(lab_w)
        for j in range(n):
            v = vals[i, j]
            cell = "" if abs(v) < min_value else _compact(v)
            row += cell.rjust(col_w[j])
        lines.append(row)
    text = "\n".join(lines)
    print(text)
    return text
