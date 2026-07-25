# utils.py — analog of the general utilities in lessR zzz.R
#
# Includes the option (style) system that replaces R's getOption()
# calls, number formatting (.fmt), and an R-style pretty() for axis
# tick values, which lessR computes upstream of the render functions.

import math

# style settings; R analog: options() set by lessR style()
_OPTIONS = {
    "main_size":     1.0,
    "lab_size":      1.0,
    "axis_size":     0.9,
    "axis_color":    "black",
    "lab_color":     "black",
    "grid_color":    "gray90",
    "grid_col":      "gray85",
    "grid_lwd":      0.5,
    "grid_lty":      None,
    "axis_lwd":      1,
    "panel_border":  "#808080",
    "panel_lwd":     1,
    "panel_fill":    "white",
    "window_fill":   "white",
    "digits_d":      2,
    "pt_color":      "#324E5C",   # rgb(50,78,92), zzz_on.R
    "trans_pt_fill": 0.10,
    "segment_color": "gray40",
    "fit_color":     "#5C4032",   # rgb(92,64,50), zzz_on.R
    "fit_lwd":       2,
    "se_fill":       "#1A1A1A19",
    "ellipse_fill":  "#92806F28",
    "ellipse_color": "gray20",
    "ellipse_lwd":   1,
    "violin_fill":   "#7485975A",
    "violin_color":  "gray15",
    "box_fill":      "#419BD2",   # rgb(65,155,210), zzz_on.R
    "box_color":     "gray15",
    "out_fill":      "#8B1A1A",   # firebrick4
    "out_color":     "#8B1A1A",
    "out2_fill":     "#EE2C2C",   # firebrick2
    "out2_color":    "#EE2C2C",
    "strip_fill":       "#7F7F7F37",
    "strip_color":      "gray40",
    "strip_text_color": "gray15",
}


def get_option(name, default=None):
    val = _OPTIONS.get(name, default)
    return default if val is None else val


def set_option(name, value):
    _OPTIONS[name] = value


def fmt(k, d=None):
    """Format one number with d decimal digits. R analog: .fmt()"""
    if d is None:
        d = get_option("digits_d", 2)
    return f"{k:.{d}f}"


# aggregation functions for stat=; sd uses n-1 as in R
# R analog: .stat_fun()
STAT_FUN = {
    "mean":   lambda s: s.mean(),
    "sum":    lambda s: s.sum(),
    "sd":     lambda s: s.std(ddof=1),
    "min":    lambda s: s.min(),
    "median": lambda s: s.median(),
    "max":    lambda s: s.max(),
}

# display label for a stat= value; R analog: .stat_lbl()
STAT_LBL = {
    "sum":       "Sum",
    "mean":      "Mean",
    "sd":        "Standard Deviation",
    "deviation": "Mean Deviation",
    "min":       "Minimum",
    "median":    "Median",
    "max":       "Maximum",
}


def category_order(s):
    """Category order: declared order for a pandas Categorical,
    else sorted unique values -- the same convention as an R factor
    built by table(). Shared by Chart() and XY()."""
    import pandas as pd
    if isinstance(s.dtype, pd.CategoricalDtype):
        return [c for c in s.cat.categories if c in set(s.dropna())]
    return sorted(s.dropna().unique().tolist())


def get_column(data, name, arg):
    """Resolve a string column name against a DataFrame, with the
    errors the string interface implies. Shared by Chart() and X()."""
    if not isinstance(name, str):
        raise TypeError(
            f"{arg} must be a string naming a column of data, "
            f"not {type(name).__name__}. (Python has no equivalent "
            "of R's unquoted variable names.)")
    if name not in data.columns:
        raise KeyError(
            f"{arg}='{name}' is not a column of data. "
            f"Columns: {', '.join(map(str, data.columns))}")
    return data[name]


def facet_values(data, values, arg):
    """Resolve a computed facet -- a pandas Series or numpy array
    of values aligned with data, the Python analog of R's facet
    expression. Length-checked against data, as R checks the
    expression length. Returns an aligned Series."""
    import numpy as np
    import pandas as pd
    if isinstance(values, pd.Series):
        if len(values) != len(data):
            raise ValueError(
                f"data has {len(data)} rows, but the {arg} Series "
                f"has {len(values)} values")
        name = values.name if values.name else arg
        return pd.Series(values.to_numpy(), index=data.index,
                         name=name)
    if isinstance(values, np.ndarray):
        if len(values) != len(data):
            raise ValueError(
                f"data has {len(data)} rows, but the {arg} array "
                f"has {len(values)} values")
        return pd.Series(values, index=data.index, name=arg)
    raise TypeError(
        f"{arg} must be a string naming a column of data, a "
        f"list of such names, or a pandas Series / numpy array "
        f"of computed values, not {type(values).__name__}")


def resolve_facet(data, facet, fun):
    """Resolve facet= into up to two aligned Series:
    (facet1, facet1_name, facet2, facet2_name). Accepts a column
    name, a list/tuple of names (facet1 = panel columns, facet2 =
    panel rows; more than two uses the first two, with a message),
    or a Series/array of computed values (facet_values). R analog:
    the --- resolve facet --- block of X.R / XY.R."""
    if facet is None:
        return None, None, None, None
    if isinstance(facet, str):
        return get_column(data, facet, "facet"), facet, None, None
    if isinstance(facet, (list, tuple)):
        if not all(isinstance(f, str) for f in facet):
            raise TypeError(
                "a facet list gives column names; for computed "
                "values pass a single pandas Series or numpy "
                "array")
        if len(facet) == 0:
            raise ValueError(
                "facet names a categorical variable for its "
                "panels")
        if len(facet) > 2:
            print(f"facet has {len(facet)} variables. {fun}() "
                  "uses the first two: "
                  f"{facet[0]}, {facet[1]}.")
        f1 = get_column(data, facet[0], "facet")
        if len(facet) == 1:
            return f1, facet[0], None, None
        f2 = get_column(data, facet[1], "facet")
        return f1, facet[0], f2, facet[1]
    ser = facet_values(data, facet, "facet")
    return ser, str(ser.name), None, None


def kde(xg, grid, h):
    """Gaussian KDE of xg evaluated on grid with bandwidth h.
    Shared by the density and violin renderers."""
    import numpy as np
    z = (grid[:, None] - xg[None, :]) / h
    return (np.exp(-0.5 * z * z).sum(axis=1)
            / (len(xg) * h * float(np.sqrt(2 * np.pi))))


def bw_nrd0(x):
    """Silverman rule-of-thumb KDE bandwidth, the default of R's
    density(). R analog: stats::bw.nrd0()"""
    import numpy as np
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        raise ValueError("bandwidth needs at least 2 data points")
    sd = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    lo = min(sd, iqr / 1.34)
    if lo == 0:
        lo = sd or abs(float(x[0])) or 1.0
    return 0.9 * lo * n ** -0.2


def band_width(x, bw_iter=10, n=512):
    """Violin bandwidth: start at bw_nrd0 and widen by 10% per
    iteration until the density curve has at most one direction
    change (a single peak) or bw_iter iterations pass.
    R analog: .band.width() (zzz.R)"""
    import numpy as np
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    bw = bw_nrd0(x)
    for _ in range(int(bw_iter)):
        grid = np.linspace(x.min() - 3 * bw, x.max() + 3 * bw, n)
        xd = np.diff(kde(x, grid, bw))
        flips = int((np.sign(xd[1:]) != np.sign(xd[:-1])).sum())
        if flips <= 1:
            break
        bw *= 1.1
    return bw


def pretty(lo, hi, n=5):
    """Nice tick values covering [lo, hi]. R analog: pretty()"""
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = 10 * mag
    for m in (1, 2, 5, 10):
        if raw <= m * mag * (1 + 1e-10):
            step = m * mag
            break
    start = math.floor(lo / step + 1e-10) * step
    end = math.ceil(hi / step - 1e-10) * step
    k = round((end - start) / step)
    vals = [start + i * step for i in range(k + 1)]
    # avoid -0.0 and float dust such as 0.30000000000000004
    return [round(v, 10) + 0.0 for v in vals]
