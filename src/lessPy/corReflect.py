# corReflect.py — analog of corReflect.R.
#
# corReflect(): reverse the sign of the correlations of the named
# variables, to flip reverse-scored items so they correlate
# positively with the rest of a scale. Reflecting a variable
# negates all its off-diagonal correlations; reflecting two of
# them leaves their mutual correlation unchanged (it flips twice).
# Returns the reflected correlation matrix; the heat map is on the
# returned frame's .attrs["plots"].

import numpy as np
import pandas as pd

from .utils import get_option


def corReflect(R, vars, main=None, heat_map=True):
    """Reflect (negate) the correlations of the variables named
    in vars within the correlation matrix R (a DataFrame or
    array; raw data is correlated first). Returns the reflected
    correlation-matrix DataFrame; its .attrs["plots"] holds the
    heat map. R analog: corReflect()"""
    Rm = pd.DataFrame(R)
    vals = Rm.to_numpy(dtype=float)
    if (Rm.shape[0] != Rm.shape[1]
            or not np.allclose(vals, vals.T, atol=1e-8)):
        Rm = Rm.select_dtypes("number").corr()
    Rm.index = Rm.columns = [str(c) for c in Rm.columns]
    names = list(Rm.columns)
    vars = [vars] if isinstance(vars, str) else list(vars)
    bad = [str(v) for v in vars if str(v) not in names]
    if bad:
        raise ValueError(
            f"variables not in the matrix: {', '.join(bad)}")

    # sign vector: -1 for a reflected variable. The outer product
    # negates each reflected variable's off-diagonal row/col and
    # leaves the diagonal (sign^2 = 1) and any pair of reflected
    # variables (two flips) unchanged. ~ corReflect.R loop
    s = np.ones(len(names))
    for v in vars:
        s[names.index(str(v))] = -1.0
    out = pd.DataFrame(np.outer(s, s) * Rm.to_numpy(dtype=float),
                       index=Rm.index, columns=Rm.columns)

    plots = {}
    if heat_map:
        plots["heatmap"] = _heatmap(out, main)
    out.attrs["plots"] = plots
    return out


def _heatmap(df, main):
    import plotly.graph_objects as go

    from .plotly_utils import plotly_style, to_hex
    labels = list(df.columns)
    Z = df.to_numpy(dtype=float).copy()
    np.fill_diagonal(Z, np.nan)
    style = plotly_style()
    fig = go.Figure(go.Heatmap(
        z=Z, x=labels, y=labels, zmin=-1, zmax=1,
        colorscale="RdBu", reversescale=True,
        colorbar=dict(title="r")))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        template=None, paper_bgcolor=to_hex(style["window_fill"]),
        title=dict(text=main or "With Reflected Item Coefficients",
                   x=0.5, xanchor="center",
                   font=dict(size=round(
                       16 * get_option("main_size", 1)))))
    return fig
