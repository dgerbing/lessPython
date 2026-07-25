# corProp.py — analog of corProp.R.
#
# corProp(): the item proportionality matrix — for each pair of
# items, how similarly they correlate with all the OTHER items
# (the correlation of their two correlation profiles, excluding
# the pair itself). A high value means two items are nearly
# interchangeable indicators, useful in scale construction.
# Returns the proportionality matrix (rounded to 2 as in R); the
# heat map is on the returned frame's .attrs["plots"].

import numpy as np
import pandas as pd

from .utils import get_option


def corProp(R, main=None, heat_map=True):
    """Item proportionality coefficients of a correlation matrix
    R (a DataFrame or array; raw data is correlated first): the
    profile similarity of each item pair across the other items.
    Returns the proportionality matrix DataFrame (rounded to 2,
    as R); its .attrs["plots"] holds the heat map. R analog:
    corProp()"""
    Rm = pd.DataFrame(R)
    vals = Rm.to_numpy(dtype=float)
    if (Rm.shape[0] != Rm.shape[1]
            or not np.allclose(vals, vals.T, atol=1e-8)):
        Rm = Rm.select_dtypes("number").corr()
    Rm.index = Rm.columns = [str(c) for c in Rm.columns]
    S = Rm.to_numpy(dtype=float)

    # Diag[i] = sum_k R[k,i]^2 ; RR = R @ R (the cross products).
    # For a pair (i,j) exclude i and j from both the cross product
    # and the two sums of squares, then normalize. ~ corProp.R
    diag = np.diag(S)
    Diag = (S ** 2).sum(axis=0)
    RR = S @ S
    cross = RR - S * (diag[:, None] + diag[None, :])
    D1 = Diag[:, None] - (diag[:, None] ** 2 + S ** 2)
    D2 = Diag[None, :] - (diag[None, :] ** 2 + S ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = cross / np.sqrt(D1 * D2)
    np.fill_diagonal(P, 1.0)
    P = np.round(P, 2)

    out = pd.DataFrame(P, index=Rm.index, columns=Rm.columns)
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
    np.fill_diagonal(Z, np.nan)            # ignore the diagonal
    style = plotly_style()
    fig = go.Figure(go.Heatmap(
        z=Z, x=labels, y=labels, zmin=-1, zmax=1,
        colorscale="RdBu", reversescale=True,
        colorbar=dict(title="prop")))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        template=None, paper_bgcolor=to_hex(style["window_fill"]),
        title=dict(text=main or "Item Proportionalities", x=0.5,
                   xanchor="center",
                   font=dict(size=round(
                       16 * get_option("main_size", 1)))))
    return fig
