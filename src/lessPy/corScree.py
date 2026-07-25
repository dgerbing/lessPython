# corScree.py — analog of corScree.R.
#
# corScree(): the scree plots for deciding the number of factors
# — the eigenvalues of a correlation matrix against their index,
# and the differences of successive eigenvalues. Prints both
# sequences and returns them with the two plotly figures.

import numpy as np
import pandas as pd

from .plotly_utils import (
    axis_format, axis_num, plot_border, plotly_style, to_hex,
    x_grid)
from .utils import fmt, get_option, pretty


class corScreeResults:
    """Results of corScree(): the eigenvalues, the differences of
    successive eigenvalues, and the two plotly figures in
    .plots."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return f"<lessPy corScree: {len(self.eigenvalues)} vars>"


def corScree(R, main=None):
    """Scree analysis of a correlation matrix R (a DataFrame or
    array; raw data is correlated first): its eigenvalues and
    their successive differences, each printed and plotted.
    Returns a corScreeResults with the figures in .plots.
    R analog: corScree()"""
    Rm = pd.DataFrame(R)
    vals = Rm.to_numpy(dtype=float)
    if (Rm.shape[0] != Rm.shape[1]
            or not np.allclose(vals, vals.T, atol=1e-8)):
        Rm = Rm.select_dtypes("number").corr()
    S = Rm.to_numpy(dtype=float)

    ev = np.linalg.eigvalsh(S)[::-1]       # descending
    ev_diff = ev[:-1] - ev[1:]             # -diff(ev)

    def line(seq):
        return " ".join(fmt(v, 3) for v in seq)

    print("\nEigenvalues\n" + "-" * 11 + "\n" + line(ev)
          + "\n\nDifferences of Successive Eigenvalues\n"
          + "-" * 37 + "\n" + line(ev_diff) + "\n")

    plots = {
        "scree": _scree_plot(ev, "Eigenvalues", main),
        "differences": _scree_plot(
            ev_diff, "Differences of Successive Eigenvalues",
            main)}
    return corScreeResults(eigenvalues=ev, differences=ev_diff,
                           plots=plots)


def _scree_plot(y, y_lab, main):
    """A line-with-markers plot of y against its 1-based index.
    ~ .plt.main segments plot in corScree.R"""
    import plotly.graph_objects as go
    style = plotly_style()
    x = np.arange(1, len(y) + 1)
    col = to_hex(get_option("fit_color", "#5C4032"))
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines+markers",
        line=dict(color=col, width=1.5),
        marker=dict(size=7, color=col), hoverinfo="x+y",
        showlegend=False))
    axT2 = pretty(float(min(y.min(), 0)), float(y.max()))
    ax_x = axis_num("Index", list(x), [str(i) for i in x])
    ax_y = axis_num(y_lab, axT2, axis_format(axT2, 2))
    ax_y.update(showgrid=True,
                gridcolor=to_hex(style["grid_col"]),
                gridwidth=1, griddash="dot")
    fig.update_layout(
        xaxis=ax_x, yaxis=ax_y,
        shapes=x_grid(list(x)) + plot_border(), template=None,
        plot_bgcolor=to_hex(style["panel_fill"]),
        paper_bgcolor=to_hex(style["window_fill"]),
        title=dict(text=main or "", x=0.5, xanchor="center",
                   font=dict(size=round(
                       16 * get_option("main_size", 1)))))
    return fig
