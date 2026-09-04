# profile_plotly.py — analog of the form="profile" path of Chart.R
#
# Renders a profile from an ALREADY-AGGREGATED table: one point for
# each level of the categorical x, connected across the levels by
# line segments. The segments assert an ordering of the categorical
# axis, which is the purpose of the form, so they are drawn unless
# turned off with segments=False.
#
#   single profile   tbl a Series indexed by the x categories
#   one per group    tbl a DataFrame, rows the by levels, columns
#                    the x categories: the interaction plot that
#                    ANOVA() draws for a two-way design
#
# R renders this form through .plt.main() with cat.x=TRUE. The
# geometry here is the plotly equivalent: a categorical x axis with
# its order frozen to the order of the table's columns, a numeric
# value axis on pretty() ticks shared by every profile, and one
# lines+markers trace per group.

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .utils import get_option, pretty
from .plotly_utils import (
    BASE_COLORS, axis_cat, axis_format, axis_num, legend_style,
    make_trans, plot_border, plotly_style, to_hex, y_grid,
)


def _series(tbl):
    """The table as a list of (group name, values) pairs over a
    common set of categories. A Series is the single unnamed
    profile; a DataFrame carries one profile per row."""
    if isinstance(tbl, pd.DataFrame):
        cats = [str(c) for c in tbl.columns]
        return cats, [(str(g), tbl.loc[g].to_numpy(dtype=float))
                      for g in tbl.index]
    cats = [str(c) for c in tbl.index]
    return cats, [(None, tbl.to_numpy(dtype=float))]


def profile_plotly(tbl, x_name=None, by_name=None,
                   fill=None, border=None, pt_size=1.5,
                   segments=True, main=None,
                   x_lab=None, y_lab=None, digits_d=2,
                   axis_fmt="K", axis_x_pre="", axis_y_pre="",
                   rotate_x=0, rotate_y=0, transparency=None,
                   style_opts=None):
    """One point per level of x, connected across the levels. With
    a by variable the table carries one row per group and each is
    drawn as its own profile, the interaction plot."""
    if style_opts is None:
        style_opts = plotly_style()
    if fill is None:
        fill = BASE_COLORS

    cats, series = _series(tbl)
    G = len(series)
    fill_list = [fill[g % len(fill)]
                 if isinstance(fill, (list, tuple)) else fill
                 for g in range(G)]
    opacity = 1 - (0 if transparency is None else transparency)

    # the value axis, shared by every profile so that the groups are
    # read against one scale. Finite values only: an empty cell of
    # the by-by-x table leaves a gap in its profile rather than
    # dropping the group
    vals = np.concatenate([v for _, v in series])
    fin = vals[np.isfinite(vals)]
    if len(fin) == 0:
        raise ValueError("no finite values to plot")
    lo, hi = float(fin.min()), float(fin.max())
    gridT = pretty(lo, hi)
    gridL = axis_format(gridT, digits_d, axis_fmt, axis_y_pre)

    fig = go.Figure()
    for g, (gname, v) in enumerate(series):
        clr = make_trans(fill_list[g], opacity)
        hover = [
            (f"{x_name}: {c}"
             + (f"<br>{by_name}: {gname}"
                if gname is not None and by_name else "")
             + f"<br>{y_lab}: {vv:,.{digits_d}f}")
            for c, vv in zip(cats, v)
        ]
        fig.add_trace(go.Scatter(
            x=cats, y=v,
            mode="lines+markers" if segments else "markers",
            name=gname,
            line=dict(color=clr, width=1.5),
            marker=dict(color=clr, size=round(6.5 * pt_size / 1.5),
                        line=dict(
                            color=to_hex(
                                get_option("pt_color", "#324E5C")
                                if border is None else border),
                            width=0.5)),
            hoverinfo="text", hovertext=hover,
            showlegend=G > 1,
        ))

    x_ax = axis_cat(x_lab)
    x_ax.update(categoryorder="array", categoryarray=cats,
                tickmode="array", tickvals=cats, ticktext=cats)
    y_ax = axis_num(y_lab, gridT, gridL)
    if rotate_x:
        x_ax["tickangle"] = -rotate_x
    if rotate_y:
        y_ax["tickangle"] = -rotate_y

    title_size = round(16 * get_option("main_size", 1))
    fig.update_layout(
        xaxis=x_ax,
        yaxis=y_ax,
        shapes=y_grid(gridT) + plot_border(),
        margin=dict(t=round(title_size * 2.2) if main else 30),
        title=(dict(text=main, x=0.5, xanchor="center",
                    y=0.98, yanchor="top",
                    font=dict(size=title_size,
                              color=to_hex(get_option(
                                  "lab_color", "black"))))
               if main else None),
        template=None,
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]),
    )
    if G > 1:
        fig.update_layout(legend=legend_style(by_name, style_opts))
    return fig
