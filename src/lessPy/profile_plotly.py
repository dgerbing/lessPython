# profile_plotly.py — the faceted profile chart, Chart(form=
# "profile", facet=). R draws it with base graphics only
# (.plt.profile.facet in plt.profileFacet.R); this is the plotly
# design of the same display, laid out as the faceted dot chart.
#
# The values arrive aggregated, one row per (x, by, facet) cell, so
# this function only places them: a point at each category of x and
# a segment joining the points of a group across the categories.
# Every panel shares the categories of x and the scale of y, which
# is what makes the panels comparable; a category absent from a
# panel keeps its place on that panel's axis and draws no point.
# With a by variable each panel carries one profile per group, the
# interaction plot of the single-panel form drawn within every
# panel, with one legend for all.
#
# The single-panel profile renders through plt_plotly(), as R's
# goes through .plt.main() -> plt.plotly().

import numpy as np
import plotly.graph_objects as go

from .utils import get_option
from .plotly_utils import (
    axis_num, facet_layout, make_trans, plot_border, plotly_style,
    to_hex, y_grid,
)


def profile_facet_plotly(cells, x_levels, series, y_tick, y_lab_tick,
                         connect=True,
                         x_lab="", y_lab="", by_name=None,
                         fill=("#4398D0",), pt_size=1.5,
                         facet_levels=(), facet_name="", n_col=None,
                         main=None, rotate_x=0, style_opts=None):
    """cells: {facet level: {series name: (x positions, values)}}.
    series: the series names in legend order ([None] without by).
    y_tick / y_lab_tick: the shared value-axis ticks and labels."""
    if style_opts is None:
        style_opts = plotly_style()
    grouped = series != [None]
    fills = [fill[i % len(fill)] for i in range(len(series))]
    px = float(pt_size) * (6.5 if grouped else 7.25)
    ln_width = 1.5
    title_size = round(16 * get_option("main_size", 1))
    pad = 0.04 * (y_tick[-1] - y_tick[0])
    y_range = [y_tick[0] - pad, y_tick[-1] + pad]
    x_pos = list(range(1, len(x_levels) + 1))

    def ann_adj(i, a, d):
        a["y"] = min(d["y"][1] + 0.04, 0.99)
        return a
    domains, anns = facet_layout(
        list(facet_levels), facet_name or "", yanchor="bottom",
        ann_adjust=ann_adj, n_col=n_col)
    n_col_use = n_col if n_col is not None else min(
        len(facet_levels), 3)

    fig = go.Figure()
    axes = {}
    shapes = []
    for i, lv in enumerate(facet_levels):
        suf = "" if i == 0 else str(i + 1)
        for j, nm in enumerate(series):
            xv, yv = cells.get(lv, {}).get(nm, ([], []))
            hover = (f"{x_lab}: %{{text}}<br>{y_lab}: %{{y}}"
                     + (f"<br>{by_name}: {nm}" if grouped else "")
                     + "<extra></extra>")
            fig.add_trace(go.Scatter(
                x=list(xv), y=list(yv),
                mode="lines+markers" if connect else "markers",
                xaxis=f"x{suf}", yaxis=f"y{suf}",
                text=[x_levels[int(p) - 1] for p in xv],
                name=None if nm is None else str(nm),
                legendgroup=None if nm is None else str(nm),
                marker=dict(symbol="circle", size=px,
                            sizemode="diameter",
                            color=make_trans(fills[j], 1),
                            line=dict(color=to_hex(fills[j]),
                                      width=1)),
                line=dict(color=to_hex(fills[j]), width=ln_width),
                hovertemplate=hover,
                showlegend=grouped and i == 0,
            ))
        first_col = i % n_col_use == 0
        x_ax = axis_num(x_lab, x_pos, list(x_levels))
        x_ax.update(range=[0.5, len(x_levels) + 0.5],
                    domain=list(domains[i]["x"]), anchor=f"y{suf}")
        if rotate_x:
            x_ax["tickangle"] = -rotate_x
        y_ax = axis_num(y_lab if first_col else "", y_tick,
                        y_lab_tick)
        y_ax.update(range=y_range, domain=list(domains[i]["y"]),
                    anchor=f"x{suf}")
        if not first_col:           # one shared value scale per row
            y_ax["showticklabels"] = False
        axes[f"xaxis{suf}"] = x_ax
        axes[f"yaxis{suf}"] = y_ax
        for g in y_grid(y_tick):
            g = dict(g)
            g.update(xref=f"x{suf} domain", yref=f"y{suf}", x0=0, x1=1)
            shapes.append(g)

    fig.update_layout(
        annotations=anns,
        shapes=shapes,
        margin=dict(t=round(title_size * 2.2) if main else 40,
                    b=8, l=20, r=20),
        showlegend=grouped,
        legend=dict(
            title=(dict(text=by_name) if grouped and by_name
                   else None),
            font=dict(size=round(16 * get_option("axis_size", 0.9)))),
        title=(dict(text=main, x=0.5, xanchor="center", y=0.98,
                    yanchor="top", font=dict(size=title_size))
               if main else None),
        template=None,
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]),
        **axes,
    )
    return fig
