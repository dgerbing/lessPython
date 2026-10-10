# sunflower_plotly.py — XY(form="sunflower")
#
# A DESIGN, not a translation: lessR draws the sunflower only with
# base R's sunflowerplot() and, faceted, a lattice panel of its own
# (.plt.sunflower.facet), so there is no plotly source to port. As
# there: one dot at each distinct (x, y), and where k > 1
# observations coincide, k petals radiating from it, the first
# vertical, evenly spaced (sunflowerplot()'s angles), 1/8 inch long.
# Dots take bar_fill_cont, petals the point color, as .plt.main().
# Plotly has no petal symbol, so the petals are line segments in data
# units, sized from the plotting region of the default figure so that
# they read as equal-length spokes. facet= draws one panel per level,
# each with its own coordinates and so its own petal counts.

import numpy as np
import plotly.graph_objects as go

from .plotly_utils import (
    axis_format, axis_num, facet_fig, facet_panels, finish_facet,
    make_trans, plot_border, plotly_style, to_hex, x_grid,
)
from .utils import get_option, pretty

_PETAL_PX = 12          # 1/8 inch at 96 dpi
_PLOT_PX = (540, 330)   # plotting region of the default figure


def _counts(x, y):
    """Distinct coordinates and how many observations share each.
    R analog: table(x, y) of .plt.main()"""
    ok = np.isfinite(x) & np.isfinite(y)
    pts, cnt = np.unique(np.column_stack((x[ok], y[ok])), axis=0,
                         return_counts=True)
    return pts[:, 0], pts[:, 1], cnt


def _petals(xx, yy, cnt, dx, dy):
    """Segments for every coordinate with more than one observation:
    cnt petals, the first vertical. R analog: sunflowerplot()"""
    sx, sy = [], []
    for x0, y0, k in zip(xx, yy, cnt):
        if k < 2:
            continue
        ang = np.pi / 2 + 2 * np.pi * np.arange(k) / k
        for a in ang:
            sx += [x0, x0 + dx * np.cos(a), None]
            sy += [y0, y0 + dy * np.sin(a), None]
    return sx, sy


def _panel_traces(x, y, xr, yr, w_px, h_px, fill, color, pt_size,
                  hover):
    xx, yy, cnt = _counts(x, y)
    dx = _PETAL_PX * (xr[1] - xr[0]) / w_px
    dy = _PETAL_PX * (yr[1] - yr[0]) / h_px
    sx, sy = _petals(xx, yy, cnt, dx, dy)
    px = max(3.0, 5.5 * float(pt_size))
    return [
        go.Scatter(x=sx, y=sy, mode="lines",
                   line=dict(color=to_hex(color), width=1),
                   hoverinfo="skip", showlegend=False),
        go.Scatter(x=xx, y=yy, mode="markers",
                   marker=dict(size=px, color=to_hex(fill),
                               line=dict(width=0)),
                   customdata=cnt, hovertemplate=hover,
                   showlegend=False)]


def _range(v):
    t = pretty(float(np.nanmin(v)), float(np.nanmax(v)))
    pad = 0.04 * (t[-1] - t[0])
    return t, (t[0] - pad, t[-1] + pad)


def sunflower_plotly(x, y, x_lab, y_lab, main=None, facet=None,
                     facet_order=None, facet_name=None, facet2=None,
                     facet2_order=None, facet2_name=None, n_col=1,
                     pt_size=1, fill=None, color=None, overlay=None,
                     digits_d=2, axis_fmt="K", axis_x_pre="",
                     axis_y_pre=""):
    """The sunflower plot of x and y, single panel or faceted.
    overlay(fig, x, y, row, col) adds a fit line or an ellipse to a
    panel. Returns a plotly Figure."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fill = get_option("bar_fill_cont", "#96AAC3") if fill is None \
        else fill
    color = get_option("pt_color", "#324E5C") if color is None \
        else color
    style_opts = plotly_style()
    axT1, xr = _range(x)
    axT2, yr = _range(y)
    hover = (f"{x_lab}: %{{x}}<br>{y_lab}: %{{y}}<br>"
             "count: %{customdata}<extra></extra>")

    if facet is None:
        fig = go.Figure(_panel_traces(x, y, xr, yr, *_PLOT_PX, fill,
                                      color, pt_size, hover))
        if overlay is not None:
            overlay(fig, x, y, None, None)
        ax_x = axis_num(x_lab, axT1, axis_format(axT1, digits_d,
                                                 axis_fmt, axis_x_pre))
        ax_y = axis_num(y_lab, axT2, axis_format(axT2, digits_d,
                                                 axis_fmt, axis_y_pre))
        ax_x.update(range=list(xr))
        ax_y.update(range=list(yr), showgrid=True,
                    gridcolor=to_hex(style_opts["grid_col"]),
                    gridwidth=1, griddash="dot")
        fig.update_layout(
            xaxis=ax_x, yaxis=ax_y, template=None,
            shapes=x_grid(axT1) + plot_border(),
            plot_bgcolor=to_hex(style_opts["panel_fill"]),
            paper_bgcolor=to_hex(style_opts["window_fill"]))
    else:
        labels, pos, sel, n_row_g, n_col = facet_panels(
            facet, facet_order, facet2, facet2_order, facet_name,
            facet2_name, n_col)
        fig = facet_fig(n_row_g, n_col)
        w_px = _PLOT_PX[0] / n_col
        h_px = _PLOT_PX[1] * 1.3 / n_row_g
        for i, m in enumerate(sel):
            row, col = pos[i]
            for tr in _panel_traces(x[m], y[m], xr, yr, w_px, h_px,
                                    fill, color, pt_size, hover):
                fig.add_trace(tr, row=row, col=col)
            if overlay is not None and m.sum() > 2:
                overlay(fig, x[m], y[m], row, col)
        ax = {"axT1": axT1,
              "axL1": axis_format(axT1, digits_d, axis_fmt, axis_x_pre),
              "axT2": axT2,
              "axL2": axis_format(axT2, digits_d, axis_fmt, axis_y_pre)}
        finish_facet(fig, labels, ax, x_lab, y_lab, gridT1=axT1,
                     style_opts=style_opts, n_col=n_col, pos=pos)
        fig.update_xaxes(range=list(xr))
        fig.update_yaxes(range=list(yr))
    if main:
        fig.update_layout(title=dict(
            text=main, x=0.5, xanchor="center",
            font=dict(size=round(16 * get_option("main_size", 1)))))
    return fig
