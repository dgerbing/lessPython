# freq_poly_plotly.py — analog of freq_poly.plotly.R
#
# Frequency polygon: per-group bin counts on SHARED breaks (the
# bins arrive from X() computed over all the data, so every
# group's polygon is comparable), vertices at the bin midpoints,
# closed to zero one bin-width beyond the first and last
# midpoints. Modeled on dn_plotly, as freq_poly.plotly() is
# modeled on dn.plotly(); binning is local (pd.cut), no
# dependency on the histogram code path. fill_area=False (X's
# area_fill "off"): lines only, with a marker at each vertex but
# not at the two zero-closing endpoints.
#
# facet= draws one panel per level on the shared breaks and a
# shared count scale, first level in the bottom panel — an
# EXTENSION beyond R, where X() still stops with "Facets not yet
# working with density or freq_poly plots". With by=, each panel
# overlays one polygon per group present in it; proportions are
# per group within a panel.

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .utils import get_option, pretty
from .plotly_utils import (
    auto_opacity, axis_format, axis_num, facet_fig,
    facet_panels, finish_facet, get_tick_fmt,
    legend_style, make_trans, plot_border, plotly_style, to_hex,
    x_grid,
)


def _poly_counts(xg, breaks, proportion):
    """Bin counts of xg on the shared breaks (right-closed, as in
    R's hist); proportions are within-group."""
    cats = pd.cut(pd.Series(xg), breaks, right=True,
                  include_lowest=True)
    yv = cats.value_counts(sort=False).to_numpy(dtype=float)
    if proportion and yv.sum() > 0:
        yv = yv / yv.sum()
    return yv


def freq_poly_plotly(x, by=None, x_name=None, by_name=None,
                     facet=None, facet_order=None, facet_name=None,
                     facet2=None, facet2_order=None,
                     facet2_name=None,
                     breaks=None, proportion=False,
                     fill=None, fill_area=True,
                     x_lab=None, y_lab=None, main=None,
                     n_col=1, axis_fmt="K", axis_x_pre="",
                     axis_y_pre="", digits_d=3,
                     style_opts=None):

    if breaks is None or len(breaks) < 2:
        raise ValueError("breaks (bin edges) required, length >= 2")
    if style_opts is None:
        style_opts = plotly_style()
    if x_name is None:
        x_name = getattr(x, "name", None) or "x"
    if x_lab is None:
        x_lab = x_name
    if y_lab is None:
        y_lab = (f"Proportion of {x_name}" if proportion
                 else f"Count of {x_name}")

    if facet is not None:
        return _fp_facet(x, facet, facet_order, x_name,
                         facet_name, breaks, proportion, fill,
                         fill_area, x_lab, y_lab, digits_d, main,
                         style_opts, by=by, by_name=by_name,
                         n_col=n_col, facet2=facet2,
                         facet2_order=facet2_order,
                         facet2_name=facet2_name,
                         axis_fmt=axis_fmt,
                         axis_x_pre=axis_x_pre)

    x = np.asarray(x, dtype=float)

    # --- groups ---
    if by is None:
        groups = ["Series 1"]
    else:
        by = pd.Series(by).astype(str)
        groups = sorted(by.dropna().unique().tolist())
    G = len(groups)

    # groups need distinguishable hues: default to the palette
    if fill is None:
        if G > 1:
            from .plotly_utils import BASE_COLORS
            fill = BASE_COLORS
        else:
            fill = get_option("bar_fill_cont", "#96AAC3")

    # --- bin geometry, counts per group on the shared breaks ---
    breaks = [float(b) for b in breaks]
    left = np.array(breaks[:-1])
    right = np.array(breaks[1:])
    mids = (left + right) / 2
    step = mids[1] - mids[0] if len(mids) > 1 else 1.0

    polys, ymax = [], 0.0
    for gname in groups:
        xg = x if by is None else x[(by == gname).to_numpy()]
        xg = xg[np.isfinite(xg)]
        yv = _poly_counts(xg, breaks, proportion)
        polys.append(yv)
        ymax = max(ymax, float(yv.max()))

    # --- ticks ---
    gridT1 = pretty(float(mids.min()), float(mids.max()))
    gridL1 = axis_format(gridT1, digits_d, axis_fmt,
                         axis_x_pre)
    gridT2 = pretty(0, ymax, n=6)
    fmt2 = get_tick_fmt(gridT2, digits_d)
    gridL2 = [f"{v:.{digits_d}f}" if fmt2 else f"{v:g}"
              for v in gridT2]

    # --- colors ---
    alpha_fill = auto_opacity(G, "fill")
    alpha_line = auto_opacity(G, "lines")
    fill_list = [fill[g % len(fill)]
                 if isinstance(fill, (list, tuple)) else fill
                 for g in range(G)]
    fill_rgba = [make_trans(c, alpha_fill) for c in fill_list]
    line_rgba = [make_trans(c, alpha_line) for c in fill_list]
    pt_rgba = [make_trans(c, 1) for c in fill_list]  # vertices

    fig = go.Figure()
    for g, gname in enumerate(groups):
        xx = np.concatenate([[mids[0] - step], mids,
                             [mids[-1] + step]])
        yy = np.concatenate([[0], polys[g], [0]])
        hover = [
            (f"{x_name}: {xv:.6g}"
             f"<br>{y_lab}: {round(float(yv), digits_d):g}")
            for xv, yv in zip(xx, yy)
        ]
        if G > 1 and by_name:
            hover = [f"{by_name}: {gname}<br>{h}" for h in hover]
        # lines only: mark the vertices, not the zero endpoints
        marker = (None if fill_area
                  else dict(color=pt_rgba[g],
                            size=[0] + [7.5] * len(mids) + [0],
                            line=dict(width=0)))
        fig.add_trace(go.Scatter(
            x=xx, y=yy,
            name=gname if G > 1 else None,
            mode="lines" if fill_area else "lines+markers",
            line=dict(color=line_rgba[g], width=2.2),
            marker=marker,
            fill="tozeroy" if fill_area else "none",
            fillcolor=fill_rgba[g] if fill_area else None,
            hoverinfo="text", hovertext=hover,
            showlegend=G > 1,
        ))

    # --- axes, grid, layout ---
    ax_x = axis_num(x_lab, gridT1, gridL1)
    ax_y = axis_num(y_lab, gridT2, gridL2)
    ax_y.update(showgrid=True,
                gridcolor=to_hex(style_opts["grid_col"]),
                gridwidth=1, griddash="dot")

    fig.update_layout(
        xaxis=ax_x,
        yaxis=ax_y,
        shapes=x_grid(gridT1) + plot_border(),
        template=None,
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]),
    )

    if G > 1:
        fig.update_layout(legend=legend_style(by_name, style_opts))

    if main:
        title_size = round(16 * get_option("main_size", 1))
        fig.update_layout(
            title=dict(text=main, x=0.5, xanchor="center",
                       y=0.99, yanchor="top",
                       font=dict(size=title_size,
                                 color=to_hex(get_option(
                                     "lab_color", "black")))),
            margin=dict(t=round(title_size * 2.2)))
    return fig


def _fp_facet(x, facet, facet_order, x_name, facet_name, breaks,
              proportion, fill, fill_area, x_lab, y_lab, digits_d,
              main, style_opts, by=None, by_name=None, n_col=1,
              facet2=None, facet2_order=None, facet2_name=None,
              axis_fmt="K", axis_x_pre=""):
    """One frequency-polygon panel per facet level: shared breaks,
    shared count scale, per-group proportions. With by=, one
    polygon per group within each panel, legend from the first
    panel showing a group; a group absent from a panel draws no
    polygon there. facet2: the two-facet grid, rows = facet2
    levels."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    x, facet = x[ok], np.asarray(facet)[ok]
    if facet2 is not None:
        facet2 = np.asarray(facet2)[ok]
    if facet_order is None:
        facet_order = sorted(set(facet))
    labels, pos, panel_sel, n_row_g, n_col = facet_panels(
        facet, facet_order, facet2, facet2_order,
        facet_name, facet2_name, n_col)
    n_f = len(labels)

    if by is None:
        groups = [None]
    else:
        by = pd.Series(by).astype(str).to_numpy()[ok]
        groups = sorted(set(by))
    G = len(groups)

    if fill is None:
        if G > 1:
            from .plotly_utils import BASE_COLORS
            fill = BASE_COLORS
        else:
            fill = get_option("bar_fill_cont", "#96AAC3")

    breaks = [float(b) for b in breaks]
    left = np.array(breaks[:-1])
    right = np.array(breaks[1:])
    mids = (left + right) / 2
    step = mids[1] - mids[0] if len(mids) > 1 else 1.0

    # polys[i][g] is None for an absent (facet, group) cell
    polys = [[None] * G for _ in range(n_f)]
    ymax = 0.0
    for i in range(n_f):
        in_lvl = panel_sel[i]
        for g, gname in enumerate(groups):
            sel = (in_lvl if gname is None
                   else in_lvl & (by == gname))
            if not sel.any():
                continue
            yv = _poly_counts(x[sel], breaks, proportion)
            polys[i][g] = yv
            ymax = max(ymax, float(yv.max()))

    axT1 = pretty(float(mids.min()), float(mids.max()))
    axT2 = pretty(0, ymax, n=6)
    fmt2 = get_tick_fmt(axT2, digits_d)
    ax = dict(axT1=axT1,
              axL1=axis_format(axT1, digits_d, axis_fmt,
                               axis_x_pre),
              axT2=axT2,
              axL2=[f"{v:.{digits_d}f}" if fmt2 else f"{v:g}"
                    for v in axT2])

    alpha_fill = auto_opacity(G, "fill")
    alpha_line = auto_opacity(G, "lines")
    fill_list = [fill[g % len(fill)]
                 if isinstance(fill, (list, tuple)) else fill
                 for g in range(G)]
    fill_rgba = [make_trans(c, alpha_fill) for c in fill_list]
    line_rgba = [make_trans(c, alpha_line) for c in fill_list]
    pt_rgba = [make_trans(c, 1) for c in fill_list]

    fig = facet_fig(n_row_g, n_col)
    seen = set()                    # legend once per group
    for i, lab in enumerate(labels):
        row, p_col = pos[i]
        panel_txt = (lab if facet2 is not None
                     else f"{facet_name}: {lab}")
        for g, gname in enumerate(groups):
            yv = polys[i][g]
            if yv is None:
                continue
            xx = np.concatenate([[mids[0] - step], mids,
                                 [mids[-1] + step]])
            yy = np.concatenate([[0], yv, [0]])
            hover = [
                (f"{x_name}: {xv:.6g}"
                 f"<br>{y_lab}: {round(float(hv), digits_d):g}"
                 f"<br>{panel_txt}")
                for xv, hv in zip(xx, yy)
            ]
            if gname is not None and by_name:
                hover = [f"{by_name}: {gname}<br>{h}"
                         for h in hover]
            marker = (None if fill_area
                      else dict(color=pt_rgba[g],
                                size=[0] + [7.5] * len(mids)
                                     + [0],
                                line=dict(width=0)))
            show = G > 1 and gname not in seen
            seen.add(gname)
            fig.add_trace(go.Scatter(
                x=xx, y=yy,
                name=gname if G > 1 else None,
                legendgroup=gname if G > 1 else None,
                mode="lines" if fill_area else "lines+markers",
                line=dict(color=line_rgba[g], width=2.2),
                marker=marker,
                fill="tozeroy" if fill_area else "none",
                fillcolor=fill_rgba[g] if fill_area else None,
                hoverinfo="text", hovertext=hover,
                showlegend=show,
            ), row=row, col=p_col)

    finish_facet(fig, labels, ax, x_lab, y_lab, n_col=n_col,
                 gridT1=axT1, style_opts=style_opts, pos=pos)
    if G > 1:
        fig.update_layout(legend=legend_style(by_name,
                                              style_opts))
    if main:
        title_size = round(16 * get_option("main_size", 1))
        fig.update_layout(
            title=dict(text=main, x=0.5, xanchor="center",
                       y=0.99, yanchor="top",
                       font=dict(size=title_size,
                                 color=to_hex(get_option(
                                     "lab_color", "black")))),
            margin=dict(t=round(title_size * 2.2)))
    return fig
