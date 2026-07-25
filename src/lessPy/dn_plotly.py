# dn_plotly.py — analog of dn.plotly.R
#
# Renders a kernel density curve (Gaussian kernel) from RAW numeric
# data, with a vertical line at each group's mean. Optional by=
# overlays one translucent curve per group with a legend.
#
# The KDE itself is computed here with numpy (R uses stats::density;
# the bandwidth arrives from X() via bw_nrd0, R's default rule), on
# a shared grid of n points extended cut*bw beyond the data range —
# the same geometry as R's density(cut = 3).
#
# facet= draws one panel per level: shared bandwidth, support, and
# density scale, mean line per panel, first level in the bottom
# panel. An EXTENSION beyond R, where X() still stops with
# "Facets not yet working with density" (X.R line ~220). With
# by=, each panel overlays one curve per group present in it.

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .utils import get_option, kde, pretty
from .plotly_utils import (
    as_plotly_color, auto_opacity, axis_format, axis_num,
    facet_fig, facet_panels, finish_facet, legend_style,
    make_trans, plot_border, plotly_style, to_hex, x_grid,
)

def dn_plotly(x, by=None, x_name=None, by_name=None,
              facet=None, facet_order=None, facet_name=None,
              facet2=None, facet2_order=None, facet2_name=None,
              fill=None, x_lab=None, y_lab="Density", main=None,
              bw=None, adjust=1, n=512, from_=None, to=None,
              full_curve=True, fill_area=True,
              kind="general", fill_normal=None,
              color_normal="gray20",
              show_histogram=False, fill_hist=None,
              hist_edges=None,
              rug=False, color_rug="black", size_rug=0.5,
              n_col=1, axis_fmt="K", axis_x_pre="",
              digits_d=3, style_opts=None):

    if style_opts is None:
        style_opts = plotly_style()
    if x_name is None:
        x_name = getattr(x, "name", None) or "x"
    if x_lab is None:
        x_lab = x_name
    fill_miss = fill is None
    if fill_miss:
        # R analog: X.R density default fill rgb(80,150,200)
        fill = "#5096C8"

    x = np.asarray(x, dtype=float)
    if bw is None:
        from .utils import bw_nrd0
        bw = bw_nrd0(x)
    h = bw * adjust

    if facet is not None:
        if fill_miss and by is not None:
            from .plotly_utils import BASE_COLORS
            fill = BASE_COLORS
        return _dn_facet(x, facet, facet_order, x_name,
                         facet_name, fill, x_lab, y_lab, main,
                         h, n, fill_area, style_opts,
                         by=by, by_name=by_name, n_col=n_col,
                         facet2=facet2,
                         facet2_order=facet2_order,
                         facet2_name=facet2_name,
                         kind=kind, fill_normal=fill_normal,
                         color_normal=color_normal,
                         show_histogram=show_histogram,
                         fill_hist=fill_hist,
                         hist_edges=hist_edges,
                         rug=rug, color_rug=color_rug,
                         size_rug=size_rug,
                         axis_fmt=axis_fmt,
                         axis_x_pre=axis_x_pre,
                         digits_d=digits_d)

    # --- groups ------------------------------------------------
    if by is None:
        groups = ["Series 1"]
    else:
        by = pd.Series(by).astype(str)
        groups = sorted(by.dropna().unique().tolist())
    G = len(groups)

    # groups need distinguishable hues: default to the palette
    if fill_miss and G > 1:
        from .plotly_utils import BASE_COLORS
        fill = BASE_COLORS

    def group_x(gname):
        xg = x if by is None else x[(by == gname).to_numpy()]
        return xg[np.isfinite(xg)]

    for gname in groups:
        n_ok = len(group_x(gname))
        if n_ok < 2:
            raise ValueError(
                f"density requires at least 2 finite observations; "
                f"group '{gname}' has {n_ok}")

    # --- shared support (R density: cut = 3 -> +- 3*bw) ---------
    if full_curve:
        ends = []
        for gname in groups:
            xg = group_x(gname)
            ends += [xg.min() - 3 * h, xg.max() + 3 * h]
        lo = min(ends) if from_ is None else from_
        hi = max(ends) if to is None else to
    else:
        fx = x[np.isfinite(x)]
        lo = fx.min() if from_ is None else from_
        hi = fx.max() if to is None else to
    if hi <= lo:
        hi = lo + max(abs(lo) * 1e-4, 1e-6)
    n = max(2, int(n))
    grid = np.linspace(lo, hi, n)
    dx = grid[1] - grid[0]

    # --- densities per group -------------------------------------
    dens = []
    ymax = 0.0
    for gname in groups:
        xg = group_x(gname)
        y = kde(xg, grid, h)
        cum = np.concatenate(
            [[0], np.cumsum((y[1:] + y[:-1]) * 0.5 * dx)])
        if cum[-1] > 0:
            cum = cum / cum[-1]
        dens.append(dict(y=y, cum=cum, mn=float(xg.mean())))
        ymax = max(ymax, float(y.max()))

    # --- normal curve and background histogram --------------------
    # R analog: dn.main.R — kind "normal"/"both" adds the normal
    # curve at the sample mean and sd; show_histogram draws a
    # faint density-scaled histogram behind the curves
    d_nrm = None
    if kind in ("normal", "both"):
        xg0 = group_x(groups[0])
        mu, sd = float(xg0.mean()), float(xg0.std(ddof=1))
        z = (grid - mu) / sd
        d_nrm = np.exp(-0.5 * z * z) / (sd * np.sqrt(2 * np.pi))
    hist_y = None
    if show_histogram and hist_edges is not None:
        e = np.asarray(hist_edges, dtype=float)
        cnt, _ = np.histogram(group_x(groups[0]), bins=e)
        tot = cnt.sum()
        if tot > 0:
            hist_w = np.diff(e)
            hist_y = cnt / (tot * hist_w)  # density scale
            hist_mids = (e[:-1] + e[1:]) / 2
    if kind == "normal":               # general curve not drawn
        ymax = 0.0
    if d_nrm is not None:
        ymax = max(ymax, float(d_nrm.max()))
    if hist_y is not None:
        ymax = max(ymax, float(hist_y.max()))

    # --- ticks ----------------------------------------------------
    gridT1 = pretty(float(lo), float(hi))
    gridL1 = axis_format(gridT1, digits_d, axis_fmt,
                         axis_x_pre)
    gridT2 = pretty(0, ymax, n=6)
    gridL2 = [f"{v:g}" for v in gridT2]

    # --- colors ---------------------------------------------------
    alpha_fill = auto_opacity(G, "fill")
    alpha_line = auto_opacity(G, "lines") if fill_area else 1
    fill_list = [fill[g % len(fill)]
                 if isinstance(fill, (list, tuple)) else fill
                 for g in range(G)]
    fill_rgba = [make_trans(c, alpha_fill) for c in fill_list]
    line_rgba = [make_trans(c, alpha_line) for c in fill_list]

    def hover(d):
        return [
            (f"{x_name}: {grid[i]:.6g}"
             f"<br>Density={round(float(d['y'][i]), 6):g}"
             f"<br>Cumulative %: {100 * d['cum'][i]:.1f}%")
            for i in range(n)
        ]

    fig = go.Figure()
    if hist_y is not None:             # faint histogram behind
        fig.add_trace(go.Bar(
            x=hist_mids, y=hist_y, width=hist_w,
            marker=dict(color=as_plotly_color(
                get_option("se_fill", "#1A1A1A19")
                if fill_hist is None else fill_hist),
                line=dict(width=0)),
            hoverinfo="skip", showlegend=False))
    if kind in ("general", "both"):
        for g, gname in enumerate(groups):
            d = dens[g]
            hv = hover(d)
            if G > 1 and by_name:
                hv = [f"{by_name}: {gname}<br>{t}" for t in hv]
            fig.add_trace(go.Scatter(
                mode="lines",
                x=grid, y=d["y"],
                name=gname if G > 1 else None,
                line=dict(color=line_rgba[g], width=1.4),
                fill="tozeroy" if fill_area else "none",
                fillcolor=fill_rgba[g] if fill_area else None,
                hoverinfo="text",
                hovertext=hv,
                showlegend=G > 1,
            ))
            # vertical line at the group mean, up to the curve
            y_mn = float(np.interp(d["mn"], grid, d["y"]))
            mean_line = dict(color=line_rgba[g], dash="dash",
                             width=1.4 if fill_area else 0.7)
            fig.add_trace(go.Scatter(
                mode="lines",
                x=[d["mn"], d["mn"]], y=[0, y_mn],
                line=mean_line,
                hoverinfo="skip", showlegend=False,
            ))
    if d_nrm is not None:              # normal curve
        nrm_fill = (None if fill_normal in (None, "transparent")
                    else as_plotly_color(fill_normal))
        fig.add_trace(go.Scatter(
            mode="lines", x=grid, y=d_nrm,
            line=dict(color=to_hex(color_normal),
                      width=1.35 if nrm_fill is None else 1),
            fill="none" if nrm_fill is None else "tozeroy",
            fillcolor=nrm_fill,
            hoverinfo="skip", showlegend=False))
    if rug:                            # ticks at each data value
        xr, yr = [], []
        y1r = -0.05 * ymax
        for v in group_x(groups[0]):
            xr += [float(v), float(v), None]
            yr += [0.0, y1r, None]
        fig.add_trace(go.Scatter(
            mode="lines", x=xr, y=yr,
            line=dict(color=to_hex(color_rug),
                      width=max(0.5, float(size_rug))),
            hoverinfo="skip", showlegend=False))

    # --- axes, grid, layout ----------------------------------------
    axis_shapes = []
    if not fill_area:      # lines only: neutral x-axis baseline
        axis_shapes = [dict(
            type="line", xref="paper", yref="paper",
            x0=0, x1=1, y0=0, y1=0,
            line=dict(color=to_hex("gray50"), width=1))]

    ax_x = axis_num(x_lab, gridT1, gridL1)
    ax_y = axis_num(y_lab, gridT2, gridL2)
    ax_y.update(showgrid=True,
                gridcolor=to_hex(style_opts["grid_col"]),
                gridwidth=1, griddash="dot")

    fig.update_layout(
        xaxis=ax_x,
        yaxis=ax_y,
        shapes=x_grid(gridT1) + plot_border() + axis_shapes,
        template=None,
        bargap=0,
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


def _dn_facet(x, facet, facet_order, x_name, facet_name, fill,
              x_lab, y_lab, main, h, n, fill_area, style_opts,
              by=None, by_name=None, n_col=1,
              facet2=None, facet2_order=None, facet2_name=None,
              kind="general", fill_normal=None,
              color_normal="gray20",
              show_histogram=False, fill_hist=None,
              hist_edges=None,
              rug=False, color_rug="black", size_rug=0.5,
              axis_fmt="K", axis_x_pre="", digits_d=3):
    """One density panel per facet level: common bandwidth,
    support, and density scale; mean line per curve. With by=,
    one translucent curve per group within each panel, legend
    from the first panel showing a group; a group absent from a
    panel draws no curve there. A cell too thin to estimate
    (< 2 finite values) draws an empty panel; only if no cell
    can plot does the call stop, as in R's .plt.dist.facet.
    facet2: the two-facet grid, rows = facet2 levels. Without
    by=, each panel takes the single-panel embellishments, as
    in R: kind= normal curve from the panel's mean and sd,
    show_histogram backdrop on shared bins, rug."""
    ok = np.isfinite(x)
    x, facet = x[ok], np.asarray(facet)[ok]
    if facet2 is not None:
        facet2 = np.asarray(facet2)[ok]
    if facet_order is None:
        facet_order = sorted(set(facet))
    labels, pos, sel, n_row_g, n_col = facet_panels(
        facet, facet_order, facet2, facet2_order,
        facet_name, facet2_name, n_col)
    n_f = len(labels)

    if by is None:
        groups = [None]
    else:
        by = pd.Series(by).astype(str).to_numpy()[ok]
        groups = sorted(set(by))
    G = len(groups)

    def cell(i, gname):
        m = sel[i]
        if gname is not None:
            m = m & (by == gname)
        return x[m]

    if not any(len(cell(i, g)) >= 2
               for i in range(n_f) for g in groups):
        raise ValueError(
            "No facet cell has enough data to plot.")

    lo = x.min() - 3 * h
    hi = x.max() + 3 * h
    grid = np.linspace(lo, hi, max(2, int(n)))
    dx = grid[1] - grid[0]

    # curves[i][g] is None for an absent or too-thin cell
    curves = [[None] * G for _ in range(n_f)]
    ymax = 0.0
    for i in range(n_f):
        for g, gname in enumerate(groups):
            xg = cell(i, gname)
            if len(xg) < 2:
                continue
            y = kde(xg, grid, h)
            cum = np.concatenate(
                [[0], np.cumsum((y[1:] + y[:-1]) * 0.5 * dx)])
            if cum[-1] > 0:
                cum = cum / cum[-1]
            curves[i][g] = dict(y=y, cum=cum,
                                mn=float(xg.mean()))
            ymax = max(ymax, float(y.max()))

    # per-panel embellishments, single series only (R's
    # .plt.dist.facet has no grouping): normal curve from the
    # panel's mean/sd, density-scaled histogram on shared bins
    solo = G == 1
    d_nrms = [None] * n_f
    if solo and kind in ("normal", "both"):
        for i in range(n_f):
            xg = cell(i, None)
            if len(xg) < 2:
                continue
            mu = float(xg.mean())
            sd = float(xg.std(ddof=1))
            if sd == 0:
                continue
            z = (grid - mu) / sd
            d_nrms[i] = (np.exp(-0.5 * z * z)
                         / (sd * np.sqrt(2 * np.pi)))
            ymax = max(ymax, float(d_nrms[i].max()))
    hist_ys = [None] * n_f
    hist_mids = hist_w = None
    if solo and show_histogram and hist_edges is not None:
        e = np.asarray(hist_edges, dtype=float)
        hist_w = np.diff(e)
        hist_mids = (e[:-1] + e[1:]) / 2
        for i in range(n_f):
            cnt, _ = np.histogram(cell(i, None), bins=e)
            tot = cnt.sum()
            if tot > 0:
                hist_ys[i] = cnt / (tot * hist_w)
                ymax = max(ymax, float(hist_ys[i].max()))

    axT1 = pretty(float(lo), float(hi))
    axT2 = pretty(0, ymax, n=6)
    ax = dict(axT1=axT1,
              axL1=axis_format(axT1, digits_d, axis_fmt,
                               axis_x_pre),
              axT2=axT2, axL2=[f"{v:g}" for v in axT2])

    alpha_fill = auto_opacity(G, "fill")
    alpha_line = auto_opacity(G, "lines") if fill_area else 1
    fill_list = [fill[g % len(fill)]
                 if isinstance(fill, (list, tuple)) else fill
                 for g in range(G)]
    fill_rgba = [make_trans(c, alpha_fill) for c in fill_list]
    line_rgba = [make_trans(c, alpha_line) for c in fill_list]

    fig = facet_fig(n_row_g, n_col)
    seen = set()                    # legend once per group
    for i, lab in enumerate(labels):
        row, p_col = pos[i]
        panel_txt = (lab if facet2 is not None
                     else f"{facet_name}: {lab}")
        if hist_ys[i] is not None:     # faint histogram behind
            fig.add_trace(go.Bar(
                x=hist_mids, y=hist_ys[i], width=hist_w,
                marker=dict(color=as_plotly_color(
                    get_option("se_fill", "#1A1A1A19")
                    if fill_hist is None else fill_hist),
                    line=dict(width=0)),
                hoverinfo="skip", showlegend=False,
            ), row=row, col=p_col)
        for g, gname in enumerate(groups):
            c = curves[i][g]
            if c is None or kind == "normal":
                continue
            hover = [
                (f"{x_name}: {grid[j]:.6g}"
                 f"<br>Density={round(float(c['y'][j]), 6):g}"
                 f"<br>Cumulative %: {100 * c['cum'][j]:.1f}%"
                 f"<br>{panel_txt}")
                for j in range(len(grid))
            ]
            if gname is not None and by_name:
                hover = [f"{by_name}: {gname}<br>{t}"
                         for t in hover]
            show = G > 1 and gname not in seen
            seen.add(gname)
            fig.add_trace(go.Scatter(
                mode="lines", x=grid, y=c["y"],
                name=gname if G > 1 else None,
                legendgroup=gname if G > 1 else None,
                line=dict(color=line_rgba[g], width=1.4),
                fill="tozeroy" if fill_area else "none",
                fillcolor=fill_rgba[g] if fill_area else None,
                hoverinfo="text", hovertext=hover,
                showlegend=show,
            ), row=row, col=p_col)
            y_mn = float(np.interp(c["mn"], grid, c["y"]))
            fig.add_trace(go.Scatter(
                mode="lines", x=[c["mn"], c["mn"]], y=[0, y_mn],
                line=dict(color=line_rgba[g], dash="dash",
                          width=1.4 if fill_area else 0.7),
                hoverinfo="skip", showlegend=False,
            ), row=row, col=p_col)
        if d_nrms[i] is not None:      # normal curve
            nrm_fill = (None
                        if fill_normal in (None, "transparent")
                        else as_plotly_color(fill_normal))
            fig.add_trace(go.Scatter(
                mode="lines", x=grid, y=d_nrms[i],
                line=dict(color=to_hex(color_normal),
                          width=1.35 if nrm_fill is None else 1),
                fill="none" if nrm_fill is None else "tozeroy",
                fillcolor=nrm_fill,
                hoverinfo="skip", showlegend=False,
            ), row=row, col=p_col)
        if solo and rug:               # ticks at each data value
            xr, yr = [], []
            y1r = -0.05 * ymax
            for v in cell(i, None):
                xr += [float(v), float(v), None]
                yr += [0.0, y1r, None]
            fig.add_trace(go.Scatter(
                mode="lines", x=xr, y=yr,
                line=dict(color=to_hex(color_rug),
                          width=max(0.5, float(size_rug))),
                hoverinfo="skip", showlegend=False,
            ), row=row, col=p_col)

    fig.update_layout(bargap=0)
    finish_facet(fig, labels, ax, x_lab, y_lab, n_col=n_col,
                 gridT1=axT1, style_opts=style_opts, pos=pos)
    if solo and rug:    # room below zero for the rug ticks
        top = float(ax["axT2"][-1])
        fig.update_yaxes(range=[-0.06 * top, top * 1.04])
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
