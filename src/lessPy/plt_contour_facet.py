# plt_contour_facet.py — analog of plt.contourFacet.R
#
# Faceted form="contour" / form="smooth" for XY(): one density
# panel per facet cell, computed on a common KDE grid and
# bandwidth so density is comparable panel-to-panel — shared
# contour levels (contour) or a shared color range on the
# smooth_power-transformed density (smooth). As in R, the smooth
# panels use kde2d with smoothScatter's bandwidth ((q95-q05)/25
# per axis, h = 4x the kernel sd) so faceted and single-panel
# smooths are consistent; contour keeps the normal-reference
# bandwidth. A cell with < 3 points draws an empty panel; only
# if no cell can estimate does the call stop. Layout follows R:
# near-square single-facet grid (n_col computed by the caller),
# rows = facet2 levels for two facets.

import numpy as np
import plotly.graph_objects as go

from .plotly_utils import (
    axis_format, facet_fig, facet_panels, finish_facet,
    plotly_style, to_hex)
from .plt_contour import _COLORSCALE, _bw_nrd, _kde2d
from .plt_smooth import _RAMP as _SMOOTH_RAMP
from .utils import get_option, pretty


def _ss_bw(v):
    """smoothScatter's kernel sd: the 5-95% quantile spread / 25.
    R analog: the ss.bw() local of .plt.contour.facet"""
    q05, q95 = np.percentile(v, [5, 95])
    return (q95 - q05) / 25


def plt_contour_facet(xv, yv, facet_arr, facet_order, facet_name,
                      facet2_arr, facet2_order, facet2_name,
                      contour_n, contour_nbins, contour_points,
                      pt_size, x_lab, y_lab, main, digits_d,
                      ellipse, ellipse_color, ellipse_lwd,
                      fit, fit_power, fit_color, fit_lwd,
                      fit_fn, ellipse_fn,
                      render="contour", smooth_power=0.25,
                      smooth_points=100, smooth_bins=128,
                      n_col=1, axis_fmt="K", axis_x_pre="",
                      axis_y_pre=""):
    """Faceted joint-density display. fit_fn/ellipse_fn: the
    caller's per-panel fit and ellipse constructors (XY._plt_fit,
    XY._ellipse_region). R analog: .plt.contour.facet()"""
    smooth = render == "smooth"
    n_grid = smooth_bins if smooth else contour_nbins

    labels, pos, sel, n_row_g, n_col = facet_panels(
        facet_arr, facet_order, facet2_arr, facet2_order,
        facet_name, facet2_name, n_col)
    n_f = len(labels)

    # common KDE grid + bandwidth: densities comparable across
    # panels
    x_ext = np.ptp(xv) * 0.12
    y_ext = np.ptp(yv) * 0.12
    lims = (xv.min() - x_ext, xv.max() + x_ext,
            yv.min() - y_ext, yv.max() + y_ext)
    if smooth:
        h_common = [4 * _ss_bw(xv), 4 * _ss_bw(yv)]
    else:
        h_common = [_bw_nrd(xv), _bw_nrd(yv)]

    dens = [None] * n_f
    z_lo = z_hi = None
    for i in range(n_f):
        xi, yi = xv[sel[i]], yv[sel[i]]
        if len(xi) < 3:
            continue
        h_i = [h if h > 0 else fb for h, fb in
               zip(h_common, (_bw_nrd(xi), _bw_nrd(yi)))]
        if min(h_i) <= 0:   # still degenerate (e.g., collinear)
            continue
        gx, gy, z = _kde2d(xi, yi, n_grid, lims, h=h_i)
        dens[i] = (gx, gy, z)
        z_lo = float(z.min()) if z_lo is None \
            else min(z_lo, float(z.min()))
        z_hi = float(z.max()) if z_hi is None \
            else max(z_hi, float(z.max()))
    if z_hi is None:
        raise ValueError(
            "No facet cell has enough data (>= 3 points) for a "
            + ("smooth density." if smooth else "contour."))
    step = (z_hi - z_lo) / contour_n
    zlim_s = (max(z_lo, 0) ** smooth_power,
              max(z_hi, 0) ** smooth_power)

    # shared plot limits and ticks across panels
    axT1 = pretty(float(xv.min()), float(xv.max()))
    axT2 = pretty(float(yv.min()), float(yv.max()))
    ax = {"axT1": axT1,
          "axL1": axis_format(axT1, digits_d, axis_fmt,
                              axis_x_pre),
          "axT2": axT2,
          "axL2": axis_format(axT2, digits_d, axis_fmt,
                              axis_y_pre)}

    ell_levels = [lv for lv in np.atleast_1d(ellipse) if lv]
    px = float(pt_size) * 7.25
    if not np.isfinite(px) or px <= 0:
        px = 5

    style_opts = plotly_style()
    fig = facet_fig(n_row_g, n_col)
    for i in range(n_f):
        row, col = pos[i]
        if dens[i] is None:
            continue
        gx, gy, z = dens[i]
        xi, yi = xv[sel[i]], yv[sel[i]]
        if smooth:
            fig.add_trace(go.Heatmap(
                x=gx, y=gy, z=(z ** smooth_power).T,
                colorscale=_SMOOTH_RAMP, zsmooth="best",
                zmin=zlim_s[0], zmax=zlim_s[1],
                showscale=False, hoverinfo="skip",
            ), row=row, col=col)
            if smooth_points > 0:       # least-dense points
                ix = np.clip(np.searchsorted(gx, xi), 0,
                             len(gx) - 1)
                iy = np.clip(np.searchsorted(gy, yi), 0,
                             len(gy) - 1)
                take = np.argsort(z[ix, iy],
                                  kind="stable")[:smooth_points]
                fig.add_trace(go.Scatter(
                    x=xi[take], y=yi[take], mode="markers",
                    marker=dict(symbol="circle", size=px,
                                sizemode="diameter",
                                color="rgba(0,0,0,0.33)",
                                line=dict(color="#FFFFFF",
                                          width=0.2)),
                    hoverinfo="skip", showlegend=False,
                ), row=row, col=col)
        else:
            fig.add_trace(go.Contour(
                x=gx, y=gy, z=z.T,
                colorscale=_COLORSCALE,
                contours=dict(start=z_lo + step, end=z_hi,
                              size=step),
                line=dict(width=0), showscale=False,
                hoverinfo="skip",
            ), row=row, col=col)
            if contour_points:
                fig.add_trace(go.Scatter(
                    x=xi, y=yi, mode="markers",
                    marker=dict(symbol="circle", size=px,
                                sizemode="diameter",
                                color="rgba(0,0,0,0.27)",
                                line=dict(color="#FFFFFF",
                                          width=0.2)),
                    hoverinfo="skip", showlegend=False,
                ), row=row, col=col)
        for lv in ell_levels:           # data ellipse(s)
            if len(xi) < 3:
                break
            ex, ey = ellipse_fn(xi, yi, float(lv))
            fig.add_trace(go.Scatter(
                x=ex, y=ey, mode="lines",
                line=dict(color=to_hex(ellipse_color),
                          width=ellipse_lwd),
                hoverinfo="skip", showlegend=False,
            ), row=row, col=col)
        if fit != "off" and len(xi) >= 3:
            xs, _, f = fit_fn(xi, yi, fit, fit_power)
            okf = np.isfinite(f)
            if okf.sum() >= 2:
                fig.add_trace(go.Scatter(
                    x=xs[okf], y=f[okf], mode="lines",
                    line=dict(color=to_hex(fit_color),
                              width=fit_lwd),
                    hoverinfo="skip", showlegend=False,
                ), row=row, col=col)

    # size the figure so each contour panel is roughly square
    # (they read as scatter panels, not wide distribution strips)
    panel = 300
    finish_facet(fig, labels, ax, x_lab, y_lab, gridT1=None,
                 style_opts=style_opts, n_col=n_col, pos=pos,
                 height=150 + panel * n_row_g,
                 width=130 + panel * n_col)
    # density panels span the data, clipped to the shared limits
    pad_x = 0.04 * (axT1[-1] - axT1[0])
    pad_y = 0.04 * (axT2[-1] - axT2[0])
    fig.update_xaxes(range=[axT1[0] - pad_x, axT1[-1] + pad_x])
    fig.update_yaxes(range=[axT2[0] - pad_y, axT2[-1] + pad_y])
    if main:
        title_size = round(16 * get_option("main_size", 1))
        fig.update_layout(
            title=dict(text=main, x=0.5, xanchor="center",
                       font=dict(size=title_size)))
    return fig
