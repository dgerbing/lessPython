# plt_smooth.py — analog of the form="smooth" branch of
# plt.main.R (~795-808), which delegates to smoothScatter()
#
# form="smooth" for XY(): the 2-D binned kernel density of
# <x, y> as a smoothed color raster, densities transformed by
# z^smooth_power, with the smooth_points lowest-density points
# overplotted. _bkde2d() ports KernSmooth::bkde2D() directly
# (linear binning, separable normal kernel) with smoothScatter's
# bandwidth, (q95 - q05)/25 per axis, so the density grid
# matches R. Renderer conventions follow plt_plotly.py: fit
# lines, SE bands, and data ellipses arrive as prepared
# coordinates computed by XY(). Deviation: the fill ramp is
# fixed at the default-theme clr.den, colorRampPalette(
# c(window_fill, hcl(240, 80, 16))) — lessPy does not port
# themes.

import numpy as np
import plotly.graph_objects as go
from scipy import stats as sps
from scipy.signal import fftconvolve

from .plotly_utils import (
    as_plotly_color, axis_num, get_tick_fmt, plot_border,
    plotly_style, to_hex, x_grid, y_grid)
from .utils import get_option

# colorRampPalette(c("white", hcl(240, 80, 16))): the density
# ramp of the default lessR theme (plt.main.R ~799)
_RAMP = [[0, "#FFFFFF"], [1, "#0041A5"]]


def _linbin2d(x, y, gx, gy):
    """Linear binning: each point's weight split among its four
    surrounding grid nodes. R analog: KernSmooth's linbin2D()"""
    m1, m2 = len(gx), len(gy)
    lx = (x - gx[0]) / (gx[1] - gx[0])
    ly = (y - gy[0]) / (gy[1] - gy[0])
    i, j = np.floor(lx).astype(int), np.floor(ly).astype(int)
    rx, ry = lx - i, ly - j
    g = np.zeros((m1, m2))
    for di, wx in ((0, 1 - rx), (1, rx)):
        for dj, wy in ((0, 1 - ry), (1, ry)):
            ii, jj = i + di, j + dj
            ok = (ii >= 0) & (ii < m1) & (jj >= 0) & (jj < m2)
            np.add.at(g, (ii[ok], jj[ok]), (wx * wy)[ok])
    return g


def _bkde2d(x, y, h, n_grid, lims):
    """Binned 2-D kernel density estimate on an n_grid x n_grid
    grid over lims=(x_lo, x_hi, y_lo, y_hi), normal kernel with
    sd h=(hx, hy) truncated at tau=3.4 sd.
    R analog: KernSmooth::bkde2D()"""
    tau = 3.4
    gx = np.linspace(lims[0], lims[1], n_grid)
    gy = np.linspace(lims[2], lims[3], n_grid)
    g = _linbin2d(x, y, gx, gy)
    kern = []
    for (a, b), hh in (((lims[0], lims[1]), h[0]),
                       ((lims[2], lims[3]), h[1])):
        L = min(int(tau * hh * (n_grid - 1) / (b - a)),
                n_grid - 1)
        fac = (b - a) / (hh * (n_grid - 1))
        half = sps.norm.pdf(np.arange(L + 1) * fac) / hh
        full = np.concatenate([half[:0:-1], half])
        kern.append(full / (full.sum() * fac * hh))
    z = fftconvolve(g, np.outer(kern[0], kern[1]),
                    mode="same") / len(x)
    return gx, gy, np.clip(z, 0, None)


def plt_smooth(xv, yv, smooth_points, smooth_size, smooth_power,
               smooth_bins, x_lab, y_lab, main, digits_d,
               ax, gridT1, gridT2, x_lim, y_lim,
               fit_lines, fit_color, fit_lwd, se_polys, se_fill,
               ellipses, ellipse_fill, ellipse_color,
               ellipse_lwd):
    """Smoothed-density display of the scatter of x and y with
    the lowest-density points overplotted, plus the standard
    scatter overlays. R analog: smoothScatter() via plt.main.R"""

    # smoothScatter bandwidth and grid: data range extended by
    # 1.5 * bandwidth per axis (the bkde2D default range.x)
    h = []
    for v in (xv, yv):
        q05, q95 = np.percentile(v, [5, 95])
        hh = (q95 - q05) / 25
        h.append(hh if hh > 0 else 1.0)
    gx, gy, z = _bkde2d(xv, yv, h, smooth_bins,
                        (xv.min() - 1.5 * h[0],
                         xv.max() + 1.5 * h[0],
                         yv.min() - 1.5 * h[1],
                         yv.max() + 1.5 * h[1]))
    dens = z ** smooth_power           # compress the peaks

    style_opts = plotly_style()
    fig = go.Figure()

    fmtx = get_tick_fmt(xv, digits_d)
    fmty = get_tick_fmt(yv, digits_d)
    xpart = f"%{{x:{fmtx}}}" if fmtx else "%{x}"
    ypart = f"%{{y:{fmty}}}" if fmty else "%{y}"
    fig.add_trace(go.Heatmap(
        x=gx, y=gy, z=dens.T,
        colorscale=_RAMP, zsmooth="best", showscale=False,
        hovertemplate=(f"{x_lab}: {xpart}<br>{y_lab}: {ypart}"
                       "<extra></extra>"),
    ))

    # the smooth_points points in the lowest-density grid cells
    # overplot the raster (smoothScatter nrpoints selection)
    n_pts = min(len(xv), int(np.ceil(smooth_points)))
    if n_pts > 0:
        ix = ((len(gx) - 1) * (xv - gx[0])
              / (gx[-1] - gx[0])).astype(int)
        iy = ((len(gy) - 1) * (yv - gy[0])
              / (gy[-1] - gy[0])).astype(int)
        sel = np.argsort(dens[ix, iy], kind="stable")[:n_pts]
        px = 2 * float(smooth_size)    # pch="." speck
        if not np.isfinite(px) or px <= 0:
            px = 2
        fig.add_trace(go.Scatter(
            x=xv[sel], y=yv[sel], mode="markers",
            marker=dict(symbol="circle", size=px,
                        sizemode="diameter", color="#000000"),
            hoverinfo="skip", showlegend=False))

    # standard scatter overlays, as in plt_plotly
    for ex, ey in ellipses or []:
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(color=to_hex(ellipse_color),
                      width=ellipse_lwd),
            fill="toself",
            fillcolor=as_plotly_color(ellipse_fill),
            hoverinfo="skip", showlegend=False))
    for bx, bnd in se_polys or []:
        fig.add_trace(go.Scatter(
            x=bx, y=bnd, mode="none", fill="toself",
            fillcolor=as_plotly_color(se_fill),
            hoverinfo="skip", showlegend=False))
    for fl in fit_lines or []:
        if len(fl["x"]) < 2:
            continue
        fig.add_trace(go.Scatter(
            x=fl["x"], y=fl["y"], mode="lines", name="Fit",
            legendgroup="fit",
            line=dict(color=to_hex(fit_color), width=fit_lwd),
            hoverinfo="skip", showlegend=True))

    ax_x = axis_num(x_lab, ax["axT1"], ax["axL1"])
    ax_y = axis_num(y_lab, ax["axT2"], ax["axL2"])
    ax_x["range"] = x_lim
    ax_y["range"] = y_lim
    fig.update_layout(
        xaxis=ax_x, yaxis=ax_y,
        shapes=x_grid(gridT1) + y_grid(gridT2) + plot_border(),
        template=None,
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]),
    )

    if main:
        title_size = round(16 * get_option("main_size", 1))
        fig.update_layout(
            title=dict(text=main, x=0.5, xanchor="center",
                       y=0.99, yanchor="top",
                       font=dict(size=title_size)),
            margin=dict(t=round(title_size * 2.2)))
    return fig
