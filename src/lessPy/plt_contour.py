# plt_contour.py — analog of plt.contour.R
#
# form="contour" for XY(): filled contours of the 2-D kernel
# density of <x, y>. The density grid ports MASS::kde2d() directly
# (product normal kernel, bandwidth.nrd) so the contours match R.
# Renderer conventions follow plt_plotly.py: the fit line and the
# data ellipses arrive as prepared coordinates computed by XY().
# Deviation: the fill ramp is fixed at the default-theme blues,
# colorRampPalette(c("white", getColors("blues"))) — lessPy does
# not port themes.

import numpy as np
import plotly.graph_objects as go
from scipy import stats as sps

from .plotly_utils import (
    axis_format, axis_num, get_tick_fmt, plot_border,
    plotly_style, to_hex)
from .utils import get_option, pretty

# getColors("blues") preceded by white: the filled-contour ramp
# of the default lessR theme
_RAMP = [
    "#FFFFFF", "#CCECFF", "#B4D8FC", "#9DC5EB", "#84B2DB",
    "#6B9FCC", "#4F8DBC", "#2D7CAE", "#006BA0", "#005B93",
    "#004C8A", "#004087", "#0040A9",
]
_COLORSCALE = [[i / (len(_RAMP) - 1), c]
               for i, c in enumerate(_RAMP)]


def _bw_nrd(v):
    """Normal reference bandwidth.
    R analog: MASS::bandwidth.nrd()"""
    q1, q3 = np.percentile(v, [25, 75])
    return (4 * 1.06 * min(v.std(ddof=1), (q3 - q1) / 1.34)
            * len(v) ** (-0.2))


def _kde2d(x, y, n, lims, h=None):
    """2-D product-normal kernel density on an n x n grid over
    lims=(x_lo, x_hi, y_lo, y_hi). h: bandwidth pair on the
    kde2d scale (kernel sd is h/4), default bandwidth.nrd per
    axis. R analog: MASS::kde2d()"""
    if h is None:
        hx, hy = _bw_nrd(x) / 4, _bw_nrd(y) / 4
    else:
        hx, hy = h[0] / 4, h[1] / 4
    if hx <= 0 or hy <= 0:
        raise ValueError(
            "cannot estimate the density: x or y has no spread")
    gx = np.linspace(lims[0], lims[1], n)
    gy = np.linspace(lims[2], lims[3], n)
    ax = sps.norm.pdf((gx[:, None] - x[None, :]) / hx)
    ay = sps.norm.pdf((gy[:, None] - y[None, :]) / hy)
    z = ax @ ay.T / (len(x) * hx * hy)
    return gx, gy, z                   # z[i, j] at (gx[i], gy[j])


def plt_contour(xv, yv, contour_n, contour_nbins, contour_points,
                pt_size, x_lab, y_lab, main, digits_d, ell95,
                ellipses, ellipse_color, ellipse_lwd,
                fit_lines, fit_color, fit_lwd, legend=False,
                axis_fmt="K", axis_x_pre="", axis_y_pre=""):
    """Filled-contour display of the joint density of x and y,
    with optional data point, ellipse, and fit line overlays.
    R analog: .plt.contour()"""

    # KDE — extend grid beyond data so contours are not clipped
    # at the data edges
    x_ext = np.ptp(xv) * 0.12
    y_ext = np.ptp(yv) * 0.12
    gx, gy, z = _kde2d(xv, yv, contour_nbins,
                       (xv.min() - x_ext, xv.max() + x_ext,
                        yv.min() - y_ext, yv.max() + y_ext))

    # 95% mass threshold, then contour-derived ranges
    dz = z / z.sum()
    zs = np.sort(dz.ravel())[::-1]
    thr = zs[np.searchsorted(np.cumsum(zs), 0.95)]
    mask = dz >= thr
    x_in = gx[mask.any(axis=1)]
    y_in = gy[mask.any(axis=0)]

    # ellipse-derived ranges: the 95% data ellipse truncated to
    # the data range
    e_x, e_y = ell95
    keep = ((e_x >= xv.min()) & (e_x <= xv.max())
            & (e_y >= yv.min()) & (e_y <= yv.max()))
    e_x, e_y = e_x[keep], e_y[keep]

    # plot limits: when the kde grid covers the full data range,
    # use the grid extent so contours are not clipped; otherwise
    # the union of the data and truncated-ellipse ranges
    def _lim(cont, data, ell, grid):
        if cont[0] > data[0] or cont[1] < data[1]:
            if len(ell) > 0:
                return [min(data[0], ell.min()),
                        max(data[1], ell.max())]
            return list(data)
        return list(grid)
    x_lim = _lim((x_in.min(), x_in.max()), (xv.min(), xv.max()),
                 e_x, (gx[0], gx[-1]))
    y_lim = _lim((y_in.min(), y_in.max()), (yv.min(), yv.max()),
                 e_y, (gy[0], gy[-1]))

    if contour_points:                 # room for edge points
        x_lim[1] += 0.015 * (x_lim[1] - x_lim[0])
        y_lim[1] += 0.015 * (y_lim[1] - y_lim[0])

    # ticks within the limits
    axT1 = pretty(x_lim[0], x_lim[1])
    axT2 = pretty(y_lim[0], y_lim[1])
    fmt1 = get_tick_fmt(axT1, digits_d)
    fmt2 = get_tick_fmt(axT2, digits_d)
    axL1 = axis_format(axT1, digits_d, axis_fmt, axis_x_pre)
    axL2 = axis_format(axT2, digits_d, axis_fmt, axis_y_pre)

    style_opts = plotly_style()
    fig = go.Figure()

    # the filled contours: contour_n bands from min to max
    # density; start at the first interior level so the region
    # below it fills white, as in filled.contour()
    lv0, lv1 = float(z.min()), float(z.max())
    step = (lv1 - lv0) / contour_n
    xpart = f"%{{x:{fmt1}}}" if fmt1 else "%{x}"
    ypart = f"%{{y:{fmt2}}}" if fmt2 else "%{y}"
    fig.add_trace(go.Contour(
        x=gx, y=gy, z=z.T,
        colorscale=_COLORSCALE,
        contours=dict(start=lv0 + step, end=lv1, size=step),
        line=dict(width=0),
        showscale=bool(legend),
        colorbar=dict(
            tickfont=dict(
                color=to_hex(get_option("axis_color", "black")),
                size=16 * get_option("axis_size", 0.9)),
            outlinewidth=0),
        hovertemplate=(f"{x_lab}: {xpart}<br>{y_lab}: {ypart}"
                       "<extra></extra>"),
    ))

    # R overlay style: translucent black fill, thin white border
    if contour_points:
        px = float(pt_size) * 7.25
        if not np.isfinite(px) or px <= 0:
            px = 5
        fig.add_trace(go.Scatter(
            x=xv, y=yv, mode="markers",
            marker=dict(symbol="circle", size=px,
                        sizemode="diameter",
                        color="rgba(0,0,0,0.27)",
                        line=dict(color="#FFFFFF", width=0.2)),
            hoverinfo="skip", showlegend=False))

    # data ellipses: line only on a contour (no fill, as in R)
    for ex_l, ey_l in ellipses or []:
        fig.add_trace(go.Scatter(
            x=ex_l, y=ey_l, mode="lines",
            line=dict(color=to_hex(ellipse_color),
                      width=ellipse_lwd),
            hoverinfo="skip", showlegend=False))

    for fl in fit_lines or []:
        if len(fl["x"]) < 2:
            continue
        fig.add_trace(go.Scatter(
            x=fl["x"], y=fl["y"], mode="lines",
            line=dict(color=to_hex(fit_color), width=fit_lwd),
            hoverinfo="skip", showlegend=False))

    ax_x = axis_num(x_lab, axT1, axL1)
    ax_y = axis_num(y_lab, axT2, axL2)
    ax_x["range"] = x_lim
    ax_y["range"] = y_lim
    fig.update_layout(
        xaxis=ax_x, yaxis=ax_y,
        shapes=plot_border(),
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
