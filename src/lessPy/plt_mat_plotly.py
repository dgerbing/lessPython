# plt_mat_plotly.py — the scatterplot matrix (SPLOM).
#
# Shared by Regression() and XY(), mirroring the single R helper
# .plt.mat (plt.mat.R). The layout matches R's pairs() panels:
#   lower triangle  scatter of (col var, row var) with a fit line
#                   and its 95% confidence band
#   upper triangle  the correlation coefficient, as text
#   diagonal        the variable name, on the se_fill background
# Regression() calls it with fit="lm"; XY() passes its own fit=
# (default "off", so no line). Points are dropped listwise across
# all variables first (R's na.omit).

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sps

from .plotly_utils import (
    as_plotly_color, axis_format, make_trans, plotly_style,
    to_hex)
from .utils import fmt, get_option, pretty


def _axis_ref(k, kind):
    """Domain reference string for the k-th subplot axis, as used
    by shapes and annotations: 'x domain', 'x2 domain', ..."""
    n = "" if k == 1 else str(k)
    return f"{kind}{n} domain"


def scatter_matrix(df, fit="lm", digits_d=2, cor_coef=True,
                   main=None, band=True, form="scatter",
                   contour_n=None, contour_nbins=50,
                   contour_points=False, smooth_bins=128,
                   smooth_power=0.25):
    """Scatterplot matrix of the columns of df, in column order.
    fit: "off" (no line, XY default), "lm" (Regression) or
    "loess". cor_coef=True puts the correlation in the upper
    triangle and the scatter below (~ .plt.mat, Regression);
    cor_coef=False draws the scatter in every off-diagonal cell,
    the symmetric matrix R's Logit uses (logit.4Pred). band adds
    the fit's 95% confidence band. form="contour" or "smooth"
    draws each cell as its own kernel density over the cell's
    range, 8 contour bands unless contour_n is given (a cell cannot
    resolve the 20 of a single plot), any fit drawn without its
    band. R analog: .plt.mat() panel.dens. Returns a plotly
    Figure."""
    if form not in ("scatter", "contour", "smooth"):
        raise ValueError(
            'A matrix renders its cells as form="scatter", "contour", '
            f'or "smooth", so form="{form}" is not\navailable for a '
            "matrix.")
    contour_n = 8 if contour_n is None else int(contour_n)
    cols = list(df.columns)
    n = len(cols)
    if n < 2:
        raise ValueError(
            "a scatterplot matrix needs at least 2 variables")

    data = df.dropna()                 # R na.omit: listwise
    vals = {c: data[c].to_numpy(dtype=float) for c in cols}
    rng = {c: pretty(float(v.min()), float(v.max()))
           for c, v in vals.items()}

    style = plotly_style()
    panel_fill = get_option("panel_fill", "white")
    window_fill = get_option("window_fill", "white")
    bg = window_fill if panel_fill == "transparent" else panel_fill
    diag_fill = get_option("se_fill", "#1A1A1A19")
    se_fill = get_option("se_fill", "#1A1A1A19")
    border = to_hex(style["panel_border"])
    pt_fill = get_option("pt_fill", get_option("pt_color",
                                               "#324E5C"))
    pt_color = get_option("pt_color", "#324E5C")
    lab_color = to_hex(style["lab_color"])
    fit_color = to_hex(get_option("fit_color", "#5C4032"))
    fit_lwd = get_option("fit_lwd", 2)

    # sizes shrink with the variable count, as R's cex.adj
    px = 0.5 * max(2.5, 7.25 * (0.80 - 0.048 * n) / 0.75)
    txt_size = max(9, round(10 * (1.6 - 0.065 * n)))

    fig = make_subplots(rows=n, cols=n, horizontal_spacing=0.008,
                        vertical_spacing=0.008)

    do_fit = fit in ("lm", "loess")

    for r in range(1, n + 1):          # r: row (top = 1)
        for c in range(1, n + 1):      # c: column (left = 1)
            k = (r - 1) * n + c        # subplot / axis index
            xr, yr = _axis_ref(k, "x"), _axis_ref(k, "y")

            # scatter in the lower triangle, and also in the
            # upper triangle when no correlations are requested
            if r != c and (r > c or not cor_coef) \
                    and form != "scatter":
                _cell_bg(fig, xr, yr, bg, border)
                _density_cell(fig, r, c, vals[cols[c - 1]],
                              vals[cols[r - 1]], rng[cols[c - 1]],
                              rng[cols[r - 1]], form, contour_n,
                              contour_nbins, contour_points,
                              smooth_bins, smooth_power,
                              fit if do_fit else "off", fit_color,
                              fit_lwd)
                continue
            if r != c and (r > c or not cor_coef):
                _cell_bg(fig, xr, yr, bg, border)
                _scatter_cell(fig, r, c, vals[cols[c - 1]],
                              vals[cols[r - 1]], fit, do_fit, px,
                              pt_fill, pt_color, fit_color,
                              fit_lwd, se_fill, band)
                continue

            # text cells (diagonal, upper correlation): an
            # invisible point anchors the axes so the domain
            # shape/text render
            _anchor(fig, r, c)
            if r == c:                 # diagonal: variable name
                _cell_bg(fig, xr, yr, diag_fill, border)
                fig.add_annotation(
                    xref=xr, yref=yr, x=0.5, y=0.5, text=cols[r - 1],
                    showarrow=False,
                    font=dict(size=txt_size, color=lab_color))
            else:                      # upper: correlation
                _cell_bg(fig, xr, yr, bg, border)
                a, b = vals[cols[r - 1]], vals[cols[c - 1]]
                rr = float(np.corrcoef(a, b)[0, 1])
                fig.add_annotation(
                    xref=xr, yref=yr, x=0.5, y=0.5,
                    text=fmt(rr, 2), showarrow=False,
                    font=dict(size=txt_size, color="black"))

    # per-column x range (col var) and per-row y range (row var)
    for c in range(1, n + 1):
        rc = rng[cols[c - 1]]
        fig.update_xaxes(range=[rc[0], rc[-1]], showgrid=False,
                         zeroline=False, showticklabels=False,
                         ticks="", col=c)
    for r in range(1, n + 1):
        rr = rng[cols[r - 1]]
        fig.update_yaxes(range=[rr[0], rr[-1]], showgrid=False,
                         zeroline=False, showticklabels=False,
                         ticks="", row=r)

    # outer scales only: x on the bottom row, y on the left
    # column, interior ticks (drop the endpoints, which would
    # collide at the panel seams)
    for c in range(1, n + 1):
        t = rng[cols[c - 1]][1:-1]
        fig.update_xaxes(
            showticklabels=True, tickvals=t,
            ticktext=axis_format(t, digits_d),
            tickfont=dict(size=8), row=n, col=c)
    for r in range(1, n + 1):
        t = rng[cols[r - 1]][1:-1]
        fig.update_yaxes(
            showticklabels=True, tickvals=t,
            ticktext=axis_format(t, digits_d),
            tickfont=dict(size=8), row=r, col=1)

    fig.update_layout(
        template=None, showlegend=False,
        plot_bgcolor=to_hex(bg),
        paper_bgcolor=to_hex(window_fill))
    if main:
        fig.update_layout(title=dict(
            text=main, x=0.5, xanchor="center",
            font=dict(size=round(16 * get_option("main_size",
                                                 1)))))
    return fig


def _anchor(fig, r, c):
    """A single invisible point, so a text-only cell's axes exist
    and its domain-referenced shape and label are drawn."""
    fig.add_trace(go.Scatter(
        x=[0.5], y=[0.5], mode="markers",
        marker=dict(opacity=0), hoverinfo="skip",
        showlegend=False), row=r, col=c)


def _cell_bg(fig, xr, yr, fill, border):
    """Fill the cell and outline it, spanning the full domain."""
    fig.add_shape(type="rect", xref=xr, yref=yr,
                  x0=0, x1=1, y0=0, y1=1, layer="below",
                  fillcolor=as_plotly_color(fill),
                  line=dict(color=border, width=1))


def _scatter_cell(fig, r, c, xv, yv, fit, do_fit, px,
                  pt_fill, pt_color, fit_color, fit_lwd, se_fill,
                  band=True):
    """Points, and (when fit is on) the fit line and, if band,
    its 95% confidence band, in subplot (row r, col c)."""
    from .XY import _loess, _plt_fit, _se_band

    if do_fit and len(xv) >= 2:
        if fit == "loess":
            xs, _, f, se_f = _loess(xv, yv, 2 / 3)
        else:                          # lm
            xs, ys_s, f = _plt_fit(xv, yv, "lm", 1)
        if band:
            if fit == "loess":
                tq = sps.t.ppf((1 + 0.95) / 2, len(xs) - 1)
                poly = (np.concatenate([xs, xs[::-1]]),
                        np.concatenate([f + tq * se_f,
                                        (f - tq * se_f)[::-1]]))
            else:
                poly = _se_band(xs, ys_s, f, 0.95)
            fig.add_trace(go.Scatter(
                x=poly[0], y=poly[1], mode="none", fill="toself",
                fillcolor=as_plotly_color(se_fill),
                hoverinfo="skip", showlegend=False), row=r, col=c)

    fig.add_trace(go.Scatter(
        x=xv, y=yv, mode="markers",
        marker=dict(symbol="circle", size=px, sizemode="diameter",
                    color=make_trans(pt_fill, 0.9), opacity=1,
                    line=dict(color=to_hex(pt_color), width=0.5)),
        hoverinfo="x+y", showlegend=False), row=r, col=c)

    if do_fit and len(xv) >= 2:
        fig.add_trace(go.Scatter(
            x=xs, y=f, mode="lines",
            line=dict(color=fit_color, width=fit_lwd),
            hoverinfo="skip", showlegend=False), row=r, col=c)


def _density_cell(fig, r, c, xv, yv, xr, yr, form, contour_n,
                  contour_nbins, contour_points, smooth_bins,
                  smooth_power, fit, fit_color, fit_lwd):
    """One cell as a kernel density over the cell's range: filled
    contour bands or the smooth shading, then the fit, unbanded.
    R analog: panel.dens of .plt.mat()"""
    from .plt_contour import _COLORSCALE, _bw_nrd, _kde2d
    from .plt_smooth import _RAMP
    from .XY import _loess, _plt_fit

    ok = np.isfinite(xv) & np.isfinite(yv)
    xo, yo = xv[ok], yv[ok]
    if len(xo) < 3 or _bw_nrd(xo) <= 0 or _bw_nrd(yo) <= 0:
        return
    lims = (float(xr[0]), float(xr[-1]), float(yr[0]), float(yr[-1]))
    n_grid = smooth_bins if form == "smooth" else contour_nbins
    gx, gy, z = _kde2d(xo, yo, n_grid, lims)
    if form == "smooth":
        fig.add_trace(go.Heatmap(
            x=gx, y=gy, z=(z ** smooth_power).T, colorscale=_RAMP,
            zsmooth="best", showscale=False, hoverinfo="skip"),
            row=r, col=c)
    else:
        lv0, lv1 = float(z.min()), float(z.max())
        step = (lv1 - lv0) / contour_n
        fig.add_trace(go.Contour(
            x=gx, y=gy, z=z.T, colorscale=_COLORSCALE,
            contours=dict(start=lv0 + step, end=lv1, size=step),
            line=dict(width=0), showscale=False, hoverinfo="skip"),
            row=r, col=c)
        if contour_points:
            fig.add_trace(go.Scatter(
                x=xo, y=yo, mode="markers",
                marker=dict(size=3, color="rgba(0,0,0,0.27)",
                            line=dict(color="white", width=0.2)),
                hoverinfo="skip", showlegend=False), row=r, col=c)
    if fit in ("lm", "loess"):
        if fit == "loess":
            xs, _, f, _ = _loess(xo, yo, 2 / 3)
        else:
            xs, _, f = _plt_fit(xo, yo, "lm", 1)
        fig.add_trace(go.Scatter(
            x=xs, y=f, mode="lines",
            line=dict(color=fit_color, width=fit_lwd),
            hoverinfo="skip", showlegend=False), row=r, col=c)
