# XY.py — analog of XY.R (pipeline; scatter / time-series core)
#
# XY(): the two-variable analytic view — the relationship between
# two NUMERICAL variables. Categorical variables belong to
# Chart(). As with Chart() and X(), the pipeline is ported, not
# the lines: XY.R's NSE, legacy parameters, base-R/lattice paths,
# and PDF device code have no Python counterpart.
#
# Ported in this increment: the scatterplot — points, by= groups,
# the least-squares fit family (lm/ls, null, exp, quad, power,
# log) and loess, both with SE bands, data ellipses — the date-x
# time-series line display, forecasting (plt_forecast.py:
# ts_source="fable" on statsmodels ETS/TSLM, and "classic"
# Holt-Winters and seasonal regression), facet= panels
# (numeric x), accompanying statistics (stats_out.py, quiet=
# to suppress), outlier/MD flagging (MD_cut/out_cut, with
# the dashed outlier-removed fit line), form="contour"
# (plt_contour.py: filled contours of the 2-D kernel density),
# and form="smooth" (plt_smooth.py: the smoothScatter() density
# raster with low-density points overplotted), and jitter_x/
# jitter_y (display only, auto for a discrete axis; the fit,
# ellipse, and statistics use the original data, as in R), and
# enhance= (the enhanced-scatterplot bundle: ellipse, MD
# outliers, lm fit, mean crosshair), facet= for a date x
# (_ts_facet: time-series panels; by= with a faceted time
# series is not yet ported), and the ts_ family: ts_unit/
# ts_agg aggregation (plt_time.py), ts_stack stacked series,
# and ts_area_fill/ts_area_split area fills.
#
# facet= orthogonality (July 2026, ~ the R facet unification):
# facet= takes a column name, a list of two names (a row x
# column grid, rows the second variable), or an aligned
# Series/array of computed values (the analog of R's facet
# expression); n_row/n_col lay out the panels. Faceted
# contour/smooth render through plt_contour_facet.py
# (~ .plt.contour.facet, shared KDE grid and bandwidth). A
# vector of x (or y) names overlays series on one panel
# (_series_overlay ~ the .plt.main overlay) and facet= panels
# the overlay (_facet_series ~ .plt.facet.series); by= with a
# vector errors, redirecting to facet=. The same numeric
# vector for both x and y draws a scatterplot matrix, via the
# shared scatter_matrix helper (plt_mat_plotly). The faceted
# scatter prints a summary table per grouping variable
# (facet_summary ~ .vbs_summary_table).
# A categorical x or y is refused with the call of the view it
# belongs to, as XY.R's master control funnel now does (Aug
# 2026); the dot plots XY() once drew for row_names and for a
# categorical variable with stat= are Chart(form="dot")'s. R's form="hexbin" is dropped by design:
# plotly has no hexbin trace.

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as sps

from .plt_add import plt_add
from .plt_contour import plt_contour
from .plt_contour_facet import plt_contour_facet
from .date_infer import date_infer
from .plt_forecast import plt_forecast
from .plt_plotly import plt_plotly
from .plt_smooth import plt_smooth
from .plt_time import plt_time
from .sunflower_plotly import sunflower_plotly
from .plt_mat_plotly import scatter_matrix
from . import x_console as xc
from .stats_out import _stat_by_levels
from .stats_out import (
    facet_summary, md_outliers, resolve_quiet, xy_stats)
from .plotly_utils import (
    BASE_COLORS, as_plotly_color, axis_format, axis_num,
    facet_fig, facet_panels, finish_facet,
    get_tick_fmt, legend_style, make_trans, plot_border,
    plotly_style, square_layout, sym_at, to_hex, x_grid,
    font_scaled, by_colors)
from .utils import (
    STAT_FUN, STAT_LBL, category_order, get_column, get_option,
    pretty, resolve_facet,
)

_FORMS = ("scatter", "smooth", "contour", "sunflower")
_FITS = ("off", "loess", "lm", "ls", "null", "exp", "quad",
         "power", "log")


def _plt_fit(xv, yv, fit, fit_power):
    """Fitted curve at the sorted x values, via least squares on
    the (possibly transformed) data. R analog: .plt.fit()"""
    od = np.argsort(xv, kind="stable")
    xs, ys = xv[od], yv[od]
    if fit == "null":
        return xs, ys, np.full(len(xs), ys.mean())

    p = float(fit_power)
    with np.errstate(invalid="ignore", divide="ignore"):
        if fit == "lm":
            z = ys
        elif fit == "quad":
            z, p = np.sqrt(ys), 2.0     # sqrt(y) ~ x, then square
        elif fit == "power":
            z = ys ** (1.0 / p)
        elif fit == "exp":
            if (ys <= 0).all():
                raise ValueError("All values of y are "
                                 "non-positive, cannot take log(y)")
            z = np.log(ys if p == 1 else ys ** (1.0 / p))
        elif fit == "log":
            z = np.exp(ys if p == 1 else ys ** (1.0 / p))
            if np.isinf(z).any():
                raise ValueError("Some values of y too large for "
                                 "exp(y). Rescale.")

    ok = np.isfinite(z)               # log/sqrt of negatives drop
    if ok.sum() < 2:
        raise ValueError(f'fit="{fit}": fewer than 2 usable '
                         "values after transformation")
    b1, b0 = np.polyfit(xs[ok], z[ok], 1)
    lin = b0 + b1 * xs
    with np.errstate(invalid="ignore"):
        if fit == "lm":
            f = lin
        elif fit in ("quad", "power"):
            f = lin ** p
        elif fit == "exp":
            f = np.exp(lin)
        else:                          # log: y = log(b0 + b1*x)
            f = np.log(lin)
    return xs, ys, f


_FIT_NEW_OK = ("lm", "quad", "power", "exp", "log")


def _fit_new_values(xv, yv, fit, fit_power, x_new):
    """Predict y at new x values from the same fitted model as
    _plt_fit (lm/quad/power/exp/log). R analog: .plt.fit y.new."""
    od = np.argsort(xv, kind="stable")
    xs, ys = xv[od], yv[od]
    p = float(fit_power)
    with np.errstate(invalid="ignore", divide="ignore"):
        if fit == "quad":
            z, p = np.sqrt(ys), 2.0
        elif fit == "lm":
            z = ys
        elif fit == "power":
            z = ys ** (1.0 / p)
        elif fit == "exp":
            z = np.log(ys if p == 1 else ys ** (1.0 / p))
        else:                          # log
            z = np.exp(ys if p == 1 else ys ** (1.0 / p))
    ok = np.isfinite(z)
    b1, b0 = np.polyfit(xs[ok], z[ok], 1)
    xn = np.asarray(x_new, dtype=float)
    lin = b0 + b1 * xn
    if fit == "lm":
        return lin
    if fit in ("quad", "power"):
        return lin ** p
    if fit == "exp":
        return np.exp(lin)
    return np.log(lin)                  # log


def _fit_new_table(groups, fit, fit_power, x_new, x_name, y_name,
                   digits_d):
    """Printed table of fitted y at the new x values (fit_new),
    ascending, one block per by group. R analog: out_y.new."""
    dd = 3 if digits_d is None else digits_d
    xn = np.sort(np.asarray(x_new, dtype=float))

    def xfmt(v):
        return str(int(v)) if v == int(v) else f"{v:.{dd}f}"

    lines = []
    for nm, xg, yg in groups:
        xg = np.asarray(xg, dtype=float)
        yg = np.asarray(yg, dtype=float)
        m = np.isfinite(xg) & np.isfinite(yg)
        if m.sum() < 2:
            continue
        preds = _fit_new_values(xg[m], yg[m], fit, fit_power, xn)
        xcol = [xfmt(v) for v in xn]
        ycol = [f"{v:.{dd}f}" for v in preds]
        wx = max(len(x_name), *(len(s) for s in xcol))
        yhdr = f"{y_name}_Fit"
        wy = max(len(yhdr), *(len(s) for s in ycol))
        lines.append("")
        if nm is not None:
            lines.append(f"{nm}")
        lines.append(f" {x_name:>{wx}} {yhdr:>{wy}}")
        for xs_, ys_ in zip(xcol, ycol):
            lines.append(f" {xs_:>{wx}} {ys_:>{wy}}")
    return lines


def _loess_rows(xs, q_pt, span, degree, rw):
    """The local fit at q_pt as linear operators on y: rows giving
    the fitted value and the slope there. Tri-cube weights on the
    floor(n*span) nearest points (times the robustness weights),
    polynomial of the given degree centered at q_pt.
    R analog: ehg127() of the loess Fortran"""
    n = len(xs)
    q = min(n, max(int(span * n + 1e-5), degree + 1))
    d = np.abs(xs - q_pt)
    h = np.partition(d, q - 1)[q - 1]
    if span > 1:                       # window widens past data,
        h *= span ** 0.5               # verified against R loess
    if h > 0:
        w = np.clip(1 - (d / h) ** 3, 0, 1) ** 3
    else:                              # window is exact x ties
        w = (d == 0).astype(float)
    w = w * rw
    nz = np.flatnonzero(w)
    xc = xs[nz] - q_pt
    X = np.column_stack([xc ** k for k in range(degree + 1)])
    XtW = X.T * w[nz]
    B = np.linalg.pinv(XtW @ X) @ XtW
    val = np.zeros(n)
    slope = np.zeros(n)
    val[nz] = B[0]
    slope[nz] = B[1]
    return val, slope


def _loess_kd(xs, span, cell=0.2):
    """Leaf cells of R's kd tree on one sorted x: the bounding box
    widened by 0.5% each side, a cell split at its median x (moved
    off ties) while it holds more than floor(n*span*cell) points.
    Returns (lower, upper, split) per leaf, the split for the walk
    down the tree. R analog: ehg126(), ehg124()"""
    n = len(xs)
    lo, hi = float(xs[0]), float(xs[-1])
    mu = 0.005 * max(hi - lo, 1e-10 * max(abs(lo), abs(hi)) + 1e-30)
    fc = int(np.floor(n * span * cell))
    nodes = []                         # (l, u, vlo, vhi), 1-based

    def build(l, u, vlo, vhi):
        leaf = (u - l) + 1 <= fc
        if not leaf:
            m = (l + u) // 2
            off = 0
            while l <= m + off < u:
                if xs[m + off - 1] == xs[m + off]:
                    off = -off
                    if off >= 0:
                        off += 1
                else:
                    m += off
                    break
            v = float(xs[m - 1])
            leaf = v == vlo or v == vhi
        if leaf:
            nodes.append((vlo, vhi))
            return
        build(l, m, vlo, v)
        build(m + 1, u, v, vhi)

    build(1, n, lo - mu, hi + mu)
    return nodes


def _loess(xv, yv, span, degree=2, family="gaussian"):
    """R's default loess at the sorted x values: local fits of the
    given degree at the vertices of a kd tree, each giving a value
    and a slope, joined by cubic Hermite interpolation within each
    cell (surface="interpolate"). family="symmetric" reweights by
    the bisquare of the residuals over 4 iterations. Also the SE of
    fit from the rows of the resulting smoother, as predict(l.ln,
    se=TRUE). span may be given as (span, degree, family).
    R analog: .plt.fit() loess(y.lv ~ x.lv, span, degree, family)"""
    if isinstance(span, tuple):
        span, degree, family = span
    od = np.argsort(xv, kind="stable")
    xs, ys = np.asarray(xv, float)[od], np.asarray(yv, float)[od]
    n = len(xs)
    cells = _loess_kd(xs, span)
    verts = sorted({v for c in cells for v in c})
    rw = np.ones(n)
    n_iter = 4 if family == "symmetric" else 1
    for it in range(n_iter):
        rows = {v: _loess_rows(xs, v, span, degree, rw) for v in verts}
        L = np.empty((n, n))
        for i, z in enumerate(xs):
            for v0, v1 in cells:       # x on a split goes left
                if z <= v1:
                    break
            hh = v1 - v0
            u = (z - v0) / hh
            g0, s0 = rows[v0]
            g1, s1 = rows[v1]
            L[i] = ((1 - u) ** 2 * (1 + 2 * u) * g0
                    + u * u * (3 - 2 * u) * g1
                    + (u * (1 - u) ** 2 * s0 - u * u * (1 - u) * s1) * hh)
        f = L @ ys
        res = ys - f
        if it < n_iter - 1:            # R lowesw(): bisquare weights
            cmad = 6 * np.median(np.abs(res))
            if cmad < np.finfo(float).tiny:
                rw = np.ones(n)
            else:
                r = np.abs(res)
                rw = np.where(r > 0.999 * cmad, 0.0,
                              np.where(r > 0.001 * cmad,
                                       (1 - (r / cmad) ** 2) ** 2, 1.0))
    l2 = (L ** 2).sum(axis=1)
    dof = n - 2 * np.trace(L) + l2.sum()   # tr[(I-L)'(I-L)]
    return xs, ys, f, np.sqrt(res @ res / dof * l2)


_QQ_LAB = {".normal": "Normal Quantiles",
           ".lognormal": "Lognormal Quantiles",
           ".exponential": "Exponential Quantiles",
           ".uniform": "Uniform Quantiles"}


def _qq_quantiles(y, key):
    """Theoretical quantiles, one per value of y in the row order of
    y (NaN where y is missing): the i-th smallest y pairs with the
    i-th smallest quantile of the distribution fit by y's moments.
    R analog: .qq.quantiles() (qq.R)"""
    y = np.asarray(y, dtype=float)
    ok = ~np.isnan(y)
    n = int(ok.sum())
    if n < 3:
        raise ValueError(
            "A q-q chart requires at least 3 non-missing values of y, "
            f"but only {n} found.")
    yv = y[ok]
    m, sd = yv.mean(), yv.std(ddof=1)
    if sd == 0:
        raise ValueError(
            "All values of y are the same, so there is no "
            "distribution\nto compare against a theoretical "
            "distribution.")
    if key == ".lognormal" and (yv <= 0).any():
        raise ValueError(
            "A lognormal distribution is defined only for positive "
            f"values,\nbut y has {(yv <= 0).sum()} value(s) of zero or "
            "less.\n\nFor data that include zero, consider .exponential")
    if key == ".exponential" and (yv < 0).any():
        raise ValueError(
            "An exponential distribution is defined only for "
            f"non-negative\nvalues, but y has {(yv < 0).sum()} "
            "negative value(s).")
    a = 3 / 8 if n <= 10 else 0.5      # R ppoints()
    p = (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)
    if key == ".normal":
        q = sps.norm.ppf(p, m, sd)
    elif key == ".lognormal":
        lg = np.log(yv)
        q = np.exp(sps.norm.ppf(p, lg.mean(), lg.std(ddof=1)))
    elif key == ".exponential":
        q = -m * np.log1p(-p)
    else:
        q = m - np.sqrt(3) * sd + p * 2 * np.sqrt(3) * sd
    out = np.full(len(y), np.nan)
    rk = np.empty(n, dtype=int)        # rank, ties.method="first"
    rk[np.argsort(yv, kind="stable")] = np.arange(n)
    out[ok] = q[rk]
    return out


def _qq_reference(fig, fit_color, fit_lwd):
    """The 45-degree reference of a q-q chart in every panel, from
    the range of that panel's plotted values, drawn beneath the
    points. R analog: plt.plotly.R qq reference"""
    panels = {}
    for tr in fig.data:
        if tr.type != "scatter" or tr.x is None or tr.y is None:
            continue
        key = (tr.xaxis or "x", tr.yaxis or "y")
        v = np.r_[np.asarray(tr.x, float), np.asarray(tr.y, float)]
        v = v[np.isfinite(v)]
        if len(v):
            lo, hi = panels.get(key, (np.inf, -np.inf))
            panels[key] = (min(lo, v.min()), max(hi, v.max()))
    refs = [go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                       xaxis=xa, yaxis=ya,
                       line=dict(color=to_hex(get_option(
                           "fit_color", "#5C4032")
                           if fit_color is None else fit_color),
                           width=1.5 if fit_lwd is None else fit_lwd),
                       showlegend=False, hoverinfo="skip")
            for (xa, ya), (lo, hi) in panels.items()]
    k = len(refs)
    fig.add_traces(refs)
    fig.data = fig.data[-k:] + fig.data[:-k] if k else fig.data
    return fig


def _ts_ylab(y_name, unit, aggregated, ts_agg):
    """Default value-axis label of a time series: the aggregation
    and the time unit, as "Total Sales by Year"; a vector of y
    (y_name None) is named by its legend, so "Total by Year", and
    with no aggregation to report either, no label. R analog: the
    y.lab of .plt.main() for a date x"""
    if unit in (None, "unknown"):
        return y_name or ""
    tu = "days" if unit == "days7" else unit
    tu = tu[:-1] if tu.endswith("s") else tu
    agg = (("Total" if ts_agg == "sum" else "Mean")
           if aggregated else "")
    if y_name is None and not agg:
        return ""
    return " ".join(p for p in (agg, y_name, "by", tu.capitalize())
                    if p)


def _se_band(xs, ys, f, level):
    """SE band polygon about a least-squares line: fit +/- t * SE
    of the conditional mean. R analog: plt.main.R se bands, from
    predict(se=TRUE) and qt(prb, n-1)."""
    n = len(xs)
    mse = ((ys - f) ** 2).sum() / (n - 2)
    xbar = xs.mean()
    sxx = ((xs - xbar) ** 2).sum()
    se = np.sqrt(mse * (1 / n + (xs - xbar) ** 2 / sxx))
    tq = sps.t.ppf((1 + level) / 2, n - 1)
    return (np.concatenate([xs, xs[::-1]]),
            np.concatenate([f + tq * se, (f - tq * se)[::-1]]))


def _ellipse_region(xv, yv, level, npoints=100):
    """Bivariate-normal data ellipse, the Murdoch and Chow
    construction of the ellipse package that XY() cites.
    R analog: ellipse::ellipse.default()"""
    r = float(np.corrcoef(xv, yv)[0, 1])
    d = np.arccos(np.clip(r, -1, 1))
    t = np.sqrt(sps.chi2.ppf(level, 2))
    theta = np.linspace(0, 2 * np.pi, npoints)
    ex = xv.mean() + t * xv.std(ddof=1) * np.cos(theta + d / 2)
    ey = yv.mean() + t * yv.std(ddof=1) * np.cos(theta - d / 2)
    return ex, ey


def _xy_facet(xv, yv, by_arr, by_order, facet_arr, facet_order,
              by_name, facet_name, fills, shape, pt_size,
              pt_opacity, fit, fit_power, se_levels, span,
              fit_color, fit_lwd, se_fill,
              ellipse, ellipse_fill, ellipse_color, ellipse_lwd,
              x_lab, y_lab, main, digits_d,
              facet2_arr=None, facet2_order=None,
              facet2_name=None, n_col=1):
    """One scatter panel per facet level on shared axes,
    following the faceted-histogram conventions (_hs_facet):
    first level on the top panel, strip labels, legend
    entries from the first panel only. facet2: the two-facet
    grid, rows = facet2 levels."""
    labels, pos, sel, n_row_g, n_col = facet_panels(
        facet_arr, facet_order, facet2_arr, facet2_order,
        facet_name, facet2_name, n_col)
    n_f = len(labels)
    groups = by_order if by_order is not None else [None]
    has_groups = by_order is not None
    px = float(pt_size) * (6.5 if has_groups else 7.25)
    if not np.isfinite(px) or px < 0:
        px = 5

    # per-panel artifacts, plus every plotted value for the
    # shared axis scaling
    panels = []
    xs_all, ys_all = [xv], [yv]
    for in_lvl in sel:
        pts, fit_lines, se_polys, ellipses = [], [], [], []
        for g, gname in enumerate(groups):
            m = in_lvl if gname is None \
                else in_lvl & (by_arr == gname)
            xg, yg = xv[m], yv[m]
            pts.append((gname, xg, yg))
            if fit != "off" and len(xg) >= 2:
                if fit == "loess":
                    xs, ys_s, f, se_f = _loess(xg, yg, span)
                    fit_lines.append({"name": gname, "x": xs,
                                      "y": f, "g": g})
                    for lv in se_levels:
                        tq = sps.t.ppf((1 + lv) / 2, len(xs) - 1)
                        se_polys.append((
                            np.concatenate([xs, xs[::-1]]),
                            np.concatenate(
                                [f + tq * se_f,
                                 (f - tq * se_f)[::-1]])))
                else:
                    xs, ys_s, f = _plt_fit(xg, yg, fit, fit_power)
                    okf = np.isfinite(f)
                    fit_lines.append({"name": gname, "x": xs[okf],
                                      "y": f[okf], "g": g})
                    if fit == "lm":
                        for lv in se_levels:
                            se_polys.append(
                                _se_band(xs, ys_s, f, lv))
            if ellipse and len(xg) >= 3:
                ellipses.append((g, *_ellipse_region(xg, yg,
                                                     ellipse)))
        panels.append((pts, fit_lines, se_polys, ellipses))
        for fl in fit_lines:
            xs_all.append(fl["x"]); ys_all.append(fl["y"])
        for bx, bnd in se_polys:
            ys_all.append(bnd)
        for _, ex, ey in ellipses:
            xs_all.append(ex); ys_all.append(ey)

    xs_all = np.concatenate(xs_all)
    ys_all = np.concatenate(ys_all)
    axT1 = pretty(float(np.nanmin(xs_all)), float(np.nanmax(xs_all)))
    axT2 = pretty(float(np.nanmin(ys_all)), float(np.nanmax(ys_all)))
    ax = {"axT1": axT1,
          "axL1": axis_format(axT1, digits_d),
          "axT2": axT2,
          "axL2": axis_format(axT2, digits_d)}

    style_opts = plotly_style()
    fig = facet_fig(n_row_g, n_col)
    for i in range(n_f):
        pts, fit_lines, se_polys, ellipses = panels[i]
        row, col = pos[i]
        for g, (gname, xg, yg) in enumerate(pts):
            fig.add_trace(go.Scatter(
                x=xg, y=yg, mode="markers",
                name=None if gname is None else str(gname),
                legendgroup=(None if gname is None
                             else str(gname)),
                marker=dict(
                    symbol=sym_at(shape, g), size=px,
                    sizemode="diameter",
                    color=make_trans(fills[g], pt_opacity),
                    opacity=1,
                    line=dict(color=to_hex(fills[g]), width=1)),
                hoverinfo="x+y" + ("+name" if has_groups else ""),
                showlegend=has_groups and i == 0,
            ), row=row, col=col)
        for g, ex, ey in ellipses:
            edge = (to_hex(fills[g]) if has_groups
                    else to_hex(ellipse_color))
            fig.add_trace(go.Scatter(
                x=ex, y=ey, mode="lines",
                line=dict(color=edge, width=ellipse_lwd),
                fill="toself",
                fillcolor=as_plotly_color(ellipse_fill),
                hoverinfo="skip", showlegend=False,
            ), row=row, col=col)
        for bx, bnd in se_polys:
            fig.add_trace(go.Scatter(
                x=bx, y=bnd, mode="none", fill="toself",
                fillcolor=as_plotly_color(se_fill),
                hoverinfo="skip", showlegend=False,
            ), row=row, col=col)
        for fl in fit_lines:
            if len(fl["x"]) < 2:
                continue
            single = fl["name"] is None
            fig.add_trace(go.Scatter(
                x=fl["x"], y=fl["y"], mode="lines",
                name="Fit" if single else f"Fit: {fl['name']}",
                legendgroup=("fit" if single
                             else str(fl["name"])),
                line=dict(color=(to_hex(fit_color) if single
                                 else to_hex(fills[fl["g"]])),
                          width=fit_lwd),
                hoverinfo="skip",
                showlegend=i == 0,
            ), row=row, col=col)

    finish_facet(fig, labels, ax, x_lab, y_lab,
                 gridT1=axT1, style_opts=style_opts,
                 n_col=n_col, pos=pos)
    # scatter panels: y spans the data, not [0, max] as counts do
    pad = 0.04 * (axT2[-1] - axT2[0])
    fig.update_yaxes(range=[axT2[0] - pad, axT2[-1] + pad])
    if has_groups:
        fig.update_layout(legend=legend_style(by_name,
                                              style_opts))
    if main:
        title_size = round(16 * get_option("main_size", 1))
        fig.update_layout(
            title=dict(text=main, x=0.5, xanchor="center",
                       font=dict(size=title_size)))
    return fig


def _series_pts(V, shared, m, g, on):
    """Panel points of series g under mask m, oriented per on=:
    "x" plots the series values on the x axis against the shared
    variable, "y" the reverse."""
    vg = V[m, g]
    sg = shared[m]
    return (vg, sg) if on == "x" else (sg, vg)


def _series_overlay(V, names, shared, on, fills, shape, pt_size,
                    pt_opacity, fit, fit_power, se_levels, span,
                    fit_lwd, se_fill,
                    ellipse, ellipse_fill, ellipse_lwd,
                    x_lab, y_lab, main, digits_d):
    """The multi-series overlay on one panel: the columns of V
    against the single shared variable, one color per series,
    optional per-series fit (in the series color), SE bands, and
    data ellipses. R analog: the .plt.main overlay of a vector
    of x (or y) variables."""
    k = V.shape[1]
    px = float(pt_size) * 6.5
    if not np.isfinite(px) or px < 0:
        px = 5
    all_m = np.ones(len(shared), dtype=bool)

    xs_all, ys_all = [], []
    art = []
    for g in range(k):
        xg, yg = _series_pts(V, shared, all_m, g, on)
        fit_lines, se_polys, ellipses = [], [], []
        if fit != "off" and len(xg) >= 2:
            if fit == "loess":
                xs, ys_s, f, se_f = _loess(xg, yg, span)
                fit_lines.append((xs, f))
                for lv in se_levels:
                    tq = sps.t.ppf((1 + lv) / 2, len(xs) - 1)
                    se_polys.append((
                        np.concatenate([xs, xs[::-1]]),
                        np.concatenate([f + tq * se_f,
                                        (f - tq * se_f)[::-1]])))
            else:
                xs, ys_s, f = _plt_fit(xg, yg, fit, fit_power)
                okf = np.isfinite(f)
                fit_lines.append((xs[okf], f[okf]))
                if fit == "lm":
                    for lv in se_levels:
                        se_polys.append(_se_band(xs, ys_s, f, lv))
        if ellipse and len(xg) >= 3:
            ellipses.append(_ellipse_region(xg, yg, ellipse))
        art.append((xg, yg, fit_lines, se_polys, ellipses))
        xs_all += [xg] + [fl[0] for fl in fit_lines] \
            + [ex for ex, _ in ellipses]
        ys_all += [yg] + [fl[1] for fl in fit_lines] \
            + [bnd for _, bnd in se_polys] \
            + [ey for _, ey in ellipses]

    xs_all = np.concatenate(xs_all)
    ys_all = np.concatenate(ys_all)
    axT1 = pretty(float(np.nanmin(xs_all)),
                  float(np.nanmax(xs_all)))
    axT2 = pretty(float(np.nanmin(ys_all)),
                  float(np.nanmax(ys_all)))

    style_opts = plotly_style()
    fig = go.Figure()
    for g, nm in enumerate(names):
        xg, yg, fit_lines, se_polys, ellipses = art[g]
        for ex, ey in ellipses:
            fig.add_trace(go.Scatter(
                x=ex, y=ey, mode="lines",
                line=dict(color=to_hex(fills[g]),
                          width=ellipse_lwd),
                fill="toself",
                fillcolor=as_plotly_color(ellipse_fill),
                hoverinfo="skip", showlegend=False))
        for bx, bnd in se_polys:
            fig.add_trace(go.Scatter(
                x=bx, y=bnd, mode="none", fill="toself",
                fillcolor=as_plotly_color(se_fill),
                hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=xg, y=yg, mode="markers", name=str(nm),
            legendgroup=str(nm),
            marker=dict(symbol=sym_at(shape, g), size=px,
                        sizemode="diameter",
                        color=make_trans(fills[g], pt_opacity),
                        opacity=1,
                        line=dict(color=to_hex(fills[g]),
                                  width=1)),
            hoverinfo="x+y+name", showlegend=True))
        for fx_l, fy_l in fit_lines:
            if len(fx_l) < 2:
                continue
            fig.add_trace(go.Scatter(
                x=fx_l, y=fy_l, mode="lines",
                legendgroup=str(nm),
                line=dict(color=to_hex(fills[g]),
                          width=fit_lwd),
                hoverinfo="skip", showlegend=False))

    ax_x = axis_num(x_lab, axT1, axis_format(axT1, digits_d))
    ax_y = axis_num(y_lab, axT2, axis_format(axT2, digits_d))
    pad = 0.04 * (axT2[-1] - axT2[0])
    ax_y["range"] = [axT2[0] - pad, axT2[-1] + pad]
    ax_y.update(showgrid=True,
                gridcolor=to_hex(style_opts["grid_col"]),
                gridwidth=1, griddash="dot")
    fig.update_layout(
        xaxis=ax_x, yaxis=ax_y,
        shapes=x_grid(axT1) + plot_border(),
        template=None,
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]),
        legend=legend_style("", style_opts),
        **square_layout(main=bool(main)))
    if main:
        title_size = round(16 * get_option("main_size", 1))
        fig.update_layout(
            title=dict(text=main, x=0.5, xanchor="center",
                       font=dict(size=title_size)))
    return fig


def _facet_series(V, names, shared, on, facet_arr, facet_order,
                  facet_name, facet2_arr, facet2_order,
                  facet2_name, n_col, fills, shape, pt_size,
                  pt_opacity, fit, fit_power, span, fit_lwd,
                  x_lab, y_lab, main, digits_d):
    """The multi-series overlay repeated across the facet
    panels: each panel the full overlay for one facet level (or
    cell), per-series fit in the series color, all panels on a
    common scale, series legend from the first panel. Default a
    single column of panels, as the generic facet path.
    R analog: .plt.facet.series()"""
    labels, pos, sel, n_row_g, n_col = facet_panels(
        facet_arr, facet_order, facet2_arr, facet2_order,
        facet_name, facet2_name, n_col)
    n_f = len(labels)
    k = V.shape[1]
    px = float(pt_size) * 6.5
    if not np.isfinite(px) or px < 0:
        px = 5

    panels = []
    xs_all, ys_all = [], []
    for i in range(n_f):
        pts, fit_lines = [], []
        for g in range(k):
            xg, yg = _series_pts(V, shared, sel[i], g, on)
            pts.append((xg, yg))
            xs_all.append(xg)
            ys_all.append(yg)
            ok = np.isfinite(xg) & np.isfinite(yg)
            if fit != "off" and ok.sum() > 2:
                if fit == "loess":
                    xs, _, f, _ = _loess(xg[ok], yg[ok], span)
                else:
                    xs, _, f = _plt_fit(xg[ok], yg[ok], fit,
                                        fit_power)
                okf = np.isfinite(f)
                fit_lines.append((g, xs[okf], f[okf]))
                xs_all.append(xs[okf])
                ys_all.append(f[okf])
        panels.append((pts, fit_lines))

    xs_all = np.concatenate(xs_all)
    ys_all = np.concatenate(ys_all)
    axT1 = pretty(float(np.nanmin(xs_all)),
                  float(np.nanmax(xs_all)))
    axT2 = pretty(float(np.nanmin(ys_all)),
                  float(np.nanmax(ys_all)))
    ax = {"axT1": axT1,
          "axL1": axis_format(axT1, digits_d),
          "axT2": axT2,
          "axL2": axis_format(axT2, digits_d)}

    style_opts = plotly_style()
    fig = facet_fig(n_row_g, n_col)
    for i in range(n_f):
        pts, fit_lines = panels[i]
        row, col = pos[i]
        for g, nm in enumerate(names):
            xg, yg = pts[g]
            fig.add_trace(go.Scatter(
                x=xg, y=yg, mode="markers", name=str(nm),
                legendgroup=str(nm),
                marker=dict(symbol=shape, size=px,
                            sizemode="diameter",
                            color=make_trans(fills[g],
                                             pt_opacity),
                            opacity=1,
                            line=dict(color=to_hex(fills[g]),
                                      width=1)),
                hoverinfo="x+y+name",
                showlegend=i == 0,
            ), row=row, col=col)
        for g, fx_l, fy_l in fit_lines:
            if len(fx_l) < 2:
                continue
            fig.add_trace(go.Scatter(
                x=fx_l, y=fy_l, mode="lines",
                legendgroup=str(names[g]),
                line=dict(color=to_hex(fills[g]),
                          width=fit_lwd),
                hoverinfo="skip", showlegend=False,
            ), row=row, col=col)

    finish_facet(fig, labels, ax, x_lab, y_lab,
                 gridT1=axT1, style_opts=style_opts,
                 n_col=n_col, pos=pos)
    pad = 0.04 * (axT2[-1] - axT2[0])
    fig.update_yaxes(range=[axT2[0] - pad, axT2[-1] + pad])
    fig.update_layout(legend=legend_style("", style_opts))
    if main:
        title_size = round(16 * get_option("main_size", 1))
        fig.update_layout(
            title=dict(text=main, x=0.5, xanchor="center",
                       font=dict(size=title_size)))
    return fig


def _xy_series(x, y, data, x_multi, by, facet, form, n_row,
               n_col, fill, transparency, shape, pt_size,
               fit, fit_power, fit_se, fit_lwd, span,
               ellipse, ellipse_fill, ellipse_lwd,
               MD_cut, out_cut,
               xlab, ylab, main, digits_d, quiet):
    """Resolve and dispatch a vector of x (or y) variables: the
    single-panel multi-series overlay, or with facet= the
    facet-series panels. R analog: the nn.col > 1 paths of
    XY.R."""
    names = list(x) if x_multi else list(y)
    shared_name = y if x_multi else x
    if not isinstance(shared_name, str):
        raise TypeError(
            "with a vector of variables, the other axis is a "
            "single column name")
    if form != "scatter":
        raise ValueError(
            'a vector of variables overlays series: '
            'form="scatter"')
    if by is not None:
        raise ValueError(
            "a multi-series overlay already colors by series. "
            "To display a second grouping variable, use facet=")
    if MD_cut > 0 or out_cut > 0:
        raise ValueError(
            "outlier flagging (MD_cut/out_cut) applies to a "
            "single scatterplot")

    arg = "x" if x_multi else "y"
    cols = [get_column(data, nm, arg) for nm in names]
    shared_ser = get_column(data, shared_name,
                            "y" if x_multi else "x")
    for nm, s in zip(names + [shared_name],
                     cols + [shared_ser]):
        if pd.api.types.is_datetime64_any_dtype(s):
            raise NotImplementedError(
                "multiple series on a date axis: use tidy "
                "(long) data with by=")
        if not pd.api.types.is_numeric_dtype(s):
            raise TypeError(
                f"the multi-series overlay is numeric, but "
                f"'{nm}' is {s.dtype}")

    f1_ser, f1_name, f2_ser, f2_name = resolve_facet(
        data, facet, "XY")
    used = cols + [shared_ser] \
        + [s for s in (f1_ser, f2_ser) if s is not None]
    keep = ~pd.concat(used, axis=1).isna().any(axis=1)
    V = np.column_stack([c[keep].to_numpy(dtype=float)
                         for c in cols])
    shared_v = shared_ser[keep].to_numpy(dtype=float)
    on = "x" if x_multi else "y"
    ser_lab = ", ".join(names)
    x_lab = ((ser_lab if x_multi else shared_name)
             if xlab is None else xlab)
    y_lab = ((shared_name if x_multi else ser_lab)
             if ylab is None else ylab)

    k = len(names)
    if isinstance(fill, (list, tuple)):
        fills = [fill[i % len(fill)] for i in range(k)]
    elif fill is not None:
        fills = [fill] * k
    else:
        fills = by_colors(k)           # theme's palette, as R
    if transparency is None:
        transparency = get_option("trans_pt_fill", 0.10)
    if ellipse is True:
        ellipse = 0.95
    if fit == "ls":
        fit = "lm"
    # SE bands off by default with several series, as by= groups
    se_lv = [lv for lv in (fit_se if isinstance(
        fit_se, (list, tuple)) else [fit_se]) if lv]
    dd = 2 if digits_d is None else digits_d
    fit_lwd_use = (get_option("fit_lwd", 2)
                   if fit_lwd is None else fit_lwd)

    if not resolve_quiet(quiet):       # per-series relationship
        all_m = np.ones(len(shared_v), dtype=bool)
        groups = [(nm, *_series_pts(V, shared_v, all_m, g, on))
                  for g, nm in enumerate(names)]
        print("\n".join(xy_stats(groups, x_lab, y_lab,
                                 digits_d=dd)))

    if f1_ser is None:
        return _series_overlay(
            V, names, shared_v, on, fills, shape, pt_size,
            1 - transparency, fit, fit_power, se_lv, span,
            fit_lwd_use, get_option("se_fill", "#1A1A1A19"),
            ellipse,
            (get_option("ellipse_fill", "#92806F28")
             if ellipse_fill is None else ellipse_fill),
            (get_option("ellipse_lwd", 1)
             if ellipse_lwd is None else ellipse_lwd),
            x_lab, y_lab, main, dd)

    f1a = f1_ser[keep].to_numpy()
    f1o = category_order(f1_ser[keep])
    f2a = f2_ser[keep].to_numpy() if f2_ser is not None else None
    f2o = (category_order(f2_ser[keep])
           if f2_ser is not None else None)
    n_col_use = (int(n_col) if n_col is not None
                 else (math.ceil(len(f1o) / int(n_row))
                       if n_row is not None else 1))
    return _facet_series(
        V, names, shared_v, on, f1a, f1o, f1_name,
        f2a, f2o, f2_name, n_col_use, fills, shape, pt_size,
        1 - transparency, fit, fit_power, span, fit_lwd_use,
        x_lab, y_lab, main, dd)


def _ts_facet(xv, yv, facet_arr, facet_order, fill0, border0,
              pt_size, pt_opacity, x_lab, y_lab, main, digits_d,
              area_fill=None, area_split=0,
              facet_name=None, facet2_arr=None,
              facet2_order=None, facet2_name=None, n_col=1,
              by_arr=None, by_order=None, by_name=None,
              line_width=1.5, frcsts=None, ts_PI=0.95):
    """One time-series panel per facet level on shared axes,
    following the faceted-scatter conventions (_xy_facet):
    first level on the top panel, strip labels. The x axis
    uses plotly's native date ticks, as the single-panel time
    series does. R analog: the lattice cont_cont path for a
    date x. facet2: the two-facet grid, rows = facet2 levels.
    by: several series per panel in the legend's colors, unfilled
    unless area_fill is asked for. frcsts: a forecast per panel
    (panel index -> plt_forecast result), every panel on one scale
    that holds the prediction intervals."""
    labels, pos, sel, n_row_g, n_col = facet_panels(
        facet_arr, facet_order, facet2_arr, facet2_order,
        facet_name, facet2_name, n_col)
    n_f = len(labels)
    px = float(pt_size) * 6.5          # panel-scaled points
    if not np.isfinite(px) or px < 0:
        px = 5
    mode_pts = "lines+markers" if px > 0 else "lines"

    y_all = [np.asarray(yv, float)]
    for fr in (frcsts or {}).values():
        y_all += [np.asarray(fr["y_lwr"], float),
                  np.asarray(fr["y_upr"], float)]
    y_all = np.concatenate(y_all)
    axT2 = pretty(float(np.nanmin(y_all)), float(np.nanmax(y_all)))
    fmt2 = get_tick_fmt(axT2, digits_d)
    ax = {"axT1": None, "axL1": None,
          "axT2": axT2,
          "axL2": axis_format(axT2, digits_d)}

    ypart = f"%{{y:{fmt2}}}" if fmt2 else "%{y}"
    hover = (f"Date: %{{x|%Y-%m-%d}}<br>{y_lab.strip()}: "
             f"{ypart}<extra></extra>")

    style_opts = plotly_style()
    fig = facet_fig(n_row_g, n_col)
    if area_fill is not None:          # fill toward area_split,
        rng_pad = 0.04 * (axT2[-1] - axT2[0])  # panel-clipped
        base = min(max(float(area_split), axT2[0] - rng_pad),
                   axT2[-1] + rng_pad)
    if by_arr is None:
        series = [(None, np.ones(len(xv), dtype=bool), fill0,
                   border0)]
    else:
        by_s = np.asarray(by_arr).astype(str)
        cols = by_colors(len(by_order))
        series = [(str(lv), by_s == str(lv), cols[g], cols[g])
                  for g, lv in enumerate(by_order)]
    for i in range(n_f):
        row, col = pos[i]
        for nm, bm, f_c, b_c in series:
            m = sel[i] & bm
            if not m.any():
                continue
            od = np.argsort(xv[m], kind="stable")
            xm, ym = xv[m][od], yv[m][od]
            if area_fill is not None:
                fig.add_trace(go.Scatter(
                    x=np.concatenate([xm, xm[-1:], xm[:1]]),
                    y=np.concatenate([ym, [base, base]]),
                    mode="none", fill="toself",
                    fillcolor=area_fill, hoverinfo="skip",
                    showlegend=False,
                ), row=row, col=col)
            hv = hover if nm is None else (
                f"{by_name}: {nm}<br>" + hover)
            fig.add_trace(go.Scatter(
                x=xm, y=ym, mode=mode_pts if line_width > 0
                else "markers",
                marker=(dict(symbol="circle", size=px,
                             sizemode="diameter",
                             color=make_trans(f_c, pt_opacity),
                             opacity=1,
                             line=dict(color=to_hex(b_c), width=1))
                        if "markers" in mode_pts or line_width <= 0
                        else None),
                line=dict(color=to_hex(b_c), width=line_width),
                name=nm, legendgroup=nm,
                hovertemplate=hv,
                showlegend=nm is not None and i == 0,
            ), row=row, col=col)       # first level top-left
        if frcsts and i in frcsts:
            _forecast_traces(fig, frcsts[i], ts_PI, row=row, col=col,
                             legend=i == min(frcsts))

    finish_facet(fig, labels, ax, x_lab, y_lab,
                 gridT1=None, style_opts=style_opts,
                 n_col=n_col, pos=pos)
    if by_arr is not None or frcsts:
        fig.update_layout(showlegend=True,
                          legend=legend_style(by_name, style_opts)
                          if by_arr is not None else None)
    # ts panels: y spans the data; native date grid on x
    pad = 0.04 * (axT2[-1] - axT2[0])
    fig.update_yaxes(range=[axT2[0] - pad, axT2[-1] + pad])
    fig.update_xaxes(showgrid=True,
                     gridcolor=to_hex(style_opts["grid_col"]),
                     gridwidth=1)
    if main:
        title_size = round(16 * get_option("main_size", 1))
        fig.update_layout(
            title=dict(text=main, x=0.5, xanchor="center",
                       font=dict(size=title_size)))
    return fig


def _area_color(col, opacity):
    """Area fill color: "on" is the violin_fill option; an
    explicit #RRGGBBAA keeps its own alpha; otherwise the point
    transparency applies, as R's .maketrans of area_fill."""
    if col == "on":
        col = get_option("violin_fill", "#7485975A")
    if (isinstance(col, str) and col.startswith("#")
            and len(col) == 9):
        return as_plotly_color(col)
    return make_trans(col, opacity)


def _apply_rotate(fig, rotate_x, rotate_y):
    """Tick-label rotation, all panels. R: rotate_x/rotate_y."""
    if rotate_x:
        fig.update_xaxes(tickangle=-float(rotate_x))
    if rotate_y:
        fig.update_yaxes(tickangle=-float(rotate_y))
    return fig


def _add_means(fig, xv, yv):
    """The add="means" crosshair set by enhance: a line at the
    mean of each variable, spanning the panel.
    R analog: plt.main.R annotations (~1403-1424)"""
    line = dict(color="#1A1A1A", width=1)  # add_color gray10
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=float(xv.mean()), x1=float(xv.mean()),
                  y0=0, y1=1, line=line)
    fig.add_shape(type="line", xref="paper", yref="y",
                  x0=0, x1=1, y0=float(yv.mean()),
                  y1=float(yv.mean()), line=line)


def _run_analysis(yv, digits_d, show_detail):
    """Runs test for a run chart: count the consecutive runs of y on
    each side of its median, optionally listing each run's members.
    Returns (median, text lines). R analog: the Run Analysis of
    .plt.txt."""
    import numpy as np
    y = np.asarray(yv, dtype=float)
    n = len(y)
    m = float(np.median(y))
    dd = 2 if digits_d is None else digits_d

    def sgn(v):
        v = float(v)
        return (v > 0) - (v < 0)

    run, members = [1], [[1]]
    for i in range(1, n):
        if y[i] != m and sgn(y[i] - m) != sgn(y[i - 1] - m):
            run.append(0)
            members.append([])
        run[-1] += 1
        members[-1].append(i + 1)
    n_runs = len(run)
    eq = [j + 1 for j in range(n) if y[j] == m]

    lines = ["", "-" * 12, "Run Analysis", "-" * 12]
    if show_detail:
        lines.append("")
        for k in range(n_runs):
            if run[k] > 1:
                idxs = " ".join(f"{j:>3}" for j in members[k])
                lines.append(
                    f"size={run[k]:>3}  Run {k + 1:>3} : {idxs}")
    lines.append(f"\nTotal number of runs: {n_runs}")
    lines.append("Total number of values that do not equal the "
                 f"median: {n - len(eq)}")
    if eq:                             # listed only with show_runs
        if show_detail:
            lines.append("\nValues ignored that equal the median")
            lines += [f"    #{j}  {y[j - 1]:.{dd}f}" for j in eq]
            lines.append(f"Total number of values ignored: {len(eq)}")
    else:
        lines.append("Total number of values ignored that equal the "
                     "median: 0")
    return m, lines


def _outlier_traces(fig, xv, yv, out_idx, labels, fill0,
                    pt_opacity, pt_size, out_shape, out_size,
                    ID_color, ID_size):
    """Overdraw the flagged outliers and label each with its ID.
    R analog: plt.plotly.R outlier points/labels (~140-185):
    open symbol, same fill and size, ID annotation below."""
    px = float(pt_size) * float(out_size) * 7.25
    if not np.isfinite(px) or px <= 0:
        px = 5
    fig.add_trace(go.Scatter(
        x=xv[out_idx], y=yv[out_idx], mode="markers",
        marker=dict(symbol=out_shape, size=px,
                    sizemode="diameter",
                    color=make_trans(fill0, pt_opacity),
                    opacity=1,
                    line=dict(color=to_hex(fill0), width=1.5)),
        hoverinfo="x+y", showlegend=False))
    font_sz = max(9, round(float(ID_size) * 14))
    for i in out_idx:
        fig.add_annotation(
            x=xv[i], y=yv[i], text=str(labels[i]),
            showarrow=False, yshift=-12,
            font=dict(color=to_hex(ID_color), size=font_sz,
                      family="Arial"))


def _forecast_traces(fig, frcst, ts_PI, row=None, col=None,
                     legend=True):
    """Overlay the forecast on the time series display: model-fit
    line, PI band with dotted boundaries, forecast line + points.
    R analog: plt.plotly.R forecast section (~lines 343-451),
    default-theme forecast hue rgb(.6, 0, 0)."""
    fit_rgba = "rgba(153,0,0,0.35)"    # muted: fit and PI lines
    fore_rgba = "rgba(153,0,0,0.90)"   # solid: forecast
    band_rgba = "rgba(153,0,0,0.15)"   # light: PI band fill
    xf = np.asarray(frcst["x_fit"])    # numpy datetimes: plain
    xh = np.asarray(frcst["x_hat"])    # Timestamps break kaleido
    yf, yh = frcst["y_fit"], frcst["y_hat"]
    at = {} if row is None else dict(row=row, col=col)
    _add = fig.add_trace

    def add(tr, **kw):                 # one legend entry per role
        if not legend:
            tr.showlegend = False
        tr.legendgroup = tr.name
        _add(tr, **at)

    fig = _Adder(fig, add)
    fig.add_trace(go.Scatter(
        x=xf, y=yf, mode="lines",
        line={"color": fit_rgba, "width": 1.5},
        name="Model fit", showlegend=True))
    fig.add_trace(go.Scatter(          # connector to forecast
        x=np.concatenate([xf[-1:], xh[:1]]),
        y=np.r_[yf[-1], yh[0]], mode="lines",
        line={"color": fit_rgba, "width": 1.5},
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(          # PI band polygon
        x=np.concatenate([xh, xh[::-1]]),
        y=np.concatenate([frcst["y_upr"], frcst["y_lwr"][::-1]]),
        mode="none", fill="toself", fillcolor=band_rgba,
        name=f"{round(ts_PI * 100)}% PI", showlegend=True,
        hoverinfo="skip"))
    for bound in (frcst["y_upr"], frcst["y_lwr"]):
        fig.add_trace(go.Scatter(
            x=xh, y=bound, mode="lines",
            line={"color": fit_rgba, "width": 1, "dash": "dot"},
            showlegend=False))
    fig.add_trace(go.Scatter(
        x=xh, y=yh, mode="lines+markers",
        marker={"symbol": "circle", "size": 6,
                "color": fore_rgba},
        line={"color": fore_rgba, "width": 2},
        name="Forecast", showlegend=True))


class _Adder:
    """fig.add_trace routed through a placement function."""
    def __init__(self, fig, add):
        self._fig, self.add_trace = fig, add


def _ts_fitted_lines(frcst, yv, x_name, y_name, ts_unit, digits_d):
    """Table of the observed y beside the model-fitted y over the
    historical dates, printed when ts_fitted=True. R analog: the
    out_fitted data frame of plt.forecast.R (~590). The classic
    ES fit starts after the first cycle, so x_fit can be a tail of
    the series; the observed column aligns to that tail."""
    dd = 2 if digits_d is None else digits_d
    xf = pd.DatetimeIndex(np.asarray(frcst["x_fit"]))
    yf = np.asarray(frcst["y_fit"], dtype=float)
    y_obs = np.asarray(yv, dtype=float)[-len(yf):]
    dfmt = {"months": "%b %Y", "years": "%Y"}.get(ts_unit)
    if ts_unit == "quarters":
        dates = [f"{d.year} Q{d.quarter}" for d in xf]
    elif dfmt:
        dates = list(xf.strftime(dfmt))
    else:
        dates = list(xf.strftime("%Y-%m-%d"))
    tbl = pd.DataFrame({x_name: dates, y_name: y_obs, "fitted": yf})
    return ["", "Fitted Values", "-" * 13,
            tbl.to_string(
                index=False,
                float_format=lambda v: f"{v:.{dd}f}")]


def XY(x, y=None, data=None, filter=None, by=None, facet=None,
       n_row=None, n_col=None,
       form="scatter", stat=None, sort="0", show_runs=False,
       center_line="off",
       contour_n=None, contour_nbins=None, contour_points=None,
       contour_legend=None,
       smooth_points=100, smooth_size=1, smooth_power=0.25,
       smooth_bins=128,
       enhance=False,
       segments_x=None, segments_y=None,
       add=None, x1=None, y1=None, x2=None, y2=None,
       fill=None, color=None, transparency=None,
       pt_size=1, pt_shape="circle", line_width=1.5,
       jitter_x=None, jitter_y=None,
       MD_cut=0, out_cut=0, out_shape="circle-open", out_size=1,
       ID=None, ID_color="gray50", ID_size=0.6,
       fit="off", fit_power=1, fit_se=None, fit_errors=False,
       fit_new=None,
       fit_color=None, fit_lwd=None, fit_span_loess=0.75,
       fit_degree_loess=2, fit_family_loess="gaussian",
       ellipse=0, ellipse_fill=None, ellipse_color=None,
       ellipse_lwd=None,
       ts_unit=None, ts_agg="sum", ts_stack=False,
       ts_area_fill=None, ts_area_split=0,
       ts_ahead=0, ts_method="es",
       ts_source="fable", ts_error=None,
       ts_trend=None, ts_seasons=None,
       ts_alpha=None, ts_beta=None, ts_gamma=None, ts_PI=0.95,
       ts_NA=None, ts_format=None, ts_fitted=False,
       ts_n_x_tics=None,
       axis_fmt="K", axis_x_pre="", axis_y_pre="",
       rotate_x=0, rotate_y=0, scale_x=None, scale_y=None,
       xlab=None, ylab=None, main=None, digits_d=None,
       quiet=None,
       plot_errors=None, span=None):
    """Analytic view of the relationship between two numerical
    variables, optionally grouped (by=). A date x displays as a
    time series. Variables are strings naming columns of the
    DataFrame `data`. Returns a plotly Figure.
    """

    # ----- validate parameters ------------------------------------
    if y is None:
        raise ValueError(
            "Argument y is required. For the distribution of a "
            "single variable, use X().")
    if data is None:
        raise ValueError(
            "data= is required: a pandas DataFrame containing the "
            "named columns")
    if form not in _FORMS:
        raise ValueError(f"form must be one of {_FORMS}")
    # setting a contour_ parameter selects the contour form, as
    # in R (XY.R ~174)
    if any(p is not None for p in
           (contour_n, contour_nbins, contour_points,
            contour_legend)):
        form = "contour"
    contour_n_user = contour_n         # a matrix cell defaults to 8
    contour_n = 20 if contour_n is None else contour_n
    contour_nbins = 50 if contour_nbins is None else contour_nbins
    contour_points = bool(contour_points)
    contour_legend = bool(contour_legend)
    if pt_shape == "sunflower":        # its earlier route, kept
        form, pt_shape = "sunflower", "circle"
    # the form is consulted before by is resolved, so a vector of by
    # gets the sunflower's own reason (XY.R)
    if form == "sunflower" and by is not None:
        raise ValueError(
            'Parameter by is not available for form="sunflower".\n\n'
            "A petal count is defined for one set of coordinates, not\n"
            "separately within each group, so overlaid groups would\n"
            "interleave their petals at a shared coordinate.\n"
            "To stratify the same two variables, use facet, which "
            "draws\neach group in its own panel with its own "
            "coordinates.")
    if form == "sunflower" and (isinstance(x, (list, tuple))
                                or isinstance(y, (list, tuple))):
        raise ValueError(
            "The petal count is defined for one set of coordinates, "
            "not\nseparately for each series, so a vector of x or y "
            'variables\nis not available for form="sunflower". To '
            'overlay the series,\nuse form="scatter".')
    if isinstance(by, (list, tuple)):
        raise ValueError(
            "Only one by variable is permitted, but more than one "
            f"specified:\n  by = {list(by)}\n\n"
            "The groups of by are overlaid within the panel, "
            "distinguished by\ncolor and optionally symbol. That one "
            "channel is carried by the\nfirst variable, so a second "
            "has no encoding left by which to\nseparate its levels.\n"
            "To stratify by a second variable, use facet, which draws "
            "each of\nits groups in a panel of its own.")
    if form != "scatter" and by is not None:
        raise ValueError(
            f'Parameter by is not available for form="{form}".\n\n'
            f"A {form} plot summarizes the joint density over the "
            "whole\nplotting region, so summaries superimposed on it "
            "would obstruct\none another rather than remain "
            "distinguishable.\nTo stratify the same two variables, "
            "use facet, which draws\neach group in its own panel with "
            "its own density.")
    if add is not None and form in ("contour", "smooth"):
        raise ValueError(
            'add is not active for "contour" or "smooth" plots')
    if add is not None and facet is not None:
        raise ValueError(
            "add= annotations apply to a single panel: "
            "no facet=")
    if axis_fmt not in ("K", ",", ".", ""):
        raise ValueError('axis_fmt: "K", ",", ".", or ""')
    if center_line not in ("off", "mean", "median", "zero"):
        raise ValueError(
            'center_line: "off", "mean", "median", or "zero"')
    if span is not None:
        raise ValueError(
            "Parameter span has been renamed to fit_span_loess.\n\n"
            "It sets the span of a loess fit, so the name now says "
            "which\nfit it belongs to, alongside fit_degree_loess "
            "and fit_family_loess.")
    if fit_degree_loess not in (1, 2):
        raise ValueError("fit_degree_loess: 1 or 2")
    if fit_family_loess not in ("gaussian", "symmetric"):
        raise ValueError('fit_family_loess: "gaussian" or "symmetric"')
    # the loess settings travel together to every fit (R: loess(...,
    # span, degree, family) in .plt.fit())
    span = (float(fit_span_loess), int(fit_degree_loess),
            fit_family_loess)
    if plot_errors is not None:
        raise ValueError(
            "Parameter plot_errors has been renamed to fit_errors. "
            "The segments it draws join each point to the fitted line, "
            "so they are the errors of the fit, and the name now says "
            "which fit they belong to, as fit, fit_color, and fit_se do.")
    if center_line != "off" and facet is not None:
        raise ValueError(
            "center_line draws on a single panel: no facet=")
    if fit_errors:
        if fit == "off":
            raise ValueError(
                "fit_errors draws residual segments to the fit "
                "line: specify fit=")
        if form != "scatter":
            raise ValueError('fit_errors applies to form="scatter"')
        if facet is not None:
            raise ValueError(
                "fit_errors draws on a single panel: no facet=")
    if (scale_x is not None or scale_y is not None) \
            and facet is not None:
        raise ValueError(
            "scale_x and scale_y apply to a single panel: "
            "no facet=")
    if fit not in _FITS:
        raise ValueError(f"fit must be one of {_FITS}")
    if fit == "ls":                    # least squares synonym
        fit = "lm"
    if MD_cut > 0 or out_cut > 0:
        if by is not None or facet is not None:
            raise ValueError(
                "outlier flagging (MD_cut/out_cut) applies to a "
                "single scatterplot: no by= or facet=")
        if form != "scatter":
            raise ValueError(
                "outlier flagging (MD_cut/out_cut) applies to "
                'the scatter form, form="scatter"')
    if (n_row is not None or n_col is not None) \
            and facet is None:
        raise ValueError("n_row and n_col lay out facet panels: "
                         "specify facet=")
    if ts_agg not in ("sum", "mean"):
        raise ValueError('ts_agg: "sum" or "mean"')
    if ts_stack and by is None:
        raise ValueError(
            "ts_stack stacks multiple time series: specify by=")
    if (by is not None and ts_area_fill is not None
            and not ts_stack):
        raise ValueError(
            "Filling the areas under multiple curves is only "
            "meaningful if the curves are stacked, so set "
            "ts_stack=True")
    if ts_ahead > 0:
        if ts_source not in ("fable", "classic"):
            raise ValueError('ts_source: "fable" or "classic"')
        if by is not None:             # XY.R: the overlay channel
            raise ValueError(            # is not available
                "A forecast draws its fit, prediction interval, and "
                "forecast\nin the series' own colors, so the overlay "
                "of by= groups is not\navailable to it. To forecast "
                "each group, use facet=, which\nforecasts each panel "
                "on one scale.")

    # for the report: the caller's own names, the parameters named in
    # the call (which the suggestions skip), the digits as given
    data_in = data
    digits_user = digits_d
    x_call = x                         # as written, keyword included
    given = {nm for nm, v in (
        ("enhance", enhance or None), ("fill", fill), ("color", color),
        ("fit", None if fit == "off" else fit),
        ("out_cut", out_cut or None), ("MD_cut", MD_cut or None),
        ("pt_shape", None if pt_shape == "circle" else pt_shape),
        ("pt_size", None if pt_size == 1 else pt_size),
        ("line_width", None if line_width == 1.5 else line_width),
        ("ts_ahead", ts_ahead or None), ("ts_unit", ts_unit),
        ("ts_agg", None if ts_agg == "sum" else ts_agg),
        ("ts_seasons", ts_seasons), ("ts_area_fill", ts_area_fill),
        ("ts_stack", ts_stack or None)) if v is not None}
    show = not resolve_quiet(quiet)

    def say(*comps):
        if show:
            print(xc.report(*comps))

    if filter is not None:
        data = data.query(filter)

    # ----- a vector of x (or y) variables: the multi-series
    # overlay; facet= panels it (the facet-series display) --------
    x_multi = isinstance(x, (list, tuple))
    y_multi = isinstance(y, (list, tuple))
    if x_multi and len(x) == 1:
        x, x_multi = x[0], False
    if y_multi and len(y) == 1:
        y, y_multi = y[0], False
    if x_multi and y_multi:
        # the same numeric vector for both x and y is a
        # scatterplot matrix (R: same.xy.expr); by= and facet=
        # are unsupported by design, as in R
        if list(x) != list(y):
            raise ValueError(
                "vectors for both x and y form a scatterplot "
                "matrix: specify the same variables for x and y")
        if by is not None or facet is not None:
            raise ValueError(
                "a scatterplot matrix does not support by= or "
                "facet=: its small points make group colors hard "
                "to distinguish, omitted by design")
        mat_df = data[list(x)].apply(
            pd.to_numeric, errors="coerce")
        return scatter_matrix(
            mat_df, fit=("off" if fit in ("off", "null")
                         else fit),
            digits_d=digits_d, main=main, form=form,
            contour_n=contour_n_user,
            contour_nbins=contour_nbins,
            contour_points=contour_points, smooth_bins=smooth_bins,
            smooth_power=smooth_power)
    # a vector of y on a date x: one series per variable, overlaid
    # and keyed by name in the legend, drawn as a by= series would
    # be; the value axis reports only the aggregation and time unit
    # (XY.R, Aug 2026)
    if y_multi and not x_multi and isinstance(x, str) \
            and x in data.columns:
        xd = data[x]
        if not pd.api.types.is_datetime64_any_dtype(xd) \
                and not pd.api.types.is_numeric_dtype(xd):
            from .date_infer import date_infer as _infer_dates
            try:
                xd = _infer_dates(xd)
            except Exception:
                pass
        if pd.api.types.is_datetime64_any_dtype(xd):
            if by is not None:
                raise ValueError(
                    "A vector of y variables overlays its variables as "
                    "the series of the\nplot, which spends the color "
                    "dimension that by requires.\nTo display a "
                    "grouping variable, use facet=, which panels the "
                    "series.")
            fcols = ([facet] if isinstance(facet, str) else
                     list(facet) if isinstance(facet, (list, tuple))
                     else [])
            long = data[[x] + fcols + list(y)].copy()
            long[x] = xd
            long = long.melt(id_vars=[x] + fcols, value_vars=list(y),
                             var_name="Series", value_name="Value")
            long["Series"] = pd.Categorical(long["Series"],
                                            categories=list(y))
            if ylab is None:
                from .plt_time import _infer_unit
                ylab = _ts_ylab(None, ts_unit if ts_unit is not None
                                else _infer_unit(long[x].to_numpy()),
                                ts_unit is not None, ts_agg)
            fig_l = XY(x, "Value", data=long, by="Series", facet=facet,
                       ts_unit=ts_unit, ts_agg=ts_agg,
                       ts_stack=ts_stack, ts_area_fill=ts_area_fill,
                       ts_area_split=ts_area_split, pt_size=pt_size,
                       line_width=line_width, n_row=n_row, n_col=n_col,
                       fill=fill, xlab=xlab, ylab=ylab, main=main,
                       digits_d=digits_d, quiet=True,
                       rotate_x=rotate_x, rotate_y=rotate_y)
            fig_l.update_layout(legend_title_text="")
            pre_l, dn_l = xc.call_names(data_in)
            say(xc.suggest_series(
                pre_l, x, list(y), dn_l, given,
                "".join(f', {k}="{v}"' if isinstance(v, str)
                        else f", {k}={v}"
                        for k, v in (("ts_unit", ts_unit),)
                        if v is not None), pt_size=pt_size))
            return fig_l

    if x_multi or y_multi:
        # a vector against a categorical variable is Chart()'s dot
        # chart, named rather than drawn (XY.R)
        one = y if x_multi else x
        if (isinstance(one, str) and one in data.columns
                and not pd.api.types.is_numeric_dtype(data[one])
                and not pd.api.types.is_datetime64_any_dtype(
                    data[one])):
            vec = x if x_multi else y
            vec_s = "[" + ", ".join(f"'{v}'" for v in vec) + "]"
            raise ValueError(
                "XY() requires a continuous x and a continuous y.\n"
                f"{one} is categorical.\n\nFor several values at each "
                f"level of {one}, use Chart()\nwith a vector of y "
                f"variables:\n  Chart('{one}', y={vec_s}, form='dot')")
        fig_s = _apply_rotate(_xy_series(
            x, y, data, x_multi, by, facet, form, n_row, n_col,
            fill, transparency, pt_shape, pt_size,
            fit, fit_power, fit_se, fit_lwd, span,
            ellipse, ellipse_fill, ellipse_lwd,
            MD_cut, out_cut,
            xlab, ylab, main, digits_d, True),
            rotate_x, rotate_y)
        # the report: one correlation per plotted variable, each
        # headed by its name (R overwrote all but the last)
        pre, dname = xc.call_names(data_in)
        names = list(x) if x_multi else list(y)
        other = data[y if x_multi else x]
        comps = [xc.suggest_scatter(pre, x, y, dname, given)]
        if (fit in ("off", "lm")
                and pd.api.types.is_numeric_dtype(other)):
            for nm in names:
                xv_ = pd.to_numeric(data[nm], errors="coerce")
                pair = (xv_, other) if x_multi else (other, xv_)
                head = (f"Variable: {nm} with {y}" if x_multi
                        else f"Variable: {x} with {nm}")
                comps.append(["", head] + xc.cor_block(
                    pair[0].to_numpy(float), pair[1].to_numpy(float),
                    nm if x_multi else x, y if x_multi else nm))
        say(*comps)
        return fig_s

    # ----- row_names: each row label identifies a single case, a
    # categorical axis that belongs to Chart(form="dot")
    # R analog: XY.R categorical refusal, x.unique branch
    _ROW_KW = ("row_names", "row.names")
    x_row = isinstance(x, str) and x in _ROW_KW
    y_row = isinstance(y, str) and y in _ROW_KW
    if x_row or y_row:
        cont = y if x_row else x
        raise TypeError(
            "XY() requires a continuous x and a continuous y.\n"
            "row_names is categorical.\n\n"
            "Each level of row_names identifies a single case, so\n"
            f"  display the values of {cont} directly with Chart():\n"
            f"  Chart('row_names', y='{cont}', form='dot')")

    # sort and the segments_x/_y droplines served the dot plot XY()
    # once drew of a categorical variable, now Chart()'s
    if stat is not None:
        if stat not in STAT_FUN:
            raise ValueError("stat must be one of "
                             + ", ".join(f'"{k}"' for k in STAT_FUN))
        if (by is not None or facet is not None
                or isinstance(x, (list, tuple))
                or isinstance(y, (list, tuple))):
            raise ValueError(
                "stat plots one statistic of y at each value of x: "
                "no by=, facet=, or vector of variables")
    for nm, v in (("sort", None if sort == "0" else sort),
                  ("segments_x", segments_x),
                  ("segments_y", segments_y)):
        if v is not None:
            raise ValueError(
                f"{nm} applies to the dot plot of a categorical "
                "variable, which is Chart(form='dot'), not XY()")

    # ----- resolve variables and filter ---------------------------
    # ".Index" is the row-number pseudo-variable (1..n) for a run
    # chart: synthesize it rather than looking up a column
    def _resolve(name, arg):
        if isinstance(name, str) and name == ".Index":
            return pd.Series(range(1, len(data) + 1),
                             index=data.index, name="Index")
        return get_column(data, name, arg)

    # a keyword in the x role names no variable, so it is matched
    # without regard to case: .index for the run chart, .normal,
    # .lognormal, .exponential, .uniform for a q-q chart (XY.R)
    qq_key = None
    if (isinstance(x, str) and x.startswith(".")
            and x not in data.columns):
        key = x.lower()
        if key == ".index":
            x = ".Index"
        elif key in _QQ_LAB:
            qq_key = key
        else:
            raise ValueError(
                "A name in the x role that begins with a period is a "
                f"keyword\nfor a generated variable, but {x} is not one "
                "of\nthe keywords, and is not a variable in the data.\n\n"
                "Available keywords, matched without regard to case:\n"
                "  .index  generates the consecutive integers of a run "
                "chart\n  " + ", ".join(_QQ_LAB) + "  each generates\n"
                "    the theoretical quantiles of the corresponding "
                "q-q chart")
    if qq_key is not None and (isinstance(y, (list, tuple))
                               or form != "scatter"):
        raise ValueError(
            "A q-q chart plots one y variable as a scatterplot "
            'against its theoretical quantiles: one y, form="scatter"')

    index_x = isinstance(x, str) and x == ".Index"
    index_y = isinstance(y, str) and y == ".Index"
    y_ser = _resolve(y, "y")
    x_ser = (pd.Series(np.nan, index=data.index, name=_QQ_LAB[qq_key])
             if qq_key is not None else _resolve(x, "x"))
    if qq_key is not None:
        x = _QQ_LAB[qq_key]
    if index_x:                 # for axis labels below
        x = "Index"
    if index_y:
        y = "Index"

    by_ser = get_column(data, by, "by") if by is not None else None
    (facet_ser, facet_name,
     facet2_ser, facet2_name) = resolve_facet(data, facet, "XY")

    # q-q: the quantiles of y, regenerated within each group of by
    # and each panel of facet, so every group reads against its own
    # distribution; both axes then share one range (XY.R)
    if qq_key is not None:
        if not pd.api.types.is_numeric_dtype(y_ser):
            raise ValueError(
                "A q-q chart displays the distribution of a continuous "
                "variable,\nbut the y variable is not numeric.")
        grp = None
        for g in (by_ser, facet_ser, facet2_ser):
            if g is not None:
                gs = g.astype(str).where(g.notna())
                grp = gs if grp is None else grp + "\x1f" + gs
        yv = y_ser.to_numpy(dtype=float)
        if grp is None:
            qv = _qq_quantiles(yv, qq_key)
        else:
            qv = np.full(len(yv), np.nan)
            for lv in grp.dropna().unique():
                i = (grp == lv).to_numpy()
                if (~np.isnan(yv[i])).sum() >= 3:
                    qv[i] = _qq_quantiles(yv[i], qq_key)
        x_ser = pd.Series(qv, index=data.index, name=x)
        if scale_x is None and scale_y is None:
            pr = pretty(float(np.nanmin(np.r_[qv, yv])),
                        float(np.nanmax(np.r_[qv, yv])))
            scale_x = scale_y = (pr[0], pr[-1], len(pr) - 1)

    # the report (x_console): ports of R's printers, built as the
    # display is; xy_stats() records the numbers for fig.stats
    pre, dname = xc.call_names(data_in)
    _lbls = ((getattr(data_in, "attrs", {}) or {})
             .get("variable_labels", {}) or {})
    x_lbl, y_lbl = _lbls.get(x), _lbls.get(y)
    dd_r = (xc.xy_digits(pd.to_numeric(y_ser, errors="coerce")
                         .to_numpy(float), digits_user)
            if not isinstance(y, (list, tuple)) else 3)

    def scatter_report(groups_, fit_stats_, md_lines_):
        xy_stats(groups_, x, y, fit_stats_, digits_d, by_name=by)
        if stat_lines is not None:     # R reports the levels alone
            return [stat_lines]
        comps = [xc.suggest_scatter(pre, x_call, y, dname, given,
                                    by=by)]
        if md_lines_ is not None:
            comps.append(md_lines_)
        # correlation for no fit or a linear one, not with by, not
        # for a q-q chart (.plt.txt)
        if fit in ("off", "lm") and by is None and qq_key is None:
            for _, xg, yg in groups_:
                comps.append(xc.cor_block(xg, yg, x, y, x_lbl, y_lbl))
        # the fit that enhance= supplies is drawn, not reported (R)
        if (fit not in ("off", "null") and fit_stats_
                and not (enhance and "fit" not in given)):
            comps.append(xc.fit_text(fit_stats_, fit, fit_power, y,
                                     dd_r, by_name=by))
        return comps

    def facet_report(xv_s):
        """XY.R's summary of x per grouping variable; x's own
        decimals, at least 2, as X() now gives them"""
        sd = max(xc.max_dd(xv_s), 2)
        summ = ["", f"---------- Summary Statistics for {x}"]
        for g, nm in ((by_ser, by), (facet_ser, facet_name),
                      (facet2_ser, facet2_name)):
            if g is not None:
                summ += [""] + xc.vbs_summary(
                    xv_s, g.to_numpy(), nm, sd,
                    order=category_order(g))
        return summ

    # stat: the statistic of y at each value of a continuous x, the
    # values first reported by level of x (XY.R, .ss.numeric(y, by=x))
    stat_lines = None
    if stat is not None:
        for a_, b_ in ((x, y), (y, x)):
            if not pd.api.types.is_numeric_dtype(data[a_]):
                raise ValueError(
                    "stat in XY() plots a statistic of y at each value "
                    "of a continuous x. For a statistic of a numerical "
                    "variable across the levels of a categorical one, "
                    f"use Chart():\n  Chart('{a_}', y='{b_}', "
                    f"stat='{stat}')")
        okx = x_ser.notna()
        stat_lines = _stat_by_levels(
            pd.to_numeric(y_ser[okx]), x_ser[okx], y, x,
            max(xc.max_dd(y_ser.to_numpy(float)) + 1, 2))
        agg = (pd.to_numeric(y_ser[okx]).groupby(x_ser[okx])
               .agg(STAT_FUN[stat]).dropna())
        data = pd.DataFrame({x: agg.index.to_numpy(float),
                             y: agg.to_numpy(float)})
        x_ser, y_ser = data[x], data[y]
        if ylab is None:
            ylab = f"{STAT_LBL[stat]} of {y}"

    # facet needs a categorical variable; a numeric one with more
    # than 25 distinct values is almost surely continuous, so stop
    # and point to a category conversion (XY.R ~1168). A category
    # dtype bypasses this, as an R factor does.
    for fs, fn in ((facet_ser, facet_name),
                   (facet2_ser, facet2_name)):
        if (fs is not None
                and pd.api.types.is_numeric_dtype(fs)
                and fs.nunique() > 25):
            raise ValueError(
                f"Parameter facet requires a categorical variable, "
                f"but '{fn}' is numeric with {fs.nunique()} unique "
                f"values.\n\nIf '{fn}' is categorical, convert it to "
                f"a category:\n  "
                f"data['{fn}'] = data['{fn}'].astype('category')")

    # a string date column (e.g. "2024-01-15", "2024 Q3") becomes
    # datetimes, so it drives the time-series display, as XY.R runs
    # date.infer on x. Non-date strings pass through unchanged.
    # ts_format= parses x with an explicit strftime pattern (XY.R
    # ~716), for a format date_infer would not recognize.
    if (ts_format is not None
            and not pd.api.types.is_datetime64_any_dtype(x_ser)):
        parsed = pd.to_datetime(x_ser, format=ts_format,
                                errors="raise")
        parsed.index, parsed.name = x_ser.index, x_ser.name
        x_ser = parsed
    elif (not pd.api.types.is_numeric_dtype(x_ser)
            and not pd.api.types.is_datetime64_any_dtype(x_ser)):
        try:
            parsed = date_infer(x_ser)
            if pd.api.types.is_datetime64_any_dtype(parsed):
                parsed.index = x_ser.index
                parsed.name = x_ser.name
                x_ser = parsed
        except (ValueError, TypeError):    # not a date column
            pass

    is_date = pd.api.types.is_datetime64_any_dtype(x_ser)
    # XY() displays the relationship of two continuous variables; a
    # categorical x or y belongs to another view, so name its call
    # rather than render. R analog: XY.R master control funnel
    x_cat = not is_date and not pd.api.types.is_numeric_dtype(x_ser)
    y_cat = not pd.api.types.is_numeric_dtype(y_ser)
    if x_cat or y_cat:
        head = "XY() requires a continuous x and a continuous y.\n"
        if x_cat and y_cat:
            raise TypeError(
                head + f"Both {x} and {y} are categorical.\n\n"
                "For two categorical variables, use Chart() with a "
                f"by variable:\n  Chart('{x}', by='{y}', "
                "form='bubble')")
        cont, cat_, cat_s = ((y, x, x_ser) if x_cat
                             else (x, y, y_ser))
        if not cat_s.dropna().duplicated().any():
            raise TypeError(
                head + f"{cat_} is categorical.\n\n"
                f"Each level of {cat_} identifies a single case, so\n"
                f"  display the values of {cont} directly with "
                f"Chart():\n  Chart('{cat_}', y='{cont}', "
                "form='dot')")
        f_txt = ""
        if facet is not None:
            f_txt = (", facet=[" + ", ".join(f"'{f}'" for f in facet)
                     + "]" if isinstance(facet, (list, tuple))
                     else f", facet='{facet}'"
                     if isinstance(facet, str)
                     else f", facet='{facet_name}'")
        raise TypeError(
            head + f"{cat_} is categorical.\n\n"
            f"For the distribution of the continuous {cont} across\n"
            f"  the levels of the categorical {cat_}, use X() with "
            f"a by variable:\n  X('{cont}', by='{cat_}'{f_txt})")

    # ts_NA= replaces missing y with a set value before the NA
    # rows are dropped, so a gap in the series becomes that value
    # (e.g. 0) rather than a hole (XY.R ~908)
    if ts_NA is not None:
        y_ser = y_ser.fillna(ts_NA)

    used = [s for s in (x_ser, y_ser, by_ser, facet_ser,
                        facet2_ser)
            if s is not None]
    keep = ~pd.concat(used, axis=1).isna().any(axis=1)
    x_ser, y_ser = x_ser[keep], y_ser[keep]
    if by_ser is not None:
        by_ser = by_ser[keep]
    if facet_ser is not None:
        facet_ser = facet_ser[keep]
    if facet2_ser is not None:
        facet2_ser = facet2_ser[keep]

    if is_date and form != "scatter":
        raise ValueError(
            f'form="{form}" describes the joint density of two '
            "numerical variables, not a time series")

    if is_date:                        # time series: date order
        od = x_ser.sort_values(kind="stable").index
        x_ser, y_ser = x_ser.loc[od], y_ser.loc[od]
        if by_ser is not None:
            by_ser = by_ser.loc[od]
        if facet_ser is not None:
            facet_ser = facet_ser.loc[od]
        if facet2_ser is not None:
            facet2_ser = facet2_ser.loc[od]
        if fit != "off" or (ellipse is not False and ellipse):
            raise NotImplementedError(
                "fit and ellipse for a time series (date x) are "
                "not yet ported")
        if MD_cut > 0 or out_cut > 0:
            raise ValueError(
                "outlier flagging (MD_cut/out_cut) applies to a "
                "scatterplot, not a time series")
    elif ts_ahead > 0:
        raise ValueError(
            "forecasting (ts_ahead) requires a date variable "
            "for x")

    if not is_date and (ts_stack or ts_area_fill is not None):
        raise ValueError(
            "ts_stack and ts_area_fill apply to a time series "
            "(date x)")

    # ----- ts aggregation (ts_unit/ts_agg) --------------------------
    # facets aggregate within each panel, below
    ts_agg_done = is_date and ts_unit is not None and facet_ser is None
    if ts_agg_done:
        x_ser, y_ser, by_ser, ts_unit = plt_time(
            x_ser, y_ser, by_ser, ts_unit, ts_agg,
            quiet=resolve_quiet(quiet))
    if is_date and ylab is None and facet_ser is None:  # a given
        # ylab names the axis
        from .plt_time import _infer_unit
        ylab = _ts_ylab(
            y, ts_unit if ts_unit is not None
            else _infer_unit(x_ser.to_numpy()), ts_agg_done, ts_agg)
    if is_date and (ts_stack or ts_area_fill is not None) \
            and pt_size == 1:
        pt_size = 0                    # R: no points with areas

    # ----- enhance: the enhanced-scatterplot bundle ----------------
    # R XY.R:445-450 — for parameters not explicitly set:
    # ellipse .95, MD_cut 6, fit "lm", the mean crosshair
    # (add="means"); default values stand in for R's missing().
    # MD flagging only where supported (single-panel scatter);
    # the crosshair follows R's annotation path (scatter and
    # smooth, single panel).
    add_means = False
    if enhance:
        if is_date:
            raise ValueError(
                "enhance applies to a scatterplot, not a time "
                "series")
        if not ellipse:
            ellipse = 0.95
        if fit == "off":
            fit = "lm"
        if (MD_cut == 0 and out_cut == 0 and form == "scatter"
                and by is None and facet is None):
            MD_cut = 6
        add_means = (form in ("scatter", "smooth")
                     and facet is None)

    # ----- sunflower: petals where observations coincide ----------
    if form == "sunflower":
        if is_date:
            raise ValueError(
                'form="sunflower" plots two continuous variables, '
                "not a time series")
        if ellipse is True:
            ellipse = 0.95
        f_col = to_hex(get_option("fit_color", "#5C4032")
                       if fit_color is None else fit_color)
        f_lwd = get_option("fit_lwd", 2) if fit_lwd is None else fit_lwd
        e_lvls = ([] if not ellipse else
                  list(ellipse) if isinstance(ellipse, (list, tuple))
                  else [ellipse])

        def overlay(fig_, xv_, yv_, row, col):
            at = {} if row is None else dict(row=row, col=col)
            for lv in e_lvls:
                ex, ey = _ellipse_region(xv_, yv_, float(lv))
                fig_.add_trace(go.Scatter(
                    x=ex, y=ey, mode="lines",
                    line=dict(color=to_hex(
                        get_option("ellipse_color", "gray20")
                        if ellipse_color is None else ellipse_color),
                        width=get_option("ellipse_lwd", 1)
                        if ellipse_lwd is None else ellipse_lwd),
                    hoverinfo="skip", showlegend=False), **at)
            if fit not in ("off", "null"):
                if fit == "loess":
                    xs_, _, f_, _ = _loess(xv_, yv_, span)
                else:
                    xs_, _, f_ = _plt_fit(xv_, yv_, fit, fit_power)
                fig_.add_trace(go.Scatter(
                    x=xs_, y=f_, mode="lines",
                    line=dict(color=f_col, width=f_lwd),
                    hoverinfo="skip", showlegend=False), **at)

        xv_f = x_ser.to_numpy(dtype=float)
        yv_f = y_ser.to_numpy(dtype=float)
        if facet is not None:
            n_lvl = len(category_order(facet_ser))
            n_col_sf = (int(n_col) if n_col is not None else
                        math.ceil(n_lvl / int(n_row)) if n_row
                        is not None else math.ceil(math.sqrt(n_lvl)))
            fig = sunflower_plotly(
                xv_f, yv_f, x if xlab is None else xlab,
                y if ylab is None else ylab, main,
                facet=facet_ser.to_numpy(),
                facet_order=category_order(facet_ser),
                facet_name=facet_name,
                facet2=(None if facet2_ser is None
                        else facet2_ser.to_numpy()),
                facet2_order=(None if facet2_ser is None
                              else category_order(facet2_ser)),
                facet2_name=facet2_name, n_col=n_col_sf,
                pt_size=pt_size, fill=fill, color=color,
                overlay=overlay,
                digits_d=2 if digits_d is None else digits_d,
                axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
                axis_y_pre=axis_y_pre)
            xy_stats([(None, xv_f, yv_f)], x, y,
                     digits_d=2 if digits_d is None else digits_d)
            say(facet_report(xv_f))
        else:
            fig = sunflower_plotly(
                xv_f, yv_f, x if xlab is None else xlab,
                y if ylab is None else ylab, main, pt_size=pt_size,
                fill=fill, color=color, overlay=overlay,
                digits_d=2 if digits_d is None else digits_d,
                axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
                axis_y_pre=axis_y_pre)
            fs = []
            if fit not in ("off", "null"):
                if fit == "loess":
                    xs_, ys_, f_, _ = _loess(xv_f, yv_f, span)
                else:
                    xs_, ys_, f_ = _plt_fit(xv_f, yv_f, fit, fit_power)
                fs = [(None, fit, ys_, f_, xs_)]
            say(*scatter_report([(None, xv_f, yv_f)], fs, None))
        return _apply_rotate(fig, rotate_x, rotate_y)

    # ----- facet: one panel per level -----------------------------
    if facet is not None:
        facet2_arr = f2_order = None
        if facet2_ser is not None:
            facet2_arr = facet2_ser.to_numpy()
            f2_order = category_order(facet2_ser)
        # near-square grid by default, like R's lattice facets;
        # explicit n_col/n_row win, and a date (time series) keeps
        # one wide panel per row. With facet2 the grid shape is set
        # from the facet1 levels in facet_panels, so this is moot.
        n_lvl = len(category_order(facet_ser))
        if n_col is not None:
            n_col_use = int(n_col)
        elif n_row is not None:
            n_col_use = math.ceil(n_lvl / int(n_row))
        elif is_date:
            n_col_use = 1
        else:
            n_col_use = math.ceil(math.sqrt(n_lvl))
        if form in ("contour", "smooth"):
            if ellipse is True:
                ellipse = 0.95
            xy_stats([(None, x_ser.to_numpy(dtype=float),
                       y_ser.to_numpy(dtype=float))], x, y,
                     digits_d=2 if digits_d is None else digits_d)
            say(facet_report(x_ser.to_numpy(dtype=float)))
            return _apply_rotate(plt_contour_facet(
                x_ser.to_numpy(dtype=float),
                y_ser.to_numpy(dtype=float),
                facet_ser.to_numpy(),
                category_order(facet_ser), facet_name,
                facet2_arr, f2_order, facet2_name,
                contour_n, contour_nbins, contour_points,
                pt_size,
                x if xlab is None else xlab,
                y if ylab is None else ylab,
                main, 2 if digits_d is None else digits_d,
                ellipse,
                (get_option("ellipse_color", "gray20")
                 if ellipse_color is None else ellipse_color),
                (get_option("ellipse_lwd", 1)
                 if ellipse_lwd is None else ellipse_lwd),
                fit, fit_power,
                (get_option("fit_color", "#5C4032")
                 if fit_color is None else fit_color),
                (get_option("fit_lwd", 2)
                 if fit_lwd is None else fit_lwd),
                _plt_fit, _ellipse_region,
                render=form, smooth_power=smooth_power,
                smooth_points=smooth_points,
                smooth_bins=smooth_bins,
                n_col=n_col_use, axis_fmt=axis_fmt,
                axis_x_pre=axis_x_pre, axis_y_pre=axis_y_pre),
                rotate_x, rotate_y)
        if is_date:                    # time-series panels
            # ts_unit aggregates within each panel and group; the
            # forecast is of each panel's series (XY.R, Aug 2026)
            f_arr = facet_ser.to_numpy()
            ts_by = by_ser.to_numpy() if by_ser is not None else None
            agg_done = ts_unit is not None
            if agg_done:
                parts = [s_.astype(str).to_numpy() for s_ in
                         (facet_ser, facet2_ser, by_ser)
                         if s_ is not None]
                key = pd.Series(["\x1f".join(t) for t in zip(*parts)],
                                index=x_ser.index)
                xa, ya, ka, ts_unit = plt_time(
                    x_ser, y_ser, key, ts_unit, ts_agg,
                    quiet=resolve_quiet(quiet))
                cols = list(zip(*[str(k).split("\x1f")
                                  for k in ka.astype(str)]))
                x_ser, y_ser = xa, ya
                f_arr = np.asarray(cols[0], dtype=object)
                ci = 1
                if facet2_ser is not None:
                    facet2_arr = np.asarray(cols[ci], dtype=object)
                    ci += 1
                if by_ser is not None:
                    ts_by = np.asarray(cols[ci], dtype=object)
            f_order = [str(v) for v in category_order(facet_ser)]
            f_arr = np.asarray(f_arr).astype(str)
            if ylab is None:
                from .plt_time import _infer_unit
                ylab = _ts_ylab(y, ts_unit if ts_unit is not None
                                else _infer_unit(x_ser.to_numpy()),
                                agg_done, ts_agg)
            frcsts = {}
            if ts_ahead > 0:
                xs_all = x_ser.to_numpy()
                ys_all = y_ser.to_numpy(dtype=float)
                for i_f, lv in enumerate(f_order):
                    mm = f_arr == lv
                    if mm.sum() < 4:
                        continue
                    od = np.argsort(xs_all[mm], kind="stable")
                    frcsts[i_f] = plt_forecast(
                        xs_all[mm][od], ys_all[mm][od], x, y,
                        ts_unit=ts_unit, ts_ahead=ts_ahead,
                        ts_method=ts_method, ts_source=ts_source,
                        ts_error=ts_error, ts_trend=ts_trend,
                        ts_seasons=ts_seasons, ts_alpha=ts_alpha,
                        ts_beta=ts_beta, ts_gamma=ts_gamma,
                        ts_PI=ts_PI, digits_d=digits_d)
            if transparency is None:
                transparency = get_option("trans_pt_fill", 0.10)
            fill0 = (get_option("pt_color", "#324E5C")
                     if fill is None else
                     (fill[0] if isinstance(fill, (list, tuple))
                      else fill))
            border0 = fill0 if color is None else color
            a_fill = (None if ts_area_fill is None else
                      _area_color(ts_area_fill,
                                  1 - transparency))
            ts_fig = _ts_facet(
                x_ser.to_numpy(),
                y_ser.to_numpy(dtype=float),
                f_arr, f_order,
                fill0, border0, pt_size, 1 - transparency,
                x if xlab is None else xlab,
                y if ylab is None else ylab,
                main, 2 if digits_d is None else digits_d,
                area_fill=a_fill,
                area_split=float(ts_area_split),
                facet_name=facet_name, facet2_arr=facet2_arr,
                facet2_order=f2_order,
                facet2_name=facet2_name, n_col=n_col_use,
                by_arr=ts_by,
                by_order=(None if by_ser is None else
                          [str(v) for v in category_order(by_ser)]),
                by_name=by, line_width=line_width,
                frcsts=frcsts, ts_PI=ts_PI)
            if ts_n_x_tics is not None:     # date-axis tick count
                ts_fig.update_xaxes(nticks=int(ts_n_x_tics))
            if show:
                for i_f, fr in frcsts.items():
                    print(f"\n{facet_name}: {f_order[i_f]}")
                    print("\n".join(fr["report"]))
                    print("Forecast\n--------")
                    print(fr["forecast"].to_string(
                        float_format=lambda v:
                        f"{v:.{(2 if digits_d is None else digits_d) + 2}f}"))
            return _apply_rotate(ts_fig, rotate_x, rotate_y)
        by_arr = by_order = None
        if by_ser is not None:
            by_arr = by_ser.to_numpy()
            by_order = category_order(by_ser)
        n_grp = 1 if by_order is None else len(by_order)
        if fill is None:
            fills_f = ([get_option("pt_color", "#324E5C")]
                       if n_grp == 1
                       else by_colors(n_grp))
        else:
            fills_f = (list(fill)
                       if isinstance(fill, (list, tuple))
                       else [fill])
        fills_f = [fills_f[i % len(fills_f)]
                   for i in range(n_grp)]
        if transparency is None:
            transparency = get_option("trans_pt_fill", 0.10)
        if fit_se is None:
            fit_se = 0.95 if by is None else 0
        se_lv = [lv for lv in (
            fit_se if isinstance(fit_se, (list, tuple))
            else [fit_se]) if lv]
        if ellipse is True:
            ellipse = 0.95
        xv_s = x_ser.to_numpy(dtype=float)
        xy_stats([(None, xv_s, y_ser.to_numpy(dtype=float))],
                 x, y, digits_d=2 if digits_d is None else digits_d)
        say(facet_report(xv_s))
        fig_f = _apply_rotate(_xy_facet(
            x_ser.to_numpy(dtype=float),
            y_ser.to_numpy(dtype=float),
            by_arr, by_order,
            facet_ser.to_numpy(), category_order(facet_ser),
            by, facet_name, fills_f, pt_shape, pt_size,
            1 - transparency, fit, fit_power, se_lv, span,
            (get_option("fit_color", "#5C4032")
             if fit_color is None else fit_color),
            (get_option("fit_lwd", 2)
             if fit_lwd is None else fit_lwd),
            get_option("se_fill", "#1A1A1A19"),
            ellipse,
            (get_option("ellipse_fill", "#92806F28")
             if ellipse_fill is None else ellipse_fill),
            (get_option("ellipse_color", "gray20")
             if ellipse_color is None else ellipse_color),
            (get_option("ellipse_lwd", 1)
             if ellipse_lwd is None else ellipse_lwd),
            x if xlab is None else xlab,
            y if ylab is None else ylab,
            main, 2 if digits_d is None else digits_d,
            facet2_arr=facet2_arr, facet2_order=f2_order,
            facet2_name=facet2_name, n_col=n_col_use),
            rotate_x, rotate_y)
        if qq_key is not None:
            _qq_reference(fig_f, fit_color, fit_lwd)
        return fig_f

    # ----- groups and colors --------------------------------------
    xv = x_ser.to_numpy() if is_date \
        else x_ser.to_numpy(dtype=float)
    yv = y_ser.to_numpy(dtype=float)
    if by_ser is None:
        groups = [(None, xv, yv)]
    else:
        by_order = category_order(by_ser)
        bv = by_ser.to_numpy()
        groups = [(nm, xv[bv == nm], yv[bv == nm])
                  for nm in by_order]

    # ----- jitter: the display coordinates only --------------------
    # auto-jitter an axis with few discrete values (R plt.main.R
    # ~634-659); the fit, ellipse, and statistics use the
    # original data, as R restores the un-jittered values
    do_jitter = (form == "scatter" and not is_date
                 and float(pt_size) > 0)
    if do_jitter:
        if jitter_x is None:
            jitter_x = (float(np.ptp(xv)) / 32
                        if len(np.unique(xv)) <= 14 < len(xv)
                        else 0)
        if jitter_y is None:
            jitter_y = (float(np.ptp(yv)) / 32
                        if len(np.unique(yv)) <= 14 < len(yv)
                        else 0)
    jitter_x = 0 if jitter_x is None else float(jitter_x)
    jitter_y = 0 if jitter_y is None else float(jitter_y)

    disp_groups = groups
    if do_jitter and (jitter_x > 0 or jitter_y > 0):
        rng = np.random.default_rng()
        xd, yd = xv, yv
        if jitter_x > 0:
            xd = xv + rng.uniform(-jitter_x, jitter_x, len(xv))
        if jitter_y > 0:
            yd = yv + rng.uniform(-jitter_y, jitter_y, len(yv))
        if by_ser is None:
            disp_groups = [(None, xd, yd)]
        else:
            disp_groups = [(nm, xd[bv == nm], yd[bv == nm])
                           for nm in by_order]

    # ----- outliers by Mahalanobis distance -----------------------
    out_idx = md_lines = md_ids = None
    if (MD_cut > 0 or out_cut > 0) and not is_date:
        md_ids = np.asarray(
            get_column(data, ID, "ID")[keep] if ID is not None
            else x_ser.index).astype(str)
        out_idx, md_lines = md_outliers(xv, yv, md_ids,
                                        MD_cut, out_cut)

    n_grp = len(groups)
    if fill is None:
        fills = ([get_option("pt_color", "#324E5C")] if n_grp == 1
                 else by_colors(n_grp))
    else:
        fills = list(fill) if isinstance(fill, (list, tuple)) \
            else [fill]
    borders = None if color is None else color
    if transparency is None:
        transparency = get_option("trans_pt_fill", 0.10)

    # ----- ts stacked / area fills ---------------------------------
    # R plt.main.R: stack polygons (~727-755) between cumulative
    # curves, hues by default, lines in the same hues on top;
    # single-series area (~678-680) fills toward ts_area_split
    # (the lattice origin), clipped to the data extent
    area_polys = []
    area_yrange = None
    if ts_stack:
        x0 = groups[0][1]
        for nm, xg, yg in groups[1:]:
            if len(xg) != len(x0) or (xg != x0).any():
                raise ValueError(
                    "ts_stack requires the same dates at every "
                    "level of by=")
        if ts_area_fill is not None:
            a_base = (list(ts_area_fill)
                      if isinstance(ts_area_fill, (list, tuple))
                      else [ts_area_fill])
            a_cols = [a_base[i % len(a_base)]
                      for i in range(n_grp)]
        else:                          # qualitative hues, as R
            a_cols = [fills[i % len(fills)]
                      for i in range(n_grp)]
        cum = None
        stacked = []
        for i, (nm, xg, yg) in enumerate(groups):
            yc = yg if cum is None else yg + cum
            if cum is None:            # fill to the series min
                lo = float(yc.min())
                xx = np.concatenate([xg, xg[-1:], xg[:1]])
                yy = np.concatenate([yc, [lo, lo]])
            else:                      # band between the curves
                xx = np.concatenate([xg, xg[::-1]])
                yy = np.concatenate([yc, cum[::-1]])
            area_polys.append(
                (xx, yy, _area_color(a_cols[i],
                                     1 - transparency)))
            stacked.append((nm, xg, yc))
            cum = yc
        groups = stacked
        disp_groups = stacked
    elif is_date and ts_area_fill is not None:
        nm0, xg, yg = groups[0]
        lo, hi = float(yg.min()), float(yg.max())
        pad_a = 0.04 * (hi - lo) if hi > lo else 1.0
        base = min(max(float(ts_area_split), lo - pad_a),
                   hi + pad_a)
        xx = np.concatenate([xg, xg[-1:], xg[:1]])
        yy = np.concatenate([yg, [base, base]])
        area_polys.append(
            (xx, yy, _area_color(ts_area_fill,
                                 1 - transparency)))
        # fill reaches the axis edge, as R fills to the y min
        area_yrange = [min(base, lo - pad_a), hi + pad_a]

    # ----- fit lines and SE bands ---------------------------------
    # with by= the bands default off, as in R (XY.R line ~484);
    # fit_errors shows residuals, not a band, so it too defaults
    # the band off (XY.R line ~483)
    if fit_se is None:
        fit_se = 0 if fit_errors else (0.95 if by is None else 0)
    se_levels = [lv for lv in (fit_se if isinstance(
        fit_se, (list, tuple)) else [fit_se]) if lv]
    if form == "contour":              # R: no SE bands on contour
        se_levels = []

    fit_lines, se_polys, fit_stats = [], [], []
    err_lines = []                     # fit_errors residual segments
    if fit != "off":
        for nm, xg, yg in groups:
            if len(xg) < 2:
                continue
            if fit == "loess":         # R: bands for lm and loess
                xs, ys_s, f, se_f = _loess(xg, yg, span)
                fit_lines.append({"name": nm, "x": xs, "y": f})
                fit_stats.append((nm, fit, ys_s, f, xs))
                if fit_errors:
                    err_lines.append((xs, ys_s, f))
                for lv in se_levels:
                    tq = sps.t.ppf((1 + lv) / 2, len(xs) - 1)
                    se_polys.append((
                        np.concatenate([xs, xs[::-1]]),
                        np.concatenate([f + tq * se_f,
                                        (f - tq * se_f)[::-1]])))
                continue
            xs, ys_s, f = _plt_fit(xg, yg, fit, fit_power)
            okf = np.isfinite(f)       # exp/log back-transform NaN
            fit_lines.append({"name": nm, "x": xs[okf],
                              "y": f[okf]})
            fit_stats.append((nm, fit, ys_s[okf], f[okf], xs[okf]))
            if fit_errors:
                err_lines.append((xs[okf], ys_s[okf], f[okf]))
            if fit == "lm":
                for lv in se_levels:
                    se_polys.append(_se_band(xs, ys_s, f, lv))

    # ----- ellipses ------------------------------------------------
    if ellipse is True:
        ellipse = 0.95
    ellipses = []
    if ellipse:
        if not 0 < ellipse < 1:
            raise ValueError("ellipse is the confidence level of "
                             "a data ellipse, between 0 and 1")
        for nm, xg, yg in groups:
            if len(xg) < 3:
                who = "" if nm is None else f" for group '{nm}'"
                raise ValueError(
                    f"ellipse: need at least 3 points{who}, "
                    f"found {len(xg)}")
            ellipses.append(_ellipse_region(xg, yg, ellipse))

    if digits_d is None:
        digits_d = 2

    # ----- contour: filled joint density (form="contour") ----------
    if form == "contour":
        fig = plt_contour(
            xv, yv, contour_n, contour_nbins, contour_points,
            pt_size,
            x_lab=x if xlab is None else xlab,
            y_lab=y if ylab is None else ylab,
            main=main, digits_d=digits_d,
            ell95=_ellipse_region(xv, yv, 0.95),
            ellipses=ellipses,
            ellipse_color=(get_option("ellipse_color", "gray20")
                           if ellipse_color is None
                           else ellipse_color),
            ellipse_lwd=(get_option("ellipse_lwd", 1)
                         if ellipse_lwd is None else ellipse_lwd),
            fit_lines=fit_lines,
            fit_color=(get_option("fit_color", "#5C4032")
                       if fit_color is None else fit_color),
            fit_lwd=(get_option("fit_lwd", 2)
                     if fit_lwd is None else fit_lwd),
            legend=contour_legend,
            axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
            axis_y_pre=axis_y_pre)
        _apply_rotate(fig, rotate_x, rotate_y)
        say(*scatter_report(groups, fit_stats, md_lines))
        return fig

    # ----- forecast (date x, ts_source="classic") -------------------
    frcst = None
    if ts_ahead > 0:
        frcst = plt_forecast(
            xv, yv, x, y, ts_unit=ts_unit, ts_ahead=ts_ahead,
            ts_method=ts_method, ts_source=ts_source,
            ts_error=ts_error, ts_trend=ts_trend,
            ts_seasons=ts_seasons, ts_alpha=ts_alpha,
            ts_beta=ts_beta, ts_gamma=ts_gamma, ts_PI=ts_PI,
            digits_d=digits_d)

    # ----- axes -----------------------------------------------------
    ys_all = np.concatenate(
        [yv] + [fl["y"] for fl in fit_lines]
        + [b[1] for b in se_polys] + [e[1] for e in ellipses]
        + [p[1] for p in area_polys]
        + ([frcst["y_fit"], frcst["y_hat"],
            frcst["y_lwr"], frcst["y_upr"]] if frcst else []))
    if scale_y is not None:            # explicit y scale
        axT2 = np.linspace(float(scale_y[0]), float(scale_y[1]),
                           int(scale_y[2]) + 1)  # n intervals
    else:
        axT2 = pretty(float(np.nanmin(ys_all)),
                      float(np.nanmax(ys_all)))
    axL2 = axis_format(axT2, digits_d, axis_fmt, axis_y_pre)
    if is_date:
        ax = {"axT1": None, "axL1": None, "axT2": axT2,
              "axL2": axL2}
        gridT1 = None
    else:
        xs_all = np.concatenate(
            [xv] + [fl["x"] for fl in fit_lines]
            + [e[0] for e in ellipses])
        if scale_x is not None:        # explicit x scale
            axT1 = np.linspace(float(scale_x[0]),
                               float(scale_x[1]),
                               int(scale_x[2]) + 1)  # n intervals
        else:
            axT1 = pretty(float(np.nanmin(xs_all)),
                          float(np.nanmax(xs_all)))
        ax = {"axT1": axT1,
              "axL1": axis_format(axT1, digits_d, axis_fmt,
                                  axis_x_pre),
              "axT2": axT2, "axL2": axL2}
        gridT1 = axT1

    # ----- render ---------------------------------------------------
    if form == "smooth":
        # plot window: data and overlay extent + 4% (R xaxs="r");
        # ticks outside the window drop, as on the scatter form
        x_lo = float(np.nanmin(xs_all))
        x_hi = float(np.nanmax(xs_all))
        y_lo = float(np.nanmin(ys_all))
        y_hi = float(np.nanmax(ys_all))
        pad_x = 0.04 * (x_hi - x_lo)
        pad_y = 0.04 * (y_hi - y_lo)
        fig = plt_smooth(
            xv, yv, smooth_points, smooth_size, smooth_power,
            smooth_bins,
            x_lab=x if xlab is None else xlab,
            y_lab=y if ylab is None else ylab,
            main=main, digits_d=digits_d,
            ax=ax, gridT1=gridT1, gridT2=ax["axT2"],
            x_lim=[x_lo - pad_x, x_hi + pad_x],
            y_lim=[y_lo - pad_y, y_hi + pad_y],
            fit_lines=fit_lines,
            fit_color=(get_option("fit_color", "#5C4032")
                       if fit_color is None else fit_color),
            fit_lwd=(get_option("fit_lwd", 2)
                     if fit_lwd is None else fit_lwd),
            se_polys=se_polys,
            se_fill=get_option("se_fill", "#1A1A1A19"),
            ellipses=ellipses,
            ellipse_fill=(get_option("ellipse_fill", "#92806F28")
                          if ellipse_fill is None
                          else ellipse_fill),
            ellipse_color=(get_option("ellipse_color", "gray20")
                           if ellipse_color is None
                           else ellipse_color),
            ellipse_lwd=(get_option("ellipse_lwd", 1)
                         if ellipse_lwd is None
                         else ellipse_lwd))
        _apply_rotate(fig, rotate_x, rotate_y)
        if add_means:
            _add_means(fig, xv, yv)
        say(*scatter_report(groups, fit_stats, md_lines))
        return fig

    fig = plt_plotly(
        disp_groups, by_name=by,
        fill=fills, border=borders, shape=pt_shape, pt_size=pt_size,
        x_lab=x if xlab is None else xlab,
        y_lab=y if ylab is None else ylab,
        ax=ax, gridT1=gridT1, gridT2=ax["axT2"],
        main=main, digits_d=digits_d,
        connect=is_date or show_runs or index_x, ln_width=line_width,
        is_date=is_date,
        pt_opacity=1 - transparency,
        area_polys=area_polys,
        fit_lines=fit_lines, fit_color=fit_color, fit_lwd=fit_lwd,
        se_polys=se_polys,
        se_fill=get_option("se_fill", "#1A1A1A19"),
        ellipses=ellipses,
        ellipse_fill=(get_option("ellipse_fill", "#92806F28")
                      if ellipse_fill is None else ellipse_fill),
        ellipse_color=(get_option("ellipse_color", "gray20")
                       if ellipse_color is None else ellipse_color),
        ellipse_lwd=(get_option("ellipse_lwd", 1)
                     if ellipse_lwd is None else ellipse_lwd),
    )

    if (form == "scatter" and not is_date and not index_x
            and not show_runs):
        # square plot box, as R's default device (a time series, a
        # .Index run chart, or a runs chart keeps its wide aspect)
        fig.update_layout(**square_layout(main=bool(main)))

    # fit_errors: residual segment from each point to its fitted
    # value, drawn over the points (R plt.main.R ~1403, rgb 130,40,35)
    if fit_errors and err_lines:
        ex, ey = [], []
        for xs_, yo_, ff_ in err_lines:
            for i in range(len(xs_)):
                ex += [xs_[i], xs_[i], None]
                ey += [yo_[i], ff_[i], None]
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(color="#822823", width=1),
            hoverinfo="skip", showlegend=False))

    # runs test: connect the points and report the run analysis
    # below; the run chart's center line is the median, so a run
    # defaults center_line to "median" (XY.R ~809)
    run_lines = None
    if show_runs:
        m_run, run_lines = _run_analysis(yv, digits_d, True)
        if center_line == "off":
            center_line = "median"

    # center line: a gray dashed reference at the mean, median, or
    # zero, labeled on the right (R plt.main.R ~988)
    if center_line != "off":
        if center_line == "mean":
            m_cl, lbl = float(np.mean(yv)), " mean"
        elif center_line == "median":
            m_cl, lbl = float(np.median(yv)), " median"
        else:                          # "zero": line but no label
            m_cl, lbl = 0.0, ""
        gray = to_hex("gray50")
        fig.add_hline(y=m_cl, line=dict(color=gray, dash="dash",
                                        width=1))
        if lbl:
            fig.add_annotation(x=1, xref="paper", y=m_cl, yref="y",
                               text=lbl, showarrow=False,
                               xanchor="left",
                               font=dict(size=10, color=gray))

    if area_yrange is not None:
        fig.update_layout(yaxis_range=area_yrange)

    if add_means:                      # enhance: mean crosshair
        _add_means(fig, xv, yv)
    if qq_key is not None:
        _qq_reference(fig, fit_color, fit_lwd)

    if scale_x is not None:            # explicit axis ranges
        fig.update_xaxes(range=[float(scale_x[0]),
                                float(scale_x[1])])
    if scale_y is not None:
        fig.update_yaxes(range=[float(scale_y[0]),
                                float(scale_y[1])])
    if is_date and ts_n_x_tics is not None:   # date-axis tick count
        fig.update_xaxes(nticks=int(ts_n_x_tics))
    _apply_rotate(fig, rotate_x, rotate_y)

    if add is not None:                # add= annotations
        add_l = list(add) if isinstance(add, (list, tuple)) \
            else [add]
        if add_l and add_l[0] == "means":   # XY.R ~1405-1410
            _add_means(fig, xv, yv)
            add_l = add_l[1:]
        if add_l:
            def _res(v, key, val):     # "mean_x"/"mean_y"
                if v is None:
                    return None
                vv = (list(v) if isinstance(v, (list, tuple))
                      else [v])
                return [val if u == key else u for u in vv]
            mx_v = None if is_date else float(xv.mean())
            my_v = float(yv.mean())
            plt_add(fig, add_l,
                    x1=_res(x1, "mean_x", mx_v),
                    x2=_res(x2, "mean_x", mx_v),
                    y1=_res(y1, "mean_y", my_v),
                    y2=_res(y2, "mean_y", my_v))

    if out_idx is not None and len(out_idx) > 0:
        _outlier_traces(fig, xv, yv, out_idx, md_ids, fills[0],
                        1 - transparency, pt_size, out_shape,
                        out_size, ID_color, ID_size)
        if fit != "off":               # second fit, outliers
            m = np.ones(len(xv), bool)  # removed, drawn dashed
            m[out_idx] = False
            if fit == "loess":
                xs2, _, f2, _ = _loess(xv[m], yv[m], span)
            else:
                xs2, _, f2 = _plt_fit(xv[m], yv[m], fit,
                                      fit_power)
                ok2 = np.isfinite(f2)
                xs2, f2 = xs2[ok2], f2[ok2]
            fig.add_trace(go.Scatter(
                x=xs2, y=f2, mode="lines",
                name="Fit (no outliers)",
                line=dict(color=to_hex(
                    get_option("fit_color", "#5C4032")
                    if fit_color is None else fit_color),
                    width=get_option("fit_lwd", 2)
                    if fit_lwd is None else fit_lwd,
                    dash="dash"),
                showlegend=True))

    # the series displays: suggestions (with the forecast report
    # and table when one is made), the run analysis of a run chart
    ser_extra = "".join(
        f", {k}={v!r}" if isinstance(v, str) else f", {k}={v}"
        for k, v in (("by", by), ("ts_unit", ts_unit),
                     ("ts_ahead", ts_ahead or None))
        if v is not None).replace("'", '"')
    if frcst is not None:
        _forecast_traces(fig, frcst, ts_PI)
        if show:
            for ln in frcst["report"]:
                print(ln)
            if ts_fitted:              # observed vs model-fit table
                print("\n".join(_ts_fitted_lines(
                    frcst, yv, x, y, ts_unit, digits_d)))
            say(xc.suggest_series(pre, x, y, dname, given,
                                  ser_extra, pt_size=pt_size))
            print("Forecast\n--------")
            print(frcst["forecast"].to_string(
                float_format=lambda v: f"{v:.{digits_d + 2}f}"))
    elif is_date:
        say(xc.suggest_series(pre, x, y, dname, given, ser_extra,
                              pt_size=pt_size))
    elif index_x:                      # run chart: R's run analysis
        if run_lines is None:
            _, run_lines = _run_analysis(yv, digits_d, False)
        say(xc.suggest_series(pre, ".Index", y, dname, given,
                              ser_extra, run=True, pt_size=pt_size),
            xc.run_summary(yv, dd_r), run_lines[1:])
    else:
        say(*scatter_report(groups, fit_stats, md_lines))
        if fit_new is not None and fit in _FIT_NEW_OK and show:
            print("\n".join(_fit_new_table(
                groups, fit, fit_power, fit_new, x, y, digits_d)))
        if (jitter_x > 0 or jitter_y > 0) and show:
            print("\nSome Parameter values (can be manually set)")
            print("-" * 55)
            print(f"size: {float(pt_size):.2f} "
                  " size of plotted points")
            if jitter_y > 0:
                print(f"jitter_y: {jitter_y:.2f} "
                      " random vertical movement of points")
            if jitter_x > 0:
                print(f"jitter_x: {jitter_x:.2f} "
                      " random horizontal movement of points")
    return fig


# font_size= scales all text of the returned figure
from .stats_out import attach_stats as _attach_stats  # noqa: E402
from .style import with_theme as _with_theme  # noqa: E402
XY = font_scaled(_attach_stats(_with_theme(XY), capture=True))
