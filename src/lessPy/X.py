# X.py — analog of X.R
#
# X(): the one-variable analytic view — the distribution of a
# single NUMERICAL variable. Categorical variables belong to
# Chart(). As with Chart(), the pipeline is ported, not the lines:
# X.R's NSE, legacy parameters, base-R/lattice paths, and PDF
# device code have no Python counterpart.
#
# Implemented forms: "histogram" (default), "freq_poly",
# "density", and the VBS family ("violin", "box", "strip", "bs",
# "vbs") via the designed vbs_plotly renderer (lessR renders VBS
# through lattice only, so that renderer is a design, not a
# translation), with by= support throughout and facet= for every
# form: VBS (one band per level), histogram (~ .bar.lattice),
# and density and freq_poly (~ .plt.dist.facet). by= combines
# with facet=: one series per group within each panel or band.
# All forms of X.R are ported.
#
# facet= orthogonality (July 2026, ~ the R facet unification):
# facet= takes a column name, a list of two names (a row x
# column grid — rows the second variable; VBS instead stacks
# one band section per second-level, all on the shared x axis),
# or an aligned Series/array of computed values (the analog of
# R's facet expression). Density facets take the single-panel
# embellishments per panel (kind=, show_histogram, rug) when
# by= is absent, as R's .plt.dist.facet.
#
# Also ported (July 2026): cumulate= (with reg=) and counts=
# for the histogram; kind= (normal curve + Shapiro-Wilk
# console report), show_histogram/fill_hist, and rug (rug
# styling implies the density form, X.R:136-138) for the
# density — per panel with facet=, as in R (by= raises,
# except show_histogram which quietly does not draw).
# VBS: box_adj (+a, b) medcouple-adjusted fences, bw_iter
# violin bandwidth search (now the default bandwidth, as R),
# out_cut/ID/ID_size outlier labels above the strip.
# n_row/n_col lay the histogram/density/freq_poly facets out
# as a grid (lattice bottom-up fill); not for the VBS bands.
# Not ported: aspect= (panel aspect is figure sizing in
# plotly), the axis-format/margin family, add= annotations,
# themes.

import math

import numpy as np
import pandas as pd
from scipy import stats as sps

from .dn_plotly import dn_plotly
from .freq_poly_plotly import freq_poly_plotly
from .hs_plotly import hs_plotly
from .plotly_utils import BASE_COLORS, axis_format, font_scaled
from .plt_add import plt_add
from .stats_out import resolve_quiet, x_stats
from .vbs_plotly import vbs_plotly
from .utils import (
    bw_nrd0, category_order, get_column, get_option, pretty,
    resolve_facet,
)

# the VBS family: single-element v/b/s, and the composites. "vb"
# and "vs" are the two-element combos R reaches only through the
# (deprecated) vbs_plot= subset string; here they are just forms.
_VBS_FORMS = ("violin", "box", "strip", "vb", "vs", "bs", "vbs")
_FORMS = ("histogram", "freq_poly", "density") + _VBS_FORMS
_STATS = ("count", "proportion", "density")


def _breaks_from_args(fx, bin_start, bin_width, bin_end, breaks):
    """Bin edges: explicit bin_* arguments win; otherwise Sturges
    via pretty(), the policy of R's hist()."""
    lo, hi = float(fx.min()), float(fx.max())
    if bin_width is not None or bin_start is not None \
            or bin_end is not None:
        if bin_width is None:
            nb = max(1, math.ceil(math.log2(len(fx))) + 1)
            step = pretty(lo, hi, nb)
            bin_width = step[1] - step[0]
        start = lo if bin_start is None else float(bin_start)
        end = hi if bin_end is None else float(bin_end)
        edges = [start]
        while edges[-1] < end - 1e-12:
            edges.append(edges[-1] + float(bin_width))
        if edges[-1] < hi:      # cover the max even past bin_end
            edges.append(edges[-1] + float(bin_width))
        return edges
    if isinstance(breaks, (list, tuple, np.ndarray)):
        return [float(b) for b in breaks]
    if isinstance(breaks, (int, float)):
        return pretty(lo, hi, int(breaks))
    if str(breaks).lower() != "sturges":
        raise ValueError('breaks must be "Sturges", a number of '
                         "bins, or a list of edges")
    nb = max(1, math.ceil(math.log2(len(fx))) + 1)
    return pretty(lo, hi, nb)


def _x_finish(fig, rotate_x, rotate_y, scale_x, axis_fmt,
              axis_x_pre, digits_d):
    """Post-render axis adjustments shared by every X() form:
    scale_x explicit ticks/range, rotate_x/rotate_y tick
    angles. R analogs: scale_x, rotate_x, rotate_y."""
    if scale_x is not None:
        tv = np.linspace(float(scale_x[0]), float(scale_x[1]),
                         int(scale_x[2]))
        fig.update_xaxes(
            tickvals=tv,
            ticktext=axis_format(tv, digits_d, axis_fmt,
                                 axis_x_pre),
            range=[float(scale_x[0]), float(scale_x[1])])
    if rotate_x:
        fig.update_xaxes(tickangle=-float(rotate_x))
    if rotate_y:
        fig.update_yaxes(tickangle=-float(rotate_y))
    return fig


def X(x, by=None, facet=None, data=None, filter=None,
      form="histogram",
      stat="count",
      fill=None, color=None,
      bin_start=None, bin_width=None, bin_end=None,
      breaks="Sturges",
      counts=False, cumulate="off", reg="snow2",
      bandwidth=None, adjust=1, full_curve=True, area_fill="on",
      kind="general", show_histogram=True,
      fill_normal=None, color_normal="gray20", fill_hist=None,
      rug=False, color_rug="black", size_rug=0.5,
      position="overlay",
      bw=None, bw_iter=10, vbs_ratio=1.1, vbs_pt_fill="black",
      violin_fill=None, box_fill=None,
      vbs_mean=False, fences=False, k=1.5,
      box_adj=False, a=-4, b=3,
      out_cut=0, ID=None, ID_size=0.6,
      jitter_x=None, jitter_y=None,
      pt_size=None, out_size=None, out_shape="circle", pt_shape=None,
      n_row=None, n_col=None,
      add=None, x1=None, y1=None, x2=None, y2=None,
      axis_fmt="K", axis_x_pre="", axis_y_pre="",
      rotate_x=0, rotate_y=0, scale_x=None,
      xlab=None, ylab=None, main=None, digits_d=None,
      quiet=None):
    """Analytic view of the distribution of one numerical variable,
    optionally grouped (by=). Variables are strings naming columns
    of the DataFrame `data`. Returns a plotly Figure.
    """

    if form not in _FORMS:
        raise ValueError(f"form must be one of {_FORMS}")
    if stat not in _STATS:
        raise ValueError(f"stat must be one of {_STATS}")
    if data is None:
        raise ValueError(
            "data= is required: a pandas DataFrame containing the "
            "named columns")
    if stat == "density":
        if form == "freq_poly":
            stat = "count"  # density is not a freq poly stat (X.R)
        else:
            form = "density"    # R analog: X.R stat="density"

    if cumulate not in ("off", "on", "both"):
        raise ValueError('cumulate: "off", "on", or "both"')
    if kind not in ("general", "normal", "both"):
        raise ValueError('kind: "general", "normal", or "both"')
    # a rug, or rug styling, implies a density plot (X.R:136-138)
    if color_rug != "black" or size_rug != 0.5:
        rug = True
    if rug and form != "density":
        form = "density"
    if cumulate != "off" and (by is not None
                              or facet is not None):
        raise ValueError(
            "cumulate applies to a single histogram: "
            "no by= or facet=")
    if counts and (by is not None or facet is not None):
        raise ValueError(
            "counts labels apply to a single histogram: "
            "no by= or facet=")
    if (kind != "general" or rug) and by is not None:
        raise ValueError(
            "kind and rug draw per-panel curves of a single "
            "series: no by=")
    if (out_cut > 0 or box_adj) and form not in _VBS_FORMS:
        raise ValueError(
            "out_cut and box_adj apply to the VBS forms: "
            'violin, box, strip, "vb", "vs", "bs", "vbs"')

    if axis_fmt not in ("K", ",", ".", ""):
        raise ValueError('axis_fmt: "K", ",", ".", or ""')
    if scale_x is not None and facet is not None:
        raise ValueError(
            "scale_x applies to a single panel: no facet=")

    if add is not None:
        # R draws X() annotations in hst.main only, n.by == 1
        if form != "histogram" or by is not None \
                or facet is not None:
            raise ValueError(
                "add= annotations apply to a single-panel "
                "histogram in X()")

    # facet grid layout (the lattice n_row/n_col); aspect= is
    # not ported — panel aspect is figure sizing in plotly
    if (n_row is not None or n_col is not None) and facet is None:
        raise ValueError("n_row and n_col lay out facet panels: "
                         "specify facet=")
    if ((n_row is not None or n_col is not None)
            and form in _VBS_FORMS):
        raise ValueError(
            "the plotly VBS draws facet levels as bands on one "
            "panel, so n_row and n_col do not apply")

    if filter is not None:
        data = data.query(filter)

    x_ser = get_column(data, x, "x")
    if not pd.api.types.is_numeric_dtype(x_ser):
        raise TypeError(
            f"X() analyzes the distribution of a numerical "
            f"variable, but '{x}' is {x_ser.dtype}. For a "
            "categorical variable use Chart().")
    by_ser = get_column(data, by, "by") if by is not None else None
    (facet_ser, facet_name,
     facet2_ser, facet2_name) = resolve_facet(data, facet, "X")

    used = [s for s in (x_ser, by_ser, facet_ser, facet2_ser)
            if s is not None]
    keep = ~pd.concat(used, axis=1).isna().any(axis=1)
    x_ser = x_ser[keep]
    if by_ser is not None:
        by_ser = by_ser[keep]
    if facet_ser is not None:
        facet_ser = facet_ser[keep]
    if facet2_ser is not None:
        facet2_ser = facet2_ser[keep]

    fx = x_ser.to_numpy(dtype=float)
    fx = fx[np.isfinite(fx)]
    if len(fx) == 0:
        raise ValueError(f"'{x}' has no finite values")

    if not resolve_quiet(quiet):       # accompanying statistics
        print("\n".join(x_stats(
            x_ser.to_numpy(dtype=float), x,
            2 if digits_d is None else digits_d)))
        if form == "density" and kind in ("normal", "both"):
            if 2 < len(fx) < 5000:     # R dn.main.R range
                W, p = sps.shapiro(fx)
                print("\nNull hypothesis is a normal population")
                print("Shapiro-Wilk normality test:  "
                      f"W = {W:.4f},  p-value = {p:.4f}")
            else:
                print("\nSample size out of range for the "
                      "Shapiro-Wilk normality test")

    # ----- violin / box / strip (VBS) ----------------------------
    if form in _VBS_FORMS:
        vbs_plot = {"violin": "v", "box": "b",
                    "strip": "s"}.get(form, form)
        fin = np.isfinite(x_ser.to_numpy(dtype=float))
        by_arr = by_order = None
        facet_arr = facet_order = None
        facet2_arr = facet2_order = None
        if facet_ser is not None:
            facet_arr = facet_ser.to_numpy()[fin]
            facet_order = category_order(facet_ser)
            # per-panel box hues, ~ .plt.fill "hues" for facets,
            # unless box_fill= names the fill(s) explicitly
            if box_fill is None:
                box_fill = [BASE_COLORS[i % len(BASE_COLORS)]
                            for i in range(len(facet_order))]
        if facet2_ser is not None:
            facet2_arr = facet2_ser.to_numpy()[fin]
            facet2_order = category_order(facet2_ser)
        if by_ser is not None:
            # by= colors points per group; vbs_pt_fill does not
            # apply, as in the X.R n.by > 1 branch
            by_arr = by_ser.to_numpy()[fin]
            by_order = category_order(by_ser)
            n_g = len(by_order)
            pt_fill = [BASE_COLORS[i % len(BASE_COLORS)]
                       for i in range(n_g)]
            pt_color = pt_fill
            pt_trans = get_option("trans_pt_fill", 0.10)
        # strip-point colors per vbs_pt_fill, X.R VBS section
        elif vbs_pt_fill == "black":
            pt_fill, pt_color, pt_trans = "black", "black", 0.10
        elif vbs_pt_fill == "default":
            pt_fill, pt_color = "black", "black"
            pt_trans = get_option("trans_pt_fill", 0.10)
        else:
            pt_fill = vbs_pt_fill
            pt_color = get_option("pt_color", "#324E5C")
            pt_trans = get_option("trans_pt_fill", 0.10)
        if fill is not None:
            pt_fill = fill
        if color is not None:
            pt_color = color
        ids = None
        if out_cut > 0:                # outlier labels
            ids = np.asarray(
                get_column(data, ID, "ID")[keep]
                if ID is not None
                else x_ser.index).astype(str)[fin]
        return _x_finish(vbs_plotly(
            fx, x_name=x, vbs_plot=vbs_plot,
            by=by_arr, by_order=by_order, by_name=by,
            facet=facet_arr, facet_order=facet_order,
            facet_name=facet_name,
            facet2=facet2_arr, facet2_order=facet2_order,
            facet2_name=facet2_name,
            violin_fill=violin_fill, box_fill=box_fill,
            bw=bw, vbs_ratio=vbs_ratio,
            pt_fill=pt_fill, pt_color=pt_color, pt_trans=pt_trans,
            pt_size=pt_size, out_size=out_size,
            out_shape=out_shape, shape=pt_shape,
            vbs_mean=vbs_mean, fences=fences, k_iqr=k,
            box_adj=box_adj, a=a, b=b, bw_iter=bw_iter,
            out_cut=out_cut, ids=ids, ID_size=ID_size,
            jitter_x=jitter_x, jitter_y=jitter_y,
            x_lab=x if xlab is None else xlab, main=main,
            digits_d=2 if digits_d is None else digits_d,
            axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
        ), rotate_x, rotate_y, scale_x, axis_fmt, axis_x_pre,
            2 if digits_d is None else digits_d)

    # facet for histogram/density: aligned array + level order
    facet_arr = (facet_ser.to_numpy()
                 if facet_ser is not None else None)
    f_order = (category_order(facet_ser)
               if facet_ser is not None else None)
    facet2_arr = (facet2_ser.to_numpy()
                  if facet2_ser is not None else None)
    f2_order = (category_order(facet2_ser)
                if facet2_ser is not None else None)
    # resolve the panel grid: explicit n_col wins, else derive
    # from n_row; default one column, the established layout
    if f_order is not None:
        if n_col is None:
            n_col = (math.ceil(len(f_order) / int(n_row))
                     if n_row is not None else 1)
        n_col = max(1, int(n_col))
    else:
        n_col = 1

    # ----- frequency polygon -------------------------------------
    if form == "freq_poly":
        edges = _breaks_from_args(fx, bin_start, bin_width,
                                  bin_end, breaks)
        return _x_finish(freq_poly_plotly(
            x_ser, by=by_ser, x_name=x, by_name=by,
            facet=facet_arr, facet_order=f_order,
            facet_name=facet_name,
            facet2=facet2_arr, facet2_order=f2_order,
            facet2_name=facet2_name,
            breaks=edges, proportion=stat == "proportion",
            fill=fill,
            fill_area=area_fill not in ("off", "transparent"),
            x_lab=x if xlab is None else xlab, y_lab=ylab,
            main=main, n_col=n_col,
            axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
            axis_y_pre=axis_y_pre,
            digits_d=3 if digits_d is None else digits_d,
        ), rotate_x, rotate_y, scale_x, axis_fmt, axis_x_pre,
            3 if digits_d is None else digits_d)

    # ----- density ---------------------------------------------
    if form == "density":
        bw = bandwidth if bandwidth is not None else bw_nrd0(fx)
        show_h = show_histogram and by is None
        return _x_finish(dn_plotly(
            x_ser, by=by_ser, x_name=x, by_name=by,
            facet=facet_arr, facet_order=f_order,
            facet_name=facet_name,
            facet2=facet2_arr, facet2_order=f2_order,
            facet2_name=facet2_name,
            fill=fill,
            x_lab=x if xlab is None else xlab,
            y_lab="Density" if ylab is None else ylab,
            main=main,
            bw=bw, adjust=adjust, full_curve=full_curve,
            fill_area=area_fill not in ("off", "transparent"),
            kind=kind, fill_normal=fill_normal,
            color_normal=color_normal,
            show_histogram=show_h, fill_hist=fill_hist,
            hist_edges=(_breaks_from_args(fx, bin_start,
                                          bin_width, bin_end,
                                          breaks)
                        if show_h else None),
            rug=rug, color_rug=color_rug, size_rug=size_rug,
            n_col=n_col,
            axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
        ), rotate_x, rotate_y, scale_x, axis_fmt, axis_x_pre,
            3 if digits_d is None else digits_d)

    # ----- histogram ---------------------------------------------
    edges = _breaks_from_args(fx, bin_start, bin_width, bin_end,
                              breaks)
    proportion = stat == "proportion"
    if ylab is None:
        prefix = "Proportion of" if proportion else "Count of"
        ylab = f"{prefix} {x}"
    fig = hs_plotly(
        x_ser, by=by_ser, x_name=x, by_name=by,
        facet=facet_arr, facet_order=f_order, facet_name=facet_name,
        facet2=facet2_arr, facet2_order=f2_order,
        facet2_name=facet2_name,
        breaks=edges, freq=True, proportion=proportion,
        fill=fill, border=color,
        cumulate=cumulate, reg=reg, counts=counts,
        x_lab=x if xlab is None else xlab, y_lab=ylab,
        digits_d=2 if digits_d is None else digits_d,
        position=position, main=main, n_col=n_col,
        axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
        axis_y_pre=axis_y_pre,
    )
    _x_finish(fig, rotate_x, rotate_y, scale_x, axis_fmt,
              axis_x_pre, 2 if digits_d is None else digits_d)
    if add is not None:
        plt_add(fig, add, x1=x1, x2=x2, y1=y1, y2=y2)
    return fig


# font_size= scales all text of the returned figure
X = font_scaled(X)
