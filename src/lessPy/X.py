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
# as a grid (reading-order fill, top-left first); not for the VBS bands.
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
from .plotly_utils import (
    BASE_COLORS, axis_format, by_colors, font_scaled,
)
from .plt_add import plt_add
from . import x_console as xc
from .stats_out import CALL_STATS, record_stats, resolve_quiet, x_stats
from .vbs_plotly import vbs_plotly
from .utils import (
    band_width, category_order, get_column, get_option, pretty,
    resolve_facet, set_option,
)

# the VBS family: single-element v/b/s, and the composites. "vb"
# and "vs" are the two-element combos R reaches only through the
# (deprecated) vbs_plot= subset string; here they are just forms.
_VBS_FORMS = ("violin", "box", "strip", "vb", "vs", "bs", "vbs")
_FORMS = ("histogram", "freq_poly", "density") + _VBS_FORMS
_STATS = ("count", "proportion")


def _norm_form(form):
    """Any one to three of v b s, in any order and any case, names
    those VBS layers: normalize to the canonical order v b s, a
    lone letter to the full name of its layer (X.R)."""
    if isinstance(form, str):
        ltr = list(form.strip().lower())
        if (1 <= len(ltr) <= 3 and set(ltr) <= set("vbs")
                and len(set(ltr)) == len(ltr)):
            f = "".join(c for c in "vbs" if c in ltr)
            return {"v": "violin", "b": "box", "s": "strip"}.get(f, f)
    return form


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
                         int(scale_x[2]) + 1)   # n intervals, as R
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
      stat=None,
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
    of the DataFrame `data`. Returns a plotly Figure, or a list of
    Figures, one per variable, when x is a list of names.
    """

    form = _norm_form(form)
    if form not in _FORMS:
        raise ValueError(
            "form must be one of: "
            + ", ".join(f'"{f}"' for f in _FORMS) + "\n\n"
            "Any subset of the violin, box, and strip layers is a "
            "form, named\nby its letters in any order and any case, "
            'as in form="vb" or\nform="SV". A single layer may also '
            'be named in full, as in\nform="violin".')
    if stat is not None:
        if stat not in _STATS:
            raise ValueError(f"stat must be one of {_STATS}")
        # stat names a frequency of the binned data, which only the
        # forms that plot a frequency have (X.R)
        if form not in ("histogram", "freq_poly"):
            why = ("A density is an area, already scaled to a "
                   "proportion,\nso a count and a proportion do not "
                   "distinguish two\ndisplays."
                   if form == "density" else
                   f'form="{form}" plots the values themselves and\n'
                   "the quantiles of their distribution, not a "
                   "frequency,\nso a count and a proportion do not "
                   "distinguish two\ndisplays.")
            raise ValueError(
                "stat sets the plotted value to a count of the cases\n"
                "in each bin or to their proportion, so it applies to "
                "the forms\nthat plot a frequency: form=\"histogram\" "
                'and\nform="freq_poly".\n\n' + why)
    else:
        stat = "count"
    if data is None:
        raise ValueError(
            "data= is required: a pandas DataFrame containing the "
            "named columns")
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
            "cumulate applies to a single histogram or frequency "
            "polygon: no by= or facet=")
    if cumulate != "off" and form not in ("histogram", "freq_poly"):
        raise ValueError(
            "cumulate accumulates the bin counts of a histogram or "
            f'frequency polygon, which form="{form}" does not plot')
    if counts and (by is not None or facet is not None):
        raise ValueError(
            "counts labels apply to a single histogram: "
            "no by= or facet=")
    if rug and by is not None:
        raise ValueError(
            "rug draws the data values of a single series: no by=")
    if kind != "general" and by is not None and facet is not None:
        raise ValueError(
            "kind with by= fits a normal curve to each group of a "
            "single panel: no facet=")
    # by overlays the groups within a panel, marked on the points
    # of the strip layer; a violin or box drawn over another is
    # not readable, so without a strip there is no layer to carry
    # the groups, and facet separates them (X.R)
    vbs_ltr = {"violin": "v", "box": "b",
               "strip": "s"}.get(form, form)
    if (form in _VBS_FORMS and by is not None
            and "s" not in vbs_ltr):
        raise ValueError(
            "The groups of by are overlaid within a panel, marked "
            "on the\npoints of the strip layer, which "
            f'form="{form}" does\nnot include. A violin or a box '
            "drawn over another is not\nreadable, so there is no "
            "layer here to carry the groups.\n\n"
            "To mark the groups, add the strip layer:\n"
            f'  X("{x}", by="{by}", form="{vbs_ltr}s")\n\n'
            f"To draw a separate {form} for each group:\n"
            f'  X("{x}", facet="{by}", form="{form}")')
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

    data_in = data                     # the caller's, for suggestions
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

    # before casewise deletion: the one-variable summary counts the
    # missing values of x, the panel table those within each panel
    x_raw = x_ser.to_numpy(dtype=float)
    grp_ok = ~pd.concat([s for s in (by_ser, facet_ser, facet2_ser)
                         if s is not None] or [x_ser.notna()],
                        axis=1).isna().any(axis=1).to_numpy()
    by_raw = by_ser.to_numpy() if by_ser is not None else None
    f_raw = facet_ser.to_numpy() if facet_ser is not None else None

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

    # the statistics for fig.stats; the console report, ports of
    # R's printers (x_console), follows the display it accompanies
    x_stats(x_ser.to_numpy(dtype=float), x,
            2 if digits_d is None else digits_d)
    show = not resolve_quiet(quiet)
    pre, dname = xc.call_names(data_in)
    x_lbl = ((getattr(data_in, "attrs", {}) or {})
             .get("variable_labels", {}) or {}).get(x)
    # parameters named in the call, which the suggestions skip
    given = {nm for nm, v in (
        ("bin_width", bin_width), ("bin_start", bin_start),
        ("bin_end", bin_end), ("out_cut", out_cut or None),
        ("fences", fences or None), ("vbs_mean", vbs_mean or None),
        ("box_adj", box_adj or None)) if v is not None}

    def say(*comps):
        if show:
            print(xc.report(*comps))

    def one_var(sug):
        """suggestions, summary, outliers: the single-group report"""
        return [sug, xc.ss_numeric(x_raw, x, x_lbl, digits_d),
                xc.outliers(fx, 2)]

    def panel_table():
        """the statistics of each panel (one facet variable)"""
        ok = grp_ok
        cells = f_raw[ok].astype(str)
        name = facet_name
        d = 2
        if by_raw is not None:
            cells = np.char.add(np.char.add(cells, ", "),
                                by_raw[ok].astype(str))
            name = f"{facet_name}, {by}"
            d = max(xc.max_dd(fx) + 1, 2) if digits_d is None \
                else digits_d
        return xc.facet_table(x_raw[ok], cells, x, name, d)

    def bins(edges):
        cnt = pd.cut(pd.Series(fx), edges, right=True,
                     include_lowest=True).value_counts(sort=False)
        return xc.bin_table(edges, cnt.to_numpy(), len(fx),
                            xc.max_dd(fx) if digits_d is None
                            else digits_d)

    # ----- violin / box / strip (VBS) ----------------------------
    if form in _VBS_FORMS:
        vbs_plot = {"violin": "v", "box": "b",
                    "strip": "s"}.get(form, form)
        # the violin's smoothing is bw, the density curve's is
        # bandwidth; both name the same quantity, so a bandwidth
        # given without a bw is honored here (X.R)
        if bw is None and bandwidth is not None:
            bw = bandwidth
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
            pt_fill = by_colors(n_g)   # theme's palette, as R
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
        fig = _x_finish(vbs_plotly(
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

        # report, ~ .param.VBS() and the pivot tables of X.R, from
        # the data as read rather than as jittered for the strip
        prm = CALL_STATS.get() or {}
        comps = [xc.suggest_vbs(
            x, pre, dname, given,
            n_by=len(by_order) if by_order is not None else 0,
            by_name=by,
            n_facet=len(facet_order) if facet_order is not None else 0,
            facet_name=facet_name)]
        ids_out = (np.asarray(get_column(data, ID, "ID")[keep])
                   .astype(str)[fin] if ID is not None else None)
        if by is None and facet is None:
            comps += [
                xc.box_summary(fx, x, x_lbl, digits_d, k_iqr=k,
                               box_adj=box_adj, a=a, b=b),
                xc.outliers(fx, digits_d, ids=ids_out, k_iqr=k,
                            box_adj=box_adj, a=a, b=b),
                xc.dup_values(fx)]
        elif facet is not None:
            comps.append(xc.dup_values(fx, facet_arr, facet_order))
        else:
            comps.append(xc.dup_values(fx, by_arr, by_order))
        comps.append(xc.vbs_params(
            vbs_plot, prm,
            show_bw=prm.get("bw") is not None))
        sd = max(xc.max_dd(fx), 2)   # integer data still needs
        summ = ["", f"---------- Summary Statistics for {x}"]
        if by is not None:
            summ += [""] + xc.vbs_summary(fx, by_arr, by, sd,
                                           order=by_order)
        if facet_arr is not None:
            summ += [""] + xc.vbs_summary(fx, facet_arr, facet_name,
                                           sd, order=facet_order)
        if facet2_arr is not None:
            summ += [""] + xc.vbs_summary(fx, facet2_arr,
                                           facet2_name, sd,
                                           order=facet2_order)
        if by is None and facet is None:
            summ += [""] + xc.vbs_summary(fx, None, x, sd)
        comps.append(summ)
        say(*comps)
        return fig

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
        fig = _x_finish(freq_poly_plotly(
            x_ser, by=by_ser, x_name=x, by_name=by,
            facet=facet_arr, facet_order=f_order,
            facet_name=facet_name,
            facet2=facet2_arr, facet2_order=f2_order,
            facet2_name=facet2_name,
            breaks=edges, proportion=stat == "proportion",
            cumulate=cumulate != "off",
            fill=fill,
            fill_area=area_fill not in ("off", "transparent"),
            x_lab=x if xlab is None else xlab, y_lab=ylab,
            main=main, n_col=n_col,
            axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
            axis_y_pre=axis_y_pre,
            digits_d=3 if digits_d is None else digits_d,
        ), rotate_x, rotate_y, scale_x, axis_fmt, axis_x_pre,
            3 if digits_d is None else digits_d)
        if facet_ser is not None:
            if facet2_ser is None:
                say(panel_table())
        elif by_ser is not None:       # X.R: summary and outliers
            say(*one_var(None)[1:])
        else:
            say(*one_var(xc.suggest_freq_poly(x, pre, dname, given)),
                bins(edges))
        return fig

    # ----- density ---------------------------------------------
    if form == "density":
        # lessR's .band.width(): bw.nrd0 widened toward one peak
        bw = bandwidth if bandwidth is not None else band_width(fx, 25)
        show_h = show_histogram and by is None
        fig = _x_finish(dn_plotly(
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
        if facet_ser is not None:
            if facet2_ser is None:
                say(panel_table())
        elif by_ser is not None:
            say(xc.by_table(x_ser.to_numpy(dtype=float),
                            by_ser.to_numpy(), x, by, pre, dname,
                            order=category_order(by_ser)))
        else:                          # dn.main.R
            st = [f"--- Bandwidth ---      for general curve: "
                  f"{bw:.4f}", ""]
            if kind in ("normal", "both"):
                if 2 < len(fx) < 5000:
                    W, p = sps.shapiro(fx)
                    record_stats(shapiro_W=float(W),
                                 shapiro_p=float(p))
                    st += [" ", "Null hypothesis is a normal population",
                           f"Shapiro-Wilk normality test:  W = {W:.4f},"
                           f"  p-value = {p:.4f}"]
                else:
                    st.append("Sample size out of range for "
                              "Shapiro-Wilk normality test.")
            sug, ss, otl = one_var(xc.suggest_density(x, pre, dname))
            say(sug, st, ss, otl)
        return fig

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
    if facet_ser is not None:
        if facet2_ser is None:         # as .bar.lattice()
            say(panel_table())
    elif by_ser is not None:
        say(xc.by_table(x_ser.to_numpy(dtype=float), by_ser.to_numpy(),
                        x, by, pre, dname,
                        order=category_order(by_ser)))
    else:
        say(*one_var(xc.suggest_hist(x, pre, dname, given)),
            bins(edges))
    return fig


def _per_variable(func):
    """A list of x variables is one display per variable, as in R,
    returned as a list of Figures; suggestions are not repeated for
    each, as R gives none for several variables."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        xs = kwargs["x"] if "x" in kwargs else (args[0] if args
                                                else None)
        if not isinstance(xs, (list, tuple)):
            return func(*args, **kwargs)
        sug = get_option("suggest", True)
        set_option("suggest", False)
        try:
            figs = []
            for xi in xs:
                if "x" in kwargs:
                    figs.append(func(*args, **{**kwargs, "x": xi}))
                else:
                    figs.append(func(xi, *args[1:], **kwargs))
            return figs
        finally:
            set_option("suggest", sug)

    return wrapper


# font_size= scales all text of the returned figure
from .stats_out import attach_stats as _attach_stats  # noqa: E402
from .style import with_theme as _with_theme  # noqa: E402
X = _per_variable(font_scaled(_attach_stats(_with_theme(X),
                                            capture=True)))
