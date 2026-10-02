# Chart.py — analog of Chart.R (and the data-prep parts of
# bc.zmain.R / agg.R that feed .bc.plotly)
#
# The R Chart() is 1800+ lines; most of that is R-specific machinery
# with no Python counterpart and is deliberately absent here:
#   - non-standard evaluation of variable names (Python interface is
#     strings naming DataFrame columns)
#   - legacy/renamed parameter migration (new package, no legacy)
#   - base-R / lattice rendering paths and PDF graphics devices
#     (plotly-only port)
# What carries over is the pipeline: resolve variables -> filter ->
# tabulate or aggregate -> sort -> render.
#
# All seven forms are implemented: bar, pie, dot, radar, bubble,
# treemap, icicle — plus "sunburst", which (as in R) is an alias
# for pie; pie with by= renders as a sunburst via the hier path.
#
# facet= draws one panel per facet level: a grid of panels for
# dot, radar, bubble (1-D), and the hier forms; the pie grid for
# a plain pie (facet becomes the grouping, as in R); and for bar
# the Trellis chart — stacked panels of horizontal count bars,
# the plotly port of R's lattice rendering.

import math

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .bc_plotly import bc_facet_plotly, bc_plotly
from .bc_items_plotly import (
    bc_items_plotly, items_fill, items_stats_lines, items_table,
)
from .plt_add import plt_add
from .bubble_plotly import bubble_plotly
from .dot_plotly import dot_plotly
from .hier_plotly import (
    hier_aggregate, hier_color_resolve, hier_plotly,
)
from .pie_plotly import pie_plotly
from .plt_plotly import plt_plotly
from .profile_plotly import profile_facet_plotly
from .radar_plotly import radar_plotly
from .plotly_utils import build_title, font_scaled
from .stats_out import chart_stats, resolve_quiet
from .utils import (
    STAT_FUN, STAT_LBL, category_order as _category_order,
    facet_values, get_column as _get_column, get_option, pretty,
)

# color theme -> HCL sequential palette family (R analog: .get_fill)
_THEME_PALETTE = {
    "colors": "blues", "dodgerblue": "blues", "blue": "blues",
    "lightbronze": "blues",
    "gray": "grays", "white": "grays", "light": "grays",
    "gold": "browns", "brown": "browns", "sienna": "browns",
    "orange": "rusts",
    "darkred": "reds", "red": "reds", "rose": "reds",
    "slatered": "reds",
    "darkgreen": "greens", "green": "greens",
    "purple": "violets",
}


def _theme_fill(theme, n):
    """A theme's fill: an n-color sequential palette in the theme's
    hue. R analog: the theme branch of the fill assignment."""
    pal = _THEME_PALETTE.get(theme)
    if pal is None:
        raise ValueError(
            f"unknown theme '{theme}'; one of: "
            f"{', '.join(sorted(_THEME_PALETTE))}")
    from .getColors import getColors
    cols = getColors(pal, n=max(1, n), quiet=True)
    return [c[:7] if len(c) == 9 else c for c in cols]  # drop alpha


def _facet_table(x_s, y_s, by_s, stat, is_agg, x_order, by_order,
                 proportion=False):
    """Frequency or stat table for one facet level, on the global
    category set(s) so panels stay aligned; absent cells fill 0,
    as in R's xtabs. Series without by, DataFrame (by x cats)
    with. proportion normalizes the per-panel counts (no y, no by)
    to sum to 1, the faceted stat_x="proportion". Used by the
    faceted radar and bubble paths."""
    if by_s is None:
        if y_s is None:
            t = x_s.groupby(x_s, observed=True).size() \
                   .reindex(x_order, fill_value=0)
            if proportion:
                tot = t.sum()
                t = t / tot if tot > 0 else t.astype(float)
        elif is_agg and stat is None:
            t = (pd.Series(y_s.values, index=x_s.values)
                 .reindex(x_order))
        else:
            t = STAT_FUN[stat](
                y_s.groupby(x_s, observed=True)).reindex(x_order)
        return t.fillna(0)

    if y_s is None:
        t = pd.crosstab(by_s, x_s)
    elif is_agg and stat is None:
        t = (pd.DataFrame({"by": by_s, "x": x_s, "y": y_s})
             .pivot(index="by", columns="x", values="y"))
    else:
        df = pd.DataFrame({"by": by_s, "x": x_s, "y": y_s})
        t = STAT_FUN[stat](
            df.groupby(["by", "x"], observed=True)["y"]
        ).unstack("x")
    return (t.reindex(index=by_order, columns=x_order)
            .fillna(0))

_FORMS = ("bar", "radar", "bubble", "dot", "pie", "icicle", "treemap",
          "profile")
_STATS = ("mean", "sum", "sd", "deviation", "min", "median", "max")
_SORTS = ("0", "-", "+")


def _sort_1d(tbl, sort):
    if sort == "-":
        return tbl.sort_values(ascending=False)
    if sort == "+":
        return tbl.sort_values(ascending=True)
    return tbl


def _sort_2d(tbl, sort):
    """Order x categories (columns) by their totals across groups."""
    if sort == "0":
        return tbl
    totals = tbl.sum(axis=0)
    asc = sort == "+"
    return tbl[totals.sort_values(ascending=asc).index]


def _dot_origin_grid(vals, origin_in=None, is_counts=None):
    """Value-axis origin and grid ticks for a dot chart. Counts
    anchor at 0; continuous data start one step below the first
    tick unless the values hug zero. R analog: .dot_origin_grid()"""
    fv = [float(v) for v in vals if np.isfinite(v)]
    if not fv:
        return (0 if origin_in is None else origin_in,
                pretty(0, 1))

    if is_counts is None:
        is_counts = all(v >= 0 and v.is_integer() for v in fv)

    origin = origin_in
    if origin is None:
        if is_counts:
            origin = 0
        else:
            fv2 = [-v for v in fv] if all(v < 0 for v in fv) else fv
            mn_v, mx_v = min(fv2), max(fv2)
            if mn_v > 0 and (mx_v - mn_v) / mn_v <= 2.40:
                origin = mn_v

    lo = min(fv) if origin is None else min(origin, min(fv))
    gridT = pretty(lo, max(lo, max(fv)))
    # nudge one step below first tick for continuous data, not counts
    if origin_in is None and not is_counts and len(gridT) > 1:
        step = gridT[1] - gridT[0]
        origin = gridT[0] - step
        gridT = pretty(min(origin, min(fv)),
                       max(origin, max(fv)))
    return origin, gridT


def _chart_items(data, items, y, by, facet, form, stat, sort,
                 sort_miss, stack100, horiz, fill, theme, labels,
                 labels_size, gap, labels_cut, xlab, ylab, main,
                 legend_title,
                 n_col, n_row, axis_fmt, axis_x_pre, rotate_x,
                 rotate_y, quiet):
    """The multi-item stacked chart and its faceted form. R analog:
    the multiple-x branch of Chart.R, .bc.main() multi, and
    .bar.stackedLattice()"""
    if by is not None:
        raise ValueError(
            "by is not available for multiple x variables: the "
            "items claim position and the responses color, so a by "
            "variable has no encoding left. Use facet, which draws "
            "each group in a panel of its own.")
    if facet is not None and form != "bar":
        raise ValueError(
            'facet with multiple x variables requires form="bar"')
    if form == "bubble":
        raise NotImplementedError(
            "the bubble plot frequency matrix of multiple x "
            'variables is not yet ported; use form="bar"')
    if form != "bar":
        raise ValueError(
            'Multiple x variables only available for form="bar"')
    if y is not None or stat is not None:
        raise ValueError(
            "A multi-item chart counts the responses to each item, "
            "so y and stat do not apply")
    if not horiz:
        raise NotImplementedError(
            "the vertical multi-item chart is not yet ported")
    cols = [_get_column(data, it, "x") for it in items]
    frq, wm, numeric = items_table(cols, items)
    fill_use = (fill if isinstance(fill, (list, tuple))
                else [fill] * len(frq.index) if fill is not None
                else items_fill(theme, len(frq.index)))
    leg = "Responses" if legend_title is None else legend_title
    show = not resolve_quiet(quiet)

    if facet is None:
        # ordered by mean response, the largest at the top: the bars
        #   build upward from the bottom, so ascending unless sort
        #   says otherwise
        if sort_miss:
            sort = "+"
        order = list(items)
        if sort != "0":
            order = list(wm.sort_values(ascending=(sort == "+"),
                                        kind="stable").index)
        if show:
            # the table lists the items top of the chart first
            print("\n".join(_items_print(
                frq[order[::-1]], wm[order[::-1]], numeric)))
        return bc_items_plotly(
            frq, order, fill_use,
            labels="%" if labels is None else labels,
            labels_size=labels_size, stack100=stack100,
            labels_cut=0.04 if labels_cut is None else labels_cut,
            x_lab="" if xlab is None else xlab,
            y_lab="" if ylab is None else ylab,
            legend_title=leg, main=main or None,
            axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
            rotate_x=rotate_x, rotate_y=rotate_y)

    # faceted: a panel is a fraction of the width of the single
    #   chart, too narrow for segment labels, and the paneled chart
    #   keeps the order given so the panels read alike
    if labels is not None and labels != "off":
        raise ValueError(
            "labels are not drawn on a faceted multi-item chart. Each "
            "panel is a fraction of the width of the single chart, so "
            "its segments are too narrow for a legible label. The "
            "counts for each panel are reported at the console.")
    if gap is not None:
        raise ValueError(
            "gap is not available for a faceted multi-item chart")
    if not sort_miss and sort != "0":
        raise ValueError(
            "sort is not available for a faceted multi-item chart: "
            "the panels keep the items in the order given, so they "
            "read alike")
    if isinstance(facet, (list, tuple)):
        fcols = [_get_column(data, f, "facet") for f in facet]
        fac = fcols[0].astype(str)
        for c in fcols[1:]:
            fac = fac + " / " + c.astype(str)
        facet_name = ", ".join(facet)
    else:
        fac = _get_column(data, facet, "facet")
        facet_name = facet
    keep = fac.notna()
    facet_order = _category_order(fac[keep])
    resp = list(frq.index)
    tbls = {}
    for lv in facet_order:
        m = keep & (fac == lv)
        f_lv, w_lv, _ = items_table([c[m] for c in cols], items) \
            if all(c[m].notna().any() for c in cols) else (
                pd.DataFrame(0, index=resp, columns=items), None, None)
        f_lv = f_lv.reindex(index=resp, fill_value=0)
        if show:
            if w_lv is None:
                w_lv = pd.Series(np.nan, index=items)
            print("\n".join(_items_print(
                f_lv, w_lv, numeric,
                f"Frequencies of Responses by Variable, "
                f"{facet_name}: {lv}")))
        t = f_lv.astype(float)
        if stack100:
            tot = t.sum(axis=0)
            t = t.div(tot.where(tot != 0), axis=1).fillna(0)
        tbls[lv] = t[items[::-1]]       # first item at the top
    tbl = pd.DataFrame([tbls[lv].sum(axis=0) for lv in facet_order],
                       index=facet_order)
    tbl.index.name = facet_name
    n_col_use = (max(1, int(n_col)) if n_col is not None
                 else max(1, math.ceil(len(facet_order) / int(n_row)))
                 if n_row is not None else 1)
    return bc_facet_plotly(
        tbl, x_name="", facet_name=facet_name,
        x_lab="" if xlab is None else xlab,
        y_lab="" if ylab is None else ylab,
        fill=fill_use, border="off",
        digits_d=2 if stack100 else 0, n_col=n_col_use,
        axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
        rotate_x=rotate_x, rotate_y=rotate_y, main=main,
        by_tbls=tbls, by_name=leg, beside=False,
        val_name=("Proportion" if stack100 else "Count"),
        legend_title=leg)


def _items_print(frq, wm, numeric, title=None):
    return items_stats_lines(
        frq, wm, title or "Frequencies of Responses by Variable",
        numeric)


def _profile_fill(theme):
    """One hue per series from the theme, as R's profile does:
    .color_range(.get_fill(theme), n), the qualitative hues for the
    default theme."""
    from .plotly_utils import BASE_COLORS
    if theme is None or theme == "colors":
        return list(BASE_COLORS)
    return _theme_fill(theme, 8)


def _max_dd(vals):
    """Most decimal digits among the first 200 values as R format()
    writes them, 7 significant digits. R analog: .max.dd()"""
    mx = 0
    for v in list(vals)[:200]:
        if v is None or not np.isfinite(v):
            continue
        txt = f"{float(v):.7g}"
        if "." in txt and "e" not in txt:
            mx = max(mx, len(txt) - txt.index(".") - 1)
    return mx


def _dot_diff_lines(cats, ydf, x_name):
    """The difference a paired dot plot presents, listed in the
    order drawn, top of the plot first. R analog: the dif.pair
    listing of Chart.R's paired dot path."""
    difs = (ydf.iloc[:, 1] - ydf.iloc[:, 0]).to_numpy(dtype=float)
    dd = min(_max_dd(np.r_[ydf.iloc[:, 0], ydf.iloc[:, 1]]) + 1, 7)
    ny = len(difs)
    mx_i = len(str(ny))
    mx_d = max(len(f"{v:.{dd}f}") for v in difs)
    mx_f = max([5] + [len(c) for c in cats])
    lines = ["", f"{ydf.columns[1]} - {ydf.columns[0]}",
             f"{'n'.rjust(mx_i)} {' diff'.rjust(mx_d)}  {x_name}",
             "-" * (mx_i + mx_d + mx_f + 2)]
    shown = (range(1, ny + 1) if ny <= 20
             else list(range(1, 11)) + list(range(ny - 9, ny + 1)))
    for i in range(1, ny + 1):
        k = ny - i
        if i in shown:
            lines.append(f"{str(i).rjust(mx_i)} "
                         f"{f'{difs[k]:.{dd}f}'.rjust(mx_d)} {cats[k]}")
    return lines + [""]


def _dot_paired(cats, ydf, sort, sort_miss, origin_x, show_diff,
                x_name, **render):
    """Order a multi-series dot plot and draw it. Two series are a
    pair, read for the gap between them, so by default they are
    ordered by that difference, ascending so the largest positive
    difference is drawn at the top, and the value axis begins at
    zero. More series are ordered by their row mean when sort is
    given. R analog: Chart.R paired dot path."""
    dif_pair = ydf.shape[1] == 2
    if dif_pair and sort_miss:
        sort = "+"
    if sort != "0":
        key = (ydf.iloc[:, 1] - ydf.iloc[:, 0] if dif_pair
               else ydf.mean(axis=1))
        order = np.argsort(key.to_numpy(dtype=float)
                           * (-1 if sort == "-" else 1),
                           kind="stable")
        cats = [cats[i] for i in order]
        ydf = ydf.iloc[order]
    if dif_pair and show_diff:
        print("\n".join(_dot_diff_lines(cats, ydf, x_name)))
    origin, gridT = _dot_origin_grid(
        ydf.to_numpy(dtype=float).ravel(),
        origin_in=0 if (dif_pair and origin_x is None) else origin_x)
    if origin is None:
        origin = 0
    return dot_plotly(cats, ydf, gridT=gridT, origin_x=origin,
                      **render)


def Chart(x, y=None, data=None, filter=None, by=None, facet=None,
          form="bar",
          n_row=None, n_col=None,
          hole=0.62,
          radius=0.50, power=0.5,
          pt_size=1, origin_x=None, origin_y=None,
          segments_x=None, segments_y=None,
          stat=None, stat_x="count",
          horiz=None, sort=None, beside=False, stack100=False,
          gap=None, scale_y=None, break_x=None,
          fill=None, color=None, transparency=None,
          fill_split=None, fill_scaled=False, fill_chroma=75,
          theme=None,
          labels=None, labels_position=None,
          labels_color=None, labels_size=None, labels_decimals=None,
          labels_cut=None, segments=None,
          legend_title=None, legend_position=None,
          legend_labels=None, legend_horiz=False,
          legend_size=None, legend_abbrev=None, legend_adjust=0,
          add=None, x1=None, y1=None, x2=None, y2=None,
          xlab=None, ylab=None, main=None,
          rotate_x=0, rotate_y=0,
          axis_fmt="K", axis_x_pre="", axis_y_pre="",
          digits_d=None, quiet=None):
    """General analytic view of one categorical variable x,
    optionally crossed with a second categorical variable (by=) or
    summarizing a numerical variable (y= with stat=). facet= names
    one more categorical variable (or a list of them, combined
    into one panel grid) that splits the chart into one panel per
    level; n_row/n_col lay those panels out as a grid.

    The parameter order follows R's Chart(), so the second
    positional argument is y, the numerical variable the chart
    aggregates: Chart("Dept", "Salary", stat="mean", data=d). A
    second CATEGORICAL variable is named -- by="Gender".

    labels_position defaults to auto-tuning for a bar chart: each
    value label sits inside its bar, or moves outside when the bar
    is too small to hold it. Pass "in" or "out" to force one.

    stack100 rescales each bar to sum to 1.0, so a stacked bar
    chart shows the composition of every x category on a common
    0-100% scale. Requires by=. labels="input" then displays the
    underlying counts, as in R.

    gap sets the spacing between bars in units of bar width (R's
    barplot space=): one value, or (within, between) with beside=.
    scale_y is (min, max, n_intervals) for the value axis, giving
    n_intervals + 1 ticks (R's axTicks(axp=)). break_x breaks
    category labels at their spaces onto separate lines; a "~" in
    a label is a non-breaking space.

    Value-axis tick labels follow the axis_fmt policies (default
    "K": "60K" for thousands, commas past 9999); axis_x_pre /
    axis_y_pre prepend a per-axis prefix (e.g. "$").

    form="dot" is Cleveland's dot plot: categories on the vertical
    axis (horiz defaults to True for this form alone), the value
    axis from zero, no title unless main= is given. With by=, one
    series of dots per level; with exactly two levels, or a pair
    of y variables (y=["Pre", "Post"]), the pair is joined by a
    segment and ordered by its difference, largest positive at
    the top. sort="0" keeps the data order instead.

    form="profile" plots one point per category of x and connects
    the points across the categories; with by= it draws one
    profile per level, the interaction plot of a two-way ANOVA.
    origin_y sets where the value axis begins; facet= draws one
    panel per level on a shared value scale.

    Variables are strings naming columns of the DataFrame `data`.
    Returns a plotly Figure; call .show() to display from a script.
    """

    # ----- validate parameters ------------------------------------
    if form == "sunburst":          # R analog: Chart.R line ~100
        form = "pie"
    if form not in _FORMS:
        raise ValueError(f"form must be one of {_FORMS}")

    # Cleveland's dot plot places the categories on the vertical
    #   axis so their labels read across. A horizontal chart lists
    #   its first category at the bottom, so sort is inverted to keep
    #   "-" meaning largest at the top. R analog: Chart.R horiz.miss
    sort_miss = sort is None
    if sort is None:
        sort = "0"
    multi_x = isinstance(x, (list, tuple))
    if horiz is None:
        # the dot plot and the multi-item chart list their categories
        #   down the vertical axis
        horiz = form == "dot" or multi_x
    if horiz and sort in ("+", "-"):
        sort = "+" if sort == "-" else "-"

    # segments joins the points of a profile; the dot chart's stems
    #   are set along its value axis
    if segments is not None and form != "profile":
        raise ValueError(
            "segments joins the points of a profile chart, so it "
            'needs form="profile"; for the segments of a dot chart '
            "set segments_x or segments_y")
    if labels_cut is not None and form != "bar":
        raise ValueError(
            "labels_cut leaves off the value labels of small bar "
            'segments, so it needs form="bar"')

    # the value axis of a profile is y, so origin_x has nothing to set
    if form == "profile" and origin_x is not None:
        raise ValueError(
            "origin_x does not apply to a profile chart: its value "
            "axis is y, so set origin_y")

    # Only the segments parameter along the value axis applies. The
    #   dot plot's orientation decides which one that is, so say which
    #   parameter to set instead rather than let the given one go
    #   inert. An unfaceted plot of several series is always
    #   horizontal.
    #   R analog: Chart.R .seg_note
    if form == "dot":
        dot_h = horiz or (facet is None and (
            by is not None or isinstance(y, (list, tuple))))
        given, instead, cat_ax, seg_ax = (
            ("segments_y", "segments_x", "y", "x") if dot_h
            else ("segments_x", "segments_y", "x", "y"))
        if (segments_y if dot_h else segments_x) is not None:
            print(f">>> {given} does not apply to this dot plot.\n"
                  f"    The categories are on the {cat_ax}-axis, so "
                  "the line segments run along the "
                  f"{seg_ax}-axis.\n"
                  f"    To control them, set  {instead}.\n")
            if dot_h:
                segments_y = None
            else:
                segments_x = None

    # hierarchical routing: treemap/icicle always; pie with by=
    # nests the by rings inside the x wedges as a sunburst
    # R analog: Chart.R hier dispatch
    hier_type = None
    if form in ("treemap", "icicle"):
        hier_type = form
    elif form == "pie" and by is not None:
        hier_type = "sunburst"
    if isinstance(by, (list, tuple)) and len(by) > 1 \
            and hier_type is None:
        raise ValueError(
            f"Only one by variable is permitted for a {form} chart, "
            "but more than one specified.\n\n"
            "The groups of by are overlaid within the display, one "
            "color\n  to each.  That one channel is carried by the "
            "first variable,\n  so a second has no encoding left by "
            "which to separate its\n  levels.\n"
            "Multiple by variables apply only to the hierarchical\n"
            "  charts -- pie/sunburst, treemap, and icicle -- where\n"
            "  nesting accepts depth and each added variable is one\n"
            "  level deeper.\n"
            f"To stratify a {form} chart by a second variable, use\n"
            "  facet, which draws each group in a panel of its own.")
    # hierarchical forms: each further by variable is one level
    #   deeper; the first plays the part of a single by elsewhere
    by_more = []
    if isinstance(by, (list, tuple)):
        by_more = list(by[1:])
        by = by[0] if by else None
    by_label = ", ".join([by] + by_more) if by is not None else None
    if hier_type is not None and stat == "deviation":
        raise ValueError('stat="deviation" is not meaningful for '
                         "hierarchical charts: negative values "
                         "cannot form part-of-whole areas")

    if isinstance(y, (list, tuple)) and form != "dot":
        raise ValueError('Multiple y variables are only supported '
                         'for form="dot" (paired dot chart)')
    if sort not in _SORTS:
        raise ValueError('sort must be "0" (none), "-" (descending) '
                         'or "+" (ascending)')
    if stat is not None and stat not in _STATS:
        raise ValueError(f"stat must be one of {_STATS}")
    if stat_x not in ("count", "proportion"):
        raise ValueError('stat_x must be "count" or "proportion"')
    if labels_position not in (None, "in", "out"):
        raise ValueError('labels_position must be "in" or "out"')
    if axis_fmt not in ("K", ",", ".", ""):
        raise ValueError('axis_fmt must be "K", ",", "." or ""')
    if stack100:
        if form != "bar":
            raise ValueError('stack100 rescales the bars of a bar '
                             'chart, so it needs form="bar"')
        if facet is not None and by is None:
            raise ValueError(
                "stack100 rescales each bar to show how the levels "
                "of a by variable divide it, so a faceted bar chart "
                "needs by= for it")
        if by is None and stat_x == "proportion":
            raise ValueError(
                'stack100 without a by variable is the same as '
                'stat_x="proportion": specify one, not both')
    if scale_y is not None:
        if len(scale_y) != 3:
            raise ValueError(
                "scale_y is (min, max, n_intervals) for the value "
                "axis: three values, giving n_intervals + 1 ticks")
        if float(scale_y[1]) <= float(scale_y[0]):
            raise ValueError("scale_y max must exceed scale_y min")
        if int(scale_y[2]) < 1:
            raise ValueError("scale_y needs at least one interval")
    # R analog: Chart.R  break_x <- !horiz && rotate_x == 0
    if break_x is None:
        break_x = not horiz and rotate_x == 0
    # the legend keys the by= levels of a two-variable bar chart,
    # which is where R defines these; say so rather than no-op
    _leg = {"legend_title": legend_title,
            "legend_position": legend_position,
            "legend_labels": legend_labels,
            "legend_horiz": legend_horiz or None,
            "legend_size": legend_size,
            "legend_abbrev": legend_abbrev,
            "legend_adjust": legend_adjust or None}
    _leg_set = [k for k, v in _leg.items() if v is not None]
    # a dot plot of several series heads its legend with legend_title
    if form == "dot" and (by is not None
                          or isinstance(y, (list, tuple))):
        _leg_set = [k for k in _leg_set if k != "legend_title"]
    if _leg_set:
        if form != "bar":
            raise ValueError(
                f"{', '.join(_leg_set)} style the legend of a "
                'two-variable bar chart, so they need form="bar"')
        if by is None:
            raise ValueError(
                f"{', '.join(_leg_set)} style the legend that a by "
                "variable creates, so they need by=")
        if facet is not None and _leg_set != ["legend_title"]:
            raise ValueError(
                f"{', '.join(_leg_set)} apply to a single-panel bar "
                "chart; a faceted (Trellis) bar chart has no by "
                "variable and no legend")

    if (n_row is not None or n_col is not None) and facet is None:
        raise ValueError("n_row and n_col lay out facet panels, "
                         "so they require a facet variable")
    if add is not None and (form != "bar"
                             or facet is not None):
        raise ValueError(
            "add= annotations apply to a single-panel bar "
            "chart in Chart()")
    if data is None:
        raise ValueError(
            "data= is required: a pandas DataFrame containing the "
            "named columns. (lessR's default of a data frame named "
            "d in the global environment has no Python analog.)")
    if stat is not None and y is None:
        raise ValueError(
            "stat requires a numerical y variable to transform, "
            "e.g., y='Salary'")

    if facet is not None:
        if isinstance(y, (list, tuple)):
            raise ValueError(
                "Multiple y variables (paired dot chart) combined "
                "with facet= is not supported. Use a single y "
                "variable with facet=, or omit facet=.")
        if stat == "deviation":
            raise ValueError("deviation for stat is not "
                             "meaningful with a facet variable")
        if form == "bubble" and by is not None:
            raise ValueError(
                "The facet option for bubble charts applies only "
                "to a single categorical variable (no by "
                "variable)")
        if stat_x == "proportion" and y is not None:
            raise ValueError(
                'stat_x="proportion" applies to counts of x, so a '
                "y variable does not apply")
        if form == "bar":
            # Trellis bar chart: panels of counts of the original
            # data, as in R (.bcParamValid / .bar.lattice)
            if sort != "0":
                raise ValueError("Sort not applicable to Trellis "
                                 "(faceted bar) charts")
            if stat_x == "proportion" and by is not None:
                raise NotImplementedError(
                    'stat_x="proportion" with by= is not yet ported')

    # ----- resolve variables and filter ---------------------------
    if filter is not None:
        data = data.query(filter)

    # row_names: the data frame's row labels as the categorical x,
    #   in their own order rather than alphabetical; the axis label
    #   is dropped unless given. R analog: Chart.R row_names
    if (isinstance(x, str) and x in ("row_names", "row.names")
            and x not in data.columns):
        names = data.index.astype(str)
        data = data.copy()
        data[x] = pd.Categorical(
            names, categories=list(dict.fromkeys(names)))
        if xlab is None:
            xlab = ""

    # a list of x variables that share one response scale: the
    # multi-item stacked chart. Its bands claim position and color
    # the responses within them, so by has no encoding left; facet
    # panels remain. R analog: Chart.R multiple-x path
    if multi_x:
        return _chart_items(
            data, list(x), y=y, by=by, facet=facet, form=form,
            stat=stat, sort=sort, sort_miss=sort_miss,
            stack100=stack100, horiz=horiz, fill=fill, theme=theme,
            labels=labels, labels_size=labels_size, gap=gap,
            labels_cut=labels_cut,
            xlab=xlab, ylab=ylab, main=main,
            legend_title=legend_title, n_col=n_col, n_row=n_row,
            axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
            rotate_x=rotate_x, rotate_y=rotate_y, quiet=quiet)

    # paired dot chart: x=labels, y=[col1, col2, ...]. Displayed
    # directly when x identifies the cases; with a repeated x each
    # column is aggregated over x by stat, which is then required.
    # R analog: Chart.R paired-dot path and its multi-column y stat
    if isinstance(y, (list, tuple)):
        x_ser = _get_column(data, x, "x")
        y_cols = [_get_column(data, yi, "y") for yi in y]
        sub = pd.concat([x_ser] + y_cols, axis=1).dropna()
        ydf = sub[list(y)]
        y_lbl = " & ".join(y)
        if sub[x].duplicated().any():
            if stat is None:
                raise ValueError(
                    "The data are not a summary (pivot) table, and "
                    f"you have numerical variables, y = {y_lbl}, so "
                    "specify stat to define the aggregation, e.g., "
                    'stat="mean"')
            if stat not in STAT_FUN and stat != "deviation":
                raise ValueError(f"stat must be one of {_STATS}")
            grp = ydf.groupby(sub[x], observed=True, sort=True)
            if stat == "deviation":
                ydf = grp.mean()
                ydf = ydf - ydf.mean()
            else:
                ydf = STAT_FUN[stat](grp)
            ydf = ydf.reindex([c for c in _category_order(sub[x])
                               if c in ydf.index])
            cats = [str(c) for c in ydf.index]
            val_lab = f"{STAT_LBL[stat]} of {y_lbl}"
        else:
            if stat is not None:
                raise ValueError(
                    "The data are a summary table, so do not specify "
                    "stat: the aggregation has already been done")
            cats = sub[x].astype(str).tolist()
            val_lab = y_lbl
        x_lab_dot = (xlab if xlab else ylab if ylab else val_lab)
        return _dot_paired(
            cats, ydf.reset_index(drop=True), sort, sort_miss,
            origin_x, show_diff=not resolve_quiet(quiet), x_name=x,
            fill=fill, border=color, pt_size=pt_size,
            x_lab=x_lab_dot, y_lab=x,
            digits_d=2 if digits_d is None else digits_d,
            pt_opacity=1 - (get_option("trans_pt_fill", 0.10)
                            if transparency is None
                            else transparency),
            main=None if not main else main,
            legend_title=legend_title,
            axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
            rotate_x=rotate_x, rotate_y=rotate_y,
            segments_x=(True if segments_x is None
                        else segments_x),
        )

    x_ser = _get_column(data, x, "x")
    by_ser = _get_column(data, by, "by") if by is not None else None
    by_more_sers = [_get_column(data, b, "by") for b in by_more]
    y_ser = _get_column(data, y, "y") if y is not None else None

    # multiple facet variables define the panel grid, not extra
    # levels: flatten to one " / " interaction, as in R
    if facet is None:
        fac_cols, facet_name = None, None
    elif isinstance(facet, (list, tuple)):
        fac_cols = [_get_column(data, f, "facet") for f in facet]
        facet_name = ", ".join(facet)
    elif isinstance(facet, str):
        fac_cols = [_get_column(data, facet, "facet")]
        facet_name = facet
    else:
        # computed facet values: a Series/array aligned with data,
        # the Python analog of R's facet expression
        ser = facet_values(data, facet, "facet")
        fac_cols = [ser]
        facet_name = str(ser.name)

    # count missing x before the removal below, or it is always 0
    #   R analog: Chart.R n.miss.x
    n_miss_x = int(x_ser.isna().sum())

    # casewise deletion over the variables in the analysis
    used = [s for s in (x_ser, by_ser, y_ser) if s is not None]
    used += by_more_sers
    used += fac_cols or []
    keep = ~pd.concat(used, axis=1).isna().any(axis=1)
    x_ser = x_ser[keep]
    by_more_sers = [b[keep] for b in by_more_sers]
    if by_ser is not None:
        by_ser = by_ser[keep]
    if y_ser is not None:
        y_ser = y_ser[keep]
    if fac_cols is None:
        facet_ser = None
    elif len(fac_cols) == 1:
        facet_ser = fac_cols[0][keep]
    else:
        facet_ser = fac_cols[0][keep].astype(str)
        for c in fac_cols[1:]:
            facet_ser = facet_ser + " / " + c[keep].astype(str)

    x_order = _category_order(x_ser)
    by_order = _category_order(by_ser) if by_ser is not None else None

    # A numeric x with many distinct values yields one category per
    #   value, usually a distribution in disguise. Advisory only: no
    #   count of levels reliably separates categorical from
    #   continuous, so the call is honored. R analog: Chart.R n.x > 12
    if (y_ser is None and pd.api.types.is_numeric_dtype(x_ser)
            and len(x_order) > 12):
        print(f">>> {x} is numeric with {len(x_order)} distinct "
              "values, so each\n    becomes its own category. For "
              "the distribution of a continuous\n    variable:  "
              f'X("{x}")\n')
    facet_order = (_category_order(facet_ser)
                   if facet_ser is not None else None)

    # theme=: a sequential palette in the theme's hue over the fill
    # elements (the by groups if present, else the x categories)
    if theme is not None and fill is None:
        n_fill = len(by_order) if by_order is not None \
            else len(x_order)
        fill = _theme_fill(theme, n_fill)

    # explicit panel-grid layout: n_col wins, else derive from
    # n_row; None keeps each renderer's own default (single-column
    # Trellis bar; up-to-3-column grid for radar/bubble/dot/hier)
    n_col_use = None
    if facet_order is not None and (n_row is not None
                                    or n_col is not None):
        if n_col is not None:
            n_col_use = max(1, int(n_col))
        else:
            n_col_use = max(1, math.ceil(len(facet_order)
                                         / int(n_row)))

    # facet panels leave less room for value labels (Chart.R:134)
    if facet_ser is not None and labels_size is None:
        labels_size = 0.85

    # facet= on a plain pie renders the pie grid, one pie per facet
    # level: the facet becomes the grouping, as in R. (Sunburst
    # routing for pie was already decided on by= above.)
    # the grid re-uses by= to group the panels, but the variable is
    # still a facet, so the title reads "across", not "by"
    facet_ttl = None
    if form == "pie" and hier_type is None and facet_ser is not None:
        facet_ttl = facet_name
        by = facet_name
        by_ser = facet_ser
        by_order = facet_order
        facet_ser = None
        facet_name = None
        facet_order = None

    # radar polygons need >= 3 axes and, with by, >= 2 groups with
    # every cell occupied. R analog: Chart.R radar checks
    if form == "radar":
        if len(x_order) < 3:
            raise ValueError(
                "radar: the categorical variable x must have at "
                "least 3 levels to form a polygon. Found "
                f"{len(x_order)} levels for {x}.")
        if by_ser is not None:
            if len(by_order) < 2:
                raise ValueError(
                    "radar: the by variable must have at least 2 "
                    "levels to define multiple polygons. Found "
                    f"{len(by_order)} levels for {by}.")
            cells = (pd.crosstab(by_ser, x_ser)
                     .reindex(index=by_order, columns=x_order,
                              fill_value=0))
            n_zero = int((cells == 0).sum().sum())
            if n_zero > 0:
                raise ValueError(
                    "radar: one or more cells are empty, with 0 "
                    f"entries ({n_zero} found). Radar polygons "
                    "assume each group has a value at every axis. "
                    "Use a larger sample, reduce the number of "
                    "levels, or choose a different chart.")

    # ----- pre-aggregated? (each x [,by] combination unique) ------
    # R analog: is.agg logic in Chart.R
    n_x = x_ser.nunique()
    if by_ser is None:
        is_agg = n_x >= len(x_ser)
    elif by_more_sers:
        # each cell of the nesting appears once in a summary table
        cells = pd.concat([x_ser, by_ser] + by_more_sers, axis=1)
        is_agg = not cells.astype(str).duplicated().any()
    else:
        is_agg = n_x * by_ser.nunique() >= len(by_ser)

    # a dot chart of counts needs repeated categories to count;
    # unique-x data needs the values themselves
    if form == "dot" and y_ser is None and is_agg:
        raise ValueError("Need to specify a numerical variable "
                         "for y, y='NAME'")

    if y_ser is not None:
        # y is the numerical variable the bars measure. The second
        # positional argument is y, as in R, so name a second
        # categorical variable explicitly with by=.
        if not is_numeric_dtype(y_ser):
            raise ValueError(
                f"y='{y}' is not a numerical variable, and y is the "
                "variable the chart aggregates. To stratify by a "
                f"second categorical variable, name it: by='{y}'")
        if is_agg and stat is not None:
            raise ValueError(
                "The data are a summary table, so do not specify "
                "stat: the aggregation has already been done")
        if not is_agg and stat is None:
            raise ValueError(
                "The data are not a summary (pivot) table, and you "
                f"have a numerical variable y='{y}', so specify "
                'stat to define the aggregation, e.g., stat="mean"')

    # accompanying statistics: frequencies, or the stat of y
    if not resolve_quiet(quiet):
        y_out = (y_ser if y_ser is not None
                 and stat in STAT_FUN else None)
        if by_more_sers and (y_ser is None or y_out is not None):
            from .stats_out import nested_stats
            print("\n".join(nested_stats(
                [x_ser, by_ser] + by_more_sers, [x, by] + by_more,
                y_out, stat, y,
                2 if digits_d is None else digits_d)))
        elif y_ser is None or y_out is not None:
            var_lbl = data.attrs.get("variable_labels", {}) or {}
            print("\n".join(chart_stats(
                x_ser, by_ser, y_out, stat, x, by, y,
                2 if digits_d is None else digits_d,
                stack100=stack100, n_miss=n_miss_x,
                x_lbl=var_lbl.get(x), y_lbl=var_lbl.get(y),
                facet_ser=facet_ser, facet_name=facet_name,
                form=form)))

    # labels default for aggregated data; for counts leave labels
    # None so bc_plotly shows the value with % lines in hover
    # R analog: Chart.R line ~716
    if labels is None and (stat is not None or
                           (is_agg and y_ser is not None)):
        # the bars carry a statistic, so the label is its value;
        #   beside only arranges the bars, and a percentage of a sum
        #   of means is not a quantity
        labels = "input"

    # ----- hierarchical forms aggregate per nesting level ---------
    # from the raw columns, not from the flat table built below
    if hier_type is not None:
        if digits_d is None:
            digits_d = 0 if y_ser is None else 2
        agg = hier_aggregate(x_ser,
                             by=([by_ser] + by_more_sers if by_more
                                 else by_ser),
                             y=y_ser, stat=stat,
                             facet=facet_ser,
                             x_name=x,
                             by_name=[by] + by_more if by_more else by,
                             y_name=y,
                             facet_name=facet_name,
                             facet_order=facet_order)
        fill_vec = hier_color_resolve(x_order, fill, x=x_ser,
                                      y=y_ser, stat=stat)
        if main is None:
            main = build_title(x, by_name=by_label, y_name=y,
                               stat=stat, facet_name=facet_name)
        elif main == "":
            main = None
        # Chart resolves the labels default before hier sees it:
        # "%" for counts (match.arg, Chart.R line 233); the shared
        # logic above already switched stat/pre-aggregated data to
        # "input". Parents carry value 0, so "%"/texttemplate modes
        # keep the inner rings meaningful.
        return hier_plotly(
            agg, fill_vec, type=hier_type,
            x_name=x, by_name=by, facet_name=facet_name,
            main=main, border=color, digits_d=digits_d,
            labels="%" if labels is None else labels,
            labels_color=("white" if labels_color is None
                          else labels_color),
            labels_size=(0.75 if labels_size is None
                         else labels_size),
            n_col=n_col_use,
        )

    # ----- faceted forms: one panel per facet level ----------------
    # (hier forms handled above; pie facet became the by grouping)

    if facet_ser is not None and form == "bar":
        # Trellis bar chart: horizontal bars per panel, the plotly
        # port of R's .bar.lattice rendering. Counts of x, or the
        # stat of y aggregated within each panel; a by variable
        # divides each bar into its levels, as the bar chart of a
        # single panel does, drawn within every panel
        agg_y = y_ser is not None and stat is not None
        if y_ser is not None and not agg_y and not is_agg:
            raise ValueError(
                "The data are not a summary (pivot) table, so "
                'specify stat to define the aggregation, e.g., '
                'stat="mean"')

        def cell_tbl(m):
            xs = x_ser[m]
            if by_ser is None:
                if y_ser is None:
                    t = xs.groupby(xs, observed=True).size()
                elif agg_y:
                    t = STAT_FUN[stat](y_ser[m].groupby(xs,
                                                        observed=True))
                else:
                    t = y_ser[m].groupby(xs, observed=True).first()
                return t.reindex(x_order)
            df = pd.DataFrame({"b": by_ser[m], "x": xs,
                               "y": 1.0 if y_ser is None
                               else y_ser[m].astype(float)})
            g = df.groupby(["b", "x"], observed=True)["y"]
            t = (g.sum() if y_ser is None
                 else STAT_FUN[stat](g) if agg_y else g.first())
            t = t.unstack("x").reindex(index=by_order,
                                       columns=x_order)
            if y_ser is None:
                t = t.fillna(0)
            if stack100:
                # each bar scaled to its own total compares
                #   composition rather than magnitude
                tot = t.sum(axis=0)
                t = t.div(tot.where(tot != 0), axis=1)
            return t

        tbls = {lv: cell_tbl(facet_ser == lv) for lv in facet_order}
        if by_ser is None:
            tbl = pd.DataFrame([tbls[lv] for lv in facet_order],
                               index=facet_order)
            by_tbls = None
        else:
            # the panel totals stand in for the shape of the grid
            tbl = pd.DataFrame([tbls[lv].sum(axis=0)
                                for lv in facet_order],
                               index=facet_order)
            by_tbls = tbls
        tbl.index.name = facet_name
        tbl.columns.name = x
        if stack100:
            val_name = f"Proportion within {x}"
        elif agg_y:
            val_name = f"{STAT_LBL[stat]} of {y}"
        elif y_ser is not None:
            val_name = y
        elif by_ser is not None:
            val_name = "Count"
        else:
            val_name = None             # Count/Proportion of x
        return bc_facet_plotly(
            tbl, x_name=x, facet_name=facet_name,
            x_lab=xlab, y_lab=ylab, fill=fill,
            border="off" if color is None else color,
            opacity=(None if transparency is None
                     else 1 - transparency),
            proportion=stat_x == "proportion" and by_ser is None,
            digits_d=(digits_d if digits_d is not None
                      else 2 if (y_ser is not None or stack100
                                 or stat_x != "count") else 0),
            n_col=n_col_use or 1,
            axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
            rotate_x=rotate_x, rotate_y=rotate_y,
            main=main,
            by_tbls=by_tbls, by_name=by, beside=beside,
            val_name=val_name, legend_title=legend_title,
        )

    if facet_ser is not None and form == "radar":
        if stat_x == "proportion" and by_ser is not None:
            raise NotImplementedError(
                'stat_x="proportion" with by= is not yet ported')
        facets = {
            str(lv): _facet_table(
                x_ser[facet_ser == lv],
                None if y_ser is None else y_ser[facet_ser == lv],
                None if by_ser is None else by_ser[facet_ser == lv],
                stat, is_agg, x_order, by_order,
                proportion=stat_x == "proportion")
            for lv in facet_order
        }
        cap = {"mean": "Mean", "sum": "Sum", "median": "Median",
               "min": "Min", "max": "Max", "sd": "SD"}
        if y_ser is None:
            val_label = "Proportion" if stat_x == "proportion" \
                else "Count"
        elif stat is not None:
            val_label = f"{cap[stat]} {y}"
        else:
            val_label = y
        trans_use = (0.4 if transparency is None and by is not None
                     else (transparency or 0))
        if digits_d is None:
            digits_d = (2 if (y_ser is not None
                              or stat_x == "proportion") else 0)
        if main is None:
            main = build_title(
                x, by_name=by,
                y_name=("Proportion" if stat_x == "proportion"
                        else y),
                stat=stat, facet_name=facet_name)
        elif main == "":
            main = None
        return radar_plotly(
            facets=facets, facet_name=facet_name,
            x_name=x, by_name=by, main=main,
            fill=fill, opacity=1 - trans_use,
            digits_d=digits_d, val_label=val_label,
            n_col=n_col_use,
        )

    if facet_ser is not None and form == "bubble":
        # one 1-D panel per facet level, all on the full category
        # set so the panels stay aligned (by= was rejected above)
        facet_tbls = {
            str(lv): _facet_table(
                x_ser[facet_ser == lv],
                None if y_ser is None else y_ser[facet_ser == lv],
                None, stat, is_agg, x_order, None,
                proportion=stat_x == "proportion")
            for lv in facet_order
        }
        prop = stat_x == "proportion"
        if digits_d is None:
            digits_d = 2 if (y_ser is not None or prop) else 0
        y_name_b = ("Proportion" if prop
                    else ("Count" if y is None else y))
        if main is None:
            main = build_title(x, y_name=y_name_b,
                               stat=stat, facet_name=facet_name)
        elif main == "":
            main = None
        return bubble_plotly(
            x_name=x, y_name=y_name_b,
            x_lab=x if xlab is None else xlab,
            main=main, fill=fill,
            border="black" if color is None else color,
            opacity=(None if transparency is None
                     else 1 - transparency),
            power=power, radius=radius,
            digits_d=digits_d,
            labels="input" if (prop and labels is None) else labels,
            labels_position=labels_position,
            labels_color=labels_color,
            labels_size=labels_size,
            facet_tbls=facet_tbls, facet_name=facet_name,
            n_col=n_col_use,
        )

    if facet_ser is not None and form == "profile":
        # one profile per group within every panel, on the categories
        # and value scale all panels share. Without y the value is a
        # count; with y and no stat the values are supplied directly,
        # one per cell. R analog: Chart.R faceted profile, drawn by
        # .plt.profile.facet() in base R; here a plotly grid
        agg_y = not is_agg and stat is not None
        if y_ser is not None and not agg_y and not is_agg:
            raise ValueError(
                "The data are not a summary (pivot) table, so "
                'specify stat to define the aggregation, e.g., '
                'stat="mean"')
        series = ([str(b) for b in by_order] if by_ser is not None
                  else [None])
        x_lv = [str(c) for c in x_order]
        pos = {c: k + 1 for k, c in enumerate(x_lv)}
        df = pd.DataFrame({
            "f": facet_ser.astype(str), "x": x_ser.astype(str),
            "g": (by_ser.astype(str) if by_ser is not None
                  else "\0"),
            "y": 1.0 if y_ser is None else y_ser.astype(float)})
        g = df.groupby(["f", "g", "x"], observed=True)["y"]
        t = (g.sum() if y_ser is None
             else STAT_FUN[stat](g) if agg_y else g.mean())
        cells = {}
        for (f, gg, xx), v in t.items():
            nm = None if by_ser is None else gg
            xs, ys = cells.setdefault(f, {}).setdefault(nm, ([], []))
            xs.append(pos[xx])
            ys.append(float(v))
        for f in cells:                 # draw left to right
            for nm, (xs, ys) in cells[f].items():
                o = np.argsort(xs)
                cells[f][nm] = ([xs[k] for k in o], [ys[k] for k in o])
        vals = t.to_numpy(dtype=float)
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        # the panels share one value scale, so the origin widens it
        if origin_y is not None:
            lo, hi = min(lo, origin_y), max(hi, origin_y)
        y_tick = pretty(lo, hi)
        if digits_d is None:
            digits_d = 0 if y_ser is None else 2
        from .plotly_utils import axis_format
        y_lab = (ylab if ylab is not None
                 else "Count" if y_ser is None
                 else f"{STAT_LBL[stat]} of {y}" if agg_y else y)
        return profile_facet_plotly(
            cells, x_lv, series, y_tick,
            axis_format(y_tick, digits_d, axis_fmt, axis_y_pre),
            connect=True if segments is None else bool(segments),
            x_lab=x if xlab is None else xlab, y_lab=y_lab,
            by_name=by,
            fill=(fill if isinstance(fill, (list, tuple))
                  else [fill] if fill is not None
                  else _profile_fill(theme)),
            pt_size=1.5 * pt_size,
            facet_levels=[str(f) for f in facet_order],
            facet_name=facet_name, n_col=n_col_use,
            main=None if not main else main, rotate_x=rotate_x)

    if facet_ser is not None and form == "dot":
        # aggregate and sort per facet panel; a panel shows only
        # its own categories. With by=, one series per level of by in
        # every panel, on the full x-by-panel grid so a combination
        # absent from the data is an empty cell. R analog: Chart.R
        # faceted dot prep and its by reshape
        prop = stat_x == "proportion"
        agg_y = not is_agg and stat is not None
        cats_l, vals_l, fac_l = [], [], []
        if by_ser is not None:
            if prop:
                raise ValueError(
                    'stat_x="proportion" with by= is not yet ported')
            rows = []
            for lv in facet_order:
                m = facet_ser == lv
                df = pd.DataFrame({"x": x_ser[m].astype(str),
                                   "by": by_ser[m].astype(str),
                                   "y": (1.0 if y_ser is None
                                         else y_ser[m].astype(float))})
                g = df.groupby(["x", "by"], observed=True)["y"]
                t = (g.sum() if y_ser is None
                     else STAT_FUN[stat](g) if agg_y else g.first())
                wide = (t.unstack("by")
                        .reindex(index=[str(c) for c in x_order],
                                 columns=[str(b) for b in by_order]))
                rows.append(wide)
                cats_l += list(wide.index)
                fac_l += [str(lv)] * len(wide)
            vals_l = pd.concat(rows, ignore_index=True)
            all_vals = vals_l.to_numpy(dtype=float).ravel()
        else:
            for lv in facet_order:
                m = facet_ser == lv
                xs = x_ser[m].astype(str)
                if y_ser is None:                # counts per panel
                    t = xs.groupby(xs).size()
                    if prop:
                        tot = t.sum()
                        t = t / tot if tot > 0 else t.astype(float)
                elif agg_y:
                    t = STAT_FUN[stat](y_ser[m].groupby(xs))
                else:                            # pre-aggregated rows
                    t = pd.Series(y_ser[m].to_numpy(dtype=float),
                                  index=xs.values)
                if sort != "0":
                    t = t.sort_values(ascending=(sort == "+"),
                                      kind="stable")
                cats_l += [str(c) for c in t.index]
                vals_l += [float(v) for v in t.to_numpy()]
                fac_l += [str(lv)] * len(t)
            all_vals = vals_l

        # an aggregated value is labeled by its statistic, as the
        #   unfaceted chart is: "Mean of Salary", not "Salary"
        if y_ser is None:
            val_lab = ("Count" if by_ser is not None
                       else f"Proportion of {x}" if prop
                       else f"Count of {x}")
        else:
            val_lab = f"{STAT_LBL[stat]} of {y}" if agg_y else y
        if horiz:
            orientation = "h"
            x_lab_arg = (xlab if xlab is not None
                         else ylab if ylab is not None else val_lab)
            y_lab_arg = x
        else:
            orientation = "v"
            x_lab_arg = xlab if xlab is not None else x
            y_lab_arg = ylab if ylab is not None else val_lab

        # the length of the segment carries the value, so the value
        #   axis begins at zero unless an origin is specified
        org_in = origin_x if horiz else origin_y
        origin, gridT = _dot_origin_grid(
            all_vals, origin_in=0 if org_in is None else org_in,
            is_counts=True if y_ser is None else None)
        if digits_d is None:
            digits_d = 2 if (y_ser is not None or prop) else 0
        return dot_plotly(
            cats_l, vals_l, orientation=orientation,
            fill=fill, border=color, pt_size=pt_size,
            x_lab=x_lab_arg, y_lab=y_lab_arg,
            digits_d=digits_d,
            pt_opacity=1 - (get_option("trans_pt_fill", 0.10)
                            if transparency is None
                            else transparency),
            gridT=gridT, origin_x=origin,
            main=None if not main else main,
            legend_title=(legend_title if legend_title is not None
                          else by),
            facet=fac_l, facet_name=facet_name, n_col=n_col_use,
            axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
            axis_y_pre=axis_y_pre,
            rotate_x=rotate_x, rotate_y=rotate_y,
            segments_x=True if segments_x is None else segments_x,
            segments_y=True if segments_y is None else segments_y,
        )

    # user-supplied axis label, before the pipeline computes a
    # default into ylab (the bubble matrix labels its y-axis with
    # the by variable, not the value label)
    ylab_user = ylab
    digits_d_user = digits_d

    # ----- build the table the renderer receives ------------------
    if y_ser is None:                        # counts of x
        if stat_x == "proportion" and by_ser is not None:
            raise NotImplementedError(
                'stat_x="proportion" with by= is not yet ported')

        if by_ser is None:
            tbl = (x_ser.groupby(x_ser).size()
                   .reindex(x_order, fill_value=0))
            if stat_x == "proportion":
                tbl = tbl / tbl.sum()
                y_name = "Proportion"
            else:
                y_name = "Count"
            tbl.index.name = x
            tbl.name = y_name
            if ylab is None:
                ylab = f"{y_name} of {x}"
            if digits_d is None:
                digits_d = 0 if stat_x == "count" else 2
        else:
            tbl = pd.crosstab(by_ser, x_ser)
            tbl = tbl.reindex(index=by_order, columns=x_order,
                              fill_value=0)
            y_name = "Count"
            if ylab is None:
                ylab = f"Count of {x}"
            if digits_d is None:
                digits_d = 0

    elif stat == "deviation":                # R analog: .agg()
        if by_ser is not None:
            raise ValueError("deviation for stat is not meaningful "
                             "with a by variable")
        means = y_ser.groupby(x_ser).mean().reindex(x_order)
        # deviation from the unweighted mean of the group means,
        # so each group counts equally regardless of its n
        tbl = means - means.mean()
        y_name = y
        if ylab is None:
            ylab = f"{STAT_LBL[stat]} of {y}"
        if digits_d is None:
            digits_d = 2

    else:                                    # y with stat (or agg)
        fun = STAT_FUN[stat] if stat is not None else None
        if by_ser is None:
            if is_agg and stat is None:
                tbl = pd.Series(y_ser.values, index=x_ser.values)
                tbl = tbl.reindex(x_order)
            else:
                tbl = fun(y_ser.groupby(x_ser)).reindex(x_order)
        else:
            df = pd.DataFrame({"by": by_ser, "x": x_ser,
                               "y": y_ser})
            if is_agg and stat is None:
                tbl = df.pivot(index="by", columns="x", values="y")
            else:
                tbl = fun(df.groupby(["by", "x"])["y"]).unstack("x")
            tbl = tbl.reindex(index=by_order, columns=x_order)
        # a profile or dot plot of several groups draws an empty
        #   cell as a missing point; the other forms need every cell
        gaps_ok = form in ("profile", "dot") and by_ser is not None
        if not gaps_ok and not np.isfinite(
                tbl.to_numpy(dtype=float)).all():
            raise ValueError(
                "The summary table of the transformed data has "
                "missing or non-finite values, likely because some "
                "cells have too few (or no) data values to compute "
                "the specified statistic")
        y_name = y
        if ylab is None:
            # pre-aggregated data label with the variable name
            # itself, as in R (Chart.R dot path, x.lab.dot)
            ylab = (f"{STAT_LBL[stat]} of {y}"
                    if stat is not None else y)
        if digits_d is None:
            digits_d = 2

    # ----- stack100: rescale each bar to sum to 1.0 ----------------
    # R analog: bc.main.R  x <- prop.table(x, 2), which normalizes
    # within each COLUMN of the by-by-x table, i.e. within each bar.
    # The counts are kept for the value labels, which R displays
    # instead of the proportions when labels="input".
    counts_tbl = None
    if stack100:
        counts_tbl = tbl.copy()
        if isinstance(tbl, pd.DataFrame):
            totals = tbl.sum(axis=0)
            if (totals == 0).any():
                zero = list(totals.index[totals == 0])
                raise ValueError(
                    "stack100 divides each bar by its total, and "
                    f"these categories of {x} have no data: {zero}")
            tbl = tbl.div(totals, axis=1)
        else:
            total = tbl.sum()
            if total == 0:
                raise ValueError("stack100 divides each bar by its "
                                 "total, and all counts are zero")
            tbl = tbl / total
        y_name = "Proportion"
        if ylab_user is None:
            if isinstance(counts_tbl, pd.Series):
                ylab = f"Proportion of {x}"
            elif beside:
                ylab = "Percentage"
            else:
                # the axis holds proportions, so it is named for them;
                #   the legend names by, so it is not repeated here
                ylab = f"Proportion within {x}"
        if digits_d_user is None:
            digits_d = 2

    if form != "radar":     # a radar's axis order stays fixed
        tbl = (_sort_1d(tbl, sort) if isinstance(tbl, pd.Series)
               else _sort_2d(tbl, sort))
        if counts_tbl is not None:      # keep the counts aligned
            counts_tbl = (counts_tbl.reindex(tbl.index)
                          if isinstance(tbl, pd.Series)
                          else counts_tbl.reindex(index=tbl.index,
                                                  columns=tbl.columns))

    # labels_decimals passes straight through: each renderer applies
    # its own per-mode default when None, which is what R's plotly
    # path does. R's BASE path instead defaults to 0 for an
    # aggregated y (bc.main.R), so the two differ there, as in R.

    # ----- render --------------------------------------------------
    opacity = None if transparency is None else 1 - transparency

    if form == "bubble":
        if main is None:
            main = build_title(x, by_name=by, y_name=y_name,
                               stat=stat)
        elif main == "":
            main = None
        return bubble_plotly(
            tbl,
            x_name=x, y_name=y_name, by_name=by,
            x_lab=x if xlab is None else xlab,
            y_lab=ylab_user,       # None -> by name (2-D only)
            main=main,
            fill=fill,
            border="black" if color is None else color,
            opacity=opacity,
            power=power, radius=radius,
            digits_d=digits_d,
            labels=labels, labels_position=labels_position,
            labels_color=labels_color,
            labels_decimals=labels_decimals,
            labels_size=0.90 if labels_size is None
                        else labels_size,
        )

    if form == "radar":
        # hover value label, R analog: .radar_aggregate() y.label
        # ("Count", "Mean Salary", or the bare variable name)
        cap = {"mean": "Mean", "sum": "Sum", "median": "Median",
               "min": "Min", "max": "Max", "sd": "SD",
               "deviation": "Deviation"}
        if y_ser is None:
            val_label = y_name              # "Count"/"Proportion"
        elif stat is not None:
            val_label = f"{cap[stat]} {y}"
        else:
            val_label = y
        # groups overlap, so default to translucent fills with by=
        trans_use = (0.4 if transparency is None and by is not None
                     else (transparency or 0))
        if main is None:
            main = build_title(x, by_name=by, y_name=y_name,
                               stat=stat)
        elif main == "":
            main = None
        return radar_plotly(
            tbl,
            x_name=x, by_name=by, main=main,
            fill=fill, opacity=1 - trans_use,
            digits_d=digits_d, val_label=val_label,
        )

    if form == "profile":
        # one point per category of x, connected across the
        #   categories, one profile per level of by: the interaction
        #   plot of the analysis of variance. The connecting segments
        #   are the purpose of the form. R analog: Chart.R profile ->
        #   .plt.main(cat.x=TRUE, segments=TRUE) -> plt.plotly()
        if isinstance(tbl, pd.DataFrame):
            groups = [(str(b), tbl.columns, tbl.loc[b])
                      for b in tbl.index]
        else:
            groups = [(None, tbl.index, tbl)]
        cats = [str(c) for c in groups[0][1]]
        pos = list(range(1, len(cats) + 1))
        # an empty cell stays NaN, which plotly draws as a gap
        groups = [(nm, pos, vals.to_numpy(dtype=float).tolist())
                  for nm, _, vals in groups]
        vals = tbl.to_numpy(dtype=float).ravel()
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        # the value axis of a profile is y, so origin_y sets where
        #   it begins, as it does for the dot chart
        if origin_y is not None:
            lo, hi = min(lo, origin_y), max(hi, origin_y)
        axT2 = pretty(lo, hi)
        from .plotly_utils import axis_format
        y_lab = (ylab_user if ylab_user is not None
                 else "Count" if y_ser is None and stat_x == "count"
                 else ylab)
        fig = plt_plotly(
            groups, by_name=by,
            fill=(fill if isinstance(fill, (list, tuple))
                  else [fill] if fill is not None
                  else _profile_fill(theme)),
            pt_size=1.5 * pt_size,
            x_lab=x if xlab is None else xlab, y_lab=y_lab,
            ax={"axT1": pos, "axL1": cats, "axT2": axT2,
                "axL2": axis_format(axT2, digits_d, axis_fmt,
                                    axis_y_pre)},
            gridT1=pos, gridT2=axT2,
            # no automatic title: it would only repeat the axis labels
            main=None if not main else main,
            digits_d=digits_d,
            connect=True if segments is None else bool(segments),
            pt_opacity=1,
            style_opts=None)
        pad = 0.04 * (axT2[-1] - axT2[0])
        fig.update_yaxes(range=[axT2[0] - pad, axT2[-1] + pad])
        fig.update_xaxes(range=[0.5, len(cats) + 0.5])
        if rotate_x:
            fig.update_xaxes(tickangle=-rotate_x)
        if rotate_y:
            fig.update_yaxes(tickangle=-rotate_y)
        return fig

    if form == "dot":
        quiet_use = resolve_quiet(quiet)
        pt_op = 1 - (get_option("trans_pt_fill", 0.10)
                     if transparency is None else transparency)
        if isinstance(tbl, pd.DataFrame):
            # by=: one series of dots per level of by, drawn as the
            #   paired geometry, one column per level. A two-level by
            #   with a stat already lists its difference as the Diff
            #   row of the summary table. R analog: Chart.R dot by
            #   reshape into the paired path
            ydf = tbl.T.reindex([c for c in x_order if c in tbl.columns])
            ydf.columns = [str(c) for c in ydf.columns]
            cats = [str(c) for c in ydf.index]
            val_lab = (xlab if xlab else ylab_user if ylab_user
                       else "Count" if y_ser is None else ylab)
            return _dot_paired(
                cats, ydf.reset_index(drop=True), sort, sort_miss,
                origin_x,
                show_diff=not quiet_use and y_ser is None,
                x_name=x,
                fill=fill, border=color, pt_size=pt_size,
                x_lab=val_lab, y_lab=x, digits_d=digits_d,
                pt_opacity=pt_op,
                main=None if not main else main,
                legend_title=(legend_title if legend_title is not None
                              else by),
                axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
                rotate_x=rotate_x, rotate_y=rotate_y,
                segments_x=(True if segments_x is None
                            else segments_x),
            )

        cats = [str(c) for c in tbl.index]
        vals = tbl.to_numpy(dtype=float)
        # the length of the segment carries the value, so the value
        #   axis begins at zero unless an origin is specified
        org_in = origin_x if horiz else origin_y
        origin, gridT = _dot_origin_grid(
            vals, origin_in=0 if org_in is None else org_in,
            is_counts=True if y_ser is None else None)
        val_lab = ylab                 # set with tbl above
        if horiz:
            # the value axis takes the plotted quantity, from xlab or
            #   the statistic; the category axis takes x's own name
            x_lab_arg = xlab if xlab is not None else val_lab
            y_lab_arg = x
        else:
            x_lab_arg = xlab if xlab is not None else x
            y_lab_arg = val_lab
        # the value axis already names the plotted quantity, so an
        #   unrequested title would only repeat it
        return dot_plotly(
            cats, vals,
            orientation="h" if horiz else "v",
            fill=fill, border=color,
            pt_size=pt_size,
            x_lab=x_lab_arg, y_lab=y_lab_arg,
            digits_d=digits_d,
            pt_opacity=pt_op,
            gridT=gridT, origin_x=origin,
            main=None if not main else main,
            axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
            axis_y_pre=axis_y_pre,
            rotate_x=rotate_x, rotate_y=rotate_y,
            segments_x=True if segments_x is None else segments_x,
            segments_y=True if segments_y is None else segments_y,
        )

    if form == "pie":
        # R analog: .build_chart_title() — auto-title unless the
        # caller supplies main; main="" suppresses the title
        if main is None:
            main = build_title(
                x, by_name=None if facet_ttl else by,
                facet_name=facet_ttl, y_name=y_name, stat=stat)
        elif main == "":
            main = None
        return pie_plotly(
            tbl,
            x_name=x, y_name=y_name, by_name=by, main=main,
            fill=fill,
            border="transparent" if color is None else color,
            opacity=1.0 if opacity is None else opacity,
            hole=hole,
            labels=labels, labels_position=labels_position,
            labels_color=labels_color,
            labels_decimals=labels_decimals,
            labels_size=1.0 if labels_size is None else labels_size,
            digits_d=digits_d,
        )

    # value-keyed bar fills: a two-color split (fill_split) or an
    # HCL gradient by distance from the split (fill_scaled). 1-D
    # bars only (a Series), as in R.
    if fill_scaled and by is not None:
        raise ValueError("fill_scaled applies only without a by "
                         "variable")
    if (fill_split is not None or fill_scaled) \
            and isinstance(tbl, pd.Series):
        from .getColors import bar_fill_colors
        fill = bar_fill_colors(tbl.to_numpy(dtype=float), fill,
                               fill_split, fill_scaled, fill_chroma)

    fig = bc_plotly(
        tbl,
        x_name=x, y_name=y_name, by_name=by,
        x_lab=xlab if xlab is not None else x,
        y_lab=ylab,
        fill=fill,
        border="off" if color is None else color,
        opacity=opacity,
        beside=beside, horiz=horiz,
        counts=counts_tbl,
        gap=gap, scale_y=scale_y, break_x=break_x,
        digits_d=digits_d, main=main,
        rotate_x=rotate_x, rotate_y=rotate_y,
        axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
        axis_y_pre=axis_y_pre,
        labels=labels, labels_position=labels_position,
        labels_size=0.90 if labels_size is None else labels_size,
        labels_color=labels_color,
        labels_decimals=labels_decimals,
        # applied only when given; R cuts by share, which a single
        #   series of a statistic does not carry, nor outside labels
        labels_cut=(None if (labels_position == "out" or (
            isinstance(tbl, pd.Series) and y_ser is not None))
            else labels_cut),
        legend_title=legend_title, legend_position=legend_position,
        legend_labels=legend_labels, legend_horiz=legend_horiz,
        legend_size=legend_size, legend_abbrev=legend_abbrev,
        legend_adjust=legend_adjust,
    )
    if add is not None:
        plt_add(fig, add, x1=x1, x2=x2, y1=y1, y2=y2)
    return fig


# font_size= scales all text of the returned figure
Chart = font_scaled(Chart)
