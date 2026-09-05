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
# form="profile" plots one point per level of x and connects them
# across the levels. With by= each group draws its own profile,
# the interaction plot of a two-way design.
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
from .plt_add import plt_add
from .bubble_plotly import bubble_plotly
from .dot_plotly import dot_plotly
from .hier_plotly import (
    hier_aggregate, hier_color_resolve, hier_plotly,
)
from .pie_plotly import pie_plotly
from .profile_plotly import profile_plotly
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
    cols = [c[:7] if len(c) == 9 else c for c in cols]  # drop alpha

    # Two grays carry the whole distinction between the fill elements,
    # with no third value between them to read against, so they are set
    # apart further than the palette spaces them: a luminance span of
    # 102, the span the palette gives three grays, rather than the 82 of
    # #969696 and #444444. R analog: the grays branch of .color_range()
    if pal == "grays" and n == 2:
        cols = ["#B3B3B3", "#4D4D4D"]        # gray70, gray30 in R
    return cols


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

_FORMS = ("bar", "radar", "bubble", "dot", "profile", "pie",
          "icicle", "treemap")
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


def Chart(x, y=None, data=None, filter=None, by=None, facet=None,
          form="bar",
          n_row=None, n_col=None,
          hole=0.65,
          radius=0.50, power=0.5,
          pt_size=1, origin_x=None, origin_y=None,
          segments_x=None, segments_y=None, segments=None,
          stat=None, stat_x="count",
          horiz=False, sort="0", beside=False, stack100=False,
          gap=None, scale_y=None, break_x=None,
          fill=None, color=None, transparency=None,
          fill_split=None, fill_scaled=False, fill_chroma=75,
          theme=None,
          labels=None, labels_position=None,
          labels_color=None, labels_size=None, labels_decimals=None,
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

    Variables are strings naming columns of the DataFrame `data`.
    Returns a plotly Figure; call .show() to display from a script.
    """

    # ----- validate parameters ------------------------------------
    if form == "sunburst":          # R analog: Chart.R line ~100
        form = "pie"
    if form not in _FORMS:
        raise ValueError(f"form must be one of {_FORMS}")

    # hierarchical routing: treemap/icicle always; pie with by=
    # nests the by rings inside the x wedges as a sunburst
    # R analog: Chart.R hier dispatch
    hier_type = None
    if form in ("treemap", "icicle"):
        hier_type = form
    elif form == "pie" and by is not None:
        hier_type = "sunburst"
    if hier_type is not None and stat == "deviation":
        raise ValueError('stat="deviation" is not meaningful for '
                         "hierarchical charts: negative values "
                         "cannot form part-of-whole areas")

    if form == "dot" and by is not None:
        raise ValueError("The by variable is not meaningful for "
                         "dot charts. Do a bar chart.")
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
        if facet is not None:
            raise ValueError(
                "stack100 rescales each bar within its by groups, "
                "and a faceted bar chart (Trellis) has no by "
                "variable. Use by= instead of facet=.")
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
    if _leg_set:
        if form != "bar":
            raise ValueError(
                f"{', '.join(_leg_set)} style the legend of a "
                'two-variable bar chart, so they need form="bar"')
        if by is None:
            raise ValueError(
                f"{', '.join(_leg_set)} style the legend that a by "
                "variable creates, so they need by=")
        if facet is not None:
            raise ValueError(
                f"{', '.join(_leg_set)} apply to a single-panel bar "
                "chart; a faceted (Trellis) bar chart has no by "
                "variable and no legend")

    if segments is not None and form != "profile":
        raise ValueError(
            'segments connects the points of a profile across the '
            'levels of x, so it needs form="profile". For the dot '
            "chart's stems, see segments_x and segments_y")
    if form == "profile":
        if facet is not None:
            raise ValueError(
                "the faceted profile is not yet ported; use by= "
                "for one profile per group in a single panel")
        if stack100 or beside or horiz:
            raise ValueError(
                "stack100, beside, and horiz shape the bars of a "
                'bar chart, so they do not apply to form="profile"')
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
            if stat is not None:
                raise ValueError(
                    "Only the original data work with Trellis "
                    "plots, no data aggregation with parameter "
                    "stat. Use by instead of facet.")
            if y is not None:
                raise ValueError(
                    "The faceted bar chart displays counts of x, "
                    "so a y variable does not apply. Use by "
                    "instead of facet.")
            if by is not None:
                raise ValueError(
                    "by and facet do not combine for bar charts: "
                    "each facet panel displays the counts of x. "
                    "Use one or the other.")

    # ----- resolve variables and filter ---------------------------
    if filter is not None:
        data = data.query(filter)

    # paired dot chart: x=labels, y=[col1, col2, ...], displayed
    # directly, never aggregated. R analog: Chart.R paired-dot path
    if isinstance(y, (list, tuple)):
        if stat is not None:
            raise ValueError("stat does not apply to a paired dot "
                             "chart; the values display directly")
        x_ser = _get_column(data, x, "x")
        y_cols = [_get_column(data, yi, "y") for yi in y]
        sub = pd.concat([x_ser] + y_cols, axis=1).dropna()
        cats = sub[x].astype(str).tolist()
        ydf = sub[list(y)]

        # sort= orders rows by row mean across all series
        if sort != "0":
            order = (ydf.mean(axis=1)
                     .sort_values(ascending=(sort == "+")).index)
            cats = sub.loc[order, x].astype(str).tolist()
            ydf = ydf.loc[order]

        origin, gridT = _dot_origin_grid(
            ydf.to_numpy(dtype=float).ravel(), origin_in=origin_x)
        if origin is None:
            origin = 0
        if main is None:
            main = build_title(x, y_name=" & ".join(y))
        elif main == "":
            main = None
        return dot_plotly(
            cats, ydf,
            pt_size=pt_size,
            x_lab="" if xlab is None else xlab, y_lab=x,
            digits_d=2 if digits_d is None else digits_d,
            pt_opacity=1 - (get_option("trans_pt_fill", 0.10)
                            if transparency is None
                            else transparency),
            gridT=gridT, origin_x=origin, main=main,
            axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
            rotate_x=rotate_x, rotate_y=rotate_y,
            segments_x=(True if segments_x is None
                        else segments_x),
        )

    x_ser = _get_column(data, x, "x")
    by_ser = _get_column(data, by, "by") if by is not None else None
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

    # casewise deletion over the variables in the analysis
    used = [s for s in (x_ser, by_ser, y_ser) if s is not None]
    used += fac_cols or []
    keep = ~pd.concat(used, axis=1).isna().any(axis=1)
    x_ser = x_ser[keep]
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
    # A frequency table alphabetizes its categories, which is what
    # category_order gives a plain column. The profile's segments
    # assert an ordering of that axis, so the categories keep the
    # order the data present them in, as they would were the column
    # a Categorical. R analog: Chart.R form="profile", x.lev
    if form == "profile" and not isinstance(x_ser.dtype,
                                            pd.CategoricalDtype):
        x_order = list(pd.unique(x_ser.dropna()))
    by_order = _category_order(by_ser) if by_ser is not None else None
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
        if y_ser is None or y_out is not None:
            print("\n".join(chart_stats(
                x_ser, by_ser, y_out, stat, x, by, y,
                2 if digits_d is None else digits_d)))

    # labels default for aggregated data; for counts leave labels
    # None so bc_plotly shows the value with % lines in hover
    # R analog: Chart.R line ~716
    if labels is None and (stat is not None or
                           (is_agg and y_ser is not None)):
        labels = "%" if beside else "input"

    # ----- hierarchical forms aggregate per nesting level ---------
    # from the raw columns, not from the flat table built below
    if hier_type is not None:
        if digits_d is None:
            digits_d = 0 if y_ser is None else 2
        agg = hier_aggregate(x_ser, by=by_ser, y=y_ser, stat=stat,
                             facet=facet_ser,
                             x_name=x, by_name=by, y_name=y,
                             facet_name=facet_name,
                             facet_order=facet_order)
        fill_vec = hier_color_resolve(x_order, fill)
        if main is None:
            main = build_title(x, by_name=by, y_name=y, stat=stat,
                               facet_name=facet_name)
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
        # Trellis bar chart: horizontal count bars per panel,
        # the plotly port of R's .bar.lattice rendering
        tbl = (pd.crosstab(facet_ser, x_ser)
               .reindex(index=facet_order, columns=x_order,
                        fill_value=0))
        tbl.index.name = facet_name
        tbl.columns.name = x
        return bc_facet_plotly(
            tbl, x_name=x, facet_name=facet_name,
            x_lab=xlab, y_lab=ylab, fill=fill,
            border="off" if color is None else color,
            opacity=(None if transparency is None
                     else 1 - transparency),
            proportion=stat_x == "proportion",
            digits_d=(digits_d if digits_d is not None
                      else (0 if stat_x == "count" else 2)),
            n_col=n_col_use or 1,
            axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
            rotate_x=rotate_x, rotate_y=rotate_y,
            main=main,
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

    if facet_ser is not None and form == "dot":
        # aggregate and sort per facet panel; a panel shows only
        # its own categories. R analog: Chart.R faceted dot prep
        prop = stat_x == "proportion"
        cats_l, vals_l, fac_l = [], [], []
        for lv in facet_order:
            m = facet_ser == lv
            xs = x_ser[m].astype(str)
            if y_ser is None:                # counts per panel
                t = xs.groupby(xs).size()
                if prop:
                    tot = t.sum()
                    t = t / tot if tot > 0 else t.astype(float)
            elif not is_agg and stat is not None:
                t = STAT_FUN[stat](y_ser[m].groupby(xs))
            else:                            # pre-aggregated rows
                t = pd.Series(y_ser[m].to_numpy(dtype=float),
                              index=xs.values)
            if sort != "0":
                t = t.sort_values(ascending=(sort == "+"))
            cats_l += [str(c) for c in t.index]
            vals_l += [float(v) for v in t.to_numpy()]
            fac_l += [str(lv)] * len(t)

        val_lab = (y if y is not None
                   else ("Proportion of " + x if prop
                         else f"Count of {x}"))
        if horiz:
            orientation = "h"
            x_lab_arg = xlab if xlab is not None else val_lab
            y_lab_arg = ylab if ylab is not None else x
        else:
            orientation = "v"
            x_lab_arg = xlab if xlab is not None else x
            y_lab_arg = ylab if ylab is not None else val_lab

        origin, gridT = _dot_origin_grid(
            vals_l,
            origin_in=origin_x if horiz else origin_y,
            is_counts=True if y_ser is None else None)
        if digits_d is None:
            digits_d = 2 if (y_ser is not None or prop) else 0
        if main is None:
            main = build_title(
                x, y_name="Proportion" if prop else y,
                stat=stat, facet_name=facet_name)
        elif main == "":
            main = None
        return dot_plotly(
            cats_l, vals_l, orientation=orientation,
            fill=fill, border=color, pt_size=pt_size,
            x_lab=x_lab_arg, y_lab=y_lab_arg,
            digits_d=digits_d,
            pt_opacity=1 - (get_option("trans_pt_fill", 0.10)
                            if transparency is None
                            else transparency),
            gridT=gridT, origin_x=origin, main=main,
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
        if not np.isfinite(tbl.to_numpy(dtype=float)).all():
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
                ylab = f"Cell % within {x} by {by}"
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

    if form == "dot":
        # tbl is always a Series here (by= was rejected above)
        cats = [str(c) for c in tbl.index]
        vals = tbl.to_numpy(dtype=float)
        origin, gridT = _dot_origin_grid(
            vals,
            origin_in=origin_x if horiz else origin_y,
            is_counts=True if y_ser is None else None)
        cat_lab = x if xlab is None else xlab
        val_lab = ylab                 # set with tbl above
        if main is None:
            main = build_title(x, y_name=y_name, stat=stat)
        elif main == "":
            main = None
        return dot_plotly(
            cats, vals,
            orientation="h" if horiz else "v",
            fill=fill, border=color,
            pt_size=pt_size,
            x_lab=val_lab if horiz else cat_lab,
            y_lab=cat_lab if horiz else val_lab,
            digits_d=digits_d,
            pt_opacity=1 - (get_option("trans_pt_fill", 0.10)
                            if transparency is None
                            else transparency),
            gridT=gridT, origin_x=origin, main=main,
            axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
            axis_y_pre=axis_y_pre,
            rotate_x=rotate_x, rotate_y=rotate_y,
            segments_x=True if segments_x is None else segments_x,
            segments_y=True if segments_y is None else segments_y,
        )

    if form == "profile":
        # tbl: a Series over the x categories, or a DataFrame with
        # one row per by group. The connecting segments are the
        # point of the form, so they are on unless turned off.
        # R analog: Chart.R form="profile" -> .plt.main(cat.x=TRUE)
        if main is None:
            main = build_title(x, by_name=by, y_name=y_name,
                               stat=stat)
        elif main == "":
            main = None
        return profile_plotly(
            tbl,
            x_name=x, by_name=by,
            fill=fill, border=color, pt_size=pt_size,
            segments=True if segments is None else segments,
            main=main,
            x_lab=x if xlab is None else xlab,
            y_lab=ylab,                # set with tbl above
            digits_d=digits_d,
            axis_fmt=axis_fmt, axis_x_pre=axis_x_pre,
            axis_y_pre=axis_y_pre,
            rotate_x=rotate_x, rotate_y=rotate_y,
            transparency=transparency,
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
