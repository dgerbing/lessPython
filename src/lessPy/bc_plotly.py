# bc_plotly.py — analog of bc.plotly.R (.bc.plotly)
#
# Renders a bar chart from ALREADY-TABULATED input, the same division
# of labor as lessR: aggregation happens upstream (later, in Chart()
# and X()); this function only renders.
#
# Input contract (mirrors the R named-vector / table contract):
#   1-D  pandas Series: index = category names, values = counts/means
#   2-D  pandas DataFrame: rows = `by` levels, columns = x categories
#        (the shape produced by pandas.crosstab(by, x))
#
# counts= carries the frequencies behind heights that Chart has
# already rescaled to proportions (stack100), so the hover
# percentages and the labels="input" text report the data.
#
# One deliberate deviation from the R source: in lessR the caller
# computes axis ticks (ax) and grid positions (gridT) upstream. Here,
# when ax is None they are computed internally with utils.pretty(),
# so bc_plotly() is callable on its own until Chart() exists.
#
# gap= and break_x= are rendering controls that R applies only in its
# base-graphics path (.bc.plotly never receives them); scale_y=
# reaches R's plotly through the ax/gridT it computes upstream.

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .utils import fmt, get_option, pretty
from .plotly_utils import (
    BASE_COLORS, as_plotly_color, axis_cat, axis_format, axis_num,
    contrast_text_for_hex, facet_fig, facet_pos, finish_facet,
    abbrev, get_tick_fmt, is_integer_valued, legend_style,
    plot_border, plotly_style, to_hex, x_grid, y_grid,
)

_LABEL_VALUES = ("%", "input", "prop", "off")


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _rep_len(v, n):
    v = _as_list(v)
    return [v[i % len(v)] for i in range(n)]


def _pick(v, name, i):
    """Color for group `name` (position i): by name if v is a dict,
    else recycled by position. R analog: .pick()"""
    if isinstance(v, dict):
        if name in v:
            return v[name]
        v = list(v.values())
    v = _as_list(v)
    if not v:
        return "black"
    return v[i % len(v)]


def _bar_labels(vals, share, labels, digits_d, counts=None,
                labels_decimals=None):
    """vals are the plotted bar heights, share the proportion each
    one represents. counts, supplied when the heights are already
    proportions (Chart's stack100), are the frequencies behind them
    that labels="input" displays in their place.
    R analog: bc.main.R  x.txt <- as.character(x.count)

    labels_decimals sets the decimal places of the label text.
    None keeps each mode's own default, which is what a direct
    call to bc_plotly() gets; Chart() always resolves a value."""
    if labels == "off":
        return ["" for _ in vals]
    if labels == "input":
        src = vals if counts is None else counts
        if labels_decimals is not None:
            d = int(labels_decimals)
        elif counts is None:
            d = digits_d
        else:
            d = 0 if is_integer_valued(counts) else digits_d
        return [fmt(v, d) for v in src]
    if labels == "%":
        d = 0 if labels_decimals is None else int(labels_decimals)
        return [f"{100 * s:.{d}f}%" for s in share]
    d = 2 if labels_decimals is None else int(labels_decimals)
    return [f"{s:.{d}f}" for s in share]         # "prop"


def _break_labels(cats, break_x):
    """Category labels for the categorical axis: a space becomes a
    line break when break_x, and "~" is the non-breaking space that
    survives it. R analog: .get.val.ln() in zzz.R"""
    if break_x:
        return [c.replace(" ", "<br>").replace("~", " ")
                for c in cats]
    return [c.replace("~", " ") for c in cats]


def _bar_gaps(gap, beside, n_groups):
    """R's barplot space= (gaps in multiples of bar width) as
    plotly's bargap / bargroupgap (fractions of a category slot).
    gap=None leaves plotly's own defaults in place.
    R analog: bc.main.R  gap default c(0.1, 1) beside, else 0.2"""
    if gap is None:
        return None, None
    if isinstance(gap, (list, tuple, np.ndarray)):
        within, between = float(gap[0]), float(gap[-1])
    else:
        within = between = float(gap)
    if within < 0 or between < 0:
        raise ValueError("gap cannot be negative")
    if not beside:
        return between / (1.0 + between), None
    # a slot holds n_groups bars, the gaps between them, and one
    # between-group gap; plotly splits what bargap leaves into
    # n_groups equal sub-slots, each bargroupgap of it empty
    k = max(1, n_groups)
    slot = k + (k - 1) * within + between
    return between / slot, (k - 1) * within / (k + (k - 1) * within)


def _decimals(v):
    """Decimal places needed to write v exactly. R analog: .max.dd()"""
    s = f"{float(v):.10f}".rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0


def bc_plotly(x, x_name=None, y_name=None, by_name=None,
              x_lab=None, y_lab=None,
              fill=None, border="off", opacity=None,
              beside=False, horiz=False, ax=None, grid=None,
              counts=None, gap=None, scale_y=None, break_x=False,
              digits_d=None, main=None, rotate_x=0, rotate_y=0,
              axis_fmt="K", axis_x_pre="", axis_y_pre="",
              labels=None, labels_size=0.90, labels_color=None,
              labels_position=None, labels_decimals=None,
              labels_autocontrast=False,
              legend_title=None, legend_position=None,
              legend_labels=None, legend_horiz=False,
              legend_size=None, legend_abbrev=None,
              legend_adjust=0, style_opts=None):

    # R analog: missing(labels); None = caller did not supply labels=
    missing_labels = labels is None
    if not missing_labels and labels not in _LABEL_VALUES:
        raise ValueError(f"labels must be one of {_LABEL_VALUES}")
    if missing_labels:
        labels = "input"    # both R paths override to "input"

    # labels_position None = auto-tune: plotly places each label
    # inside its bar, or outside when the bar is too small to hold
    # it. Stacked segments always label inside — outside placement
    # would stack the labels on top of the next segment.
    if labels_position not in (None, "in", "out"):
        raise ValueError('labels_position must be "in" or "out"')

    two_d = isinstance(x, pd.DataFrame)
    if not two_d and not isinstance(x, pd.Series):
        raise TypeError("x must be a pandas Series (1-D) or "
                        "DataFrame (2-D, rows = by levels)")

    # counts: the frequencies behind heights that are already
    # proportions (Chart's stack100), so the hover percentages and
    # the labels="input" text come from the data, not the heights
    prop_scaled = counts is not None
    cnt = None
    if prop_scaled:
        cnt = np.asarray(counts, dtype=float)
        if cnt.shape != np.asarray(x, dtype=float).shape:
            raise ValueError("counts must have the same shape as x")

    # derive names from pandas metadata when not given
    if x_name is None:
        x_name = (x.columns.name if two_d else x.index.name) or "x"
    if y_name is None:
        y_name = "Count" if two_d else (x.name or "Count")
    if two_d and by_name is None:
        by_name = x.index.name or "by"
    x_lab = x_name if x_lab is None else x_lab
    y_lab = y_name if y_lab is None else y_lab
    if digits_d is None:
        digits_d = get_option("digits_d", 2)
    if style_opts is None:
        style_opts = plotly_style()
    if fill is None:
        fill = BASE_COLORS

    title_size = round(16 * get_option("main_size", 1))
    alpha_fill = 0.92 if opacity is None else float(opacity)

    # labels_size arrives as a cex value (e.g. 0.90); convert to px
    labels_px = max(8, round(labels_size * 14))

    # % lines in hover only when data are counts (labels="input"
    # passed explicitly signals means / deviations)
    show_pct_hover = missing_labels or labels != "input"

    stacked = two_d and not beside
    if stacked and labels_position == "out":
        raise ValueError(
            'labels_position="out" is not meaningful for a stacked '
            "bar chart (a by= variable without beside=)")
    if stacked:
        text_pos = "inside"
    elif labels_position == "out":
        text_pos = "outside"
    elif labels_position == "in":
        text_pos = "inside"
    else:
        text_pos = "auto"           # per-bar in/out, the auto-tune
    # a label drawn outside sits on the panel background, not the
    # bar fill, so it gets the axis text color (R: Chart.R ~287)
    out_col = (to_hex(labels_color) if labels_color is not None
               else to_hex(get_option("axis_color", "black")))

    def text_colors(fill_hex, n):
        if labels_color is not None:
            return _rep_len(to_hex(labels_color), n)
        if labels_autocontrast:
            if isinstance(fill_hex, list):
                return [contrast_text_for_hex(c) for c in fill_hex]
            return _rep_len(contrast_text_for_hex(fill_hex), n)
        # default: white labels inside the bar (like pie/sunburst);
        # labels drawn OUTSIDE keep out_col, dark on the panel
        return _rep_len("white", n)

    # ------------------ 1-D ------------------
    if not two_d:
        cats = [str(c) for c in x.index]
        vals = x.to_numpy(dtype=float)

        total_n = np.nansum(vals)
        share = (vals / total_n if total_n > 0
                 else np.full(len(vals), np.nan))

        tick_fmt = get_tick_fmt(vals, digits_d)
        pct_line = ("<br>% of total: %{customdata:.1%}"
                    if show_pct_hover else "")
        if not horiz:
            hover = (f"{x_name}: %{{x}}<br>"
                     f"{y_name}: %{{y:{tick_fmt}}}"
                     f"{pct_line}<extra></extra>")
        else:
            hover = (f"{x_name}: %{{y}}<br>"
                     f"{y_name}: %{{x:{tick_fmt}}}"
                     f"{pct_line}<extra></extra>")

        fill_vec = as_plotly_color(_rep_len(fill, len(cats)))
        border_vec = as_plotly_color(_rep_len(border, len(cats)))

        txt = _bar_labels(vals, share, labels, digits_d, counts=cnt,
                          labels_decimals=labels_decimals)
        txt_col = text_colors(fill_vec, len(cats))

        plt = go.Figure(go.Bar(
            x=cats if not horiz else vals,
            y=vals if not horiz else cats,
            orientation=None if not horiz else "h",
            marker=dict(
                color=fill_vec,
                opacity=alpha_fill,
                line=dict(color=border_vec, width=1),
            ),
            customdata=share,
            hovertemplate=hover,
            text=txt,
            textposition=text_pos,
            insidetextanchor="middle",
            textfont=dict(size=labels_px),
            insidetextfont=dict(color=txt_col),
            outsidetextfont=dict(color=out_col),
            cliponaxis=False,
        ))

        cat_array = list(dict.fromkeys(cats))
        stack_max = np.nanmax(vals) if len(vals) else 1
        stack_min = np.nanmin(vals) if len(vals) else 0

    # ------------------ 2-D ------------------
    else:
        groups = [str(g) for g in x.index]
        # the key text: legend_labels renames the by levels and
        # legend_abbrev truncates them. R analog: legend_labels /
        # legend_abbrev of Chart()
        if legend_labels is not None:
            if len(legend_labels) != len(groups):
                raise ValueError(
                    f"legend_labels has {len(legend_labels)} entries "
                    f"for {len(groups)} levels of the by variable")
            key_names = [str(v) for v in legend_labels]
        else:
            key_names = list(groups)
        key_names = [abbrev(k, legend_abbrev) for k in key_names]
        cats = [str(c) for c in x.columns]
        mat = x.to_numpy(dtype=float)

        # percentages come from the counts when the heights have
        # already been rescaled to proportions (stack100)
        base = mat if cnt is None else cnt
        total_n = np.nansum(base)
        share_total = (base / total_n if total_n > 0
                       else np.full(base.shape, np.nan))
        x_totals = np.nansum(base, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            share_within_x = np.where(x_totals != 0,
                                      base / x_totals, np.nan)

        tick_fmt = get_tick_fmt(mat.ravel(), digits_d)
        pct_lines_2d = (
            f"<br>% of {x_name}: %{{customdata[1]:.1%}}"
            "<br>% of total: %{customdata[0]:.1%}"
            if show_pct_hover else "")
        if not horiz:
            hover = (f"{x_name}: %{{x}}<br>"
                     f"{by_name}: %{{fullData.name}}<br>"
                     f"{y_name}: %{{y:{tick_fmt}}}"
                     f"{pct_lines_2d}<extra></extra>")
        else:
            hover = (f"{x_name}: %{{y}}<br>"
                     f"{by_name}: %{{fullData.name}}<br>"
                     f"{y_name}: %{{x:{tick_fmt}}}"
                     f"{pct_lines_2d}<extra></extra>")

        plt = go.Figure()

        for i, gname in enumerate(groups):
            row = mat[i]
            fill_i = as_plotly_color(_pick(fill, gname, i))
            border_i = as_plotly_color(_pick(border, gname, i))

            cd = np.stack([share_total[i], share_within_x[i]],
                          axis=-1)
            # a stack100 bar IS its within-x percentage, so that is
            # what "%"/"prop" report there (R: .fmt(x * 100))
            txt = _bar_labels(
                row,
                share_within_x[i] if prop_scaled else share_total[i],
                labels, digits_d,
                counts=None if cnt is None else cnt[i],
                labels_decimals=labels_decimals)
            txt_col = text_colors(fill_i, len(row))

            plt.add_trace(go.Bar(
                x=cats if not horiz else row,
                y=row if not horiz else cats,
                orientation=None if not horiz else "h",
                name=key_names[i],
                legendgroup=gname,
                marker=dict(
                    color=fill_i,
                    opacity=alpha_fill,
                    line=dict(color=border_i, width=1),
                ),
                customdata=cd,
                hovertemplate=hover,
                text=txt,
                textposition=text_pos,
                insidetextanchor="middle",
                textfont=dict(size=labels_px),
                insidetextfont=dict(color=txt_col),
                outsidetextfont=dict(color=out_col),
                cliponaxis=False,
            ))

        plt.update_layout(barmode="group" if beside else "stack")

        cat_array = list(dict.fromkeys(cats))
        if beside:
            stack_max = np.nanmax(mat)
            stack_min = np.nanmin(mat)
        else:   # stacked: positive and negative runs sum separately
            stack_max = np.nanmax(
                np.nansum(np.clip(mat, 0, None), axis=0))
            stack_min = np.nanmin(
                np.nansum(np.clip(mat, None, 0), axis=0))

    # --- axes & background ------------------------------------------
    # ax not supplied: compute tick positions here (see header note);
    # value-axis labels through the axis_fmt policies ("60K"), as in
    # R (.axis.format), with the prefix of the physical value axis
    val_range = None
    if ax is None and scale_y is not None:
        # (min, max, n_intervals) gives n_intervals + 1 ticks, the
        # axTicks(axp=) contract; R caps the tick decimals at 2
        lo, hi = float(scale_y[0]), float(scale_y[1])
        tickvals = list(np.linspace(lo, hi, int(scale_y[2]) + 1))
        nd = min(2, max(_decimals(t) for t in tickvals))
        ax = {"tickvals": tickvals,
              "ticktext": axis_format(
                  tickvals, nd, axis_fmt,
                  axis_x_pre if horiz else axis_y_pre)}
        val_range = [lo, hi]
    if ax is None:
        tickvals = pretty(min(0.0, float(stack_min)),
                          max(0.0, float(stack_max)))
        ax = {"tickvals": tickvals,
              "ticktext": axis_format(
                  tickvals, digits_d, axis_fmt,
                  axis_x_pre if horiz else axis_y_pre)}
    if grid is None:
        grid = ax["tickvals"]

    # category labels wrap at their spaces when break_x; the bar
    # coordinates stay the raw category values
    cat_text = _break_labels(cat_array, break_x)
    cat_ticks = dict(tickmode="array",
                     tickvals=list(range(len(cat_array))),
                     ticktext=cat_text)

    bargap, bargroupgap = _bar_gaps(
        gap, beside and two_d, len(x.index) if two_d else 1)

    border_shapes = plot_border()

    if horiz:
        y_axis_cat = axis_cat(x_lab)
        y_axis_cat.update(categoryorder="array",
                          categoryarray=cat_array, **cat_ticks)
        if rotate_y:
            y_axis_cat["tickangle"] = -rotate_y
        val_axis = axis_num(y_lab, ax["tickvals"], ax["ticktext"])
        if rotate_x:
            val_axis["tickangle"] = -rotate_x
        if val_range is not None:
            val_axis["range"] = val_range
        plt.update_layout(
            xaxis=val_axis,
            yaxis=y_axis_cat,
            shapes=x_grid(grid) + border_shapes,
            template=None,
        )
    else:
        x_axis_cat = axis_cat(x_lab)
        x_axis_cat.update(categoryorder="array",
                          categoryarray=cat_array,
                          tickangle=-rotate_x, **cat_ticks)
        val_axis = axis_num(y_lab, ax["tickvals"], ax["ticktext"])
        if rotate_y:
            val_axis["tickangle"] = -rotate_y
        if val_range is not None:
            val_axis["range"] = val_range
        plt.update_layout(
            xaxis=x_axis_cat,
            yaxis=val_axis,
            shapes=y_grid(grid) + border_shapes,
            template=None,
        )

    if bargap is not None:
        plt.update_layout(bargap=bargap)
    if bargroupgap is not None:
        plt.update_layout(bargroupgap=bargroupgap)

    plt.update_layout(
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]),
    )

    if by_name:
        plt.update_layout(legend=legend_style(
            by_name, style_opts, title=legend_title,
            position=legend_position, horiz=legend_horiz,
            size=legend_size, adjust=legend_adjust,
            abbrev_n=legend_abbrev))

    if main:
        plt.update_layout(
            title=dict(text=main, x=0.5, xanchor="center",
                       y=0.99, yanchor="top",
                       font=dict(size=title_size)),
            margin=dict(t=round(title_size * 2.2)),
        )

    # R's .finalize_plotly_widget() is RStudio-viewer machinery with
    # no Python counterpart; the figure itself is the return value
    return plt


def bc_facet_plotly(x, x_name=None, facet_name=None,
                    x_lab=None, y_lab=None,
                    fill=None, border="off", opacity=None,
                    proportion=False, digits_d=0, main=None,
                    n_col=1, axis_fmt="K", axis_x_pre="",
                    rotate_x=0, rotate_y=0,
                    style_opts=None):
    """Faceted bar chart of counts: one panel per facet level,
    stacked on a shared count axis with horizontal bars, as in the
    lattice Trellis bar chart (barchart(x ~ Count | facet)) that
    R renders for Chart(x, facet=). Input: DataFrame with rows =
    facet levels, columns = x categories, values = counts.
    n_col > 1 lays the panels out as a grid (the lattice n_col),
    filled bottom-up as in finish_facet.
    R analog: .bar.lattice T.type="bar" (plotly-only port)"""

    if not isinstance(x, pd.DataFrame):
        raise TypeError("x must be a pandas DataFrame with rows = "
                        "facet levels, columns = x categories")
    if x_name is None:
        x_name = x.columns.name or "x"
    if facet_name is None:
        facet_name = x.index.name or "facet"
    if style_opts is None:
        style_opts = plotly_style()
    if fill is None:
        fill = BASE_COLORS

    fac_levels = [str(g) for g in x.index]
    cats = [str(c) for c in x.columns]
    mat = x.to_numpy(dtype=float)
    n_f = len(fac_levels)

    if proportion:                # per-panel, ~ prop.table margin
        row_tot = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, row_tot, out=np.zeros_like(mat),
                        where=row_tot > 0)
    val_name = "Proportion" if proportion else "Count"
    if x_lab is None:
        x_lab = f"{val_name} of {x_name}"
    if y_lab is None:
        y_lab = x_name

    alpha_fill = 0.92 if opacity is None else float(opacity)
    fill_vec = as_plotly_color(_rep_len(fill, len(cats)))
    border_vec = as_plotly_color(_rep_len(border, len(cats)))

    axT1 = pretty(0.0, float(np.nanmax(mat)) if mat.size else 1.0)
    ax = {"axT1": axT1,
          "axL1": axis_format(axT1, digits_d, axis_fmt,
                              axis_x_pre)}

    n_col = max(1, min(int(n_col or 1), n_f))
    fig = facet_fig(math.ceil(n_f / n_col), n_col)
    for i, lv in enumerate(fac_levels):
        row = mat[i]
        hover = [
            (f"{x_name}: {c}"
             f"<br>{val_name}: "
             + (f"{v:.{max(2, digits_d)}f}" if proportion
                else f"{v:g}")
             + f"<br>{facet_name}: {lv}")
            for c, v in zip(cats, row)]
        r, c = facet_pos(i, n_f, n_col)   # first level bottom-left
        fig.add_trace(go.Bar(
            x=row, y=cats, orientation="h",
            marker=dict(color=fill_vec, opacity=alpha_fill,
                        line=dict(color=border_vec, width=1)),
            hoverinfo="text", hovertext=hover,
            showlegend=False,
        ), row=r, col=c)

    finish_facet(fig, fac_levels, ax, x_lab, y_lab, gridT1=axT1,
                 style_opts=style_opts, y_cat=cats, n_col=n_col)
    if rotate_x:
        fig.update_xaxes(tickangle=-rotate_x)
    if rotate_y:
        fig.update_yaxes(tickangle=-rotate_y)

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
