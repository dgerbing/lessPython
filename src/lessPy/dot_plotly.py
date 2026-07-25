# dot_plotly.py — analog of dot.plotly.R
#
# Renders a Cleveland dot chart from ALREADY-AGGREGATED input:
#   single-series  cats + 1-D vals: one dot per category, with an
#                  origin->value stem ("lollipop") per dot
#   paired         cats + DataFrame vals (2+ numeric columns): one
#                  series per column, horizontal, with a legend
#                  (e.g., x=Name, y=[Pre, Post])
#   faceted        cats + vals + facet= (one level per
#                  observation): a grid of single-series panels,
#                  one per facet level, on a shared value axis
#                  with dot hues keyed by category name

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .utils import get_option
from .plotly_utils import (
    BASE_COLORS, axis_cat, axis_format, axis_num, facet_layout,
    get_tick_fmt, make_trans, plot_border, plotly_style, to_hex,
    x_grid, y_grid,
)

# R plotting shape codes -> plotly symbol names
_SYMBOLS = {21: "circle", 22: "square", 23: "diamond",
            24: "triangle-up", 25: "triangle-down"}


def _marker_style(shape, pt_size, fill, border, pt_opacity):
    """Per-point marker geometry. R analog: .dot_marker_style()"""
    px = float(pt_size) * 10
    if not np.isfinite(px) or px <= 0:
        px = 5
    fill1 = fill[0] if isinstance(fill, (list, tuple)) else fill
    border1 = (border[0] if isinstance(border, (list, tuple))
               else border)
    return {
        "symbol": _SYMBOLS.get(shape, "circle"),
        "px": px,
        "color": make_trans(fill1, pt_opacity),
        "border": to_hex(border1),
    }


def _marker_list(ms, n, color_override=None):
    """marker= dict for a scatter trace. R analog: .dot_marker_list()"""
    return dict(
        symbol=ms["symbol"],
        size=[ms["px"]] * n,
        sizemode="diameter",
        color=(ms["color"] if color_override is None
               else color_override),
        opacity=1,
        line=dict(color=ms["border"], width=1),
    )


def _add_marker_trace(fig, cats, vals, marker, orientation,
                      hover_tmpl=None, name=None, showlegend=False,
                      axis_suf=""):
    """R analog: .dot_marker_trace()"""
    xv, yv = (cats, vals) if orientation == "v" else (vals, cats)
    kw = dict(mode="markers", x=xv, y=yv, marker=marker,
              showlegend=showlegend,
              xaxis=f"x{axis_suf}", yaxis=f"y{axis_suf}")
    if hover_tmpl is not None:
        kw["hovertemplate"] = hover_tmpl
    if name is not None:
        kw["name"] = name
    fig.add_trace(go.Scatter(**kw))


def _add_segment_trace(fig, cats, vals, seg_color, orientation,
                       seg_start, axis_suf=""):
    """None-separated origin->value stems; cats/vals must be
    pre-filtered to finite values. R analog: .dot_segment_trace()"""
    n = len(vals)
    if n == 0:
        return
    seg_cat, seg_val = [], []
    for c, v in zip(cats, vals):
        seg_cat += [c, c, None]
        seg_val += [seg_start, v, None]
    if orientation == "v":
        seg_x, seg_y = seg_cat, seg_val
    else:
        seg_x, seg_y = seg_val, seg_cat
    fig.add_trace(go.Scatter(
        mode="lines", x=seg_x, y=seg_y,
        xaxis=f"x{axis_suf}", yaxis=f"y{axis_suf}",
        line=dict(color=seg_color, width=1),
        hoverinfo="skip", showlegend=False,
    ))


def _val_axis(label, gridT, origin_x, ticktext=None):
    """Numeric (value) axis spec. R analog: .dot_val_axis()"""
    ax = axis_num(label, gridT, ticktext)
    if origin_x is not None and gridT:
        step = gridT[1] - gridT[0] if len(gridT) > 1 else 0
        ax["range"] = [min(origin_x, min(gridT)), max(gridT) + step]
        ax["autorange"] = False
    return ax


def _cat_axis(label, cats):
    """Category axis, order frozen. R analog: .dot_cat_axis()"""
    ax = axis_cat(label)
    ax.update(categoryorder="array", categoryarray=cats,
              tickmode="array", tickvals=cats, ticktext=cats)
    return ax


def _dot_title(main):
    """R analog: .dot_title()"""
    if not main:
        return None
    return dict(text=main, x=0.5, xanchor="center",
                y=0.98, yanchor="top",
                font=dict(size=round(16 * get_option("main_size", 1)),
                          color=to_hex(get_option("lab_color",
                                                  "black"))))


def _hover(orientation, x_lab, y_lab, val_spec):
    if orientation == "v":
        cat_label = x_lab or "Category"
        val_label = y_lab or "Value"
        return (f"{cat_label}: %{{x}}<br>"
                f"{val_label}: %{{y{val_spec}}}<extra></extra>")
    cat_label = y_lab or "Category"
    val_label = x_lab or "Value"
    return (f"{val_label}: %{{x{val_spec}}}<br>"
            f"{cat_label}: %{{y}}<extra></extra>")


def dot_plotly(x, y, orientation="v",
               fill=None, border=None, shape=21, pt_size=1,
               x_lab="", y_lab="", digits_d=2, pt_opacity=0.95,
               gridT=None, origin_x=None, main=None,
               facet=None, facet_name=None, n_col=None,
               axis_fmt="K", axis_x_pre="", axis_y_pre="",
               rotate_x=0, rotate_y=0,
               segments_x=True, segments_y=True, style_opts=None):

    cats, vals = x, y
    if style_opts is None:
        style_opts = plotly_style()
    if fill is None:
        fill = BASE_COLORS
    if border is None:
        border = get_option("pt_color", "#324E5C")

    title_size = round(16 * get_option("main_size", 1))
    seg_color = to_hex(style_opts["segment_color"])

    def val_ticks(orient):
        # value-axis labels through the axis_fmt policies ("60K"),
        # prefixed for the physical axis the values land on
        if not gridT:
            return None
        return axis_format(gridT, digits_d, axis_fmt,
                           axis_x_pre if orient == "h"
                           else axis_y_pre)

    def apply_rotate(x_ax, y_ax):
        if rotate_x:
            x_ax["tickangle"] = -rotate_x
        if rotate_y:
            y_ax["tickangle"] = -rotate_y

    # ---- paired dot plot (vals is a DataFrame, 2+ columns) --------
    # Single panel, cats on y, values on x; one legend entry per
    # y column.
    if isinstance(vals, pd.DataFrame):
        cats_all = [str(c) for c in cats]
        ms = _marker_style(21, pt_size, fill, border, pt_opacity)
        seg_start = 0 if origin_x is None else origin_x

        fig = go.Figure()
        for si, col in enumerate(vals.columns):
            vals_si = vals[col].to_numpy(dtype=float)
            fill_si = (fill[si % len(fill)]
                       if isinstance(fill, (list, tuple)) else fill)
            clr = make_trans(fill_si, pt_opacity)
            if segments_x:
                fin = np.isfinite(vals_si)
                _add_segment_trace(
                    fig, [c for c, f in zip(cats_all, fin) if f],
                    vals_si[fin].tolist(), seg_color, "h",
                    seg_start)
            _add_marker_trace(
                fig, cats_all, vals_si.tolist(),
                marker=_marker_list(ms, len(vals_si),
                                    color_override=clr),
                orientation="h", name=str(col), showlegend=True)

        x_ax = _val_axis(x_lab, gridT, origin_x, val_ticks("h"))
        y_ax = _cat_axis(y_lab, cats_all)
        apply_rotate(x_ax, y_ax)
        fig.update_layout(
            xaxis=x_ax,
            yaxis=y_ax,
            shapes=(x_grid(gridT) if gridT else []) + plot_border(),
            margin=dict(t=round(title_size * 2.2) if main else 30),
            title=_dot_title(main),
            legend=dict(font=dict(size=round(
                16 * get_option("axis_size", 0.9)))),
            template=None,
            plot_bgcolor=to_hex(style_opts["panel_fill"]),
            paper_bgcolor=to_hex(style_opts["window_fill"]),
        )
        return fig

    # ---- faceted dot plot --------------------------------------------
    # Grid of single-series panels, one per facet level, on a shared
    # value axis. Per-panel segment start is clamped at-or-below the
    # panel minimum so dots never sit to the left of (or below)
    # their stem. R analog: dot.plotly.R faceted path
    if facet is not None:
        fac_char = [None if f is None
                    or (isinstance(f, float) and np.isnan(f))
                    else str(f) for f in facet]
        fac_levels = sorted({f for f in fac_char if f is not None})

        cats_in = [str(c) for c in cats]
        vals_in = np.asarray(list(vals), dtype=float)

        ms = _marker_style(shape, pt_size, fill, border,
                           pt_opacity)
        # color keyed by category name so hues match across panels
        cat_levels = list(dict.fromkeys(cats_in))
        col_map = {c: make_trans(
                       to_hex(fill[i % len(fill)]
                              if isinstance(fill, (list, tuple))
                              else fill),
                       pt_opacity)
                   for i, c in enumerate(cat_levels)}

        val_fmt = get_tick_fmt(vals_in[np.isfinite(vals_in)],
                               digits_d)
        val_spec = f":{val_fmt}" if val_fmt else ""
        hover_tmpl = _hover(orientation, x_lab, y_lab, val_spec)
        draw_seg = segments_x if orientation == "h" else segments_y

        def ann_adj(i, a, d):
            a["y"] = min(d["y"][1] + 0.04, 0.99)
            return a
        domains, anns = facet_layout(
            fac_levels, facet_name or "",
            yanchor="bottom", ann_adjust=ann_adj, n_col=n_col)

        fig = go.Figure()
        axes = {}
        for i, lv in enumerate(fac_levels):
            suf = "" if i == 0 else str(i + 1)
            keep = [k for k, f in enumerate(fac_char) if f == lv]
            cats_i = [cats_in[k] for k in keep]
            vals_i = vals_in[keep]
            fin = np.isfinite(vals_i)

            if fin.any() and draw_seg:
                # never above the panel's data minimum — avoid
                # backward stems
                panel_min = float(vals_i[fin].min())
                seg_start = (min(origin_x, panel_min)
                             if origin_x is not None else 0)
                _add_segment_trace(
                    fig, [c for c, f in zip(cats_i, fin) if f],
                    vals_i[fin].tolist(), seg_color, orientation,
                    seg_start, axis_suf=suf)
            _add_marker_trace(
                fig, cats_i,
                [None if not np.isfinite(v) else v
                 for v in vals_i],
                marker=_marker_list(
                    ms, len(vals_i),
                    color_override=[col_map[c] for c in cats_i]),
                orientation=orientation, hover_tmpl=hover_tmpl,
                axis_suf=suf)

            val_lbl = x_lab if orientation == "h" else y_lab
            cat_lbl = y_lab if orientation == "h" else x_lab
            val_ax = _val_axis(val_lbl if i == 0 else "",
                               gridT, origin_x,
                               val_ticks(orientation))
            cat_ax = _cat_axis(cat_lbl, cats_i)
            x_ax, y_ax = ((val_ax, cat_ax)
                          if orientation == "h"
                          else (cat_ax, val_ax))
            apply_rotate(x_ax, y_ax)
            x_ax.update(domain=list(domains[i]["x"]),
                        anchor=f"y{suf}")
            y_ax.update(domain=list(domains[i]["y"]),
                        anchor=f"x{suf}")
            axes[f"xaxis{suf}"] = x_ax
            axes[f"yaxis{suf}"] = y_ax

        fig.update_layout(
            annotations=anns,
            margin=dict(t=round(title_size * 2.2) if main else 40,
                        b=8, l=20, r=20),
            showlegend=False,
            title=_dot_title(main),
            template=None,
            plot_bgcolor=to_hex(style_opts["panel_fill"]),
            paper_bgcolor=to_hex(style_opts["window_fill"]),
            **axes,
        )
        return fig

    # ---- single-series dot plot ------------------------------------
    pt_opacity = float(pt_opacity)
    if not np.isfinite(pt_opacity):
        pt_opacity = 0.95
    pt_opacity = max(0.0, min(1.0, pt_opacity))

    pairs = [(str(c), v) for c, v in zip(cats, vals)
             if c is not None and not (isinstance(c, float)
                                       and np.isnan(c))]
    if not pairs:
        return go.Figure()
    cats = [p[0] for p in pairs]
    vals = np.array([p[1] for p in pairs], dtype=float)
    cats_all = list(cats)          # full list for categoryarray

    ms = _marker_style(shape, pt_size, fill, border, pt_opacity)

    val_fmt = get_tick_fmt(vals[np.isfinite(vals)], digits_d)
    val_spec = f":{val_fmt}" if val_fmt else ""
    hover_tmpl = _hover(orientation, x_lab, y_lab, val_spec)

    fig = go.Figure()
    # segments first so dots sit on top
    draw_seg = segments_x if orientation == "h" else segments_y
    if draw_seg:
        fin = np.isfinite(vals)
        _add_segment_trace(
            fig, [c for c, f in zip(cats, fin) if f],
            vals[fin].tolist(), seg_color, orientation,
            seg_start=0 if origin_x is None else origin_x)

    # one palette color per category, same hues as the bar chart
    pt_cols = make_trans(
        to_hex([fill[i % len(fill)] if isinstance(fill, (list, tuple))
                else fill for i in range(len(vals))]),
        pt_opacity)
    _add_marker_trace(
        fig, cats, [None if not np.isfinite(v) else v for v in vals],
        marker=_marker_list(ms, len(vals), color_override=pt_cols),
        orientation=orientation, hover_tmpl=hover_tmpl)

    # grid lines run perpendicular to the value axis
    if gridT:
        grid_shapes = (y_grid(gridT) if orientation == "v"
                       else x_grid(gridT))
    else:
        grid_shapes = []
    cat_ax = _cat_axis(x_lab if orientation == "v" else y_lab,
                       cats_all)
    val_ax = _val_axis(x_lab if orientation == "h" else y_lab,
                       gridT, origin_x, val_ticks(orientation))

    if orientation == "h":
        margin = dict(l=70,
                      t=round(title_size * 2.2) if main else 30,
                      b=max(40, 60 if x_lab else 0))
        apply_rotate(val_ax, cat_ax)
        fig.update_layout(xaxis=val_ax, yaxis=cat_ax)
    else:
        margin = dict(t=round(title_size * 2.2) if main else 30,
                      b=max(12, 60 if x_lab else 0))
        apply_rotate(cat_ax, val_ax)
        fig.update_layout(xaxis=cat_ax, yaxis=val_ax)

    fig.update_layout(
        shapes=grid_shapes + plot_border(),
        margin=margin,
        title=_dot_title(main),
        template=None,
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]),
    )
    return fig
