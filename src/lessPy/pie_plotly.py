# pie_plotly.py — analog of piechart.plotly.R
#
# Renders a donut/pie from ALREADY-TABULATED input:
#   1-D  pandas Series: index = slice names, values = counts/means
#        -> one pie
#   2-D  pandas DataFrame: rows = `by` levels, columns = slices
#        -> a grid of pies, one per by level, group name in each hole
#
# Deliberate deviations from the R source:
#   - The theme-dependent sequential fill branch (.scale.clr) is not
#     ported; without a theme system, default fills are BASE_COLORS,
#     the equivalent of lessR's default "colors" theme (hues).
#   - R's defensive plotly_build() post-processing of slice borders
#     works around R-plotly quirks; plotly.py sets marker.line
#     directly, so the same result needs no post-build step.

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .utils import get_option
from .plotly_utils import (
    BASE_COLORS, as_plotly_color, auto_text_color, make_trans,
    plotly_style, to_hex,
)

_LABEL_VALUES = ("%", "input", "prop", "off")


def _pick_by_names_or_recycle(v, needed_names):
    """dict: pick by slice name; else recycle across slices.
    R analog: .pick_by_names_or_recycle()"""
    n = len(needed_names)
    if isinstance(v, dict):
        base = list(v.values()) or ["black"]
        return [v.get(nm, base[i % len(base)])
                for i, nm in enumerate(needed_names)]
    if v is None:
        return ["black"] * n
    if not isinstance(v, (list, tuple)):
        v = [v]
    return [v[i % len(v)] for i in range(n)]


def _num_str(vals, total, labels, digits_d, labels_decimals=None):
    """labels_decimals sets the decimal places; None falls back to
    digits_d, except for "prop", whose values are proportions that
    digits_d (0 for counts) would flatten to "0"."""
    denom = max(total, 1e-12)
    digits_d = (int(labels_decimals) if labels_decimals is not None
                else (2 if labels == "prop" else digits_d))
    if labels == "%":
        return [f"{100 * v / denom:.{digits_d}f}%" for v in vals]
    if labels == "prop":
        return [f"{v / denom:.{digits_d}f}" for v in vals]
    if labels == "input":
        return [f"{v:.{digits_d}f}" for v in vals]
    return ["" for _ in vals]                     # "off"


def _slice_text(slices, num_str, labels):
    if labels == "off":
        return list(slices)
    return [f"{s}<br>{n}" for s, n in zip(slices, num_str)]


def _text_colors(labels_color, fills_rgba, panel_fill):
    if labels_color is None or labels_color == "adjust":
        return auto_text_color(fills_rgba, bg=panel_fill)
    return [labels_color] * len(fills_rgba)


def pie_plotly(x, x_name=None, y_name=None, by_name=None, main=None,
               fill=None, border=None, opacity=1.0,
               hole=0.65, ncols=None,
               labels=None, labels_position="in",
               labels_color=None, labels_size=1.0,
               labels_decimals=None,
               digits_d=2,
               group_labels=True, group_label_size=14,
               style_opts=None):

    if labels is None:
        labels = "%"                 # R analog: match.arg default
    if labels not in _LABEL_VALUES:
        raise ValueError(f"labels must be one of {_LABEL_VALUES}")

    two_d = isinstance(x, pd.DataFrame)
    if not two_d and not isinstance(x, pd.Series):
        raise TypeError("x must be a pandas Series (1-D) or "
                        "DataFrame (2-D, rows = by levels)")

    if x_name is None:
        x_name = (x.columns.name if two_d else x.index.name) or "x"
    if y_name is None:
        y_name = "Count" if two_d else (x.name or "Count")
    if two_d and by_name is None:
        by_name = x.index.name or "Group"
    if style_opts is None:
        style_opts = plotly_style()
    if fill is None:
        fill = BASE_COLORS
    if border is None:
        border = "transparent"

    title_size = round(16 * get_option("main_size", 1))
    alpha_fill = float(opacity)
    if not math.isfinite(alpha_fill):
        alpha_fill = 1.0
    alpha_fill = max(0.0, min(1.0, alpha_fill))

    # None = the default (R's labels_position %||% "in")
    pos_in = labels_position is None or \
        str(labels_position).lower() == "in"
    txt_pos = "inside" if pos_in else "outside"
    val_spec = f":.{max(0, int(digits_d))}f"

    def title_layout():
        if main:
            return dict(text=main, y=0.94, yanchor="top",
                        font=dict(
                            size=title_size,
                            color=to_hex(get_option("lab_color",
                                                    "black"))))
        return None

    def panel_bg(fig):
        # simulated panel background; only when not white
        bg_plot = str(to_hex(style_opts["panel_fill"])).upper()
        bg_paper = str(to_hex(style_opts["window_fill"])).upper()
        if bg_plot != "#FFFFFF" or bg_paper != "#FFFFFF":
            fig.update_layout(paper_bgcolor=bg_paper)
            fig.add_shape(type="rect", xref="paper", yref="paper",
                          x0=0, y0=0, x1=1, y1=1, layer="below",
                          fillcolor=bg_plot, line=dict(width=0))

    # --- SINGLE PIE: 1-D --------------------------------------------
    if not two_d:
        slices = [str(s) for s in x.index]
        values = x.to_numpy(dtype=float)
        tot = np.nansum(values)

        fill_vec = _pick_by_names_or_recycle(fill, slices)
        border_vec = _pick_by_names_or_recycle(border, slices)

        num_str = _num_str(values, tot, labels, digits_d,
                           labels_decimals)
        text_vec = _slice_text(slices, num_str, labels)

        txt_size = round(12 * 1.38 * labels_size *
                         get_option("axis_size", 0.9))

        fills_rgba = make_trans(fill_vec, alpha_fill)
        txt_colors = _text_colors(labels_color, fills_rgba,
                                  style_opts["panel_fill"])

        overall_pct = values / (tot if tot > 0 else 1)

        # inside labels need room: clamp an over-large hole
        hole_use = 0.62 if (pos_in and hole > 0.62) else hole

        font = dict(color=txt_colors, size=txt_size)
        fig = go.Figure(go.Pie(
            labels=slices,
            values=values,
            sort=False,
            direction="clockwise",
            hole=hole_use,
            domain=dict(x=[0, 1], y=[0.03, 0.93]),
            text=text_vec,
            textinfo="text",
            textposition=txt_pos,
            insidetextorientation="radial",
            textfont=font,
            insidetextfont=font,
            outsidetextfont=font,
            automargin=not pos_in,
            marker=dict(
                colors=fills_rgba,
                line=dict(color=as_plotly_color(border_vec),
                          width=1.6),
            ),
            customdata=overall_pct,
            hovertemplate=(
                f"{x_name}: %{{label}}"
                f"<br>{y_name}: %{{value{val_spec}}}"
                "<br>% of total: %{customdata:.2%}"
                "<extra></extra>"),
            showlegend=False,
        ))

        fig.update_layout(
            uniformtext=dict(minsize=8, mode="show"),
            margin=dict(t=round(title_size * 2.2),
                        r=20, b=30, l=20),
            title=title_layout(),
        )
        panel_bg(fig)
        return fig

    # --- GROUPED INPUT (2-D): PIE GRID ------------------------------
    groups = [str(g) for g in x.index]
    slices = [str(c) for c in x.columns]
    mat = x.to_numpy(dtype=float)
    grand_total = np.nansum(mat)

    k = len(groups)
    nc = (math.ceil(math.sqrt(k)) if ncols is None or ncols < 1
          else int(ncols))
    nr = math.ceil(k / nc)

    txt_size = round(12 * labels_size * get_option("axis_size", 0.9))
    by_title = by_name if by_name else "Group"

    fig = go.Figure()

    for i, grp in enumerate(groups):
        col, row = i % nc, i // nc
        x0, x1 = col / nc, (col + 1) / nc
        y0, y1 = 1 - (row + 1) / nr, 1 - row / nr
        shrink = 0.92
        y_mid, y_half = (y0 + y1) / 2, (y1 - y0) * shrink / 2
        dom = dict(x=[x0, x1], y=[y_mid - y_half, y_mid + y_half])

        vals = mat[i].copy()
        vals[~np.isfinite(vals)] = np.nan
        val_sum = np.nansum(vals)
        overall_pct = vals / (grand_total if grand_total > 0 else 1)

        num_str = _num_str(vals, val_sum, labels, digits_d,
                           labels_decimals)
        text_vec = _slice_text(slices, num_str, labels)

        fills_this = make_trans(
            _pick_by_names_or_recycle(fill, slices), alpha_fill)
        borders_this = _pick_by_names_or_recycle(border, slices)
        txt_colors = _text_colors(labels_color, fills_this,
                                  style_opts["panel_fill"])

        font = dict(color=txt_colors, size=txt_size)
        fig.add_trace(go.Pie(
            labels=slices,
            values=vals,
            name=f"{by_title}: {grp}",
            legendgroup="pies",
            sort=False,
            direction="clockwise",
            hole=hole,
            domain=dom,
            text=text_vec,
            textinfo="text",
            textposition=txt_pos,
            insidetextorientation="radial",
            textfont=font,
            insidetextfont=font,
            outsidetextfont=font,
            automargin=not pos_in,
            marker=dict(
                colors=fills_this,
                line=dict(color=as_plotly_color(borders_this),
                          width=1),
            ),
            customdata=overall_pct,
            hovertemplate=(
                f"{by_title}: {grp}"
                f"<br>{x_name}: %{{label}}"
                f"<br>{y_name}: %{{value{val_spec}}}"
                f"<br>% of {grp}: %{{percent}}"
                "<br>% of total: %{customdata:.2%}"
                "<extra></extra>"),
            showlegend=False,
        ))

        if group_labels and hole > 0:
            fig.add_annotation(
                x=(dom["x"][0] + dom["x"][1]) / 2,
                y=(dom["y"][0] + dom["y"][1]) / 2,
                xref="paper", yref="paper",
                text=grp, showarrow=False,
                xanchor="center", yanchor="middle",
                font=dict(size=group_label_size, color="#666666"),
            )

    fig.update_layout(
        uniformtext=dict(minsize=10, mode="show"),
        margin=dict(t=round(title_size * 2.2), r=20, b=20, l=20),
        title=title_layout(),
    )
    panel_bg(fig)
    return fig
