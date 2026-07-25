# bubble_plotly.py — analog of bubble.plotly.R
#
# Renders bubbles from ALREADY-TABULATED input:
#   1-D  pandas Series: index = categories -> one row of bubbles
#        along the x-axis, sized by value, with stems below
#   2-D  pandas DataFrame: rows = `by` levels, columns = x
#        categories -> bubble matrix (balloon plot), one bubble
#        per cell, sized by the cell value
#
# Bubble diameter: max_px * v^power / max(v^power), with
# max_px = 2 * radius * dpi — size encodes magnitude, power
# tempers the visual dominance of large values.
#
# facet_tbls= renders the faceted 1-D grid: an ordered dict
# {facet level: 1-D Series on the full category set}, one panel
# per level, with bubble sizes and category hues normalized
# globally so the panels stay comparable.

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .utils import fmt, get_option
from .plotly_utils import (
    BASE_COLORS, contrast_text_for_hex, facet_layout, make_trans,
    plotly_style, to_hex,
)

_LABEL_VALUES = ("%", "input", "prop", "off")


def _diam_builder(all_values, power, radius, dpi):
    """Diameter function normalized to the global max.
    R analog: bubble.diam.builder()"""
    v = np.maximum(0, np.asarray(all_values, dtype=float))
    with np.errstate(invalid="ignore"):
        max_ru = np.nanmax(v ** power) if v.size else np.nan
    max_px_diam = 2 * radius * dpi

    def diam(vals):
        vals = np.maximum(0, np.asarray(vals, dtype=float))
        if not np.isfinite(max_ru) or max_ru <= 0:
            return np.zeros(len(vals))
        return max_px_diam * (vals ** power) / max_ru
    return diam


def _label_text(vals, share_tot, labels, digits_d, lbl_d):
    if labels == "input":
        return [fmt(v, digits_d) for v in vals]
    if labels == "%":
        return [f"{100 * s:.{lbl_d}f}%" for s in share_tot]
    return [f"{s:.{lbl_d}f}" for s in share_tot]     # "prop"


def bubble_plotly(x=None, x_name=None, y_name=None, by_name=None,
                  x_lab=None, y_lab=None, main=None,
                  fill=None, border="black", opacity=None,
                  power=0.5, radius=0.50,
                  digits_d=0,
                  labels=None, labels_position="in",
                  labels_color=None, labels_size=0.90,
                  labels_decimals=None,
                  label_min_px=26, label_autocontrast=True,
                  facet_tbls=None, facet_name=None, n_col=None,
                  style_opts=None):

    if labels is None:
        labels = "%"                 # R analog: match.arg default
    if labels not in _LABEL_VALUES:
        raise ValueError(f"labels must be one of {_LABEL_VALUES}")

    if facet_tbls is not None:      # panels replace the x table
        x = next(iter(facet_tbls.values()))
    two_d = facet_tbls is None and isinstance(x, pd.DataFrame)
    if not two_d and not isinstance(x, pd.Series):
        raise TypeError("x must be a pandas Series (1-D) or "
                        "DataFrame (2-D, rows = by levels)")

    if x_name is None:
        x_name = (x.columns.name if two_d else x.index.name) or "x"
    if y_name is None:
        y_name = "Count" if two_d else (x.name or "Count")
    if two_d and by_name is None:
        by_name = x.index.name or "by"
    if style_opts is None:
        style_opts = plotly_style()
    if fill is None:
        fill = BASE_COLORS

    title_size = round(16 * get_option("main_size", 1))
    dpi = get_option("plotly_dpi", 96)

    alpha_fill = 0.85 if opacity is None else float(opacity)
    if not np.isfinite(alpha_fill):
        alpha_fill = 0.85
    alpha_fill = max(0.0, min(1.0, alpha_fill))

    # "prop" plots proportions, which digits_d (0 for counts)
    # would flatten to "0"
    lbl_d = (int(labels_decimals) if labels_decimals is not None
             else (2 if labels == "prop"
                   else max(0, int(digits_d))))
    trace_mode = "markers" if labels == "off" else "markers+text"
    # None = the default (R's labels_position %||% "in")
    txt_pos = ("middle center"
               if labels_position is None
               or str(labels_position).lower() == "in"
               else "top center")
    txt_size = round(14 * labels_size * get_option("axis_size", 0.9))

    lab_font = round(15 * get_option("lab_size", 1))
    tick_font = round(16 * get_option("axis_size", 0.9))

    def label_colors(base_hex, n):
        if labels_color is not None:
            return [to_hex(labels_color)] * n
        if label_autocontrast:
            return [contrast_text_for_hex(h) for h in base_hex]
        return ["black"] * n

    def title_layout(fig):
        if main:
            fig.update_layout(
                title=dict(text=main, x=0.5, xanchor="center",
                           y=0.98, yanchor="top",
                           font=dict(size=title_size,
                                     color=to_hex(get_option(
                                         "lab_color", "black")))),
                margin=dict(t=round(title_size * 2.2),
                            r=25, b=30, l=25))

    # ---------- faceted 1-D grid ------------------------------------
    # One 1-D bubble panel per facet level; sizes and category hues
    # are normalized globally so the panels stay comparable.
    # R analog: bubble.plotly.R faceted 1-D path
    if facet_tbls is not None:
        fac_levels = [str(k) for k in facet_tbls.keys()]
        tabs = list(facet_tbls.values())

        all_vals = np.concatenate(
            [t.to_numpy(dtype=float) for t in tabs])
        diam_fun = _diam_builder(all_vals, power, radius, dpi)
        max_diam = float(np.nanmax(diam_fun(all_vals))) \
            if all_vals.size else 1.0
        if not np.isfinite(max_diam) or max_diam <= 0:
            max_diam = 1.0

        # color keyed by category name, hues shared across panels
        cats_all = [str(c) for c in tabs[0].index]
        cat_hex_map = {c: to_hex(fill[i % len(fill)]
                                 if isinstance(fill, (list, tuple))
                                 else fill)
                       for i, c in enumerate(cats_all)}

        def ann_adj(i, a, d):
            a["y"] = min(d["y"][1] + 0.04, 0.99)
            return a
        domains, anns = facet_layout(
            fac_levels, facet_name or "",
            yanchor="bottom", ann_adjust=ann_adj, n_col=n_col)

        target_r = 0.35
        y_top = target_r * 1.20
        seg_color = to_hex(style_opts["grid_color"])
        hover = (f"{x_name}: %{{customdata.xcat}}"
                 f"<br>{y_name}: %{{hovertext}}"
                 f"<br>% of {x_name}: %{{customdata.pct_x:.1%}}"
                 "<br>% of total: %{customdata.pct_tot:.1%}"
                 "<extra></extra>")

        fig = go.Figure()
        stems = []
        axes = {}
        for i, (lv, t) in enumerate(zip(fac_levels, tabs)):
            suf = "" if i == 0 else str(i + 1)
            cats = [str(c) for c in t.index]
            vals = t.to_numpy(dtype=float)
            n = len(cats)

            diam_px = diam_fun(vals)
            total = np.nansum(vals)
            share_tot = (vals / total if total > 0
                         else np.zeros(n))
            cat_hex = [cat_hex_map[c] for c in cats]
            fill_rgba = make_trans(cat_hex, alpha_fill)
            txt_cols = label_colors(cat_hex, n)

            fmt_val = [fmt(v, digits_d) for v in vals]
            if labels == "off":
                label_show = [""] * n
            else:
                txt = _label_text(vals, share_tot, labels,
                                  digits_d, lbl_d)
                label_show = [tx if np.isfinite(dp)
                              and dp >= label_min_px else ""
                              for tx, dp in zip(txt, diam_px)]
            customdata = [dict(xcat=c, bycat="", pct_x=float(s),
                               pct_tot=float(s))
                          for c, s in zip(cats, share_tot)]

            fig.add_trace(go.Scatter(
                mode=trace_mode,
                x=cats,
                y=[0] * n,
                xaxis=f"x{suf}",
                yaxis=f"y{suf}",
                hovertext=fmt_val,
                hovertemplate=hover,
                customdata=customdata,
                marker=dict(
                    symbol="circle",
                    size=diam_px.tolist(),
                    sizemode="diameter",
                    color=fill_rgba,
                    line=dict(color=to_hex(border), width=1),
                    sizemin=12,
                ),
                text=label_show,
                textposition=txt_pos,
                textfont=dict(size=txt_size, color=txt_cols),
                cliponaxis=False,
                showlegend=False,
            ))

            # stems from the panel bottom to the base of each bubble
            radius_data = (diam_px / max_diam) * target_r
            stems += [dict(type="line",
                           xref=f"x{suf}", yref=f"y{suf}",
                           x0=k, x1=k,
                           y0=-1, y1=-float(radius_data[k]),
                           line=dict(color=seg_color, width=1),
                           layer="below")
                      for k in range(n)]

            axes[f"xaxis{suf}"] = dict(
                type="category",
                domain=list(domains[i]["x"]),
                anchor=f"y{suf}",
                title=dict(text=x_lab if x_lab is not None
                           else x_name,
                           standoff=10,
                           font=dict(size=lab_font)),
                showgrid=False, zeroline=False, showline=True,
                ticks="outside",
                tickfont=dict(size=tick_font),
            )
            axes[f"yaxis{suf}"] = dict(
                visible=False, fixedrange=True,
                domain=list(domains[i]["y"]),
                anchor=f"x{suf}",
                range=[-1, y_top],
            )

        fig.update_layout(
            hoverlabel=dict(align="left"),
            annotations=anns,
            shapes=stems,
            margin=dict(t=round(title_size * 2.4) if main else 40,
                        b=8, l=20, r=20),
            showlegend=False,
            template=None,
            plot_bgcolor=to_hex(style_opts["panel_fill"]),
            paper_bgcolor=to_hex(style_opts["window_fill"]),
            **axes,
        )
        if main:
            fig.update_layout(title=dict(
                text=main, x=0.5, xanchor="center",
                y=0.98, yanchor="top",
                font=dict(size=title_size,
                          color=to_hex(get_option("lab_color",
                                                  "black")))))
        return fig

    # ---------- single 1-D chart ------------------------------------
    if not two_d:
        cats = [str(c) for c in x.index]
        vals = x.to_numpy(dtype=float)
        n = len(cats)

        diam_fun = _diam_builder(vals, power, radius, dpi)
        diam_px = diam_fun(vals)
        max_diam = float(np.nanmax(diam_px)) if n else 1.0
        if not np.isfinite(max_diam) or max_diam <= 0:
            max_diam = 1.0

        total = np.nansum(vals)
        share_tot = vals / total if total > 0 else np.zeros(n)

        # one palette color per category, as in the bar chart
        cat_hex = to_hex([fill[i % len(fill)]
                          if isinstance(fill, (list, tuple))
                          else fill for i in range(n)])
        fill_rgba = make_trans(cat_hex, alpha_fill)
        txt_cols = label_colors(cat_hex, n)

        fmt_val = [fmt(v, digits_d) for v in vals]
        if labels == "off":
            label_show = [""] * n
        else:
            txt = _label_text(vals, share_tot, labels, digits_d,
                              lbl_d)
            label_show = [t if np.isfinite(dp) and dp >= label_min_px
                          else "" for t, dp in zip(txt, diam_px)]

        customdata = [dict(xcat=c, bycat="", pct_x=float(s),
                           pct_tot=float(s))
                      for c, s in zip(cats, share_tot)]
        hover = (f"{x_name}: %{{customdata.xcat}}"
                 f"<br>{y_name}: %{{hovertext}}"
                 f"<br>% of {x_name}: %{{customdata.pct_x:.1%}}"
                 "<br>% of total: %{customdata.pct_tot:.1%}"
                 "<extra></extra>")

        fig = go.Figure(go.Scatter(
            mode=trace_mode,
            x=cats,
            y=[0] * n,
            hovertext=fmt_val,
            hovertemplate=hover,
            customdata=customdata,
            marker=dict(
                symbol="circle",
                size=diam_px.tolist(),
                sizemode="diameter",
                color=fill_rgba,
                line=dict(color=to_hex(border), width=1),
                sizemin=12,
            ),
            text=label_show,
            textposition=txt_pos,
            textfont=dict(size=txt_size, color=txt_cols),
            cliponaxis=False,
            showlegend=False,
        ))

        # stems from the plot bottom to the base of each bubble
        target_r = 0.35
        radius_data = (diam_px / max_diam) * target_r
        y_top = target_r * 1.20
        seg_color = to_hex(style_opts["grid_color"])
        stems = [dict(type="line", xref="x", yref="y",
                      x0=i, x1=i, y0=-1, y1=-float(radius_data[i]),
                      line=dict(color=seg_color, width=1),
                      layer="below")
                 for i in range(n)]

        fig.update_layout(
            hoverlabel=dict(align="left"),
            xaxis=dict(
                type="category",
            automargin=True,
                title=dict(text=x_lab if x_lab is not None
                           else x_name,
                           standoff=10,
                           font=dict(size=lab_font)),
                showgrid=False, zeroline=False, showline=True,
                ticks="outside",
                tickfont=dict(size=tick_font),
            ),
            yaxis=dict(visible=False, fixedrange=True,
                       range=[-1, y_top]),
            shapes=stems,
            template=None,
            plot_bgcolor=to_hex(style_opts["panel_fill"]),
            paper_bgcolor=to_hex(style_opts["window_fill"]),
        )
        title_layout(fig)
        return fig

    # ---------- 2-D bubble matrix ------------------------------------
    groups = [str(g) for g in x.index]
    cats = [str(c) for c in x.columns]
    mat = x.to_numpy(dtype=float)
    n_g = len(groups)

    total = np.nansum(mat)
    share_tot = mat / total if total > 0 else np.zeros(mat.shape)
    col_tot = np.nansum(mat, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        share_x = np.where(col_tot > 0, mat / col_tot, 0.0)

    diam_fun = _diam_builder(mat.ravel(), power, radius, dpi)

    # one color per by group
    grp_hex = to_hex([fill[i % len(fill)]
                      if isinstance(fill, (list, tuple)) else fill
                      for i in range(n_g)])
    grp_rgba = make_trans(grp_hex, alpha_fill)
    grp_txt = label_colors(grp_hex, n_g)

    hover = (f"{x_name}: %{{customdata.xcat}}"
             f"<br>{by_name}: %{{customdata.bycat}}"
             f"<br>{y_name}: %{{hovertext}}"
             f"<br>% of {x_name}: %{{customdata.pct_x:.1%}}"
             "<br>% of total: %{customdata.pct_tot:.1%}"
             "<extra></extra>")

    fig = go.Figure()
    for i, g in enumerate(groups):
        row = mat[i]
        diam_px = diam_fun(row)
        fmt_val = [fmt(v, digits_d) for v in row]
        if labels == "off":
            label_show = [""] * len(row)
        else:
            txt = _label_text(row, share_tot[i], labels, digits_d,
                              lbl_d)
            label_show = [t if np.isfinite(dp) and dp >= label_min_px
                          else "" for t, dp in zip(txt, diam_px)]
        customdata = [dict(xcat=c, bycat=g, pct_x=float(px),
                           pct_tot=float(pt))
                      for c, px, pt in zip(cats, share_x[i],
                                           share_tot[i])]
        fig.add_trace(go.Scatter(
            mode=trace_mode,
            x=cats,
            y=[g] * len(cats),
            hovertext=fmt_val,
            hovertemplate=hover,
            customdata=customdata,
            marker=dict(
                symbol="circle",
                size=diam_px.tolist(),
                sizemode="diameter",
                color=grp_rgba[i],
                line=dict(color=to_hex(border), width=1),
            ),
            text=label_show,
            textposition=txt_pos,
            textfont=dict(size=txt_size, color=grp_txt[i]),
            cliponaxis=False,
            showlegend=False,
        ))

    fig.update_layout(
        hoverlabel=dict(align="left"),
        xaxis=dict(
            type="category",
            automargin=True,
            title=dict(text=x_lab if x_lab is not None else x_name,
                       font=dict(size=lab_font)),
            showgrid=True,
            gridcolor=to_hex(style_opts["grid_color"]),
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=tick_font),
        ),
        yaxis=dict(
            type="category",
            automargin=True,
            title=dict(text=y_lab if y_lab is not None else by_name,
                       font=dict(size=lab_font)),
            showgrid=False, zeroline=False,
            categoryorder="array",
            categoryarray=list(reversed(groups)),
            tickfont=dict(size=tick_font),
        ),
        template=None,
        shapes=[dict(type="rect", xref="paper", yref="paper",
                     x0=0, x1=1, y0=0, y1=1,
                     line=dict(color=to_hex(
                         style_opts["panel_border"]), width=1),
                     layer="below")],
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]),
    )
    title_layout(fig)
    return fig
