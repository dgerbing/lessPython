# hs_plotly.py — analog of hs.plotly.R
#
# Renders a histogram from RAW numeric data plus pre-computed bin
# breaks (binning policy — Sturges, bin_width, etc. — is decided
# upstream in X(), as in lessR where .hst.main supplies h$breaks).
# Optional by= draws one translucent series per group, overlaid
# (default) or stacked.
#
# Small extension over the R source: proportion=True scales each
# group's counts to proportions (R's plotly path always passes
# counts and handles proportions only in the base-R renderer).
#
# facet= draws one panel per level on shared bins and a shared
# count scale (~ .bar.lattice T.type="hist"), first level in the
# bottom panel; proportions are per panel. With by=, each panel
# overlays (or stacks) one translucent series per group, and
# proportions and hover shares are per group within a panel.

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .utils import get_option, pretty
from .plotly_utils import (
    auto_opacity, axis_format, axis_num, facet_fig,
    facet_panels, finish_facet, get_tick_fmt,
    legend_style, make_trans, plot_border, plotly_style, to_hex,
    x_grid,
)


def hs_plotly(x, by=None, x_name=None, by_name=None,
              facet=None, facet_order=None, facet_name=None,
              facet2=None, facet2_order=None, facet2_name=None,
              breaks=None, freq=True, proportion=False,
              fill=None, border=None,
              cumulate="off", reg="snow2", counts=False,
              x_lab=None, y_lab=None,
              ax=None, gridT1=None,
              digits_d=2, position="overlay", main=None,
              n_col=1, axis_fmt="K", axis_x_pre="",
              axis_y_pre="", style_opts=None):

    if position not in ("overlay", "stack"):
        raise ValueError('position must be "overlay" or "stack"')
    if breaks is None or len(breaks) < 2:
        raise ValueError("breaks (bin edges) required, length >= 2")
    if style_opts is None:
        style_opts = plotly_style()
    if facet is not None:
        return _hs_facet(x, facet, facet_order, x_name,
                         facet_name, breaks, proportion, fill,
                         border, x_lab, y_lab, digits_d, main,
                         style_opts, by=by, by_name=by_name,
                         position=position, n_col=n_col,
                         axis_fmt=axis_fmt,
                         axis_x_pre=axis_x_pre,
                         axis_y_pre=axis_y_pre,
                         facet2=facet2,
                         facet2_order=facet2_order,
                         facet2_name=facet2_name)
    fill_miss = fill is None
    if fill_miss:
        fill = get_option("bar_fill_cont", "#96AAC3")
    if border is None:
        border = get_option("bar_color_cont", "#8496AF")
    if x_name is None:
        x_name = getattr(x, "name", None) or "x"
    if x_lab is None:
        x_lab = x_name
    if y_lab is None:
        y_lab = f"Count of {x_name}"

    x = np.asarray(x, dtype=float)
    breaks = [float(b) for b in breaks]

    # --- bin geometry ---
    left = np.array(breaks[:-1])
    right = np.array(breaks[1:])
    mids = (left + right) / 2
    widths = right - left
    n_bins = len(mids)

    # --- groups ---
    if by is None:
        groups = ["Series 1"]
    else:
        by = pd.Series(by).astype(str)
        groups = sorted(by.dropna().unique().tolist())
    G = len(groups)

    # groups need distinguishable hues: default to the palette
    if fill_miss and G > 1:
        from .plotly_utils import BASE_COLORS
        fill = BASE_COLORS

    # --- counts per bin (right-closed bins, as in R's hist) ---
    def bin_counts(xg):
        cats = pd.cut(xg, breaks, right=True, include_lowest=True)
        return cats.value_counts(sort=False).to_numpy(dtype=float)

    y_mat = np.zeros((n_bins, G))
    for g, gname in enumerate(groups):
        xg = x if by is None else x[(by == gname).to_numpy()]
        xg = xg[np.isfinite(xg)]
        y_mat[:, g] = bin_counts(pd.Series(xg))

    # density: normalize each group to unit area; proportion: to 1
    if not freq:
        for g in range(G):
            n_g = y_mat[:, g].sum()
            if n_g > 0:
                y_mat[:, g] = y_mat[:, g] / (n_g * widths)
    elif proportion:
        for g in range(G):
            n_g = y_mat[:, g].sum()
            if n_g > 0:
                y_mat[:, g] = y_mat[:, g] / n_g

    # cumulative histogram: cumsum the (single) series; "both"
    # overdraws the regular bars in the reg color (hst.main.R)
    y_orig = None
    if cumulate in ("on", "both"):
        y_orig = y_mat[:, 0].copy()
        y_mat[:, 0] = np.cumsum(y_mat[:, 0])
        y_lab = f"Cumulative {y_lab}"

    rng_txt = [f"({lv:g}, {rv:g}]" for lv, rv in zip(left, right)]

    def hover_group(yv):
        if freq and not proportion:
            amt = yv
            cum_lbl = "Cumulative"
        else:
            amt = yv * widths if not freq else yv
            cum_lbl = ("Cumulative (area)" if not freq
                       else "Cumulative")
        total = amt.sum()
        share = amt / total if total > 0 else np.full(n_bins, np.nan)
        cum = np.cumsum(amt)
        cum_pct = cum / total if total > 0 else share
        return [
            (f"Bin: {rng_txt[i]}"
             f"<br>{y_lab}: {round(yv[i], digits_d):g}"
             f"<br>% of total: {100 * share[i]:.1f}%"
             f"<br>{cum_lbl}: {round(cum[i], digits_d):g}"
             f"<br>Cumulative %: {100 * cum_pct[i]:.1f}%")
            for i in range(n_bins)
        ]

    # colors
    alpha_fill = auto_opacity(
        G if G > 1 else 1,
        "stack" if (G > 1 and position == "stack") else "overlay")
    fills = [make_trans(
        fill[g % len(fill)] if isinstance(fill, (list, tuple))
        else fill, alpha_fill) for g in range(G)]
    borders = [to_hex(
        border[g % len(border)] if isinstance(border, (list, tuple))
        else border) for g in range(G)]

    fig = go.Figure()
    for g, gname in enumerate(groups):
        yv = y_mat[:, g]
        if cumulate != "off":          # shares of a cumsum are
            hover = [                  # not meaningful
                f"Bin: {rng_txt[i]}<br>{y_lab}: "
                f"{round(yv[i], digits_d):g}"
                for i in range(n_bins)]
        else:
            hover = hover_group(yv)
        if G > 1 and by_name:
            hover = [f"{by_name}: {gname}<br>{h}" for h in hover]
        extra = {}
        if counts:                     # labels above the bars
            extra = dict(
                text=[f"{round(v, digits_d):g}" for v in yv],
                textposition="outside", cliponaxis=False,
                textfont=dict(color=to_hex(get_option(
                    "lab_color", "black"))))
        fig.add_trace(go.Bar(
            x=mids,
            y=yv,
            name=gname if G > 1 else None,
            width=widths * 0.98,
            marker=dict(
                color=fills[g],
                line=dict(color=borders[g], width=1),
            ),
            hoverinfo="text",
            hovertext=hover,
            showlegend=G > 1,
            **extra,
        ))
    if cumulate == "both":             # regular bars on top
        reg_hex = "#EEE9E9" if reg == "snow2" else to_hex(reg)
        fig.add_trace(go.Bar(
            x=mids, y=y_orig, width=widths * 0.98,
            marker=dict(color=reg_hex,
                        line=dict(color=borders[0], width=1)),
            hoverinfo="skip", showlegend=False))

    # axes, grid, layout
    if ax is None:
        from .utils import pretty
        fx = x[np.isfinite(x)]
        axT1 = pretty(float(fx.min()), float(fx.max()))
        ymax = (y_mat.sum(axis=1).max()
                if (G > 1 and position == "stack")
                else y_mat.max())
        axT2 = pretty(0, float(ymax))
        ax = dict(
            axT1=axT1,
            axL1=axis_format(axT1, digits_d, axis_fmt,
                             axis_x_pre),
            axT2=axT2,
            axL2=axis_format(axT2, digits_d, axis_fmt,
                             axis_y_pre))
    if gridT1 is None:
        gridT1 = breaks               # grid at the bin boundaries

    ax_x = axis_num(x_lab, ax["axT1"], ax["axL1"])
    ax_y = axis_num(y_lab, ax["axT2"], ax["axL2"])
    ax_y.update(showgrid=True,
                gridcolor=to_hex(style_opts["grid_col"]),
                gridwidth=1, griddash="dot")

    fig.update_layout(
        xaxis=ax_x,
        yaxis=ax_y,
        shapes=x_grid(gridT1) + plot_border(),
        template=None,
        barmode=("stack" if (G > 1 and position == "stack")
                 else "overlay"),
        bargap=0,
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]),
    )

    if G > 1:
        fig.update_layout(legend=legend_style(by_name, style_opts))

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


def _hs_facet(x, facet, facet_order, x_name, facet_name, breaks,
              proportion, fill, border, x_lab, y_lab, digits_d,
              main, style_opts, by=None, by_name=None,
              position="overlay", n_col=1, axis_fmt="K",
              axis_x_pre="", axis_y_pre="",
              facet2=None, facet2_order=None, facet2_name=None):
    """One histogram panel per facet level: shared bins, shared
    count scale, per-panel proportions. ~ .bar.lattice hist.
    With by=, one translucent series per group within each panel,
    overlaid (default) or stacked, legend from the bottom panel;
    proportions are then per group within a panel. facet2: the
    two-facet grid, rows = facet2 levels."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    x, facet = x[ok], np.asarray(facet)[ok]
    if facet2 is not None:
        facet2 = np.asarray(facet2)[ok]
    if facet_order is None:
        facet_order = sorted(set(facet))
    labels, pos, sel, n_row_g, n_col = facet_panels(
        facet, facet_order, facet2, facet2_order,
        facet_name, facet2_name, n_col)

    if by is None:
        groups = [None]
    else:
        by = pd.Series(by).astype(str).to_numpy()[ok]
        groups = sorted(set(by))
    G = len(groups)
    stack = G > 1 and position == "stack"

    if fill is None:
        if G > 1:
            from .plotly_utils import BASE_COLORS
            fill = BASE_COLORS
        else:
            fill = get_option("bar_fill_cont", "#96AAC3")
    if border is None:
        border = get_option("bar_color_cont", "#8496AF")
    if x_lab is None:
        x_lab = x_name
    if y_lab is None:
        y_lab = (f"Proportion of {x_name}" if proportion
                 else f"Count of {x_name}")

    breaks = [float(b) for b in breaks]
    left = np.array(breaks[:-1])
    right = np.array(breaks[1:])
    mids = (left + right) / 2
    widths = right - left
    n_bins = len(mids)
    rng_txt = [f"({lv:g}, {rv:g}]"
               for lv, rv in zip(left, right)]

    # y_mats[i][:, g]: bin heights for facet panel i, group g
    y_mats = []
    for in_lvl in sel:
        y_mat = np.zeros((n_bins, G))
        for g, gname in enumerate(groups):
            sel = (in_lvl if gname is None
                   else in_lvl & (by == gname))
            cats = pd.cut(pd.Series(x[sel]), breaks, right=True,
                          include_lowest=True)
            yv = cats.value_counts(sort=False).to_numpy(
                dtype=float)
            if proportion and yv.sum() > 0:
                yv = yv / yv.sum()
            y_mat[:, g] = yv
        y_mats.append(y_mat)

    ymax = max(float(m.sum(axis=1).max() if stack else m.max())
               for m in y_mats)
    axT1 = pretty(float(x.min()), float(x.max()))
    axT2 = pretty(0, ymax)
    fmt2 = get_tick_fmt(axT2, digits_d)
    ax = dict(axT1=axT1,
              axL1=axis_format(axT1, digits_d, axis_fmt,
                               axis_x_pre),
              axT2=axT2,
              axL2=[f"{v:.{digits_d}f}" if fmt2 else f"{v:g}"
                    for v in axT2])

    alpha_fill = auto_opacity(G if G > 1 else 1,
                              "stack" if stack else "overlay")
    fills = [make_trans(
        fill[g % len(fill)] if isinstance(fill, (list, tuple))
        else fill, alpha_fill) for g in range(G)]
    borders = [to_hex(
        border[g % len(border)] if isinstance(border, (list, tuple))
        else border) for g in range(G)]

    pct_lbl = "% of panel" if G == 1 else "% of group"
    fig = facet_fig(n_row_g, n_col)
    for i, lab in enumerate(labels):
        p_row, p_col = pos[i]
        panel_txt = (lab if facet2 is not None
                     else f"{facet_name}: {lab}")
        for g, gname in enumerate(groups):
            yv = y_mats[i][:, g]
            total = yv.sum()
            hover = [
                (f"Bin: {rng_txt[j]}"
                 f"<br>{y_lab}: {round(yv[j], digits_d):g}"
                 f"<br>{pct_lbl}: "
                 f"{100 * yv[j] / total if total else 0:.1f}%"
                 f"<br>{panel_txt}")
                for j in range(n_bins)
            ]
            if gname is not None and by_name:
                hover = [f"{by_name}: {gname}<br>{h}"
                         for h in hover]
            fig.add_trace(go.Bar(
                x=mids, y=yv, width=widths * 0.98,
                name=gname if G > 1 else None,
                legendgroup=gname if G > 1 else None,
                marker=dict(color=fills[g],
                            line=dict(color=borders[g], width=1)),
                hoverinfo="text", hovertext=hover,
                showlegend=G > 1 and i == 0,
            ), row=p_row, col=p_col)  # first level bottom

    fig.update_layout(bargap=0,
                      barmode="stack" if stack else "overlay")
    finish_facet(fig, labels, ax, x_lab, y_lab,
                 gridT1=breaks, style_opts=style_opts,
                 n_col=n_col, pos=pos)
    if G > 1:
        fig.update_layout(legend=legend_style(by_name,
                                              style_opts))
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
