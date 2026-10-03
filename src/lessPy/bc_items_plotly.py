# bc_items_plotly.py — the multi-item stacked chart, Chart(x=[...]),
# for a set of items that share one response scale, such as Likert
# items. R analog: the multi-x path of Chart.R, its frequency table
# from .bc.main() (multi), drawn there with base graphics, and the
# faceted form .bar.stackedLattice(); this is the plotly design.
#
# Each item is one horizontal bar, divided into the counts of its
# responses in a divergent palette, so the bars compose the response
# distributions of all the items on one scale. Unfaceted, the items
# are ordered by their mean response, the largest at the top; each
# segment carries its percent of the item, omitted below 4%. With
# facet=, one panel per level, the items in the order given, first
# at the top, on one count scale (drawn by bc_facet_plotly).

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .utils import get_option, pretty
from .plotly_utils import (
    as_plotly_color, auto_text_color, axis_cat, axis_format,
    axis_num, legend_style, plot_border, plotly_style, to_hex,
    x_grid, y_grid,
)

# theme -> the two sequential palettes of the divergent fill
# R analog: .get_fill(theme, diverge=TRUE)
_DIVERGE = {
    "gray": ("grays", "grays"), "white": ("grays", "grays"),
    "darkred": ("turquoises", "reds"), "red": ("turquoises", "reds"),
    "rose": ("turquoises", "reds"),
    "slatered": ("turquoises", "reds"),
    "darkgreen": ("violets", "greens"),
    "green": ("violets", "greens"), "purple": ("violets", "greens"),
}


def items_fill(theme, n):
    """The divergent palette of n response colors for a theme, the
    low responses in the first hue. R analog: .color_range(
    .get_fill(theme, diverge=TRUE), n)"""
    from .getColors import getColors
    p1, p2 = _DIVERGE.get(theme or "colors", ("browns", "blues"))
    return [c[:7] for c in getColors(p1, p2, n=max(n, 2),
                                     quiet=True)][:n]


def _resp_str(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def items_table(cols, names):
    """Responses x items table of counts over the union of the
    response values, with the weighted mean response of each item.
    Items that do not share one response set compose no single
    chart: raise. R analog: .bc.main() multi tabulation and the
    one_plot test of Chart.R"""
    # one_plot: every item's responses fall within the largest set
    sets = [set(c.dropna().unique()) for c in cols]
    widest = max(sets, key=len)
    if any(s - widest for s in sets):
        raise ValueError(
            "The x variables do not share one set of response "
            "categories, so they do not compose a single multi-item "
            "chart. Chart each variable separately.")
    first = cols[0]
    if isinstance(first.dtype, pd.CategoricalDtype):
        resp = list(first.cat.categories)
        for c in cols[1:]:
            if isinstance(c.dtype, pd.CategoricalDtype):
                resp += [v for v in c.cat.categories if v not in resp]
    else:
        vals = set()
        for c in cols:
            vals |= set(c.dropna().unique())
        resp = sorted(vals)
    frq = pd.DataFrame(
        {nm: c.value_counts().reindex(resp, fill_value=0)
         for nm, c in zip(names, cols)}, index=resp)
    frq.index = [_resp_str(r) for r in resp]
    numeric = all(isinstance(r, (int, float, np.integer, np.floating))
                  for r in resp)
    wt = (np.array(resp, dtype=float) if numeric
          else np.arange(1, len(resp) + 1, dtype=float))
    with np.errstate(invalid="ignore", divide="ignore"):
        wm = pd.Series((frq.to_numpy(dtype=float) * wt[:, None]).sum(0)
                       / frq.to_numpy(dtype=float).sum(0),
                       index=names)
    return frq, wm, numeric


def items_stats_lines(frq, wm, title, numeric=True, with_sum=False):
    """'Frequencies of Responses by Variable': one row per item,
    its counts and its mean. R analog: .prntbl(t(x), 0) with the
    Mean column of .bc.main() multi. with_sum adds the Sum column of
    the bubble plot frequency matrix, whose Mean column follows at
    one space and is sized to the mean at three decimals, as
    .dpmat.main() writes it"""
    t = frq.T.copy()
    if with_sum:
        t["Sum"] = t.sum(axis=1)
    rows = [str(r) for r in t.index]
    max_c1 = max(len(r) for r in rows) + 2
    widths = [max(4, max(len(str(c)),
                         max(len(f"{v:.0f}") for v in t[c])) + 1)
              for c in t.columns]
    mx = max(len(f"{round(v, 3 if with_sum else 2):.15g}")
             for v in wm if np.isfinite(v))
    sep = " " if with_sum else "   "
    hdr = " " * max_c1 + "".join(str(c).rjust(w)
                                 for c, w in zip(t.columns, widths))
    lines = [title, "", hdr + sep + "Mean".rjust(mx + 1)]
    for r in t.index:
        ln = ("  " + str(r)).rjust(max_c1) + "".join(
            f"{v:.0f}".rjust(w) for v, w in zip(t.loc[r], widths))
        lines.append(ln + sep + f"{wm[r]:.3f}".rjust(mx + 1))
    if not numeric:
        lines += ["", "Computation of the mean based on coding "
                  "response categories from 1 to "
                  f"{len(frq.index)}"]
    return lines + [""]


def bc_items_plotly(frq, order, fill, labels="%", labels_size=None,
                    labels_cut=0.04, stack100=False, x_lab="",
                    y_lab="", legend_title="Responses", main=None,
                    digits_d=0, axis_fmt="K", axis_x_pre="",
                    rotate_x=0, rotate_y=0, horiz=True,
                    style_opts=None):
    """One bar per item, stacked by response. frq: responses x items
    counts; order: the items bottom to top, or left to right when
    horiz=False (the vertical chart)."""
    if style_opts is None:
        style_opts = plotly_style()
    cnt = frq[order].to_numpy(dtype=float)
    tot = cnt.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(tot > 0, cnt / tot, 0.0)
    mat = share if stack100 else cnt
    if labels_size is None:
        labels_size = 0.8 - 0.008 * len(order)
    txt_size = round(16 * labels_size)
    fills = [as_plotly_color(f) for f in fill]
    txt_cols = auto_text_color(fills, bg=style_opts["panel_fill"])

    fig = go.Figure()
    for i, r in enumerate(frq.index):
        if labels == "off":
            txt = None
        else:
            txt = [("" if s < labels_cut else
                    f"{100 * s:.0f}%" if labels == "%" else
                    f"{s:.2f}" if labels == "prop" else
                    f"{c:.0f}")
                   for s, c in zip(share[i], cnt[i])]
        fig.add_trace(go.Bar(
            x=mat[i] if horiz else list(order),
            y=list(order) if horiz else mat[i],
            orientation="h" if horiz else "v", name=str(r),
            marker=dict(color=fills[i % len(fills)],
                        line=dict(width=0)),
            text=txt, textposition="inside",
            insidetextanchor="middle", textangle=0,
            textfont=dict(size=txt_size,
                          color=txt_cols[i % len(txt_cols)]),
            customdata=np.stack([cnt[i], share[i]], axis=-1),
            hovertemplate=(("%{y}" if horiz else "%{x}")
                           + f"<br>{legend_title}: {r}"
                           "<br>Count: %{customdata[0]:.0f}"
                           "<br>% of item: %{customdata[1]:.1%}"
                           "<extra></extra>"),
        ))

    top = 1.0 if stack100 else float(np.nanmax(tot)) if tot.size else 1
    axT = pretty(0.0, top)
    if stack100:
        d = 2
    else:
        d = digits_d
    val_ax = axis_num(x_lab if horiz else y_lab, axT,
                      axis_format(axT, d, axis_fmt, axis_x_pre))
    val_ax.update(range=[0, axT[-1] * (1 if stack100 else 1.02)])
    cat_ax = axis_cat(y_lab if horiz else x_lab)
    cat_ax.update(categoryorder="array", categoryarray=list(order))
    x_ax, y_ax = (val_ax, cat_ax) if horiz else (cat_ax, val_ax)
    if rotate_x:
        x_ax["tickangle"] = -rotate_x
    if rotate_y:
        y_ax["tickangle"] = -rotate_y
    leg = legend_style(legend_title, style_opts)
    leg["traceorder"] = "normal"
    fig.update_layout(
        barmode="stack", bargap=0.2,
        xaxis=x_ax, yaxis=y_ax,
        shapes=(x_grid(axT) if horiz else y_grid(axT)) + plot_border(),
        legend=leg, template=None,
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]),
    )
    if main:
        title_size = round(16 * get_option("main_size", 1))
        fig.update_layout(
            title=dict(text=main, x=0.5, xanchor="center", y=0.98,
                       yanchor="top", font=dict(size=title_size)),
            margin=dict(t=round(title_size * 2.2)))
    return fig
