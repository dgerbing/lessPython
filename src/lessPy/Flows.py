# Flows.py — analog of Flows.R.
#
# Flows(): a Sankey flow diagram, stage1 -> stage2 -> optional
# stage3, with ribbon widths weighted by value. Each source's
# flow keeps its color through every stage. lessR's Flows is
# already plotly-based, so this is a direct go.Sankey port.
# Returns the plotly figure.

import numpy as np
import pandas as pd

from .plotly_utils import BASE_COLORS, make_trans, to_hex
from .utils import get_column, get_option


def Flows(value, stage1, stage2, stage3=None, data=None,
          title=None, fill=None, link_alpha=0.55,
          nodes_gray=False, neutral_gray="gray80",
          border_gray="gray60", lift_y=0, labels_size=1.0,
          digits_d=0):
    """Sankey flow diagram of value across the stages stage1 ->
    stage2 (-> stage3). value, stage1, stage2, stage3 are column
    names in data. Each source's flow keeps its color. Returns the
    plotly figure. R analog: Flows()"""
    import plotly.graph_objects as go
    if data is None:
        raise ValueError("data= is required: a DataFrame with "
                         "the value and stage columns")
    v = get_column(data, value, "value").to_numpy(dtype=float)
    s1 = get_column(data, stage1, "stage1").astype(str)
    s2 = get_column(data, stage2, "stage2").astype(str)
    s3 = (get_column(data, stage3, "stage3").astype(str)
          if stage3 is not None else None)

    sources = list(pd.unique(s1))          # first-seen order
    mids = list(pd.unique(s2))
    dests = list(pd.unique(s3)) if s3 is not None else []
    nA, nB, nC = len(sources), len(mids), len(dests)
    nodes = sources + mids + dests
    i_src = {lv: i for i, lv in enumerate(sources)}
    i_mid = {lv: nA + i for i, lv in enumerate(mids)}
    i_dest = {lv: nA + nB + i for i, lv in enumerate(dests)}

    # source palette (one color per stage1 level); the flow color
    df = pd.DataFrame({"s1": s1, "s2": s2, "v": v})
    pal = {lv: to_hex(fill[i % len(fill)]) if fill else
           to_hex(BASE_COLORS[i % len(BASE_COLORS)])
           for i, lv in enumerate(sources)}

    # links: stage1->stage2, then (colored by stage1) stage2->stage3
    src, tgt, val, lcol = [], [], [], []
    for (a, b), g in df.groupby(["s1", "s2"], sort=False):
        src.append(i_src[a])
        tgt.append(i_mid[b])
        val.append(float(g["v"].sum()))
        lcol.append(make_trans(pal[a], link_alpha))
    if dests:
        df3 = pd.DataFrame({"s1": s1, "s2": s2, "s3": s3, "v": v})
        for (a, b, c), g in df3.groupby(["s1", "s2", "s3"],
                                        sort=False):
            src.append(i_mid[b])
            tgt.append(i_dest[c])
            val.append(float(g["v"].sum()))
            lcol.append(make_trans(pal[a], link_alpha))

    # node colors and positions
    if nodes_gray:
        node_color = [to_hex(neutral_gray)] * len(nodes)
    else:
        node_color = ([pal[lv] for lv in sources]
                      + [to_hex(neutral_gray)] * (nB + nC))
    x_mid = 0.5 if nC else 0.98
    node_x = ([0.02] * nA + [x_mid] * nB + [0.98] * nC)

    def y_even(n):
        return ([0.5] if n <= 1
                else list(np.linspace(0.1, 0.9, n)))
    node_y = y_even(nA) + y_even(nB) + (y_even(nC) if nC else [])

    arrow = "→"
    if title:
        ttl = f"<b>{title}</b>"
    elif dests:
        ttl = (f"<b>{stage1} {arrow} {stage2} {arrow} "
               f"{stage3}</b>")
    else:
        ttl = f"<b>{stage1} {arrow} {stage2}</b>"

    h = max(0.80, min(0.94, 0.94 - 0.50 * abs(lift_y)))
    y1 = max(0.0, min(0.02 + lift_y, 1 - h))
    y_dom = [y1, min(1.0, y1 + h)]

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        domain=dict(x=[0, 1], y=y_dom),
        node=dict(
            label=nodes, color=node_color, x=node_x, y=node_y,
            pad=12, thickness=16,
            line=dict(color=to_hex(border_gray), width=1),
            hovertemplate="%{label}<extra></extra>"),
        link=dict(
            source=src, target=tgt, value=val, color=lcol,
            hovertemplate=(f"%{{source.label}} {arrow} "
                           "%{target.label}<br>"
                           f"{value}: %{{value:,.{digits_d}f}}"
                           "<extra></extra>"))))
    fig.update_layout(
        title=dict(text=ttl, x=0.5,
                   y=min(0.98, y_dom[1] + 0.03),
                   xanchor="center", yanchor="top",
                   font=dict(size=round(18 * labels_size))),
        margin=dict(t=90, r=30, b=30, l=30),
        font=dict(size=round(15 * labels_size)),
        template=None,
        paper_bgcolor=to_hex(get_option("window_fill", "white")))
    return fig
