# hier_plotly.py — analog of hier.plotly.R (.hier.plotly) plus its
# data preparation .hier_aggregate() and a simplified
# .hier_color_resolve() (both in zzz_plotly.R; placed here per the
# single-caller convention — only the hier path uses them).
#
# Renders the three hierarchical part-of-whole views from RAW
# (case-level) data: sunburst, treemap, icicle. Unlike the other
# renderers, aggregation happens here at every nesting level, so
# the input is the raw x (and optional by / y) columns, not a
# pre-built table.
#
# Node model (plotly ids/labels/parents/values):
#   level 1: one node per x category, parent "" (the tree roots)
#   level 2: one node per (x, by) combination, nested inside x
#   values are set on leaves only, branchvalues="remainder"
#
# Deliberate deviations from the R source:
#   - Theme-dependent sequential palettes (.scale.clr) are not
#     ported; fills default to BASE_COLORS keyed by top-level
#     category (lessR's default "colors" theme). A sequential
#     palette NAMED in fill= is ported (read by magnitude off the
#     palette's ramp, R's pal_explicit branch), and with facet=
#     each panel reads its own shades off that ramp
#     (fill_vec_byfac); other fills are the same in every panel.

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .getColors import _SEQ_NAMES

# the palettes read by magnitude, R's seq.pals of .hier_color_resolve()
_HIER_SEQ = set(_SEQ_NAMES) | {"terrain"}
from .utils import STAT_FUN, get_option
from .plotly_utils import (
    BASE_COLORS, facet_layout, plotly_style, to_hex,
)

_LABEL_VALUES = ("input", "%", "prop", "off")


def hier_aggregate(x, by=None, y=None, stat=None, facet=None,
                   x_name="X", by_name=None, y_name="y",
                   facet_name=None, facet_order=None):
    """Build the plotly node table (ids/labels/parents/values/
    customdata/root) for sunburst/treemap/icicle. With facet=,
    one node table per facet level (nd_byfac), each carrying a
    facet line in its hover meta; nd is then None.
    R analog: .hier_aggregate()"""
    path = pd.DataFrame({x_name: x.astype(str).values})
    lvl_names = [x_name]
    # a list of by variables nests one level deeper for each, the
    #   depth that the hierarchical forms accept
    by_list = (list(by) if isinstance(by, (list, tuple))
               else [] if by is None else [by])
    nm_list = (list(by_name) if isinstance(by_name, (list, tuple))
               else [by_name or "by"])
    for k, b in enumerate(by_list):
        by_col = nm_list[k] if k < len(nm_list) else f"by{k + 1}"
        path[by_col] = b.astype(str).values
        lvl_names.append(by_col)
    n_levels = len(lvl_names)

    if y is None:
        vals = np.ones(len(path))
    else:
        vals = y.to_numpy(dtype=float)
        vals = np.where(np.isfinite(vals), vals, 0.0)

    STAT = (stat or "sum").lower()
    stat_fun = STAT_FUN[STAT]
    non_additive = STAT != "sum"

    if y is None:
        stat_label = "Count"
    elif non_additive:
        stat_label = f"{STAT.capitalize()} of {y_name}"
    else:
        stat_label = y_name

    df = path.copy()
    df["..val"] = vals

    def build_nodes(sub, facet_label=None):
        ids, labels, parents, values, custom, root = \
            [], [], [], [], [], []
        for lvl in range(1, n_levels + 1):
            cols = lvl_names[:lvl]
            g = sub.groupby(cols, sort=True)
            agg_stat = g["..val"].apply(stat_fun)
            node_n = g.size()

            for key, sv in agg_stat.items():
                key = (key,) if lvl == 1 else key
                label = str(key[-1])
                if lvl == 1:
                    parent = ""
                    node_id = f"I_{label}"
                else:
                    parent = "I_" + "_".join(
                        str(k) for k in key[:-1])
                    node_id = f"{parent}_{label}"
                meta_parts = [f"{lvl_names[k]}: {key[k]}"
                              for k in range(lvl)]
                if facet_label is not None:
                    meta_parts.append(
                        f"{facet_name or 'Facet'}: {facet_label}")
                ids.append(node_id)
                labels.append(label)
                parents.append(parent)
                values.append(float(sv) if lvl == n_levels
                              else 0.0)
                custom.append(dict(meta="<br>".join(meta_parts),
                                   stat=float(sv),
                                   n=int(node_n[key if lvl > 1
                                                 else key[0]]),
                                   level=lvl))
                root.append(str(key[0]))
        return dict(ids=ids, labels=labels, parents=parents,
                    values=values, customdata=custom, root=root)

    if facet is None:
        nd = build_nodes(df)
        nd_byfac = None
    else:
        fac = facet.astype(str).to_numpy()
        if facet_order is None:
            facet_order = sorted(set(fac))
        nd = None
        nd_byfac = {str(lv): build_nodes(df[fac == str(lv)],
                                         facet_label=lv)
                    for lv in facet_order}

    return dict(nd=nd, nd_byfac=nd_byfac, stat_label=stat_label,
                non_additive=non_additive, STAT=STAT,
                n_levels=n_levels, has_y=y is not None)


def _seq_palette_fill(top_levels, pal, x, y, stat):
    """A named sequential palette read by magnitude: each top-level
    category takes the shade of the palette's 256-color ramp at the
    position of its count (or stat of y) between the smallest and
    the largest, so larger is darker in the palette's own hues.
    R analog: the pal_explicit branch of .hier_color_resolve()."""
    from .getColors import getColors, hcl
    if y is None:
        stats = x.value_counts()
    else:
        vals = pd.to_numeric(y, errors="coerce")
        vals = vals.where(np.isfinite(vals), 0)
        stats = vals.groupby(x.astype(str)).agg(
            STAT_FUN[(stat or "sum").lower()])
    stats = stats.reindex([str(lv) for lv in top_levels]).to_numpy(
        dtype=float)
    lo, hi = stats.min(), stats.max()
    if len(top_levels) < 2 or hi - lo <= 0:
        # no range to read a shade off: one mid-luminance color
        #   in the default theme's hue, as R
        return {lv: hcl(240, 75, 60) for lv in top_levels}
    ramp = [c[:7] for c in getColors(pal, n=256, quiet=True)]
    pos = (stats - lo) / (hi - lo)       # 0 = smallest, 1 = largest
    idx = np.round(pos * (len(ramp) - 1)).astype(int)
    return {lv: ramp[i] for lv, i in zip(top_levels, idx)}


def hier_color_resolve_byfac(top_levels, fill, x, y, stat, facet,
                             facet_order):
    """Per-panel fills for a faceted chart: with a named sequential
    palette, each panel's categories take their shades from that
    panel's own counts (or stat of y), so each panel spans the
    palette; None for any other fill, which every panel shares.
    R analog: the fill_vec_byfac branch of .hier_color_resolve()"""
    if not (isinstance(fill, str) and fill.lower() in _HIER_SEQ):
        return None
    fac = facet.astype(str)
    out = {}
    for lv in facet_order:
        m = (fac == str(lv)).to_numpy()
        present = set(x[m].astype(str))
        tops = [t for t in top_levels if str(t) in present]
        if tops:
            out[str(lv)] = _seq_palette_fill(
                tops, fill.lower(), x[m].astype(str),
                None if y is None else y[m], stat)
    return out


def hier_color_resolve(top_levels, fill=None, x=None, y=None,
                       stat=None):
    """Hex color per top-level category, inherited down the tree.
    R analog: .hier_color_resolve(). A sequential palette name
    (fill="greens") is read by magnitude from x and y; the
    theme-dependent sequential branch (.scale.clr) is not
    ported."""
    if fill is None:
        fill = BASE_COLORS
    if (isinstance(fill, str) and fill.lower() in _HIER_SEQ
            and x is not None):
        return _seq_palette_fill(top_levels, fill.lower(), x, y,
                                 stat)
    if isinstance(fill, dict):
        base = list(fill.values()) or ["#CCCCCC"]
        return {lv: to_hex(fill.get(lv, base[i % len(base)]))
                for i, lv in enumerate(top_levels)}
    if not isinstance(fill, (list, tuple)):
        fill = [fill]
    return {lv: to_hex(fill[i % len(fill)])
            for i, lv in enumerate(top_levels)}


def hier_plotly(agg, fill_vec, type="sunburst", fill_vec_byfac=None,
                x_name="X", by_name=None, facet_name=None,
                main=None, border=None, digits_d=None,
                labels=None, labels_color="white",
                labels_size=0.75, n_col=None, style_opts=None):

    if labels is None:
        labels = "input"             # R analog: match.arg default
    if labels not in _LABEL_VALUES:
        raise ValueError(f"labels must be one of {_LABEL_VALUES}")
    if type not in ("sunburst", "treemap", "icicle"):
        raise ValueError('type must be "sunburst", "treemap", '
                         'or "icicle"')
    if style_opts is None:
        style_opts = plotly_style()

    nd = agg["nd"]
    nd_byfac = agg.get("nd_byfac")
    non_additive = agg["non_additive"]
    n_levels = agg["n_levels"]
    has_y = agg["has_y"]
    stat_label = agg["stat_label"]

    title_size = round(16 * get_option("main_size", 1))
    label_fsize = 18 * labels_size * get_option("axis_size", 0.9)
    border_hex = to_hex("white" if border is None else border)

    digits = 2 if digits_d is None else max(0, int(digits_d))
    if non_additive and digits == 0:
        digits += 2                  # means etc. need decimals

    fmt_spec = f"%{{customdata.stat:,.{digits}f}}"

    # texttemplate reaches customdata on parent nodes (value 0),
    # where textinfo+text would suppress the text
    use_template = ((non_additive and labels != "off")
                    or labels == "prop"
                    or labels == "input")
    if labels == "input":
        txtinfo = "none"                 # value drawn via texttemplate
    elif labels == "%":
        txtinfo = "label+percent root"
    elif labels == "prop":
        txtinfo = "none"
    else:                            # "off": name only, no value
        txtinfo = "label"

    if use_template and labels != "prop":
        texttemplate = f"%{{label}}<br>{fmt_spec}"
    elif labels == "prop":
        texttemplate = "%{label}<br>%{percentRoot:.3f}"
    else:
        texttemplate = None

    hover_lines = ["%{customdata.meta}",
                   f"{stat_label}: {fmt_spec}"]
    if not non_additive:
        hover_lines.append("% of parent: %{percentParent:.2%}")
    hover_lines.append("% of total: %{percentRoot:.2%}")
    hover = "<br>".join(hover_lines) + "<extra></extra>"

    trace_cls = {"sunburst": go.Sunburst,
                 "treemap": go.Treemap,
                 "icicle": go.Icicle}[type]

    kw = {}
    if txtinfo is not None and texttemplate is None:
        kw["textinfo"] = txtinfo
    if texttemplate is not None:
        kw["texttemplate"] = texttemplate

    def make_trace(nd_i, dom_x, dom_y, cols=None):
        cols = fill_vec if not cols else cols
        node_cols = [cols.get(r, list(cols.values())[0])
                     for r in nd_i["root"]]
        return trace_cls(
            ids=nd_i["ids"],
            labels=nd_i["labels"],
            parents=nd_i["parents"],
            values=nd_i["values"],
            customdata=nd_i["customdata"],
            domain=dict(x=list(dom_x), y=list(dom_y)),
            branchvalues="remainder",
            textfont=dict(size=label_fsize, color=labels_color),
            hovertemplate=hover,
            marker=dict(colors=node_cols,
                        line=dict(color=border_hex, width=1)),
            **kw,
        )

    lab_color = to_hex(get_option("lab_color", "black"))

    # ---- faceted grid: one panel per facet level -------------------
    # R analog: hier.plotly.R faceted section
    if nd_byfac is not None:
        fac_levels = list(nd_byfac.keys())

        if type == "treemap":
            def dom_adj(d):
                span = d["y"][1] - d["y"][0]
                y0 = max(0.0, d["y"][0] - 0.06)
                y1 = y0 + span
                if y1 > 1:
                    y1, y0 = 1.0, 1.0 - span
                d["y"] = (y0, y1)
                return d

            def ann_adj(i, a, d):
                a["y"] = min(d["y"][1] - 0.025, 0.99)
                a["x"] = (d["x"][0] + d["x"][1]) / 2
                return a
            rev_dom = True
        elif type == "icicle":
            def dom_adj(d):
                span = d["y"][1] - d["y"][0] * 1.12
                y1 = max(0.0, d["y"][1] - 0.10)
                y0 = max(0.0, y1 - span)
                if y1 > 1:
                    y1, y0 = 1.0, 1.0 - span
                d["y"] = (y0, y1)
                return d

            def ann_adj(i, a, d):
                a["y"] = min(d["y"][1] + 0.02, 0.99)
                a["x"] = a["x"] + 0.10
                a["xanchor"] = "right"
                return a
            rev_dom = False
        else:                        # sunburst
            def dom_adj(d):
                span = d["y"][1] - d["y"][0]
                y1 = min(1.0, d["y"][1] - 0.06)
                y0 = max(0.0, y1 - span)
                d["y"] = (y0, y1)
                return d

            def ann_adj(i, a, d):
                a["y"] = min(d["y"][1] - 0.025, 0.99)
                a["x"] = (d["x"][0] + d["x"][1]) / 2
                return a
            rev_dom = False

        domains, anns = facet_layout(
            fac_levels, facet_name or "",
            domain_adjust=dom_adj, reverse_domains=rev_dom,
            y_base=-0.018, y_row_shift=-0.003, yanchor="bottom",
            ann_adjust=ann_adj, n_col=n_col)

        fig = go.Figure()
        for i, lv in enumerate(fac_levels):
            nd_i = nd_byfac[lv]
            if nd_i["ids"]:
                fig.add_trace(make_trace(
                    nd_i, domains[i]["x"], domains[i]["y"],
                    cols=(fill_vec_byfac or {}).get(str(lv))))

        t_margin = round(title_size * 2.8)
        fig.update_layout(
            annotations=anns,
            margin=dict(
                t=t_margin - 5 if type == "treemap" else t_margin,
                b=8,
                l=0 if type == "icicle" else 10,
                r=30 if type == "icicle" else 20),
            title=dict(
                text=main,
                font=dict(size=title_size, color=lab_color),
                x=0.55 if type == "icicle" else 0.50,
                xanchor="center"),
            template=None,
            paper_bgcolor=to_hex(style_opts["window_fill"]),
        )
        return fig

    # ---- single panel ----------------------------------------------
    dom_y_top = 0.95 if type == "sunburst" else 1
    fig = go.Figure(make_trace(nd, [0, 1], [0, dom_y_top]))

    if type == "icicle":
        title_x = 0.64 if n_levels > 1 else 0.71
    else:
        title_x = 0.50

    if type in ("treemap", "icicle") and main:
        # plotly's built-in title expands the top margin from the
        # trace label font, squeezing the chart; an annotation in
        # paper coordinates gives full control
        fig.update_layout(
            margin=dict(t=round(title_size * 2.1), b=8,
                        l=20, r=30 if type == "icicle" else 20),
            annotations=[dict(
                text=main, x=title_x, xref="paper",
                xanchor="center", y=1, yref="paper",
                yanchor="bottom", showarrow=False,
                font=dict(size=title_size, color=lab_color))],
        )
    else:
        fig.update_layout(
            margin=dict(t=round(title_size * 2.0), b=8, l=20,
                        r=30 if type == "icicle" else 20),
            title=dict(
                text=main,
                font=dict(size=title_size, color=lab_color),
                x=title_x, xanchor="center",
                y=0.97 if not has_y else 1.00, yanchor="top"),
        )

    fig.update_layout(
        template=None,
        paper_bgcolor=to_hex(style_opts["window_fill"]),
    )
    return fig
