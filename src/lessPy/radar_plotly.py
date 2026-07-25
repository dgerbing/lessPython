# radar_plotly.py — analog of radar.plotly.R (.radar.plotly)
#
# Renders a radar (spider) chart from ALREADY-TABULATED input:
#   1-D  pandas Series: index = x categories (the axes) -> one
#        polygon
#   2-D  pandas DataFrame: rows = `by` levels, columns = x
#        categories -> one polygon per by level, with a legend
#   facets=  ordered dict {facet level: Series or DataFrame}, all
#        sharing the same x categories (and by levels) -> a grid
#        of radar panels on a common radial scale
#
# The R renderer consumes the .radar_aggregate() bundle, whose
# facet="(All)"/by="(All)" slices collapse to exactly this
# Series/crosstab shape, so Chart()'s standard table feeds it
# directly.

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .utils import get_option
from .plotly_utils import (
    BASE_COLORS, facet_layout, make_trans, plotly_style, to_hex,
)


def _theta_from_levels(k):
    """Angular tick positions, degrees, one per category axis.
    R analog: .plotly_theta_from_levels()"""
    if k < 1:
        raise ValueError("x must have at least 1 category")
    return [i * 360 / k for i in range(k)]


def radar_plotly(x=None, x_name=None, by_name=None, main=None,
                 fill=None, opacity=1.0, digits_d=2,
                 val_label="Count", facets=None, facet_name=None,
                 n_col=None, style_opts=None):

    # normalize input to an ordered list of (facet level, table);
    # a non-faceted call is the single panel (None, x)
    if facets is not None:
        fac_levels = [str(k) for k in facets.keys()]
        tabs = list(facets.values())
    else:
        fac_levels = [None]
        tabs = [x]
    n_fac = len(tabs)
    first = tabs[0]

    two_d = isinstance(first, pd.DataFrame)
    for t in tabs:
        if not isinstance(t, pd.DataFrame if two_d
                          else pd.Series):
            raise TypeError(
                "each radar table must be a pandas Series (1-D) "
                "or DataFrame (2-D, rows = by levels), the same "
                "shape in every facet panel")

    if x_name is None:
        x_name = (first.columns.name if two_d
                  else first.index.name) or "x"
    if two_d and by_name is None:
        by_name = first.index.name or "by"
    if style_opts is None:
        style_opts = plotly_style()
    if fill is None:
        fill = BASE_COLORS

    if two_d:
        groups = [str(g) for g in first.index]
        cats = [str(c) for c in first.columns]
        mats = [t.to_numpy(dtype=float) for t in tabs]
    else:
        groups = ["(All)"]
        cats = [str(c) for c in first.index]
        mats = [t.to_numpy(dtype=float).reshape(1, -1)
                for t in tabs]

    n_by = len(groups)
    rmax = np.nanmax([np.nanmax(m) for m in mats])
    if not np.isfinite(rmax):
        raise ValueError("radar: all values are missing; "
                         "nothing to plot")

    title_size = round(16 * get_option("main_size", 1))

    theta_deg = _theta_from_levels(len(cats))
    theta_closed = theta_deg + [theta_deg[0]]
    text_labels = cats + [cats[0]]

    fill_vec = [fill[i % len(fill)]
                if isinstance(fill, (list, tuple)) else fill
                for i in range(n_by)]
    cols_hex = to_hex(fill_vec)
    fills_rgba = make_trans(cols_hex, opacity)

    val_spec = f":.{max(0, int(digits_d or 0))}f"
    show_leg = n_by > 1

    # ---- facet domains + panel title annotations -------------------
    # single panel: most of the space, with breathing room (as in R)
    if n_fac == 1:
        domains = [dict(x=(0.08, 0.95), y=(0.05, 0.88))]
        anns = []
    else:
        # compress panels: shift down and shrink so the main title
        # has clear space (R analog: .radar.plotly domain_adjust)
        def dom_adj(d):
            span = (d["y"][1] - d["y"][0]) * 0.72
            y1 = min(0.78, d["y"][1] - 0.10)
            y0 = max(0.0, y1 - span)
            d["y"] = (y0, y1)
            return d

        def ann_adj(i, a, d):
            a["y"] = min(d["y"][1] + 0.12, 0.97)
            return a

        domains, anns = facet_layout(
            fac_levels, facet_name or "",
            domain_adjust=dom_adj,
            y_base=-0.018, y_row_shift=-0.003, yanchor="bottom",
            ann_adjust=ann_adj, n_col=n_col)

    fig = go.Figure()
    for i, (lv, mat) in enumerate(zip(fac_levels, mats)):
        subplot_name = "polar" if i == 0 else f"polar{i + 1}"
        fac_line = (f"{facet_name or 'Facet'}: {lv}<br>"
                    if n_fac > 1 else "")
        for j, grp in enumerate(groups):
            r_j = mat[j].tolist()
            r_closed = r_j + [r_j[0]]
            by_line = (f"{by_name}: {grp}<br>" if n_by > 1 else "")
            fig.add_trace(go.Scatterpolar(
                theta=theta_closed,
                thetaunit="degrees",
                r=r_closed,
                mode="lines+markers",
                line=dict(width=1.5, color=cols_hex[j]),
                marker=dict(size=4, color=cols_hex[j]),
                fill="toself",
                fillcolor=fills_rgba[j],
                name=grp,
                text=text_labels,
                hovertemplate=(
                    f"{x_name}: %{{text}}<br>"
                    f"{by_line}"
                    f"{fac_line}"
                    f"{val_label}: %{{r{val_spec}}}"
                    "<extra></extra>"),
                subplot=subplot_name,
                showlegend=show_leg and i == 0,
            ))

    # ---- per-panel polar layout blocks ------------------------------
    # R analog: .plotly_apply_polar_layout()
    lab_cex = get_option("lab_size", 1.0)
    ax_cex = get_option("axis_size", 0.9)
    for i, d in enumerate(domains):
        nm = "polar" if i == 0 else f"polar{i + 1}"
        fig.update_layout(**{nm: dict(
            domain=dict(x=list(d["x"]), y=list(d["y"])),
            radialaxis=dict(
                visible=True, showline=True,
                range=[0, rmax * 1.08],
                tickfont=dict(size=round(16.0 * ax_cex)),
            ),
            angularaxis=dict(
                type="linear", direction="clockwise",
                rotation=10, tickmode="array",
                tickvals=theta_deg, ticktext=cats,
                tickfont=dict(size=round(14.5 * lab_cex)),
            ),
        )})

    # legend OUTSIDE the paper (x > 1) so plotly never auto-shifts
    # the polar domain to avoid legend overlap
    r_margin = 110 if show_leg else 20
    fig.update_layout(
        title=dict(
            text=main, x=0.5, xanchor="center",
            y=0.97, yanchor="top",
            font=dict(size=title_size,
                      color=to_hex(get_option("lab_color",
                                              "black")))),
        margin=dict(t=round(title_size * 2.2), b=10,
                    l=20 if n_fac == 1 else r_margin,
                    r=r_margin),
        showlegend=show_leg,
        annotations=anns,
        legend=dict(
            x=1.02, xanchor="left", y=1.0, yanchor="top",
            bordercolor=to_hex(style_opts["panel_border"]),
            borderwidth=1,
            font=dict(size=round(18 * ax_cex)),
        ),
        template=None,
        paper_bgcolor=to_hex(style_opts["window_fill"]),
    )
    return fig
