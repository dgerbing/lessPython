# plt_plotly.py — analog of plt.plotly.R
#
# Renders the XY() scatterplot / time-series view. Deviations from
# the R renderer, whose interface reflects the base-R path it
# shares state with:
#   - fit lines, SE bands, and ellipses arrive as prepared
#     line/polygon coordinate lists (XY computes them), not as
#     per-point vectors threaded through a shared data frame
#   - bubble mode (size=) and the outlier/MD path are not ported:
#     bubble displays are internal to Chart(), and outlier
#     flagging awaits the accompanying-statistics port
#   - a date x-axis uses plotly's native date ticks rather than
#     pretty() tick vectors

import numpy as np
import plotly.graph_objects as go

from .plotly_utils import (
    as_plotly_color, axis_base, axis_num, get_tick_fmt, make_trans,
    plot_border, plotly_style, sym_at, to_hex, x_grid, y_grid,
)
from .utils import get_option


def _hover_fmt(xv, yv, x_lab, y_lab, digits_d, is_date):
    """Hover template for one trace. R analog: .hover.fmt()"""
    fmty = get_tick_fmt(yv, digits_d)
    ypart = f"%{{y:{fmty}}}" if fmty else "%{y}"
    if is_date:
        x_lab, xpart = "Date", "%{x|%Y-%m-%d}"
    else:
        fmtx = get_tick_fmt(xv, digits_d)
        xpart = f"%{{x:{fmtx}}}" if fmtx else "%{x}"
    return (f"{x_lab}: {xpart}<br>{y_lab.strip()}: {ypart}"
            "<extra></extra>")


def plt_plotly(groups, by_name=None,
               fill=("#324E5C",), border=None, shape="circle",
               pt_size=1,
               x_lab="", y_lab="", ax=None, gridT1=None,
               gridT2=None, main=None, digits_d=2,
               connect=False, is_date=False, pt_opacity=0.90,
               line_width=1.5,
               area_polys=None,
               fit_lines=None, fit_color=None, fit_lwd=None,
               se_polys=None, se_fill=None,
               ellipses=None, ellipse_fill=None,
               ellipse_color=None, ellipse_lwd=1,
               style_opts=None):
    """Scatterplot / time-series renderer for XY().

    groups: list of (name, x, y) — one entry, name None, when
    there is no by variable. R analog: plt.plotly()
    """
    if style_opts is None:
        style_opts = plotly_style()
    if fit_color is None:
        fit_color = get_option("fit_color", "#5C4032")
    if fit_lwd is None:
        fit_lwd = get_option("fit_lwd", 2)

    n_grp = len(groups)
    has_groups = n_grp > 1

    def recycle(cols):
        cols = (list(cols) if isinstance(cols, (list, tuple))
                else [cols])
        return [cols[i % len(cols)] for i in range(n_grp)]

    fills = recycle(fill)
    borders = fills if border is None else recycle(border)

    fig = go.Figure()

    # areas under/between time series: prepared polygons from
    # XY() (ts_area_fill / ts_stack), drawn beneath the lines
    if area_polys:
        for pax, pay, pcol in area_polys:
            fig.add_trace(go.Scatter(
                x=pax, y=pay, mode="none", fill="toself",
                fillcolor=pcol, hoverinfo="skip",
                showlegend=False))

    # points: pixel diameter, R's single/grouped scale factors
    px = float(pt_size) * (6.5 if has_groups else 7.25)
    if not np.isfinite(px) or px < 0:
        px = 5
    # width of the segments that connect adjacent points, from
    # line_width of XY(). A width of 0 draws no segments, as in R,
    # where .plt.main() tests  ln.width > 0
    lw = float(line_width) if line_width is not None else 1.5
    if not np.isfinite(lw) or lw < 0:
        lw = 1.5
    connect = connect and lw > 0

    mode_pts = "lines+markers" if connect else "markers"
    if connect and px == 0:
        mode_pts = "lines"

    for i, (nm, xv, yv) in enumerate(groups):
        hover = _hover_fmt(xv, yv, x_lab, y_lab, digits_d, is_date)
        if has_groups:
            hover = hover.replace(
                "<extra></extra>", f"<br>{by_name}: {nm}"
                                   "<extra></extra>")
        marker = dict(
            symbol=sym_at(shape, i), size=px, sizemode="diameter",
            color=make_trans(fills[i], pt_opacity), opacity=1,
            line=dict(color=to_hex(borders[i]), width=1),
        ) if "markers" in mode_pts else None
        fig.add_trace(go.Scatter(
            x=xv, y=yv, mode=mode_pts,
            name=None if nm is None else str(nm),
            legendgroup=None if nm is None else str(nm),
            marker=marker,
            line=(dict(color=to_hex(borders[i]), width=lw)
                  if connect else None),
            hovertemplate=hover,
            # connected series show a legend-only line sample below
            showlegend=has_groups and not connect,
        ))
        if has_groups and connect and len(xv) >= 2:
            fig.add_trace(go.Scatter(
                x=xv[:2], y=yv[:2], mode="lines",
                name=str(nm), legendgroup=str(nm),
                line=dict(color=to_hex(borders[i]), width=lw),
                hoverinfo="skip", showlegend=True,
            ))

    # ellipses: border per group when grouped, else ellipse_color
    if ellipses:
        for k, (ex, ey) in enumerate(ellipses):
            edge = (to_hex(borders[k]) if has_groups
                    else to_hex(ellipse_color))
            fig.add_trace(go.Scatter(
                x=ex, y=ey, mode="lines",
                line=dict(color=edge, width=ellipse_lwd),
                fill="toself",
                fillcolor=as_plotly_color(ellipse_fill),
                hoverinfo="skip", showlegend=False,
            ))

    # SE bands under their fit lines
    if se_polys:
        for bx, bnd_y in se_polys:
            fig.add_trace(go.Scatter(
                x=bx, y=bnd_y, mode="none", fill="toself",
                fillcolor=as_plotly_color(se_fill),
                hoverinfo="skip", showlegend=False,
            ))

    # fit lines: single fit in fit_color; group fits in group hues
    if fit_lines:
        fmtx_all = np.concatenate([f["x"] for f in fit_lines])
        fmtx = get_tick_fmt(fmtx_all, digits_d)
        xpart = f"%{{x:{fmtx}}}" if fmtx else "%{x}"
        for i, fl in enumerate(fit_lines):
            if len(fl["x"]) < 2:
                continue
            fmty = get_tick_fmt(fl["y"], digits_d)
            ypart = f"%{{y:{fmty}}}" if fmty else "%{y}"
            nm = fl["name"]
            single = nm is None
            lbl = "Fit" if single else f"Fit ({nm})"
            fig.add_trace(go.Scatter(
                x=fl["x"], y=fl["y"], mode="lines",
                name="Fit" if single else f"Fit: {nm}",
                legendgroup="fit" if single else str(nm),
                line=dict(
                    color=(to_hex(fit_color) if single
                           else to_hex(fills[i])),
                    width=fit_lwd),
                hovertemplate=(
                    f"{x_lab}: {xpart}<br>{lbl}: {ypart}"
                    "<extra></extra>"),
                showlegend=True,
            ))

    # axes, grids, legend, background
    if is_date:
        ax_x = axis_base()
        ax_x["title"]["text"] = x_lab
        ax_x.update(showgrid=True,
                    gridcolor=to_hex(style_opts["grid_col"]),
                    gridwidth=1)
        shapes = y_grid(gridT2) + plot_border()
    else:
        ax_x = axis_num(x_lab, ax["axT1"], ax["axL1"])
        shapes = x_grid(gridT1) + y_grid(gridT2) + plot_border()
    ax_y = axis_num(y_lab, ax["axT2"], ax["axL2"])

    leg_color = to_hex(style_opts["lab_color"])
    fig.update_layout(
        xaxis=ax_x, yaxis=ax_y,
        shapes=shapes,
        template=None,
        legend=dict(
            title=dict(
                text=by_name if has_groups else None,
                font=dict(
                    size=round(15 * get_option("lab_size", 1)),
                    family="Arial", color=leg_color)),
            font=dict(
                size=round(16 * get_option("axis_size", 0.9)),
                family="Arial", color=leg_color),
            orientation="v", x=1.05, y=0.5,
            xanchor="left", yanchor="middle",
            bgcolor=to_hex(style_opts["window_fill"]),
            bordercolor="#CCCCCC", borderwidth=1,
            itemsizing="constant",
        ),
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]),
    )

    if main:
        title_size = round(16 * get_option("main_size", 1))
        fig.update_layout(
            title=dict(text=main, x=0.5, xanchor="center",
                       y=0.99, yanchor="top",
                       font=dict(size=title_size)),
            margin=dict(t=round(title_size * 2.2)),
        )
    return fig
