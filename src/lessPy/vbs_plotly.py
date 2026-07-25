# vbs_plotly.py — the violin/box/scatterplot (VBS) composite
#
# A DESIGN, not a translation: lessR renders VBS only through
# lattice (plt.lattice.R con_cat panel + plt.panel.R), so there is
# no plotly source to port. The geometry and styling reproduce the
# lattice look: a mirrored-KDE violin, a narrow box with whiskers
# and outlier points layered by severity (firebrick4 inside the
# outer fences, firebrick2 beyond), and jittered strip points
# between the inner fences. Statistics follow R exactly:
# boxplot.stats on Tukey hinges (fivenum), fences at k and 2k
# hinge-IQRs. Auto point size ports the one-variable and grouped
# formulas of .param.VBS(). With by=, as in the lattice panel:
# one violin and box over all the data (box fill dropped to 0.25
# alpha so points show through), strip points colored per group
# with varying symbols (.plt.shapes "vary" order), outliers in
# their group's color rather than firebrick, and a horizontal
# legend at the top. With facet=, one full VBS per level in its
# own band on the shared value axis — statistics computed per
# panel, first level in the BOTTOM band (the lattice as.table
# default), a shaded strip label above each band, and per-panel
# box hues (.plt.fill) under the common slate violin fill.
# Deviations, by design:
#   - jitter_y is the strip half-spread as a fraction of the
#     violin half-width (lattice jitter factors do not map to
#     plotly coordinates); jitter_x keeps R's jitter() semantics,
#     amount = factor * range/50, applied to the data as in X.R
#   - violin bandwidth defaults to band_width() — bw_nrd0
#     widened 10% per iteration toward a single peak
#     (~ .band.width, bw_iter); with facet=, each panel gets
#     its own bandwidth
#
# box_adj draws the Hubert-Vandervieren adjusted boxplot:
# fences scaled by exp(a*mc)/exp(b*mc) with mc the medcouple
# (_medcouple ~ robustbase::mc). out_cut labels up to out_cut
# outliers (largest |x|) with their IDs above the strip,
# ~ plt.lattice.R 707-734.

import numpy as np
import plotly.graph_objects as go

from .plotly_utils import (
    as_plotly_color, axis_format, axis_num, get_tick_fmt,
    make_trans,
    plot_border, plotly_style, sym_at, to_hex, x_grid,
)
from .utils import band_width, fmt, get_option, kde, pretty

_BAND = 3.0        # y extent of one facet band (unit is [-1, 1])
_STRIP = (1.30, 1.80)              # facet strip band, unit-local


def five_num(x):
    """Tukey five-number summary: min, lower hinge, median, upper
    hinge, max. R analog: fivenum()"""
    xs = np.sort(x[np.isfinite(x)])
    n = len(xs)
    n4 = np.floor((n + 3) / 2) / 2
    d = np.array([1, n4, (n + 1) / 2, n + 1 - n4, n])
    return 0.5 * (xs[np.floor(d).astype(int) - 1]
                  + xs[np.ceil(d).astype(int) - 1])


def _medcouple(x):
    """Medcouple robust skew measure (Brys, Hubert & Struyf):
    the median of the kernel h(xi, xj) over xi <= med <= xj,
    with the signum kernel for ties at the median.
    R analog: robustbase::mc(doScale=FALSE)"""
    x = np.sort(np.asarray(x, dtype=float))[::-1]  # decreasing
    med = float(np.median(x))
    zp = x[x >= med]                   # decreasing
    zm = x[x <= med]                   # decreasing
    num = zp[:, None] + zm[None, :] - 2 * med
    den = zp[:, None] - zm[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        h = num / den
    p = int((x == med).sum())
    if p > 0:                          # ties at the median:
        i = np.arange(len(zp))         # h = sign(p - 1 - i - j)
        j = np.arange(len(zm))
        ti = i - (len(zp) - p)         # tie-local index in zp
        tj = j                         # ties lead zm
        tie = ((ti[:, None] >= 0)
               & (tj[None, :] <= p - 1))
        sgn = np.sign(p - 1 - ti[:, None] - tj[None, :])
        h = np.where(tie, sgn, h)
    return float(np.median(h))


def _adj_fences(q1, q3, iqr, k_iqr, mcv, a, b):
    """Inner and outer fences, Hubert-Vandervieren adjusted by
    the medcouple when box_adj (mcv nonzero from _medcouple).
    R analog: plt.lattice.R ~612-627 / robustbase::adjboxStats"""
    if mcv >= 0:
        e_lo, e_hi = np.exp(a * mcv), np.exp(b * mcv)
    else:
        e_lo, e_hi = np.exp(-b * mcv), np.exp(-a * mcv)
    fnc_in = (q1 - k_iqr * e_lo * iqr, q3 + k_iqr * e_hi * iqr)
    fnc_out = (q1 - 2 * k_iqr * e_lo * iqr,
               q3 + 2 * k_iqr * e_hi * iqr)
    return fnc_in, fnc_out


def _auto_pt_size(x, n_ux, mx_c):
    """Point size (cex) from the one-variable rules of
    .param.VBS(): shrinks with n, more with heavy replication or
    a compressed IQR."""
    n = len(x)
    rep_prop = (n - n_ux) / n
    reps = rep_prop > 0.15 and mx_c > 0.10 * n
    if not reps:
        cex = (1.096 - 0.134 * np.log(n) if n < 2535
               else 0.226 - 0.023 * np.log(n))
    else:
        cex = 0.842 - 0.109 * np.log(mx_c)
    if cex < 0.01:
        cex = 0.015 if n < 25000 else 0.006
    q75, q25 = np.percentile(x, [75, 25])
    rng = x.max() - x.min()
    rt = (q75 - q25) / rng if rng > 0 else 1
    if rt < 0.18:
        cex *= 0.147 + 4.490 * rt
    return cex


def _seg(x0, x1, y0, y1, color, width, dash=None):
    line = dict(color=color, width=width)
    if dash:
        line["dash"] = dash
    return go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                      line=line, hoverinfo="skip",
                      showlegend=False)


def _col_for(cols, i):
    cols = (list(cols) if isinstance(cols, (list, tuple))
            else [cols])
    return cols[i % len(cols)]


def vbs_plotly(x, x_name=None, vbs_plot="vbs",
               by=None, by_order=None, by_name=None,
               facet=None, facet_order=None, facet_name=None,
               facet2=None, facet2_order=None, facet2_name=None,
               violin_fill=None, violin_color=None, bw=None,
               box_fill=None, box_color=None,
               pt_fill="black", pt_color="black", pt_trans=0.10,
               pt_size=None, out_size=None, out_shape="circle",
               shape=None,
               vbs_ratio=1.1, vbs_mean=False, fences=False,
               k_iqr=1.5, box_adj=False, a=-4, b=3, bw_iter=10,
               out_cut=0, ids=None, ID_size=0.6,
               jitter_x=None, jitter_y=None,
               x_lab=None, main=None, digits_d=2,
               axis_fmt="K", axis_x_pre="",
               style_opts=None):
    """Violin/box/strip composite of one numeric vector, laid out
    horizontally on a hidden y axis. by= colors the strip points
    per group (pt_fill/pt_color then lists per level of
    by_order); facet= draws one VBS band per level of
    facet_order (box_fill then a list per level). facet2=
    stacks one full band set per facet2 level, all sections on
    the one shared x axis, each under its own  Var = "level"
    section strip — the hybrid stand-in for R's lattice
    facet1 x facet2 grid, keeping the band design's full-width
    comparability."""
    if style_opts is None:
        style_opts = plotly_style()

    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    x = x[ok]
    if ids is not None:
        ids = np.asarray(ids)[ok]
    if by is not None:
        by = np.asarray(by)[ok]
        if by_order is None:
            by_order = sorted(set(by))
    if facet is not None:
        facet = np.asarray(facet)[ok]
        if facet_order is None:
            facet_order = sorted(set(facet))
    if facet2 is not None:
        facet2 = np.asarray(facet2)[ok]
        if facet2_order is None:
            facet2_order = sorted(set(facet2))
    n = len(x)
    n_grp = len(by_order) if by is not None else 1
    n_fac = len(facet_order) if facet is not None else 1
    if facet2 is not None:
        n_fac *= len(facet2_order)   # total bands, all sections
    ux, cnt = np.unique(x, return_counts=True)
    n_ux = len(ux)
    # max duplication count; 0 when all values unique (X.R)
    mx_c = (int(cnt.max())
            if n_ux < 1001 and n_ux < n else 0)

    violin = "v" in vbs_plot
    box = "b" in vbs_plot
    strip = "s" in vbs_plot

    # grouped replication measures, .param.VBS() group branch;
    # by and facet share the branch, as .param.VBS() does
    grp = by if by is not None else facet
    grp_ord = by_order if by is not None else facet_order
    if grp is not None:
        mx_gc = max((grp == nm).sum() for nm in grp_ord)
        mc_w = max(int(np.unique(x[grp == nm],
                                 return_counts=True)[1].max())
                   for nm in grp_ord if (grp == nm).any())
        reps = (n - n_ux) / n > 0.15 and mc_w > 0.05 * n

    # jitter_x breaks ties in the data itself, as in X.R
    if jitter_x is None:
        if grp is None:
            jitter_x = 1.1 * (1 - np.exp(-0.03 * mx_c))
        else:
            jitter_x = (0.086 + 0.141 * np.log(mx_gc)
                        if reps else 0)
    if strip and jitter_x > 0 and x.max() > x.min():
        amt = jitter_x * (x.max() - x.min()) / 50
        # seeded so the plot (and its stats/outlier split) is
        # reproducible; a distinct seed from the strip y-jitter
        # (default_rng(n) below) keeps the two offsets independent
        x = x + np.random.default_rng(n + 1).uniform(-amt, amt, n)

    # geometry: violin half-width, box half-height (plt.lattice,
    # denom shrinks by 0.5 per facet level)
    hv = vbs_ratio / 2
    n_adj = n if n < 1000 else 3 * n
    base = (4.10 - 0.000065 * n_adj if n_adj <= 25000
            else 3.25 - 0.00003 * n_adj)
    hb = (vbs_ratio / max(1.5, base - 0.5 * n_fac)) / 2

    # point sizes: auto cex rules, same px scale as the scatter
    if pt_size is not None:
        cex = pt_size
    elif grp is None:
        cex = _auto_pt_size(x, n_ux, mx_c)
    else:                       # .param.VBS() group-branch rules
        n_lvl = len(grp_ord)
        cex = (0.72 - 0.124 * np.log(mc_w) if reps
               else 0.926 - 0.108 * np.log(mx_gc)
               - 0.023 * n_lvl)
        cex = max(cex, 0.025)
    px = max(1.0, cex * 7.25)
    out_cex = out_size if out_size is not None \
        else 0.58 + 0.40 * cex
    px_out = max(1.0, out_cex * 7.25)

    if violin_fill is None:
        violin_fill = get_option("violin_fill", "#7485975A")
    if violin_color is None:
        violin_color = get_option("violin_color", "gray15")
    if box_fill is None:
        box_fill = get_option("box_fill", "#419BD2")
    if box_color is None:
        box_color = get_option("box_color", "gray15")
    box_color = to_hex(box_color)

    fig = go.Figure()
    val_fmt = get_tick_fmt(x, digits_d)
    vpart = f"%{{x:{val_fmt}}}" if val_fmt else "%{x}"
    rng_ = np.random.default_rng(n)      # reproducible spread
    jy = 0.5 if jitter_y is None else float(jitter_y)

    # facet units, first level in the BOTTOM band (lattice);
    # facet2 stacks one section of facet bands per level, bottom
    # section first, each topped by a  Var = "level"  strip
    _SEC_EXTRA = 1.8       # extra y units above each section
    sec_strips = []        # (y0, y1, label) per facet2 section
    if facet is None:
        units = [dict(mask=np.ones(n, dtype=bool), yc=0.0,
                      strip=None, ci=0, hov="")]
    elif facet2 is None:
        units = [dict(mask=facet == lv, yc=ui * _BAND,
                      strip=str(lv), ci=ui,
                      hov=f"<br>{facet_name}: {lv}")
                 for ui, lv in enumerate(facet_order)]
    else:
        from .plotly_utils import facet_cell_label
        k1 = len(facet_order)
        stride = k1 * _BAND + _SEC_EXTRA
        units = []
        for i2, lv2 in enumerate(facet2_order):
            y_base = i2 * stride
            for i1, lv1 in enumerate(facet_order):
                units.append(dict(
                    mask=(facet == lv1) & (facet2 == lv2),
                    yc=y_base + i1 * _BAND,
                    strip=str(lv1), ci=i1,
                    hov=(f"<br>{facet_name}: {lv1}"
                         f"<br>{facet2_name}: {lv2}")))
            top = y_base + (k1 - 1) * _BAND
            sec_strips.append((top + _STRIP[1] + 0.25,
                               top + _STRIP[1] + 0.95,
                               facet_cell_label(facet2_name,
                                                lv2)))

    fence_ext = []                       # axis range extension

    for ui, u in enumerate(units):
        um = u["mask"]
        xu = x[um]
        yc = u["yc"]
        if len(xu) == 0:
            continue
        lvl_part = u["hov"]

        # box statistics on Tukey hinges; fences at k and 2k
        # IQRs, medcouple-adjusted when box_adj (mc 0: exp = 1)
        mn, q1, md, q3, mx = five_num(xu)
        iqr = q3 - q1
        mcv = _medcouple(xu) if box_adj else 0.0
        fnc_in, fnc_out = _adj_fences(q1, q3, iqr, k_iqr,
                                      mcv, a, b)
        inside_u = (xu >= fnc_in[0]) & (xu <= fnc_in[1])
        lo_whisk = xu[inside_u].min()
        hi_whisk = xu[inside_u].max()
        if fences:
            fence_ext += [fnc_in[0], fnc_in[1]]

        # violin: mirrored KDE, max width = vbs_ratio
        if violin and len(xu) > 1:
            h = (bw if bw is not None
                 else band_width(xu, bw_iter))
            grid = np.linspace(xu.min() - 3 * h,
                               xu.max() + 3 * h, 512)
            dens = kde(xu, grid, h)
            half = dens / dens.max() * hv
            fig.add_trace(go.Scatter(
                x=np.concatenate([grid, grid[::-1]]),
                y=np.concatenate([yc + half,
                                  yc - half[::-1]]),
                mode="lines",
                line=dict(color=to_hex(violin_color), width=1),
                fill="toself",
                fillcolor=as_plotly_color(violin_fill),
                hoverinfo="skip", showlegend=False,
            ))

        # box, whiskers, median
        if box and len(xu) > 1:
            d = digits_d
            bf = _col_for(box_fill, u["ci"])
            fig.add_trace(go.Scatter(
                x=[q1, q3, q3, q1, q1],
                y=[yc - hb, yc - hb, yc + hb, yc + hb, yc - hb],
                mode="lines",
                line=dict(color=box_color, width=2),
                fill="toself",
                # translucent with groups so points show through
                fillcolor=(make_trans(bf, 0.25) if n_grp > 1
                           else as_plotly_color(bf)),
                text=(f"Median: {fmt(md, d)}<br>"
                      f"Hinges: {fmt(q1, d)}, {fmt(q3, d)}<br>"
                      f"Whiskers: {fmt(lo_whisk, d)}, "
                      f"{fmt(hi_whisk, d)}" + lvl_part),
                hoverinfo="text", showlegend=False,
            ))
            fig.add_trace(_seg(q1, lo_whisk, yc, yc,
                               box_color, 1))
            fig.add_trace(_seg(q3, hi_whisk, yc, yc,
                               box_color, 1))
            for w in (lo_whisk, hi_whisk):    # whisker caps
                fig.add_trace(_seg(w, w, yc - hb, yc + hb,
                                   box_color, 1))
            fig.add_trace(_seg(md, md, yc - hb, yc + hb,
                               box_color, 2))
            if vbs_mean:
                m = xu.mean()
                fig.add_trace(_seg(
                    m, m, yc - hb, yc + hb,
                    to_hex(get_option("out_fill")), 2))
            if fences:
                for f in fnc_in:
                    fig.add_trace(_seg(f, f, yc - hb, yc + hb,
                                       "darkred", 1.5))

        # strip points, per by group within the unit
        if by is None:
            groups = [(None, np.ones(len(xu), dtype=bool))]
        else:
            byu = by[um]
            groups = [(nm, byu == nm) for nm in by_order]

        if strip and len(xu) > 0:
            for i, (nm, g) in enumerate(groups):
                # outliers render through their own traces below
                pts = xu[inside_u & g]
                if len(pts) == 0:
                    continue
                yj = yc + rng_.uniform(-jy * hv, jy * hv,
                                       len(pts))
                # by groups differ by color only; shape= (scalar or
                # per-group vector) overrides for every group
                sym = "circle" if shape is None else sym_at(shape, i)
                hover = f"{x_lab}: {vpart}"
                if nm is not None:
                    hover += f"<br>{by_name}: {nm}"
                hover += lvl_part
                fig.add_trace(go.Scatter(
                    x=pts, y=yj, mode="markers",
                    name=None if nm is None else str(nm),
                    marker=dict(
                        symbol=sym, size=px,
                        color=make_trans(_col_for(pt_fill, i),
                                         1 - pt_trans),
                        line=dict(
                            color=to_hex(_col_for(pt_color, i)),
                            width=0.5)),
                    hovertemplate=hover + "<extra></extra>",
                    showlegend=nm is not None and ui == 0,
                ))
        if box or strip:
            reg = (((xu >= fnc_out[0]) & (xu < fnc_in[0]))
                   | ((xu > fnc_in[1]) & (xu <= fnc_out[1])))
            ext = (xu < fnc_out[0]) | (xu > fnc_out[1])
            for i, (nm, g) in enumerate(groups):
                if nm is None:  # firebrick severity colors
                    sets = [(xu[reg], get_option("out_fill"),
                             get_option("out_color"), "outlier"),
                            (xu[ext], get_option("out2_fill"),
                             get_option("out2_color"),
                             "extreme outlier")]
                else:           # group colors, as in the panel
                    gf = _col_for(pt_fill, i)
                    gc = _col_for(pt_color, i)
                    sets = [(xu[reg & g], gf, gc, "outlier"),
                            (xu[ext & g], gf, gc,
                             "extreme outlier")]
                for vals, o_fill, o_color, tag in sets:
                    if len(vals) == 0:
                        continue
                    hover = f"{x_lab}: {vpart} ({tag})"
                    if nm is not None:
                        hover += f"<br>{by_name}: {nm}"
                    hover += lvl_part
                    fig.add_trace(go.Scatter(
                        x=vals, y=np.full(len(vals), float(yc)),
                        mode="markers",
                        marker=dict(
                            symbol=out_shape, size=px_out,
                            color=to_hex(o_fill),
                            line=dict(color=to_hex(o_color),
                                      width=1)),
                        hovertemplate=hover + "<extra></extra>",
                        showlegend=False,
                    ))

            # label up to out_cut outliers, the largest |x|,
            # IDs rotated above the band (plt.lattice 707-734)
            if out_cut > 0 and ids is not None:
                idu = ids[um]
                lo_i = np.where(xu < fnc_in[0])[0]
                lo_i = lo_i[np.argsort(xu[lo_i],
                                       kind="stable")][:out_cut]
                hi_i = np.where(xu > fnc_in[1])[0]
                hi_i = hi_i[np.argsort(-xu[hi_i],
                                       kind="stable")][:out_cut]
                cand = np.concatenate([lo_i, hi_i]).astype(int)
                keep_i = cand[np.argsort(-np.abs(xu[cand]),
                                         kind="stable")][:out_cut]
                id_font = dict(
                    color=to_hex(get_option("ID_color",
                                            "gray50")),
                    size=max(8, round(float(ID_size) * 14)))
                for ii in keep_i:
                    fig.add_annotation(
                        x=float(xu[ii]), y=yc + 0.55,
                        text=str(idu[ii]), showarrow=False,
                        textangle=-90, yanchor="bottom",
                        font=id_font)

    # ---- axes, grid, strips, layout --------------------------------
    axT1 = pretty(float(min([x.min()] + fence_ext)),
                  float(max([x.max()] + fence_ext)))
    axL1 = axis_format(axT1, digits_d, axis_fmt,
                       axis_x_pre)
    ax_x = axis_num(x_lab, axT1, axL1)

    shapes = x_grid(axT1) + plot_border()
    annotations = []
    if n_fac > 1:               # shaded strip label per band
        s_fill = as_plotly_color(get_option("strip_fill"))
        s_line = to_hex(get_option("strip_color"))
        s_text = to_hex(get_option("strip_text_color"))
        s_font = round(14 * get_option("axis_size", 0.9))
        strip_items = [(u["yc"] + _STRIP[0],
                        u["yc"] + _STRIP[1], u["strip"])
                       for u in units if u["strip"] is not None]
        for y0, y1, txt in strip_items + sec_strips:
            shapes.append({
                "type": "rect", "xref": "paper", "yref": "y",
                "x0": 0, "x1": 1, "y0": y0, "y1": y1,
                "fillcolor": s_fill,
                "line": {"color": s_line, "width": 1},
            })
            annotations.append(dict(
                xref="paper", yref="y", x=0.5,
                y=(y0 + y1) / 2, text=txt,
                showarrow=False,
                font=dict(size=s_font, color=s_text),
            ))

    if sec_strips:
        y_top = sec_strips[-1][1] + 0.05
    elif n_fac > 1:
        y_top = units[-1]["yc"] + _STRIP[1] + 0.05
    else:
        y_top = 1.05
    fig.update_layout(
        xaxis=ax_x,
        yaxis=dict(visible=False, range=[-1.05, y_top],
                   fixedrange=True),
        shapes=shapes,
        annotations=annotations,
        template=None,
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]),
        height=(450 if n_fac == 1
                else 120 + 190 * n_fac
                + 40 * len(sec_strips)),
    )
    if n_grp > 1:               # lattice key: top, one row
        leg_color = to_hex(style_opts["lab_color"])
        fig.update_layout(legend=dict(
            orientation="h",
            x=0.5, xanchor="center", y=1.02, yanchor="bottom",
            font=dict(size=round(14 * get_option("axis_size",
                                                 0.9)),
                      family="Arial", color=leg_color),
            bgcolor=to_hex(style_opts["window_fill"]),
            bordercolor=to_hex("gray80"), borderwidth=1,
            itemsizing="constant",
        ))
    if main:
        title_size = round(16 * get_option("main_size", 1))
        fig.update_layout(
            title=dict(text=main, x=0.5, xanchor="center",
                       y=0.98, yanchor="top",
                       font=dict(size=title_size)),
            margin=dict(t=round(title_size * 2.2
                                + (26 if n_grp > 1 else 0))),
        )
    return fig
