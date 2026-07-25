# ANOVA.py — analog of ANOVA.R (one- and two-factor designs).
#
# ANOVA(): analysis of variance with the lessR output pipeline —
# background, descriptive statistics per cell, the ANOVA summary
# table, association and effect sizes, and Tukey multiple
# comparisons, plus the plotly graphics. Three designs, from the
# formula:
#   Y ~ X          one-way between groups
#   Y ~ X1 * X2    two-way between groups (crossed)
#   Y ~ X + Block  one-way randomized blocks (blocking second)
# For more complex designs use statsmodels directly.
#
# The model is a formula string; the factors are treated as
# categorical. Rmd= writes a Quarto (.qmd) report (anova_rmd.py,
# ~ av.Rmd.R). Numerics through statsmodels (ols + anova_lm) and
# scipy (the studentized range for Tukey), imported lazily. As
# elsewhere, the pipeline is ported, not the lines. Returns an
# ANOVAResults object; figures in .plots are not auto-shown.

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .plotly_utils import (
    BASE_COLORS, axis_cat, axis_format, axis_num, make_trans,
    plot_border, plotly_style, to_hex, x_grid)
from .Regression import _getdigits, _prntbl
from .utils import fmt, get_column, get_option, pretty


class ANOVAResults:
    """Numeric results and figures of ANOVA(): descriptive
    statistics, the ANOVA table, effect sizes, Tukey comparisons
    (DataFrames/dicts), and the plotly figures in .plots."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return (f"<lessPy ANOVA: {self.formula} "
                f"[{self.design}], n={self.n_keep}>")


def _anova_formula(my_formula):
    """ "Y ~ X" -> (Y, [X], "oneway"); "Y ~ X1 * X2" ->
    (Y, [X1,X2], "two-between"); "Y ~ X + Block" ->
    (Y, [X,Block], "blocks"). R analog: the design detection of
    ANOVA.R."""
    if not isinstance(my_formula, str):
        raise TypeError(
            "the model is a formula string, such as "
            '"Score ~ Group"')
    if my_formula.count("~") != 1:
        raise ValueError('the formula has one "~":  "Y ~ X"')
    lhs, rhs = (s.strip() for s in my_formula.split("~"))
    if not lhs:
        raise ValueError("the formula names the response "
                         'before the "~"')
    if "*" in rhs:
        facs = [t.strip() for t in rhs.split("*")]
        design = "two-between"
    elif "+" in rhs:
        facs = [t.strip() for t in rhs.split("+")]
        design = "blocks"
    else:
        facs = [rhs]
        design = "oneway"
    if len(facs) not in (1, 2) or any(not f for f in facs):
        raise ValueError(
            "ANOVA analyzes one or two factors:\n"
            "  one-way between groups:    Y ~ X\n"
            "  one-way randomized blocks: Y ~ X + Blocks\n"
            "  two-way between groups:    Y ~ X1 * X2")
    return lhs, facs, design


def _levels(s):
    """Factor levels in order: declared for a Categorical, else
    sorted, as an R factor."""
    from .utils import category_order
    return [str(lv) for lv in category_order(
        s if isinstance(s.dtype, pd.CategoricalDtype)
        else s.astype(str))]


def _tukey(levels, means, ns, msw, df_w):
    """Tukey HSD pairwise comparisons, level_j - level_i for
    i<j in level order, the diff, its 95% family-wise interval,
    and adjusted p from the studentized range. means and ns are
    per level. Matches R's TukeyHSD ordering and signs.
    R analog: TukeyHSD()"""
    from itertools import combinations
    from scipy.stats import studentized_range
    k = len(levels)
    qc = float(studentized_range.ppf(0.95, k, df_w))
    rows, idx = [], []
    for a, b in combinations(levels, 2):        # a before b
        diff = means[b] - means[a]
        se = np.sqrt(msw / 2 * (1 / ns[a] + 1 / ns[b]))
        p = float(studentized_range.sf(abs(diff) / se, k, df_w))
        rows.append([diff, diff - qc * se, diff + qc * se, p])
        idx.append(f"{b}-{a}")
    return pd.DataFrame(rows, index=idx,
                        columns=["diff", "lwr", "upr", "p adj"])


def _raw_means_ns(yv, gv, levels):
    """Raw group means and counts, keyed by level."""
    means = {lv: float(yv[gv == lv].mean()) for lv in levels}
    ns = {lv: int((gv == lv).sum()) for lv in levels}
    return means, ns


def _seq_marginal_means(dfo, y_name, fcols, i, gv, levels):
    """The model.tables marginal means for factor i in the
    sequential (Type I) fit: the first factor is unadjusted
    (raw means), each later factor adjusted for the earlier
    ones. R analog: model.tables(aov, "means"), used by
    TukeyHSD.aov."""
    import statsmodels.formula.api as smf
    g = float(dfo[y_name].mean())

    def fitted(k):
        if k < 0:
            return np.full(len(dfo), g)
        terms = " + ".join(f"C({fcols[j]})" for j in range(k + 1))
        return smf.ols(f"{y_name} ~ {terms}",
                       data=dfo).fit().fittedvalues.to_numpy()

    proj = fitted(i) - fitted(i - 1)
    return {lv: g + float(proj[gv == lv].mean()) for lv in levels}


def ANOVA(my_formula, data=None, filter=None, brief=False,
          digits_d=None, res_rows=None, res_sort="zresid",
          Rmd=None, Rmd_data=None, Rmd_format="html",
          Rmd_browser=True, jitter_x=0.4, graphics=True):
    """Analysis of variance of a formula string — one-way
    (Y ~ X), two-way between groups (Y ~ X1 * X2), or randomized
    blocks (Y ~ X + Block). Reports descriptive statistics, the
    ANOVA table, effect sizes, and Tukey comparisons, with the
    plotly graphics. Always prints, as in R; returns an
    ANOVAResults object with the figures in .plots."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    if data is None:
        raise ValueError(
            "data= is required: a pandas DataFrame containing "
            "the model's variables")
    if res_sort not in ("zresid", "fitted", "off"):
        raise ValueError('res_sort: "zresid", "fitted", "off"')
    if Rmd is not None:
        if Rmd_format not in ("html", "pdf", "docx", "word",
                              "none"):
            raise ValueError('Rmd_format: "html", "pdf", '
                             '"docx", or "none"')
        if brief:
            raise ValueError(
                "a Quarto report needs the full analysis, so "
                "Rmd= is not available with brief=True")
    if filter is not None:
        data = data.query(filter)

    y_name, facs, design = _anova_formula(my_formula)
    formula = (f"{y_name} ~ " + (" * ".join(facs)
               if design == "two-between"
               else " + ".join(facs)))

    y_ser = get_column(data, y_name, "response")
    if not pd.api.types.is_numeric_dtype(y_ser):
        raise TypeError(f"the response '{y_name}' is numeric")
    fac_sers = [get_column(data, f, "factor") for f in facs]

    used = pd.concat([y_ser] + fac_sers, axis=1)
    keep = ~used.isna().any(axis=1)
    n_obs = len(data)
    n_keep = int(keep.sum())
    yv = y_ser[keep].to_numpy(dtype=float)
    fvals = [s[keep].astype(str).to_numpy() for s in fac_sers]
    flevs = [_levels(s[keep]) for s in fac_sers]
    row_labels = y_ser[keep].index.astype(str)

    if digits_d is None:
        digits_d = _getdigits(yv, 2)
    d = digits_d

    # statsmodels model with categorical factors in level order
    df = pd.DataFrame({y_name: yv})
    fcols = []
    for f, v, lv in zip(facs, fvals, flevs):
        col = f"_f{len(fcols)}"
        df[col] = pd.Categorical(v, categories=lv)
        fcols.append(col)
    rhs = (f"C({fcols[0]})" if design == "oneway"
           else f"C({fcols[0]}) * C({fcols[1]})"
           if design == "two-between"
           else f"C({fcols[0]}) + C({fcols[1]})")
    fit = smf.ols(f"{y_name} ~ {rhs}", data=df).fit()

    lines = [f"\n  BACKGROUND", ""]
    lines += _background(y_name, facs, flevs, n_obs, n_keep,
                         design)

    if design == "oneway":
        out = _oneway(fit, yv, fvals[0], y_name, facs[0],
                      flevs[0], n_keep, d, brief, graphics,
                      jitter_x, lines, formula, n_obs)
    else:
        out = _twoway(fit, yv, fvals, y_name, facs, flevs,
                      n_keep, d, brief, graphics, design, lines,
                      formula, n_obs, df, fcols)

    # residuals listing (shared), unless brief or res_rows=0
    out.residuals = None
    if not brief:
        res_tbl = _residuals_section(
            fit, facs, fvals, y_name, yv, row_labels, n_keep, d,
            res_rows, res_sort, lines)
        out.residuals = res_tbl

    print("\n".join(lines))

    if Rmd is not None:
        from .anova_rmd import anova_rmd
        anova_rmd(out, y_name, facs, formula, design,
                  list(data.columns), Rmd, Rmd_data, Rmd_format,
                  Rmd_browser)
    return out


def _residuals_section(fit, facs, fvals, y_name, yv, row_labels,
                       n_keep, d, res_rows, res_sort, lines):
    """Fitted values, residuals, and internally standardized
    residuals (rstandard), sorted by |z-resid| (or fitted),
    the first res_rows rows. ~ ANOVA.R residuals block"""
    if res_rows is None:
        res_rows = n_keep if n_keep < 20 else 20
    if res_rows == "all":
        res_rows = n_keep
    res_rows = min(int(res_rows), n_keep)
    if res_rows == 0:
        return None

    zres = fit.get_influence().resid_studentized_internal
    tbl = pd.DataFrame(
        {f: v for f, v in zip(facs, fvals)}, index=row_labels)
    tbl[y_name] = yv
    tbl["fitted"] = np.asarray(fit.fittedvalues)
    tbl["residual"] = np.asarray(fit.resid)
    tbl["z-resid"] = zres
    if res_sort == "zresid":
        tbl = tbl.reindex(
            tbl["z-resid"].abs().sort_values(
                ascending=False).index)
    elif res_sort == "fitted":
        tbl = tbl.reindex(
            tbl["fitted"].abs().sort_values(
                ascending=False).index)

    lines += ["", "", "  RESIDUALS", "",
              "Fitted Values, Residuals, Standardized Residuals"]
    if res_sort == "zresid":
        lines.append("   [sorted by Standardized Residuals, "
                     "ignoring + or - sign]")
    elif res_sort == "fitted":
        lines.append("   [sorted by Fitted Value, ignoring "
                     "+ or - sign]")
    more = ("cases (rows) of data, or res_rows=\"all\"]"
            if res_rows < n_keep else "]")
    lines.append(f"   [res_rows = {res_rows}, out of "
                 f"{n_keep} {more}")
    dd = d - 1 if d > 2 else d
    show = tbl.head(res_rows).copy()
    for c in (y_name, "fitted", "residual", "z-resid"):
        show[c] = [fmt(v, dd) for v in show[c]]
    lines += _prntbl(show, dd).split("\n")
    return tbl


def _background(y_name, facs, flevs, n_obs, n_keep, design):
    out = [f"Response Variable: {y_name}", ""]
    for i, (f, lv) in enumerate(zip(facs, flevs), start=1):
        label = ("Factor Variable" if len(facs) == 1
                 else f"Factor Variable {i}"
                 if design != "blocks"
                 else ("Factor of Interest" if i == 1
                       else "Blocking Factor"))
        out.append(f"{label}: {f}")
        out.append(f"  Levels: {', '.join(lv)}")
    out += ["",
            f"Number of cases (rows) of data:  {n_obs}",
            f"Number of cases retained for analysis:  {n_keep}"]
    return out


def _desc_oneway(yv, gv, levels, d):
    """Per-group n, mean, sd, min, max, and the grand mean."""
    rows = []
    for lv in levels:
        v = yv[gv == lv]
        rows.append([int(v.size), v.mean(), v.std(ddof=1),
                     v.min(), v.max()])
    tbl = pd.DataFrame(rows, index=levels,
                       columns=["n", "mean", "sd", "min", "max"])
    return tbl, float(yv.mean())


def _anova_table(fit, anova_lm, typ, y_name):
    """The ANOVA table (df, Sum Sq, Mean Sq, F-value, p-value)
    with the factor rows and Residuals, from statsmodels."""
    t = anova_lm(fit, typ=typ)
    t = t.rename(columns={"sum_sq": "Sum Sq", "df": "df",
                          "mean_sq": "Mean Sq", "F": "F-value",
                          "PR(>F)": "p-value"})
    if "Mean Sq" not in t.columns:
        t["Mean Sq"] = t["Sum Sq"] / t["df"]
    return t[["df", "Sum Sq", "Mean Sq", "F-value", "p-value"]]


def _fmt_anova(tbl, labels, d):
    """Format the ANOVA table as lessR does: factor rows with
    F and p, the Residuals row without."""
    out = []
    w1 = max(len("Residuals"), max(len(x) for x in labels))
    hdr = (" " * w1 + f"{'df':>6}{'Sum Sq':>12}{'Mean Sq':>11}"
           f"{'F-value':>10}{'p-value':>10}")
    out.append(hdr)
    for lbl, (_, r) in zip(labels, tbl.iterrows()):
        line = (f"{lbl:<{w1}}{int(r['df']):>6}"
                f"{fmt(r['Sum Sq'], d):>12}"
                f"{fmt(r['Mean Sq'], d):>11}")
        if lbl != "Residuals":
            line += (f"{fmt(r['F-value'], d):>10}"
                     f"{fmt(r['p-value'], 4):>10}")
        out.append(line)
    return out


def _oneway(fit, yv, gv, y_name, x_name, levels, n_keep, d,
            brief, graphics, jitter_x, lines, formula, n_obs):
    from statsmodels.stats.anova import anova_lm
    p = len(levels)
    desc, grand = _desc_oneway(yv, gv, levels, d)

    lines += ["", "  DESCRIPTIVE STATISTICS", ""]
    lines += _prntbl(desc, d).split("\n")
    lines += ["", f"Grand Mean: {fmt(grand, d + 1)}"]

    tbl = _anova_table(fit, anova_lm, 1, y_name)
    tbl.index = [x_name, "Residuals"]
    lines += ["", "", "  ANOVA", "",
              f"-- Summary Table for {y_name}", ""]
    lines += _fmt_anova(tbl, [x_name, "Residuals"], d)

    ssb = float(tbl.loc[x_name, "Sum Sq"])
    ssw = float(tbl.loc["Residuals", "Sum Sq"])
    msw = float(tbl.loc["Residuals", "Mean Sq"])
    df_w = int(tbl.loc["Residuals", "df"])
    sst = ssb + ssw
    rsq = ssb / sst
    rsq_adj = 1 - ((n_keep - 1) / (n_keep - p)) * (1 - rsq)
    omsq = (ssb - (p - 1) * msw) / (sst + msw)
    lines += ["", "",
              f"-- Association and Effect Size for {y_name}", "",
              f"R Squared: {fmt(rsq, 3)}",
              f"R Sq Adjusted: {fmt(rsq_adj, 3)}",
              f"Omega Squared: {fmt(omsq, 3)}"]
    cohen_f = np.nan
    if omsq > 0:
        cohen_f = np.sqrt(omsq / (1 - omsq))
        lines += ["", f"Cohen's f: {fmt(cohen_f, 3)}"]

    tukey = None
    lines += ["", "", "  TUKEY MULTIPLE COMPARISONS OF MEANS"]
    if not brief:
        means, ns = _raw_means_ns(yv, gv, levels)
        tukey = _tukey(levels, means, ns, msw, df_w)
        lines += ["", "Family-wise Confidence Level: 0.95"]
        lines += _prntbl(tukey, d).split("\n")

    plots = {}
    if graphics:
        plots["means"] = _anova_means_plot(
            yv, gv, y_name, x_name, levels, jitter_x, d)

    return ANOVAResults(
        formula=formula, design="oneway", n_obs=n_obs,
        n_keep=n_keep, digits_d=d, response=y_name,
        factors=[x_name], descriptive=desc, grand_mean=grand,
        anova=tbl,
        effects={"R_squared": rsq, "R_sq_adjusted": rsq_adj,
                 "omega_squared": omsq, "cohen_f": cohen_f},
        tukey=tukey, plots=plots)


def _anova_means_plot(yv, gv, y_name, x_name, levels, jitter_x,
                      d):
    """Scatterplot of the response by factor level, jittered
    points with each cell mean marked and a level line.
    ~ .ANOVAz1 means plot"""
    style = plotly_style()
    pos = {lv: i for i, lv in enumerate(levels)}
    rng = np.random.default_rng(0)
    xn = np.array([pos[g] for g in gv], dtype=float)
    xn = xn + rng.uniform(-jitter_x / 2, jitter_x / 2, len(xn))
    pt = get_option("pt_color", "#324E5C")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xn, y=yv, mode="markers",
        marker=dict(symbol="circle", size=6,
                    color=make_trans(pt, 0.85), opacity=1,
                    line=dict(color=to_hex(pt), width=1)),
        hoverinfo="y", showlegend=False))
    means = [yv[gv == lv].mean() for lv in levels]
    for i, mval in enumerate(means):
        fig.add_shape(type="line", xref="x", yref="y",
                      x0=-0.5, x1=len(levels) - 0.5,
                      y0=mval, y1=mval, layer="below",
                      line=dict(color=to_hex("gray70"), width=1))
    fig.add_trace(go.Scatter(
        x=list(range(len(levels))), y=means, mode="markers",
        marker=dict(symbol="diamond", size=13,
                    color=to_hex(get_option("fit_color",
                                            "#5C4032"))),
        hoverinfo="x+y", showlegend=False))
    axT2 = pretty(float(yv.min()), float(yv.max()))
    ax_x = axis_cat(x_name)
    ax_x.update(tickmode="array",
                tickvals=list(range(len(levels))),
                ticktext=levels, range=[-0.5, len(levels) - 0.5])
    ax_y = axis_num(y_name, axT2, axis_format(axT2, d))
    ax_y.update(showgrid=True,
                gridcolor=to_hex(style["grid_col"]),
                gridwidth=1, griddash="dot")
    fig.update_layout(
        xaxis=ax_x, yaxis=ax_y, shapes=fig.layout.shapes
        + tuple(plot_border()), template=None,
        plot_bgcolor=to_hex(style["panel_fill"]),
        paper_bgcolor=to_hex(style["window_fill"]),
        title=dict(text="Scatterplot with Cell Means",
                   x=0.5, xanchor="center",
                   font=dict(size=round(
                       16 * get_option("main_size", 1)))))
    return fig


def _twoway(fit, yv, fvals, y_name, facs, flevs, n_keep, d,
            brief, graphics, design, lines, formula, n_obs, df,
            fcols):
    """Two-way between groups (Y ~ X1 * X2) or randomized blocks
    (Y ~ X1 + X2). ~ .ANOVAz2"""
    from statsmodels.stats.anova import anova_lm
    bet = design == "two-between"
    x1v, x2v = fvals[0], fvals[1]
    l1, l2 = flevs[0], flevs[1]
    f1, f2 = facs[0], facs[1]
    p, q = len(l1), len(l2)
    dp = pd.DataFrame({"y": yv, "a": x1v, "b": x2v})
    cell_n = dp.pivot_table(index="a", columns="b", values="y",
                            aggfunc="size").reindex(
                                index=l1, columns=l2)
    balanced = bool(cell_n.stack().nunique() == 1)

    if bet:
        lines += ["", "Two-way Between Groups ANOVA"]
    else:
        lines += ["", "Randomized Blocks ANOVA",
                  f"  Factor of Interest:  {f1}",
                  f"  Blocking Factor:     {f2}", "",
                  f"Note: For the F statistic for {f1} to be "
                  "distributed as F, the",
                  f"  population covariances of {y_name} must "
                  "be spherical."]

    lines += ["", "  DESCRIPTIVE STATISTICS", ""]
    if bet and not brief:
        lines += ["-- Cell Sample Sizes", "",
                  ("Equal cell sizes, so balanced design"
                   if balanced else
                   "Unequal cell sizes, so unbalanced design")]
        if not balanced:
            lines.append("ANOVA based on Type II Sums of Squares")
        lines += [""]
        cn = cell_n.T.astype(int)      # factor2 rows, factor1 cols
        lines += _prntbl(cn, 0, int_cols=list(cn.columns)
                         ).split("\n")
        cell_m = dp.pivot_table(index="b", columns="a",
                                values="y", aggfunc="mean"
                                ).reindex(index=l2, columns=l1)
        lines += ["", "-- Cell Means", ""]
        lines += _prntbl(cell_m, d).split("\n")

    # marginal means and grand mean
    m1 = dp.groupby("a")["y"].mean().reindex(l1)
    m2 = dp.groupby("b")["y"].mean().reindex(l2)
    grand = float(yv.mean())
    lines += ["", "-- Marginal Means", "", f1]
    lines += _prntbl(pd.DataFrame([m1.to_numpy()], columns=l1,
                                  index=[""]), d).split("\n")
    lines += ["", f2]
    lines += _prntbl(pd.DataFrame([m2.to_numpy()], columns=l2,
                                  index=[""]), d).split("\n")
    lines += ["", f"-- Grand Mean: {fmt(grand, d + 1)}"]
    if bet and not brief:
        cell_s = dp.pivot_table(index="b", columns="a",
                                values="y", aggfunc="std"
                                ).reindex(index=l2, columns=l1)
        lines += ["", "-- Cell Standard Deviations", ""]
        lines += _prntbl(cell_s, d).split("\n")

    # ANOVA table
    typ = 2 if (bet and not balanced) else 1
    t = anova_lm(fit, typ=typ)
    t = t.rename(columns={"sum_sq": "Sum Sq", "df": "df",
                          "PR(>F)": "p-value", "F": "F-value"})
    t["Mean Sq"] = t["Sum Sq"] / t["df"]
    inter = f"{f1}:{f2}"
    ren = {f"C({fcols[0]})": f1, f"C({fcols[1]})": f2,
           f"C({fcols[0]}):C({fcols[1]})": inter,
           "Residual": "Residuals"}
    t.index = [ren.get(i, i) for i in t.index]
    order = ([f1, f2, inter, "Residuals"] if bet
             else [f1, f2, "Residuals"])
    t = t.reindex(order)[["df", "Sum Sq", "Mean Sq", "F-value",
                          "p-value"]]
    lines += ["", "", "  ANOVA", "", "-- Summary Table"
              + (" from Type II Sums of Squares"
                 if typ == 2 else ""), ""]
    lines += _fmt_anova(t, order, d)

    msw = float(t.loc["Residuals", "Mean Sq"])
    df_w = int(t.loc["Residuals", "df"])

    # effect sizes use the Type I (sequential) F-values, as R's
    # summary(aov), even when the displayed table is Type II
    t1 = anova_lm(fit, typ=1).rename(
        columns={"F": "F-value"})
    t1.index = [ren.get(i, i) for i in t1.index]
    fa = float(t1.loc[f1, "F-value"])
    fb = float(t1.loc[f2, "F-value"])
    lines += ["", "", "-- Association and Effect Size", ""]
    eff = {}
    if bet:
        fab = float(t1.loc[inter, "F-value"])
        nh = round(1 / np.mean(1 / cell_n.to_numpy().ravel()))
        oa = ((p - 1) * (fa - 1)) / ((p - 1) * (fa - 1) + nh * p * q)
        ob = ((q - 1) * (fb - 1)) / ((q - 1) * (fb - 1) + nh * p * q)
        oab = (((p - 1) * (q - 1) * (fab - 1))
               / ((p - 1) * (q - 1) * (fab - 1) + nh * p * q))
        eff = {"omega_sq_" + f1: oa, "omega_sq_" + f2: ob,
               "omega_sq_interaction": oab}
        lines += [f"Partial Omega Squared for {f1}: {fmt(oa, 3)}",
                  f"Partial Omega Squared for {f2}: {fmt(ob, 3)}",
                  f"Partial Omega Squared for {f1} & {f2}: "
                  f"{fmt(oab, 3)}", ""]
        for nm, o in ((f1, oa), (f2, ob), (f"{f1} & {f2}", oab)):
            if o > 0:
                lines.append(f"Cohen's f for {nm}: "
                             f"{fmt(np.sqrt(o / (1 - o)), 3)}")
    else:
        oa = ((p - 1) * (fa - 1)) / ((p - 1) * (fa - 1) + q * p)
        intra = (fb - 1) / ((p - 1) + fb)
        eff = {"omega_sq_" + f1: oa, "intraclass_" + f2: intra}
        lines += [f"Partial Omega Squared for {f1}: {fmt(oa, 3)}",
                  f"Partial Intraclass Correlation for {f2}: "
                  f"{fmt(intra, 3)}", ""]
        if oa > 0:
            lines.append(f"Cohen's f for {f1}: "
                         f"{fmt(np.sqrt(oa / (1 - oa)), 3)}")
        if intra > 0:
            lines.append(f"Cohen's f for {f2}: "
                         f"{fmt(np.sqrt(intra / (1 - intra)), 3)}")

    # Tukey
    tukey = None
    lines += ["", "", "  TUKEY MULTIPLE COMPARISONS OF MEANS"]
    if not brief:
        tukey = {}
        lines += ["", "Family-wise Confidence Level: 0.95",
                  "", f"Factor: {f1}"]
        m1m = _seq_marginal_means(df, y_name, fcols, 0, x1v, l1)
        _, n1 = _raw_means_ns(yv, x1v, l1)
        tukey[f1] = _tukey(l1, m1m, n1, msw, df_w)
        lines += _prntbl(tukey[f1], d).split("\n")
        if bet:                        # blocks: factor of interest only
            lines += ["", f"Factor: {f2}"]
            m2m = _seq_marginal_means(df, y_name, fcols, 1, x2v,
                                      l2)
            _, n2 = _raw_means_ns(yv, x2v, l2)
            tukey[f2] = _tukey(l2, m2m, n2, msw, df_w)
            lines += _prntbl(tukey[f2], d).split("\n")
            cl = np.array([f"{a}:{b}" for a, b in zip(x1v, x2v)])
            clv = [f"{a}:{b}" for b in l2 for a in l1]
            clv = [c for c in clv if c in set(cl)]
            cmn, cnn = _raw_means_ns(yv, cl, clv)
            lines += ["", "Cell Means"]
            tukey["cells"] = _tukey(clv, cmn, cnn, msw, df_w)
            lines += _prntbl(tukey["cells"], d).split("\n")

    plots = {}
    if graphics:
        if bet:
            plots["interaction"] = _interaction_plot(
                dp, y_name, f1, f2, l1, l2, d)
        else:
            plots["data"] = _blocks_plot(
                x1v, yv, x2v, y_name, f1, f2, l1, l2, d,
                "Data Values")
            plots["fitted"] = _blocks_plot(
                x1v, np.asarray(fit.fittedvalues), x2v,
                "Fitted", f1, f2, l1, l2, d, "Fitted Values")

    return ANOVAResults(
        formula=formula, design=design, n_obs=n_obs,
        n_keep=n_keep, digits_d=d, response=y_name,
        factors=[f1, f2], cell_n=cell_n,
        marginal_means={f1: m1, f2: m2}, grand_mean=grand,
        anova=t, effects=eff, tukey=tukey, plots=plots)


def _cat_line_fig(title, x_name, y_name, levels_x, series, d,
                  legend_title):
    """A categorical-x line-with-markers figure: one colored
    line per by-group across the x levels. Shared by the
    interaction and blocks plots."""
    style = plotly_style()
    fig = go.Figure()
    ys_all = []
    for gi, (gname, yv_line) in enumerate(series):
        col = to_hex(BASE_COLORS[gi % len(BASE_COLORS)])
        ys_all += [v for v in yv_line if v == v]
        fig.add_trace(go.Scatter(
            x=list(range(len(levels_x))), y=yv_line,
            mode="lines+markers", name=str(gname),
            line=dict(color=col, width=2),
            marker=dict(size=9, color=col),
            connectgaps=True, hoverinfo="x+y+name"))
    axT2 = pretty(min(ys_all), max(ys_all))
    ax_x = axis_cat(x_name)
    ax_x.update(tickmode="array",
                tickvals=list(range(len(levels_x))),
                ticktext=levels_x,
                range=[-0.4, len(levels_x) - 0.6])
    ax_y = axis_num(y_name, axT2, axis_format(axT2, d))
    ax_y.update(showgrid=True,
                gridcolor=to_hex(style["grid_col"]),
                gridwidth=1, griddash="dot")
    fig.update_layout(
        xaxis=ax_x, yaxis=ax_y, shapes=plot_border(),
        template=None, legend=dict(title=dict(text=legend_title)),
        plot_bgcolor=to_hex(style["panel_fill"]),
        paper_bgcolor=to_hex(style["window_fill"]),
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=round(
                       16 * get_option("main_size", 1)))))
    return fig


def _interaction_plot(dp, y_name, f1, f2, l1, l2, d):
    """Cell means of the response across factor 1, one line per
    factor 2 level. ~ .ANOVAz2 interaction plot"""
    cm = dp.pivot_table(index="a", columns="b", values="y",
                        aggfunc="mean").reindex(index=l1,
                                                columns=l2)
    series = [(b, [cm.loc[a, b] for a in l1]) for b in l2]
    return _cat_line_fig(f"Cell Means of {y_name}", f1, y_name,
                         l1, series, d, f2)


def _blocks_plot(x1v, yvals, x2v, y_name, f1, f2, l1, l2, d,
                 title):
    """Values across the factor of interest, one line per block.
    ~ .ANOVAz2 data / fitted plots"""
    dp = pd.DataFrame({"y": yvals, "a": x1v, "b": x2v})
    series = []
    for b in l2:
        sub = dp[dp["b"] == b].set_index("a")["y"]
        series.append((b, [sub.get(a, np.nan) for a in l1]))
    return _cat_line_fig(title, f1, y_name, l1, series, d, f2)
