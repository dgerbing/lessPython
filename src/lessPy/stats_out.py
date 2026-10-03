# stats_out.py — accompanying statistics: the console numbers
# that pair with each analytic view, a core differentiator of
# the framework. R analogs: SummaryStats.R and bx.stats.R for
# X(), the correlation and fit report of XY.R, and the
# frequency / chi-square output of the Chart() bar path.
#
# This increment prints (quiet= suppresses, default from the
# "quiet" option, as R getOption("quiet")). Returning a stats
# object is deferred: the functions return the plotly Figure,
# which notebooks display automatically; wrapping it would
# break that.

import numpy as np
import pandas as pd
from scipy import stats as sps

from .utils import STAT_FUN, STAT_LBL, fmt, get_option


def resolve_quiet(quiet):
    return get_option("quiet", False) if quiet is None else quiet


def x_stats(xv, x_name, digits_d=2):
    """n, missing, mean and sd, five-number summary, and IQR
    outliers for the distribution X() displays."""
    v = np.asarray(xv, float)
    miss = int(np.isnan(v).sum())
    v = v[np.isfinite(v)]
    n = len(v)
    d = digits_d
    lines = [f"--- {x_name} ---",
             f"n: {n}   missing: {miss}"]
    if n < 2:
        return lines
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    out_v = np.sort(v[(v < q1 - 1.5 * iqr)
                      | (v > q3 + 1.5 * iqr)])
    lines += [
        f"mean: {fmt(v.mean(), d)}   sd: {fmt(v.std(ddof=1), d)}",
        f"min: {fmt(v.min(), d)}   1st Qu: {fmt(q1, d)}   "
        f"median: {fmt(med, d)}   3rd Qu: {fmt(q3, d)}   "
        f"max: {fmt(v.max(), d)}"]
    if len(out_v):
        vals = "  ".join(fmt(o, d) for o in out_v[:12])
        more = ("" if len(out_v) <= 12
                else f"  ... {len(out_v)} total")
        lines.append(f"outliers (1.5 IQR): {vals}{more}")
    else:
        lines.append("outliers (1.5 IQR): none")
    return lines


def facet_summary(x_vec, grp_vec, grp_label, digits_d=2):
    """Summary table of x over the levels of one grouping
    variable: n, Mean, Median, SD, IQR, Min, Max per level.
    Printed with the faceted scatterplot, one table per by=
    and facet variable. R analog: .vbs_summary_table()"""
    s = pd.Series(np.asarray(x_vec, dtype=float))
    g = pd.Series(np.asarray(grp_vec).astype(str))
    rows = []
    for lv, z in s.groupby(g, sort=True):
        z = z.dropna()
        rows.append({
            grp_label: lv,
            "n": len(z),
            "Mean": fmt(z.mean(), digits_d),
            "Median": fmt(z.median(), digits_d),
            "SD": fmt(z.std(ddof=1), digits_d),
            "IQR": fmt(z.quantile(.75) - z.quantile(.25),
                       digits_d),
            "Min": fmt(z.min(), digits_d),
            "Max": fmt(z.max(), digits_d),
        })
    return pd.DataFrame(rows).to_string(index=False)


def _cor_block(xg, yg, label, d):
    n = len(xg)
    if n < 4:
        return [f"correlation{label}: n = {n}, too few points"]
    r = float(np.corrcoef(xg, yg)[0, 1])
    tval = r * np.sqrt((n - 2) / max(1 - r * r, 1e-12))
    p = 2 * sps.t.sf(abs(tval), n - 2)
    zlo = np.arctanh(r) - 1.959964 / np.sqrt(n - 3)
    zhi = np.arctanh(r) + 1.959964 / np.sqrt(n - 3)
    return [
        f"correlation{label}: r = {fmt(r, 3)}",
        f"  n = {n}, t = {fmt(tval, 2)} (df = {n - 2}), "
        f"p-value = {fmt(p, 4)}",
        f"  95% CI: {fmt(np.tanh(zlo), 3)} to "
        f"{fmt(np.tanh(zhi), 3)}"]


def xy_stats(groups, x_name, y_name, fit_stats=None,
             digits_d=2):
    """Correlation (t test, Fisher-z CI) for the relationship
    XY() displays, overall or per by= group, plus fit-line MSE
    when a fit line is drawn."""
    lines = [f"--- {x_name} and {y_name} ---"]
    for nm, xg, yg in groups:
        label = "" if nm is None else f" ({nm})"
        lines += _cor_block(np.asarray(xg, float),
                            np.asarray(yg, float), label,
                            digits_d)
    for fs in (fit_stats or []):
        nm, fit, ys, f = fs
        label = "" if nm is None else f" ({nm})"
        mse = float(np.mean((ys - f) ** 2))
        tss = float(((ys - ys.mean()) ** 2).sum())
        rsq = 1 - float(((ys - f) ** 2).sum()) / tss \
            if tss > 0 else np.nan
        lines.append(
            f'fit="{fit}"{label}:  MSE: {fmt(mse, digits_d + 1)}'
            f"   R-squared: {fmt(rsq, 3)}")
    return lines


def md_outliers(xv, yv, ids, MD_cut, out_cut):
    """Bivariate outliers by Mahalanobis distance from the
    centroid: MD_cut an absolute threshold, out_cut a proportion
    (< 1) or count (>= 1) of the most extreme points. Returns the
    flagged indices and the sorted MD/ID table.
    R analog: .plt.MD (plt.MD.R)"""
    v = np.column_stack((xv, yv))
    center = v.mean(axis=0)
    cov = np.cov(v, rowvar=False, ddof=1)
    diff = v - center
    dst = np.einsum("ij,jk,ik->i", diff,
                    np.linalg.pinv(cov), diff)

    if MD_cut > 0:                     # absolute threshold
        out_idx = np.flatnonzero(dst >= MD_cut)
    elif 0 < out_cut < 1:              # a proportion
        out_idx = np.flatnonzero(
            dst > np.quantile(dst, 1 - out_cut))
    else:                              # a count
        cut = np.sort(dst)[::-1][:int(out_cut)].min()
        out_idx = np.flatnonzero(dst >= cut)

    ids = np.asarray(ids).astype(str)
    ord_ = np.argsort(dst)[::-1]
    n_lines = min(len(out_idx) + 3, len(dst))
    w_id = max(len(s) for s in ids)
    w_md = max(len(fmt(d, 2)) for d in dst)
    lines = [">>> Outlier analysis with Mahalanobis Distance",
             "",
             f"{'MD':>{w_md}} {'ID':>{w_id + 1}}",
             f"{'-----':>{w_md}} {'-----':>{w_id + 1}}"]
    for i in range(n_lines):
        if i == len(out_idx) and len(out_idx) > 0:
            lines.append("")
        j = ord_[i]
        lines.append(f"{fmt(dst[j], 2):>{w_md}} "
                     f"{ids[j]:>{w_id + 1}}")
    if n_lines < len(dst):
        lines.append(f"{'...':>{w_md}} {'...':>{w_id + 1}}")
    return out_idx, lines


# ----- Chart() console tables -------------------------------------
# Ports of the printers behind lessR's Chart() text output, line for
# line in layout: .ss.factor() (1-D and 2-D counts), .ss.numeric()
# with by (the stat of y per level), .ss.real() ("Plotted Values",
# bar only), and the "Summary Table for" crosstab of Chart.R. The
# width arithmetic follows the R source so the columns line up as
# R's do. R prints through print.out_all, which ends each line with
# a space; those trailing spaces are not reproduced.

def _fmtc(k, w=0, j="right"):
    """R .fmtc(): a string padded to width w."""
    s = str(k)
    return s.rjust(w) if j == "right" else s.ljust(w)


def _fmti(k, w=0):
    """R .fmti(): an integer right-justified to width w."""
    return str(int(k)).rjust(w)


def _fmtr(k, d, w=0):
    """R .fmt(): a real with d decimals right-justified to w."""
    return f"{k:.{d}f}".rjust(w)


def _r_chr(v, d):
    """R as.character(round(v, d)): no trailing zeros."""
    return f"{round(float(v), d):.15g}"


def _lvl_str(v):
    """A category name as R's table() names it: a whole-valued
    float reads 1, not 1.0."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _counts_1d(x_ser, x_name, n_miss, x_lbl=None, width=80):
    """One categorical variable: title, missing count, frequencies
    and proportions (horizontal, or vertical past width), and the
    chi-square test of equal probabilities. R analog: the one
    variable branch of .ss.factor() via print.out_all."""
    cnt = x_ser.groupby(x_ser, observed=True, sort=True).size()
    cnt = cnt[cnt > 0]
    names = [_lvl_str(v) for v in cnt.index]
    x = cnt.to_numpy()
    tot = int(x.sum())
    xp = x / tot
    w = len(str(tot))

    max_ln = [max(6, max(len(nm), len(str(v))) + 1)
              for nm, v in zip(names, x)]
    hdr = " " * 13
    for nm, ml in zip(names, max_ln):
        hdr += " " + _fmtc(nm, ml)
    hdr += " " + _fmtc("Total", w + 6)
    if len(hdr) <= width:
        frq = "Frequencies: "
        prp = "Proportions: "
        for v, p, ml in zip(x, xp, max_ln):
            frq += " " + _fmti(v, ml)
            prp += " " + _fmtr(p, 3, ml)
        frq += " " + _fmti(tot, w + 6)
        prp += " " + _fmtc("1.000", w + 6)
        counts = [hdr, frq, prp]
    else:                          # vertical display
        mx_nm = max(max(len(nm) for nm in names), len("Total"))
        mx_fr = len(str(tot)) + 2
        xnm = x_name if len(x_name) <= 13 else x_name[:13]
        dash = "-" * (mx_nm + mx_fr + 9)
        counts = [_fmtc(xnm, mx_nm) + " " + _fmtc("Count", mx_fr)
                  + " " + _fmtc("Prop", 6), dash]
        for nm, v, p in zip(names, x, xp):
            counts.append(_fmtc(nm, mx_nm) + " " + _fmti(v, mx_fr)
                          + " " + _fmtr(p, 3, 7))
        counts += [dash, _fmtc("Total", mx_nm) + " "
                   + _fmti(tot, mx_fr) + " " + _fmtc("1.000", 7)]

    ttl = x_name if x_lbl is None else f"{x_name}: {x_lbl}"
    lines = [f"--- {ttl} ---"]
    if n_miss is not None:
        lines += ["", f"Missing Values: {n_miss}"]
    lines += [""] + counts
    if len(x) > 1:
        chi, p = sps.chisquare(x)
        lines += ["",
                  "Chi-squared test of null hypothesis of equal "
                  "probabilities",
                  f"  Chisq = {chi:.3f}, df = {len(x) - 1}, "
                  f"p-value = {p:.3f}"]
        if (tot / len(x)) < 5:     # expected count is n/k for each
            lines += [">>> Low cell expected frequencies, so "
                      "chi-squared approximation may not be accurate",
                      ""]
    return lines


def _prnfreq(tbl, typ, max_ln, max_c1, n_dash, ttl, x_name,
             y_name, dgt=3):
    """A titled 2-D table, rows by levels, columns x levels.
    R analog: .prnfreq() in ss.factor.R, horizontal layout."""
    tx = [ttl, "-" * n_dash, ""]
    tx.append(_fmtc(x_name, max_c1 + 3))
    buf1 = dgt - 3 if typ == "r" else 0
    line = (y_name or "").ljust(max_c1)
    for c, ml in zip(tbl.columns, max_ln):
        line += _fmtc(c, ml + buf1)
    tx.append(line)
    for r in tbl.index:
        line = ("  " + str(r)).ljust(max_c1)
        for c, ml in zip(tbl.columns, max_ln):
            v = tbl.loc[r, c]
            line += (_fmtr(v, dgt, ml + buf1) if typ == "r"
                     else _fmti(v, ml))
        tx.append(line)
    return tx


def _counts_2d(x_ser, by_ser, x_name, by_name, stack100=False):
    """Two categorical variables: joint and marginal frequencies,
    then Cramer's V and the chi-square test of independence, or
    with stack100 the proportions within each column the chart is
    drawn from. R analog: the two variable branch of .ss.factor()
    and its use in bc.main.R."""
    ct = pd.crosstab(by_ser, x_ser)
    ct.index = [_lvl_str(v) for v in ct.index]
    ct.columns = [_lvl_str(v) for v in ct.columns]
    xx = ct.copy()
    xx["Sum"] = xx.sum(axis=1)
    xx.loc["Sum"] = xx.sum(axis=0)

    max_c1 = max([len(by_name or "")]
                 + [len(str(r)) for r in xx.index]) + 2
    max_c1 = max(max_c1, 5)
    max_ln = [max(4, max(len(str(c)),
                         max(len(str(int(v))) for v in xx[c])) + 1)
              for c in xx.columns]

    lines = [""]                   # R's empty title, then a blank
    lines += _prnfreq(xx, "i", max_ln, max_c1, 30,
                      "Joint and Marginal Frequencies",
                      x_name, by_name)

    if stack100:
        max_ln = [max(6, m) for m in max_ln]
        col = (ct / ct.sum(axis=0)).round(3)
        col.loc["Sum"] = col.sum(axis=0).round(3)
        mx = max(max_ln)
        buf = mx - 4               # R: digits_d - 3 < max.c1
        lines += [""] + _prnfreq(
            col, "r", [m + buf for m in max_ln[:len(col.columns)]],
            max_c1, 35, "Cell Proportions within Each Column",
            x_name, by_name)
        return lines

    obs = ct.to_numpy()
    chi, p, dof, exp = sps.chi2_contingency(obs, correction=False)
    lines.append("")
    if np.isfinite(chi):
        min_rc = min(obs.shape) - 1
        V = np.sqrt(chi / (min_rc * obs.sum()))
        phi = " (phi)" if dof == 1 else ""
        lines += [f"Cramer's V{phi}: {V:.3f}", "",
                  "Chi-square Test of Independence:",
                  f"     Chisq = {chi:.3f}, df = {dof}, "
                  f"p-value = {p:.3f}"]
        if (exp < 5).any():
            lines.append(">>> Low cell expected frequencies, "
                         "chi-squared approximation may not be "
                         "accurate")
    else:
        lines += ["Cross-tabulation table not well-formed, usually "
                  "too many zeros",
                  "Cramer's V and the chi-squared analysis not "
                  "possible"]
    return lines


def _stat_by_levels(y_ser, x_ser, y_name, x_name, digits_d,
                    y_lbl=None, x_lbl=None):
    """n, miss, mean, sd, min, mdn, max of y for each level of x.
    R analog: .ss.numeric(y, by=x, brief=TRUE) as Chart() calls
    it, its width arithmetic included."""
    d = digits_d
    groups = [(lv, g) for lv, g in
              y_ser.groupby(x_ser, observed=True, sort=True)]
    names = [_lvl_str(lv) for lv, _ in groups]
    vecs = [pd.to_numeric(g, errors="coerce") for _, g in groups]
    n_lines = len(groups)

    max_char = max(len(nm) for nm in names)
    max_lv = max_char
    max_n = max(len(str(int(v.notna().sum()))) for v in vecs)
    max_nm = max(len(str(int(v.isna().sum()))) for v in vecs)
    if n_lines == 1:
        max_lv = max(max_lv, 3)

    max_ln = 0
    for v in vecs:
        m, s = v.mean(), v.std(ddof=1)
        if pd.isna(s):
            n_ln = len(_r_chr(m, d)) + d
        else:
            n_ln = max(len(_r_chr(m, d)), len(_r_chr(s, d))) + d
        max_ln = max(max_ln, n_ln)
    if max_ln < 5:
        max_ln += 1
    if max_ln < 10:
        max_ln += 1
    if max_ln < 4:
        max_ln += 2
    if max_ln < 8:
        max_ln += 1
    nbuf = 3 if n_lines == 1 else 5

    t1 = y_name if y_lbl is None else f"{y_name}: {y_lbl}"
    t2 = x_name if x_lbl is None else f"{x_name}: {x_lbl}"
    lines = [t1, "  - by levels of - ", t2, ""]
    lines.append(" ".join([
        _fmtc("n", len(str(max_n)) + nbuf + max_lv - 2),
        _fmtc("miss", len(str(max_nm)) + 5),
        _fmtc("mean", max_ln), _fmtc("sd", max_ln),
        _fmtc("min", max_ln), _fmtc("mdn", max_ln),
        _fmtc("max", max_ln)]))
    for nm, v in zip(names, vecs):
        lvl = _fmtc(nm.ljust(max_char + 1), max_lv)
        n, miss = int(v.notna().sum()), int(v.isna().sum())
        row = [lvl, _fmti(n, max_n + 1), _fmti(miss, max_nm + 5)]
        vv = v.dropna()
        if n == 1:
            row.append(_fmtr(vv.mean(), d, max_ln))
        elif n > 1:
            row += [_fmtr(s, d, max_ln) for s in
                    (vv.mean(), vv.std(ddof=1), vv.min(),
                     vv.median(), vv.max())]
        lines.append(" ".join(row))
    return lines


def _plotted_values(vals, digits_d):
    """The aggregated values a bar chart draws. R analog: the one
    variable branch of .ss.real(), horizontal layout."""
    names = [_lvl_str(k) for k in vals.index]
    max_ln = [max(6, max(len(nm), len(_fmtr(v, digits_d))) + 1)
              for nm, v in zip(names, vals)]
    hdr = "".join(" " + _fmtc(nm, ml) for nm, ml in zip(names, max_ln))
    val = "".join(_fmtr(v, digits_d, ml + 1)
                  for v, ml in zip(vals, max_ln))
    return [" Plotted Values", " " + "-" * 14, hdr, val]


def _r_table_lines(tbl, digits_d):
    """A numeric 2-D table as R's print() of a table draws it: one
    shared width, row names left, columns right-justified."""
    cells = [[_fmtr(v, digits_d) if pd.notna(v) else "NA"
              for v in tbl.loc[r]] for r in tbl.index]
    cols = [_lvl_str(c) for c in tbl.columns]
    w = max([len(c) for c in cols] + [len(s) for row in cells
                                      for s in row])
    rows = [_lvl_str(r) for r in tbl.index]
    rw = max(len(r) for r in rows)
    lines = [" " * rw + "".join(" " + c.rjust(w) for c in cols)]
    for r, row in zip(rows, cells):
        lines.append(r.ljust(rw) + "".join(" " + s.rjust(w)
                                           for s in row))
    return lines


def _summary_table(x_ser, y_ser, grp_ser, stat, y_name, digits_d,
                   facet_ser=None, facet_name=None, diff_row=False):
    """The stat of y in each cell of the grouping by x, headed
    "Summary Table for". R analog: Chart.R's xtabs() print, with a
    second grouping printed slice by slice as R prints a 3-D
    table."""
    agg = STAT_LBL.get(stat, stat.title())
    lines = [f"Summary Table for {agg} of {y_name}", ""]
    fun = STAT_FUN[stat]

    def table(mask):
        df = pd.DataFrame({"g": grp_ser[mask], "x": x_ser[mask],
                           "y": pd.to_numeric(y_ser[mask])})
        return (df.groupby(["g", "x"], observed=True)["y"].agg(fun)
                .unstack("x"))

    if facet_ser is None:
        tbl = table(slice(None))
        if diff_row and len(tbl) == 2:
            # a paired dot plot is read for the difference between
            #   the two groups, so list it with them as a third row,
            #   rounded to the table's own precision
            rn = [_lvl_str(r) for r in tbl.index]
            tbl.loc["Diff"] = (tbl.iloc[1] - tbl.iloc[0]).round(digits_d)
            lines += _r_table_lines(tbl, digits_d)
            lines += ["", f"Diff: {rn[1]}-{rn[0]}"]
        else:
            lines += _r_table_lines(tbl, digits_d)
    else:
        for lv in _category_order_sorted(facet_ser):
            lines += [f", , {facet_name} = {_lvl_str(lv)}", ""]
            lines += _r_table_lines(table(facet_ser == lv), digits_d)
            lines.append("")
    return lines


def _category_order_sorted(s):
    vals = s.dropna().unique()
    if isinstance(s.dtype, pd.CategoricalDtype):
        return [c for c in s.cat.categories if c in set(vals)]
    return sorted(vals)


def chart_stats(x_ser, by_ser, y_ser, stat, x_name, by_name, y_name,
                digits_d=2, stack100=False, n_miss=None, x_lbl=None,
                y_lbl=None, facet_ser=None, facet_name=None,
                form="bar"):
    """The text output that accompanies Chart(), as lessR prints
    it: counts of one variable with the chi-square test of equal
    probabilities; the joint and marginal frequencies with Cramer's
    V and the test of independence (column proportions instead
    with stack100); the stat of y over the levels of x (plus the
    plotted values for a bar chart); and the summary table of the
    stat when a by or facet variable groups the bars. A facet
    without by takes by's place, as in R."""
    if y_ser is None:
        if by_ser is not None and facet_ser is not None:
            # each panel crosses x with by, so describe each panel:
            #   one cross-tabulation per facet level
            lines = []
            for lv in _category_order_sorted(facet_ser):
                m = facet_ser == lv
                lines += ["", f"{facet_name} = {_lvl_str(lv)}"]
                lines += _counts_2d(x_ser[m], by_ser[m], x_name,
                                    by_name, stack100)[1:]
            return lines + [""]
        grp, grp_name = ((by_ser, by_name) if by_ser is not None
                         else (facet_ser, facet_name))
        if grp is None:
            return _counts_1d(x_ser, x_name, n_miss, x_lbl) + [""]
        return _counts_2d(x_ser, grp, x_name, grp_name,
                          stack100) + [""]

    if by_ser is None and facet_ser is None:
        lines = _stat_by_levels(y_ser, x_ser, y_name, x_name,
                                digits_d, y_lbl, x_lbl) + [""]
        if form == "bar":
            vals = (pd.to_numeric(y_ser).groupby(x_ser, observed=True,
                                                 sort=True)
                    .agg(STAT_FUN[stat]))
            lines += [""] + _plotted_values(vals, digits_d) + [""]
        return lines

    if by_ser is None:
        return _summary_table(x_ser, y_ser, facet_ser, stat, y_name,
                              digits_d) + [""]
    return _summary_table(x_ser, y_ser, by_ser, stat, y_name,
                          digits_d, facet_ser, facet_name,
                          diff_row=(form == "dot")) + [""]


def nested_stats(cols, names, y_ser, stat, y_name, digits_d=2):
    """The joint frequencies (or the stat of y) along every path of
    a nesting of several categorical variables, the table behind a
    treemap, icicle, or sunburst with more than one by variable.
    R prints nothing for three or more dimensions."""
    df = pd.DataFrame({nm: c.map(_lvl_str) for nm, c in
                       zip(names, cols)})
    if y_ser is None:
        t = df.groupby(names, observed=True, sort=True).size()
        out = pd.DataFrame({"n": t, "prop": t / t.sum()})
        body = out.to_string(formatters={
            "n": "{:d}".format, "prop": "{:.3f}".format})
        title = "Joint Frequencies of " + ", ".join(names)
    else:
        df["..y"] = pd.to_numeric(y_ser).to_numpy()
        g = df.groupby(names, observed=True, sort=True)["..y"]
        out = pd.DataFrame({"n": g.size(), stat: STAT_FUN[stat](g)})
        body = out.to_string(formatters={
            "n": "{:d}".format,
            stat: (lambda v: f"{v:.{digits_d}f}")})
        title = (f"{STAT_LBL.get(stat, stat)} of {y_name} by "
                 + ", ".join(names))
    return [title, "-" * len(title), ""] + body.split("\n") + [""]


# ----- the statistics returned with a Chart() figure ----------------

class ChartStats:
    """The statistics behind a Chart(), attached to the returned figure
    as fig.stats. Each entry reads as an attribute, fig.stats.freq, or
    by name, fig.stats["freq"], named as R's Chart() names its returned
    list where it has a counterpart: freq, prop, values, p_value,
    n_miss. text holds the console report, kept when quiet=True
    suppresses its display. Not a dict subclass, so that an entry such
    as values is not shadowed by a dict method. R analog: the list
    that Chart() returns invisibly"""

    def __init__(self, *args, **kw):
        for k, v in dict(*args, **kw).items():
            setattr(self, k, v)

    def __getitem__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            raise KeyError(name) from None

    def __setitem__(self, name, value):
        self.__dict__[name] = value

    def __contains__(self, name):
        return name in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def get(self, name, default=None):
        return self.__dict__.get(name, default)

    def update(self, *args, **kw):
        for k, v in dict(*args, **kw).items():
            self.__dict__[k] = v

    def __repr__(self):
        keys = ", ".join(k for k in self.__dict__ if k != "text")
        return (self.get("text", "").rstrip() + "\n\n"
                f"[ChartStats: {keys}]")

    __str__ = __repr__


def _ordered_counts(s):
    t = s.groupby(s, observed=True).size()
    return t[t > 0]


def _chisq_1d(cnt):
    if len(cnt) < 2:
        return {}
    chi, p = sps.chisquare(cnt.to_numpy())
    return dict(chisq=float(chi), df=len(cnt) - 1, p_value=float(p))


def _xtab_stats(ct):
    obs = ct.to_numpy()
    out = dict(freq=ct, prop=ct / obs.sum(),
               prop_col=ct / ct.sum(axis=0),
               prop_row=ct.div(ct.sum(axis=1), axis=0))
    chi, p, dof, _ = sps.chi2_contingency(obs, correction=False)
    if np.isfinite(chi):
        out.update(chisq=float(chi), df=int(dof), p_value=float(p),
                   cramer_v=float(np.sqrt(chi / ((min(obs.shape) - 1)
                                                 * obs.sum()))))
    return out


def chart_stats_data(x_ser, by_ser, y_ser, stat, facet_ser=None,
                     n_miss=None):
    """The numbers of the Chart() report as pandas objects: counts
    with proportions and the chi-square test of equal probabilities;
    a cross-tabulation with cell, column, and row proportions,
    Cramer's V, and the test of independence; or the summary of y
    over the levels of x and the plotted values, or the table of the
    statistic over the grouping"""
    st = ChartStats()
    if y_ser is None:
        grp = by_ser if by_ser is not None else facet_ser
        if grp is None:
            cnt = _ordered_counts(x_ser)
            st.update(n_dim=1, freq=cnt, prop=cnt / cnt.sum(),
                      n_miss=n_miss, **_chisq_1d(cnt))
        elif by_ser is not None and facet_ser is not None:
            # one cross-tabulation per panel
            panels = {}
            for lv in _category_order_sorted(facet_ser):
                m = facet_ser == lv
                panels[lv] = ChartStats(
                    _xtab_stats(pd.crosstab(by_ser[m], x_ser[m])))
            st.update(n_dim=3, panels=panels)
        else:
            st.update(n_dim=2, **_xtab_stats(pd.crosstab(grp, x_ser)))
        return st

    fun = STAT_FUN[stat]
    if by_ser is None and facet_ser is None:
        y = pd.to_numeric(y_ser)
        g = y.groupby(x_ser, observed=True, sort=True)
        summary = pd.DataFrame({
            "n": g.count(), "miss": g.size() - g.count(),
            "mean": g.mean(), "sd": g.std(ddof=1), "min": g.min(),
            "mdn": g.median(), "max": g.max()})
        st.update(n_dim=1, summary=summary, values=fun(g))
        return st
    grp = by_ser if by_ser is not None else facet_ser
    df = pd.DataFrame({"g": grp, "x": x_ser, "y": pd.to_numeric(y_ser)})
    tbl = (df.groupby(["g", "x"], observed=True)["y"].agg(fun)
           .unstack("x"))
    st.update(n_dim=2, summary=tbl)
    return st
