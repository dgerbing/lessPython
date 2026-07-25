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

from .utils import fmt, get_option


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


def chart_stats(x_ser, by_ser, y_ser, stat_lbl, x_name,
                by_name, y_name, digits_d=2):
    """Frequencies with proportions for one categorical
    variable; cross-tabulation with a chi-square test with by=;
    the per-level summary when y= is charted."""
    lines = [f"--- {x_name} ---"]
    if y_ser is not None:              # stat of y per level
        g = y_ser.groupby(x_ser, observed=True)
        agg = {"sd": "std"}.get(stat_lbl, stat_lbl)
        tbl = pd.DataFrame({"n": g.size(),
                            stat_lbl: g.agg(agg)})
        lines += tbl.to_string(
            float_format=lambda v: fmt(v, digits_d)).split("\n")
    elif by_ser is None:               # one-way frequencies
        cnt = x_ser.groupby(x_ser, observed=True).size()
        tbl = pd.DataFrame({"n": cnt,
                            "prop": cnt / cnt.sum()})
        tbl.loc["Total"] = [cnt.sum(), 1.0]
        lines += tbl.to_string(
            float_format=lambda v: fmt(v, digits_d)).split("\n")
    else:                              # two-way + chi-square
        ct = pd.crosstab(by_ser, x_ser)
        lines += ct.to_string().split("\n")
        chi2, p, dof, _ = sps.chi2_contingency(ct.to_numpy())
        lines += ["",
                  f"chi-square({dof}) = {fmt(chi2, 2)}, "
                  f"p-value = {fmt(p, 4)}"]
    return lines
