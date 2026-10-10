# x_console.py — the console report of X(), ports of lessR's printers
#
# R analogs: .ss.numeric() (brief, one variable), .bx.stats() (the
# outlier listing and the VBS statistics), .hst.stats() (the bin
# table and the histogram suggestions), dn.main.R (bandwidth,
# Shapiro-Wilk, the density suggestions), the grouped table that
# X.R prints through .ss.pivot(), and .param.VBS() (suggestions,
# duplicated values, parameter values) with .vbs_summary_table().
#
# Each function returns a list of lines, a component of R's
# print.out_all(); report() joins the components with one blank
# line between them, as print.out_all() does. Trailing blanks are
# dropped (R's cat(x, "\n") leaves one at the end of every line).
#
# The suggestions are written in Python syntax, naming the module
# and the DataFrame as the caller's own code names them.

import inspect
import sys

import numpy as np
import pandas as pd

from .stats_out import _fmtc, _fmti, _fmtr, _r_chr, _stat_by_levels
from .utils import get_option
from .vbs_plotly import _adj_fences, _medcouple, five_num


# ----- R number formatting --------------------------------------

def r_format1(v):
    """R format() of one number: 7 significant digits, trailing
    zeros dropped, fixed notation."""
    s = f"{float(v):.7g}"
    if "e" in s:
        s = np.format_float_positional(float(s), trim="-")
    return s


def r_chr(v):
    """R as.character() of one number: 15 significant digits."""
    s = f"{float(v):.15g}"
    if "e" in s:
        s = np.format_float_positional(float(s), trim="-")
    return s


def _n_dec(s):
    return len(s) - s.index(".") - 1 if "." in s else 0


def max_dd(x):
    """Decimal digits of the most precise of the first 200 values.
    R analog: .max.dd()"""
    m = 0
    for v in list(np.asarray(x, dtype=float))[:200]:
        if np.isfinite(v):
            m = max(m, _n_dec(r_format1(v)))
    return m


def r_format_vec(vals):
    """R format() of a numeric vector: the decimals that show each
    value to 7 significant digits, the most of them for all, then
    one common width, right-justified."""
    v = np.asarray(vals, dtype=float)
    fin = v[np.isfinite(v)]
    rgt = max((_n_dec(r_format1(z)) for z in fin), default=0)
    out = [f"{z:.{rgt}f}" if np.isfinite(z) else "NA" for z in v]
    w = max((len(s) for s in out), default=0)
    return [s.rjust(w) for s in out]


def decdig(vals):
    """Round a column of statistics as lessR's pivot() does before
    it prints: integers kept, otherwise 3 decimals, or one past the
    leading zeros of values below 1. R analog: .decdig()"""
    v = np.asarray(vals, dtype=float)
    fin = v[np.isfinite(v)]
    if len(fin) == 0 or all(float(z).is_integer() for z in fin):
        return v
    lead0 = _lead0(fin)
    locs = [r_format1(z).find(".") + 1 or len(r_format1(z)) + 1
            for z in fin]
    dgs = lead0 + 1 if lead0 > 0 else 3
    if min(locs) > 2:
        dgs = 3
    return np.round(v, dgs)


def _lead0(vals):
    """Leading zeros after the decimal point of values below 1.
    R analog: .lead0()"""
    n_max = 0
    for v in vals:
        s = r_format1(v)
        if "." in s and v < 1:
            n = len(s.split(".")[1]) - len(s.split(".")[1].lstrip("0"))
            n_max = max(n_max, n)
    return n_max


def title(x_name, x_lbl=None):
    """--- x --- heading, with the variable label. R .title2()"""
    t = x_name if x_lbl is None else f"{x_name}: {x_lbl}"
    return f"--- {t} ---"


# ----- the caller's names, for runnable suggestions ---------------

def call_names(data):
    """The prefix that reaches lessPy in the caller's namespace
    ("lp." for import lessPy as lp, "" for from lessPy import X)
    and the caller's name of the DataFrame (default "d")."""
    mod = sys.modules.get("lessPy")
    pre, dname = "lp.", "d"
    f = inspect.currentframe()
    try:
        while f is not None and str(
                f.f_globals.get("__name__", "")).startswith("lessPy"):
            f = f.f_back
        if f is not None:
            ns = dict(f.f_globals)
            ns.update(f.f_locals)
            mods = [k for k, v in ns.items()
                    if v is mod and not k.startswith("_")]
            if mods:
                pre = mods[0] + "."
            elif "X" in ns:
                pre = ""
            names = [k for k, v in ns.items()
                     if v is data and not k.startswith("_")]
            if names:
                dname = names[0]
    finally:
        del f
    return pre, dname


# ----- components ------------------------------------------------

def ss_numeric(xv, x_name, x_lbl=None, digits_d=None):
    """n, miss, mean, sd, min, mdn, max of one variable, missing
    values included in the count. R analog: .ss.numeric(x,
    brief=TRUE) with one line"""
    v = np.asarray(xv, dtype=float)
    miss = int(np.isnan(v).sum())
    v = v[~np.isnan(v)]
    n = len(v)
    d = digits_d
    if d is None:
        d = max_dd(v) + 1
        if d == 1:
            d = 2
    max_lv = 3
    max_n = len(str(n))
    max_nm = len(str(miss))
    m = v.mean() if n > 0 else np.nan
    s = v.std(ddof=1) if n > 1 else np.nan
    if np.isnan(s):
        max_ln = len(_r_chr(m, d)) + d if n > 0 else 0
    else:
        max_ln = max(len(_r_chr(m, d)), len(_r_chr(s, d))) + d
    for lim, add in ((5, 1), (10, 1), (4, 2), (8, 1)):
        if max_ln < lim:
            max_ln += add
    lines = [title(x_name, x_lbl), ""]
    lines.append(" ".join([
        _fmtc("n", len(str(max_n)) + 3 + max_lv - 2),
        _fmtc("miss", len(str(max_nm)) + 5),
        _fmtc("mean", max_ln), _fmtc("sd", max_ln),
        _fmtc("min", max_ln), _fmtc("mdn", max_ln),
        _fmtc("max", max_ln)]))
    row = [_fmtc("", max_lv), _fmti(n, max_n + 1),
           _fmti(miss, max_nm + 5)]
    if n == 1:
        row.append(_fmtr(m, d, max_ln))
    elif n > 1:
        row += [_fmtr(z, d, max_ln) for z in
                (m, s, v.min(), np.median(v), v.max())]
    lines.append(" ".join(row))
    lines.append("")
    return lines


def _box_stats(x, k_iqr=1.5, box_adj=False, a=-4, b=3):
    """Whiskers and outliers on Tukey hinges, as boxplot.stats(), or
    medcouple-adjusted as robustbase::adjboxStats()"""
    mn, q1, md, q3, mx = five_num(x)
    iqr = q3 - q1
    mcv = _medcouple(x) if box_adj else 0.0
    (lo_f, hi_f), _ = _adj_fences(q1, q3, iqr, k_iqr, mcv, a, b)
    out = (x < lo_f) | (x > hi_f)
    inside = x[~out]
    return (inside.min(), q1, md, q3, inside.max()), x[out]


def outliers(x, digits_d=2, ids=None, k_iqr=1.5, box_adj=False,
             a=-4, b=3):
    """The boxplot outliers, small and large, with their IDs.
    R analog: .bx.stats()$txotl"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if digits_d is None:
        digits_d = max(max_dd(x) + 1, 2)
    st, out = _box_stats(x, k_iqr, box_adj, a, b)
    if len(out) == 0:
        return ["No (Box plot) outliers"]
    ids = (np.asarray(ids).astype(str) if ids is not None
           else np.array([""] * len(x)))
    d1 = digits_d - 1

    def side(mask, decreasing):
        xv, iv = x[mask], ids[mask]
        o = np.argsort(-xv if decreasing else xv, kind="stable")
        return ([f"{z:.{d1}f}" for z in xv[o]], [str(i) for i in iv[o]])

    lo_mask, hi_mask = x < st[0], x > st[4]
    if lo_mask.any():
        xc_lo, id_lo = side(lo_mask, False)
        lo_len = len(xc_lo)
        max_lo = max(len(s) for s in xc_lo)
        max_id_lo = max(len(s) for s in id_lo)
    else:
        xc_lo, id_lo, max_lo, max_id_lo, lo_len = [""], ["    "], 0, 5, 0
    if hi_mask.any():
        xc_hi, id_hi = side(hi_mask, True)
        hi_len = len(xc_hi)
        max_hi = max(len(s) for s in xc_hi)
        max_id_hi = max(len(s) for s in id_hi)
    else:
        xc_hi, id_hi, max_hi, max_id_hi, hi_len = [""], ["   "], 0, 5, 0

    max_list = 18
    n_lines = min(max(lo_len, hi_len), max_list)
    for i in range(1, n_lines + 1):    # R grows the shorter side
        if i > lo_len:
            xc_lo.append("")
            id_lo.append("")
        if i > hi_len:
            xc_hi.append("")
            id_hi.append("")

    def left(s, w):
        return str(s).ljust(max(w, 0))

    adj = len("Small") - max_id_lo
    buf = 3 - adj if adj > 0 else 3
    lines = [" ", f"--- Outliers ---     from the boxplot: {len(out)}",
             "",
             " ".join([left("Small", max_id_lo), _fmtc("", max_lo),
                       _fmtc("", max(buf, 0)), left("Large", max_id_hi)]),
             " ".join([left("-----", max_id_lo), _fmtc("", max_lo),
                       _fmtc("", max(buf, 0)), left("-----", max_id_hi)])]
    for i in range(n_lines):
        lines.append(" ".join([
            left(id_lo[i], max_id_lo), _fmtc(xc_lo[i], max_lo), "   ",
            left(id_hi[i], max_id_hi), _fmtc(xc_hi[i], max_hi)]))
    if max(lo_len, hi_len) > n_lines:
        lines += ["", f"+ {len(out) - max_list} more outliers"]
    return lines


def box_summary(x, x_name, x_lbl=None, digits_d=None, n_total=None,
                k_iqr=1.5, box_adj=False, a=-4, b=3):
    """The statistics of the VBS display, whiskers included.
    R analog: .bx.stats()$txstat"""
    x = np.asarray(x, dtype=float)
    miss = int(np.isnan(x).sum())
    v = x[~np.isnan(x)]
    d = digits_d
    if d is None:
        d = max(max_dd(v) + 1, 2)
    st, _ = _box_stats(v, k_iqr, box_adj, a, b)
    q1, q3 = np.percentile(v, [25, 75])

    def f(z):
        return f"{z:.{d}f}"

    return [title(x_name, x_lbl),
            f"Present: {len(v)}",
            f"Missing: {miss}",
            f"Total  : {len(x)}",
            "",
            f"Mean         : {f(v.mean())}",
            f"Stnd Dev     : {f(v.std(ddof=1))}",
            f"IQR          : {f(q3 - q1)}",
            f"Skew         : {f(_medcouple(v))}   [medcouple, -1 to 1]",
            "",
            f"Minimum      : {f(v.min())}",
            f"Lower Whisker: {f(st[0])}",
            f"1st Quartile : {f(q1)}",
            f"Median       : {f(np.median(v))}",
            f"3rd Quartile : {f(q3)}",
            f"Upper Whisker: {f(st[4])}",
            f"Maximum      : {f(v.max())}"]


def bin_table(edges, counts, n, mx_dd):
    """Bin width, number of bins, and the frequency distribution.
    R analog: .hst.stats()$tx"""
    br = np.asarray(edges, dtype=float)
    cnt = np.asarray(counts, dtype=float)
    mids = (br[:-1] + br[1:]) / 2
    bw = float(f"{br[1] - br[0]:.10g}")
    max_dg = max((len(r_chr(b)) for b in br if len(r_chr(b)) < 17),
                 default=0)
    max_dg_mid = max((len(r_chr(m)) for m in mids
                      if len(r_chr(m)) < 19), default=0)
    x_br = [s.rjust(max_dg) for s in r_format_vec(br)]
    x_mid = [s.rjust(max_dg_mid) for s in r_format_vec(mids)]
    if _lead0(mids) > mx_dd:
        x_mid = [f"{m:.{mx_dd}f}" for m in mids]
    bn = [f"{x_br[i]} > {x_br[i + 1]}" for i in range(len(br) - 1)]
    prop = cnt / n
    cols = [
        ("Bin", bn), ("Midpnt", x_mid),
        ("Count", [f"{c:.0f}" for c in cnt]),
        ("  Prop", [f"{p:.2f}" for p in prop]),
        ("Cumul.c", [f"{c:.0f}" for c in np.cumsum(cnt)]),
        ("Cumul.p", [f"{p:.2f}" for p in np.cumsum(prop)])]
    widths = [max(len(nm) + 1, max(len(s) for s in vals)) + 1
              for nm, vals in cols]
    lines = ["", f"Bin Width: {r_chr(bw)}",
             f"Number of Bins: {len(br) - 1}", "",
             "".join(_fmtc(nm, w) for (nm, _), w in zip(cols, widths)),
             "-" * sum(widths)]
    for i in range(len(bn)):
        lines.append("".join(_fmtc(vals[i], w)
                             for (_, vals), w in zip(cols, widths)))
    lines.append("")
    return lines


def _df_lines(cols, sep=" ", lead=" "):
    """Columns (name, strings) right-justified to their widths,
    as R prints a data frame without row names."""
    widths = [max(len(nm), max((len(s) for s in vals), default=0))
              for nm, vals in cols]
    n_rows = len(cols[0][1]) if cols else 0
    out = [lead + sep.join(nm.rjust(w)
                           for (nm, _), w in zip(cols, widths))]
    for i in range(n_rows):
        out.append(lead + sep.join(vals[i].rjust(w)
                                   for (_, vals), w in zip(cols, widths)))
    return out


def _group_stats(x, g, order=None):
    """Per-group statistics, groups in the given order (the levels of
    a categorical, as R's factor levels) or sorted."""
    s = pd.Series(np.asarray(x, dtype=float))
    gs = pd.Series(np.asarray(g)).astype(str)
    lvls = ([str(v) for v in order] if order is not None
            else sorted(gs.unique()))
    rows = []
    for lv in lvls:
        z = s[(gs == lv).to_numpy()]
        if len(z) == 0:
            continue
        zz = z.dropna()
        q1, q3 = (np.percentile(zz, [25, 75]) if len(zz)
                  else (np.nan, np.nan))
        rows.append(dict(lv=lv, n=len(zz), na=int(z.isna().sum()),
                         Mean=zz.mean(), Median=zz.median(),
                         SD=zz.std(ddof=1), IQR=q3 - q1,
                         Min=zz.min(), Max=zz.max()))
    return rows


def by_table(x, by, x_name, by_name, pre="lp.", dname="d",
             order=None):
    """Statistics of x for each group of by, then the pointer to
    the test of the mean difference for two groups. R analog: the
    grouped-stats block of X.R through .ss.pivot()"""
    rows = _group_stats(x, by, order)
    cols = [(by_name, [r["lv"] for r in rows]),
            ("n", [str(r["n"]) for r in rows]),
            ("na", [str(r["na"]) for r in rows])]
    for k in ("Mean", "Median", "SD", "IQR", "Min", "Max"):
        cols.append((k, r_format_vec(decdig([r[k] for r in rows]))))
    lines = [f"{x_name} by {by_name}", ""] + _df_lines(cols)
    if len(rows) == 2:
        lines += ["", "For inferential analysis of the mean difference:",
                  f'> {pre}ttest("{x_name} ~ {by_name}", data={dname})',
                  ""]
    else:
        lines += [""]
    return lines


def facet_table(x, cells, x_name, cells_name, digits_d=2):
    """Statistics of x in each panel, missing values counted per
    panel. R analog: .ss.numeric(x, by=cells, brief=TRUE) as
    .bar.lattice() calls it, at getOption("digits_d"): lessR's
    default 2 for facet alone, the one-variable digits (.max.dd + 1)
    left by the earlier summary when by= divides the panels"""
    return _stat_by_levels(pd.Series(np.asarray(x, dtype=float)),
                           pd.Series(np.asarray(cells)).astype(str),
                           x_name, cells_name, digits_d)


def vbs_summary(x, g, label, digits_d, order=None):
    """n, Mean, Median, SD, IQR, Min, Max of x, per group of g or
    for all of x. R analog: .vbs_summary_table() with .df_char()"""
    if g is None:
        rows = _group_stats(x, np.repeat(label, len(x)))
    else:
        rows = _group_stats(x, g, order)
    cols = [(label, [r["lv"] for r in rows]),
            ("n", [str(r["n"]) for r in rows])]
    for k in ("Mean", "Median", "SD", "IQR", "Min", "Max"):
        cols.append((k, [f"{r[k]:.{digits_d}f}" for r in rows]))
    return _df_lines(cols, sep="  ", lead="")


def dup_values(x, grp=None, grp_order=None):
    """The most repeated value(s) of x, overall or per group.
    R analog: .get.dup() of .param.VBS()"""
    x = np.asarray(x, dtype=float)
    ux = np.unique(x)
    if len(ux) >= 1001:
        return []
    if grp is None:
        cnt = pd.Series(x).value_counts()
        mx_c = 0 if len(ux) == len(x) else int(cnt.max())
        if mx_c <= 1:
            return ["Number of duplicated values: 0"]
        lvls, cols = [""], [cnt]
    else:
        g = np.asarray(grp).astype(str)
        lvls = [str(lv) for lv in grp_order]
        cols = [pd.Series(x[g == lv]).value_counts() for lv in lvls]
    max_lvl = max((len(lv) for lv in lvls), default=0)
    buf = max_lvl - 4 if max_lvl - 5 >= 0 else 0
    # R's for (i in 1:buf) runs twice when buf is 0
    h1 = "Level" + " " * (buf if buf > 0 else 2)
    h0 = h1.replace("Level", "    ")
    lines = ["", f"{h0} Max Dupli-", f"{h1} cations   Values", "-" * 30]
    for lv, cnt in zip(lvls, cols):
        mx = int(cnt.max()) if len(cnt) else 0
        lvl_c = "" if grp is None else lv.ljust(max(max_lvl, 6))
        if mx > 1:
            vals = sorted(cnt[cnt == mx].index)
            more = " ..." if len(vals) > 8 else ""
            vals = " ".join(r_chr(v) for v in vals[:8])
            lines.append(f"{lvl_c}    {mx:3d}     {vals}{more}")
        else:
            lines.append(f"{lvl_c}      0")
    return lines


def vbs_params(vbs_plot, prm, show_bw=True):
    """Point size, outlier size, jitter, and bandwidth as plotted.
    R analog: .get.param() of .param.VBS()"""
    if "s" not in vbs_plot and "v" not in vbs_plot:
        return []
    lines = ["", "---------- Parameter values (can be manually set)", ""]
    if "s" in vbs_plot:
        lines += [
            f"pt_size: {prm['pt_size']:.2f}   size of plotted points",
            f"out_size: {prm['out_size']:.2f}  size of plotted "
            "outlier points",
            f"jitter_y: {prm['jitter_y']:.2f} random vertical movement "
            "of points",
            f"jitter_x: {prm['jitter_x']:.2f}  random horizontal "
            "movement of points"]
    if "v" in vbs_plot and show_bw:
        lines.append(f"bw: {prm['bw']:.2f}       set bandwidth higher "
                     "for smoother edges")
    return lines


# ----- suggestions ------------------------------------------------

def _x_call(pre, x_name, dname, args):
    return f'{pre}X("{x_name}", data={dname}{args})'


def suggest_hist(x_name, pre, dname, given):
    if not get_option("suggest", True):
        return []
    lines = [">>> Suggestions"]
    for p, what in (("bin_width", "set the width of each bin"),
                    ("bin_start", "set the start of the first bin"),
                    ("bin_end", "set the end of the last bin")):
        if p not in given:
            lines.append(f"{p}: {what}")
    lines.append(_x_call(pre, x_name, dname, ', form="density"')
                 + "  # smoothed curve + histogram")
    lines.append(_x_call(pre, x_name, dname, ', form="vbs"')
                 + "  # Violin/Box/Scatterplot (VBS) plot")
    return lines


def suggest_density(x_name, pre, dname):
    if not get_option("suggest", True):
        return []
    return [">>> Suggestions",
            "bin_width: set the width of each bin",
            _x_call(pre, x_name, dname, ', form="histogram"')
            + "  # histogram only",
            _x_call(pre, x_name, dname, ', form="vbs"')
            + "  # Violin/Box/Scatterplot (VBS) plot"]


def suggest_freq_poly(x_name, pre, dname, given):
    if not get_option("suggest", True):
        return []
    lines = [">>> Suggestions"]
    if "bin_width" not in given:
        lines.append("bin_width: set the width of each bin")
    lines.append(_x_call(pre, x_name, dname, ', form="histogram"')
                 + "  # histogram")
    lines.append(_x_call(pre, x_name, dname,
                         ', form="freq_poly", area_fill="off"')
                 + "  # polygon outline only")
    return lines


def suggest_vbs(x_name, pre, dname, given, n_by=0, by_name=None,
                n_facet=0, facet_name=None):
    """R analog: the suggestions of .param.VBS()"""
    if not get_option("suggest", True):
        return []
    lines = [">>> Suggestions"]
    t = ["", "", ""]
    c = ["", "", ""]
    if "out_cut" not in given:
        t[0], c[0] = ", out_cut=2", "Label two outliers ..."
    if "fences" not in given:
        t[1] = ", fences=True"
        if c[0] == "":
            c[1] = "Show inner fences"
    if "vbs_mean" not in given:
        t[2] = ", vbs_mean=True"
        if c[1] == "":
            c[2] = " Show mean"
    if any(t):
        lines.append(_x_call(pre, x_name, dname,
                             "".join(t) + ', form="vbs"')
                     + "  # " + "".join(c))
    if "box_adj" not in given:
        lines.append(_x_call(pre, x_name, dname,
                             ', box_adj=True, form="vbs"')
                     + "  # Adjust whiskers for asymmetry")
    nm = (facet_name if n_facet == 2 else by_name if n_by == 2
          else None)
    if nm is not None:
        lines.append(f'{pre}ttest("{x_name} ~ {nm}", data={dname})')
    nm = (facet_name if n_facet > 2 else by_name if n_by > 2
          else None)
    if nm is not None:
        lines.append(f'{pre}ANOVA("{x_name} ~ {nm}", data={dname})')
    return lines


def report(*components):
    """Join the components with one blank line between them, as
    print.out_all() does, skipping empty ones."""
    lines = []
    for comp in components:
        if not comp:
            continue
        if lines:
            lines.append("")
        lines += comp
    return "\n".join(s.rstrip() for s in lines)


# ===== XY() ======================================================
# R analogs: .cr.main(brief=TRUE) (the correlation), the fit text and
# suggestions of .plt.txt(), .plt.MD(), and the run analysis.

def xy_digits(yv, digits_d=None):
    """.plt.txt(): one past the decimals of y, at least 3"""
    if digits_d is None:
        digits_d = max_dd(yv) + 1
    return max(digits_d, 3)


def cor_block(xv, yv, x_name, y_name, x_lbl=None, y_lbl=None):
    """Pearson correlation, its t test and 95% confidence interval.
    R analog: .cr.main(brief=TRUE) through cor.test()"""
    from scipy import stats as sps
    x = np.asarray(xv, dtype=float)
    y = np.asarray(yv, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    lines = ["", ">>> Pearson's product-moment correlation"]
    if x_lbl is not None or y_lbl is not None:
        lines += ["", x_name if x_lbl is None else f"{x_name}: {x_lbl}",
                  y_name if y_lbl is None else f"{y_name}: {y_lbl}"]
    lines += ["", "Number of paired values with neither missing, "
              f"n = {n}"]
    if n < 4:
        return lines
    r = float(np.corrcoef(x, y)[0, 1])
    df = n - 2
    t = r * np.sqrt(df / max(1 - r * r, 1e-300))
    p = 2 * sps.t.sf(abs(t), df)
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    q = sps.norm.ppf(0.975)
    lb, ub = np.tanh(z - q * se), np.tanh(z + q * se)

    def f3(v):
        return f"{round(float(v), 3):.3f}"

    lines += [f"Sample Correlation of {x_name} and {y_name}: r = {r:.3f}",
              "",
              f"Hypothesis Test of 0 Correlation:  t = {f3(t)},  "
              f"df = {df},  p-value = {f3(p)}",
              f"95% Confidence Interval for Correlation:  {f3(lb)} to "
              f"{f3(ub)}", ""]
    return lines


def _lin_fit(xs, ys):
    b1, b0 = np.polyfit(xs, ys, 1)
    e = ys - (b0 + b1 * xs)
    tss = ((ys - ys.mean()) ** 2).sum()
    rsq = 1 - (e @ e) / tss if tss > 0 else np.nan
    return b0, b1, (e @ e) / (len(ys) - 2), rsq


# the linearizing transform of y for each curved fit, as .plt.fit()
_FIT_OPS = {"quad": ("sqrt()", "square"),
            "power": ("the root of the\n   reciprocal of the power",
                      "of the power"),
            "exp": ("log()", "exp()"),
            "log": ("exp()", "log()")}


def fit_text(fit_stats, fit, fit_power, y_name, digits_d,
             by_name=None, var_label=False):
    """The fit line(s) and their MSE. lm and the curved fits report
    the line of the (linearized) regression; loess its MSE alone.
    R analog: the out_reg text of .plt.txt()"""
    d = digits_d
    p = 2 if fit == "quad" else fit_power
    lines = []
    for i, fs in enumerate(fit_stats):
        nm, ys, f = fs[0], np.asarray(fs[2], float), np.asarray(fs[3], float)
        xs = np.asarray(fs[4], float) if len(fs) > 4 else None
        msg = ""
        if fit in _FIT_OPS and (i == 0 or nm is not None):
            msg = ("Regressed linearized data of transformed data "
                   f"values of {y_name} with {_FIT_OPS[fit][0]}")
        # R: paste(msg, by.name, ": ", by.cat, "  ") or by.cat ("")
        head = ("  " if nm is None
                else f"Variable: {nm}  " if var_label
                else f"{by_name}: {nm}  ")
        if msg:
            lines += msg.split("\n")
        if fit == "loess" or xs is None:
            mse = ((ys - f) ** 2).sum() / (len(ys) - 2)
            md = "Loess" if fit == "loess" else "Linear"
            lines += [f"{head} {md} Model MSE = {mse:,.{d}f}", ""]
            continue
        yt = ys
        if fit in ("quad", "power"):
            yt = ys ** (1 / p)
        elif fit == "exp":
            yt = np.log(ys if p == 1 else ys ** (1 / p))
        elif fit == "log":
            yt = np.exp(ys if p == 1 else ys ** (1 / p))
        ok = np.isfinite(yt)
        b0, b1, mse, rsq = _lin_fit(xs[ok], yt[ok])
        lines += [f"{head}Line: b0 = {b0:.{d}f}    b1 = {b1:.{d}f}",
                  f"  Linear Model MSE = {mse:,.{d}f}   Rsq = {rsq:.3f}",
                  ""]
        if fit in _FIT_OPS:
            mse_nl = ((ys - f) ** 2).sum() / (len(ys) - 2)
            lines[-1:] = ["", "Fit to the data with back transform "
                          f"{_FIT_OPS[fit][1]} of linear regression model",
                          f"Model MSE = {mse_nl:,.{d}f}", "", ""]
    return lines


def _xy_call(pre, x, y, dname, extra=""):
    def q(v):
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(f'"{s}"' for s in v) + "]"
        return f'"{v}"'
    return f"{pre}XY({q(x)}, {q(y)}, data={dname}{extra}"


def suggest_scatter(pre, x, y, dname, given, by=None, rng=None):
    """The scatterplot suggestions, R's alternates drawn at random as
    .plt.txt() does with runif(). R analog: .plt.txt() contcont"""
    if not get_option("suggest", True):
        return []
    rng = np.random.default_rng() if rng is None else rng
    fc = _xy_call(pre, x, y, dname,
                  f', by="{by}"' if by is not None else "")
    lines = [">>> Suggestions  or  enter: style(suggest=False)"]
    if "enhance" not in given:
        lines.append(f"{fc}, enhance=True)  # many options")
    if rng.random() > 0.5:
        if "fill" not in given:
            lines.append(f'{fc}, fill="skyblue")  # interior fill color '
                         "of points")
    elif "color" not in given:
        lines.append(f'{fc}, color="red")  # exterior edge color of '
                     "points")
    if "fit" not in given:
        lines.append(f'{fc}, fit="lm", fit_se=[.90, .99])  # fit line, '
                     "stnd errors")
    if rng.random() > 0.5:
        if "out_cut" not in given:
            lines.append(f"{fc}, out_cut=.10)  # label top 10% from "
                         "center as outliers")
    elif "MD_cut" not in given:
        lines.append(f"{fc}, MD_cut=6)  # Mahalanobis distance from "
                     "center > 6 is an outlier")
    if by is not None and "pt_shape" not in given:
        lines.append(f'{fc}, pt_shape="diamond")  # diamond for points')
    return lines


def suggest_series(pre, x, y, dname, given, extra="", run=False,
                   pt_size=1):
    """Time-series and run-chart suggestions. R analog: .plt.txt()
    date.var and run blocks"""
    if not get_option("suggest", True):
        return []
    lines = [">>> Suggestions  or  enter: style(suggest=False)"]
    fc = _xy_call(pre, x, y, dname)
    full = _xy_call(pre, x, y, dname, extra)
    if not run:
        if "ts_ahead" not in given:
            lines.append(f"{fc}, ts_ahead=4)  # exponential smoothing "
                         "forecast 4 time units")
        if "ts_unit" not in given:
            lines.append(f'{fc}, ts_unit="years")  # aggregate time by '
                         "yearly sum")
        if "ts_agg" not in given:
            lines.append(f'{fc}, ts_unit="years", ts_agg="mean")  '
                         "# aggregate by yearly mean")
        if "ts_ahead" in given and "ts_seasons" not in given:
            lines.append(f"{fc}, ts_ahead=4, ts_seasons=False)  # turn "
                         "off exponential smoothing seasonal effect")
    if "pt_size" not in given and pt_size > 0:
        lines.append(f"{full}, pt_size=0)  # just line segments, "
                     "no points")
    if "line_width" not in given:
        if pt_size > 0:
            lines.append(f"{full}, line_width=0)  # just points, no "
                         "line segments")
        else:
            lines.append(f'{full}, line_width=0, ts_area_fill="on")  '
                         "# just area")
    if "ts_area_fill" not in given and "ts_stack" not in given:
        lines.append(f'{full}, ts_area_fill="on")  # default color fill')
    return lines


def run_summary(yv, digits_d):
    """The summary of y above a run analysis, untitled.
    R analog: .ss.numeric(y, x.name="*NONE*", brief=TRUE)"""
    lines = ss_numeric(yv, "", None, digits_d)
    return [" "] + lines[2:]
