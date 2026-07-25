# ttest.py — analog of ttest.R (tt.2group, tt.1group,
# tt.formula).
#
# ttest(): the t-test of a mean or a mean difference, with the
# lessR output pipeline — Describe, Assumptions (normality and,
# for two groups, homogeneity of variance), Infer (the equal-
# variance test and, for two groups, the Welch test), Effect
# Size (Cohen's d), Practical Importance, and Needed Sample Size,
# plus a plotly density plot. Modes, from the arguments:
#   ttest("Y ~ Group", data=d)      two independent groups
#   ttest("V1", "V2", data=d)       two independent groups
#   ttest("V1", "V2", data=d, paired=True)   paired
#   ttest("V", data=d, mu=100)      one group vs mu
#   ttest("V", data=d)              one group, CI only
#   ttest(n1=, m1=, s1=, n2=, m2=, s2=)      two groups from stats
#   ttest(n=, m=, s=, mu=)          one group from stats
#
# The larger-mean group is always reported first, as R. Numerics
# through scipy.stats (imported lazily); the pipeline is ported,
# not the lines. Returns a ttestResults object; the figure is in
# .plots and is not auto-shown.

import numpy as np
import pandas as pd

from .Regression import _getdigits
from .utils import fmt, get_column, get_option


class ttestResults:
    """Numeric results of ttest(): the per-group summary, the
    equal-variance and (two-group) Welch inference, the effect
    size, and the plotly figure in .plots."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return f"<lessPy ttest: {self.kind}, n={self.n_total}>"


def _F(x, d):
    return fmt(x, d)


def ttest(x=None, y=None, data=None, filter=None, paired=False,
          n=None, m=None, s=None, mu=None,
          n1=None, n2=None, m1=None, m2=None, s1=None, s2=None,
          Ynm="Y", Xnm="X", X1nm="Group1", X2nm="Group2",
          brief=False, digits_d=None, conf_level=0.95,
          alternative="two_sided", mmd=None, msmd=None,
          Edesired=None, graph=True):
    """t-test of a mean (one group, vs mu) or a mean difference
    (two independent groups or paired). Always prints, as in R;
    returns a ttestResults object with the density figure in
    .plots."""
    if alternative not in ("two_sided", "less", "greater"):
        raise ValueError(
            'alternative: "two_sided", "less", "greater"')
    if data is not None and filter is not None:
        data = data.query(filter)

    from_stats = n is not None or n1 is not None
    two_group = n1 is not None or (not from_stats and y is not None
                                   and not paired)

    if from_stats:
        if n1 is not None:
            return _two_group_stats(
                n1, m1, s1, n2, m2, s2, Ynm, Xnm, X1nm, X2nm,
                brief, digits_d, conf_level, alternative, mmd,
                msmd, Edesired)
        return _one_group(
            None, n, m, s, mu, Ynm, brief, conf_level,
            alternative, digits_d, mmd, msmd, Edesired, False,
            graph=False)

    # from data: resolve x (and y) as column names or arrays
    if isinstance(x, str) and "~" in x:
        yv, xg, Ynm, Xnm, X1nm, X2nm = _formula(x, data)
        return _two_group_data(
            yv, xg, Ynm, Xnm, X1nm, X2nm, brief, digits_d,
            conf_level, alternative, mmd, msmd, Edesired, graph)

    xv = _resolve(x, data, "x")
    if y is None:
        Ynm = x if isinstance(x, str) else Ynm
        return _one_group(
            xv, None, None, None, mu, Ynm, brief, conf_level,
            alternative, digits_d, mmd, msmd, Edesired, False,
            graph=graph)

    yv = _resolve(y, data, "y")
    if paired:
        if len(xv) != len(yv):
            raise ValueError("paired samples must be equal length")
        keep = ~(np.isnan(xv) | np.isnan(yv))
        diff = yv[keep] - xv[keep]         # y - x, as R
        return _one_group(
            diff, None, None, None, 0.0, "Difference", brief,
            conf_level, alternative, digits_d, mmd, msmd,
            Edesired, True, graph=graph)

    X1nm = x if isinstance(x, str) else X1nm
    X2nm = y if isinstance(y, str) else X2nm
    Ynm = "Y"
    return _two_group_data(
        {X1nm: xv, X2nm: yv}, None, Ynm, Xnm, X1nm, X2nm, brief,
        digits_d, conf_level, alternative, mmd, msmd, Edesired,
        graph)


def _resolve(v, data, arg):
    if isinstance(v, str):
        if data is None:
            raise ValueError(f"{arg}='{v}' is a column name, so "
                             "data= is required")
        return get_column(data, v, arg).to_numpy(dtype=float)
    return np.asarray(v, dtype=float)


def _formula(f, data):
    if data is None:
        raise ValueError("a formula needs data=")
    lhs, rhs = (t.strip() for t in f.split("~"))
    y_ser = get_column(data, lhs, "response")
    g_ser = get_column(data, rhs, "grouping")
    if not pd.api.types.is_numeric_dtype(y_ser):
        raise TypeError(f"the response '{lhs}' must be numeric")
    levels = list(pd.unique(g_ser.dropna().astype(str)))
    if len(levels) != 2:
        raise ValueError(
            f"the grouping variable '{rhs}' must have exactly "
            f"two values; found {len(levels)}. Use ANOVA for "
            "more than two groups.")
    levels = sorted(levels)
    gs = g_ser.astype(str)
    groups = {lv: y_ser[gs == lv].to_numpy(dtype=float)
              for lv in levels}
    return groups, None, lhs, rhs, levels[0], levels[1]


# ------------------------------------------------------------
# formatting helpers for the group summary
# ------------------------------------------------------------

def _alt_word(alt):
    return {"two_sided": "two.sided", "less": "less",
            "greater": "greater"}[alt]


def _tcrit(conf_level, df, alt):
    from scipy.stats import t as tdist
    if alt == "two_sided":
        return float(tdist.ppf(1 - (1 - conf_level) / 2, df))
    return float(tdist.ppf(conf_level, df))


def _pval(tvalue, df, alt):
    from scipy.stats import t as tdist
    if alt == "two_sided":
        return float(2 * tdist.sf(abs(tvalue), df))
    if alt == "less":
        return float(tdist.cdf(tvalue, df))
    return float(tdist.sf(tvalue, df))


# ------------------------------------------------------------
# two independent groups
# ------------------------------------------------------------

def _two_group_data(groups, _g, Ynm, Xnm, X1nm, X2nm, brief,
                    digits_d, conf_level, alternative, mmd, msmd,
                    Edesired, graph):
    YA0 = groups[X1nm]
    YB0 = groups[X2nm]
    n1m = int(np.isnan(YA0).sum())
    n2m = int(np.isnan(YB0).sum())
    YA = YA0[~np.isnan(YA0)]
    YB = YB0[~np.isnan(YB0)]
    if len(YA) < 2 or len(YB) < 2:
        raise ValueError("need at least two cases per sample")
    # the larger-mean group is reported first, as R
    if YA.mean() < YB.mean():
        YA, YB = YB, YA
        X1nm, X2nm = X2nm, X1nm
        n1m, n2m = n2m, n1m
    d = digits_d if digits_d is not None else _getdigits(
        np.concatenate([YA, YB]), 3)
    return _two_group(
        len(YA), YA.mean(), YA.std(ddof=1),
        len(YB), YB.mean(), YB.std(ddof=1),
        Ynm, Xnm, X1nm, X2nm, brief, d, conf_level, alternative,
        mmd, msmd, Edesired, YA, YB, n1m, n2m, graph)


def _two_group_stats(n1, m1, s1, n2, m2, s2, Ynm, Xnm, X1nm,
                     X2nm, brief, digits_d, conf_level,
                     alternative, mmd, msmd, Edesired):
    if m1 < m2:                            # larger mean first
        n1, n2 = n2, n1
        m1, m2 = m2, m1
        s1, s2 = s2, s1
        X1nm, X2nm = X2nm, X1nm
    d = digits_d if digits_d is not None else 3
    return _two_group(n1, m1, s1, n2, m2, s2, Ynm, Xnm, X1nm,
                      X2nm, brief, d, conf_level, alternative,
                      mmd, msmd, Edesired, None, None, 0, 0,
                      False)


def _two_group(n1, m1, s1, n2, m2, s2, Ynm, Xnm, X1nm, X2nm,
               brief, d, conf_level, alternative, mmd, msmd,
               Edesired, YA, YB, n1m, n2m, graph):
    from scipy.stats import f as fdist
    from_data = YA is not None
    v1, v2 = s1 ** 2, s2 ** 2
    sd_d = d if from_data else d - 1
    clpct = f"{round(conf_level * 100, 2):g}%"
    alt = alternative

    L = ["", f"Compare {Ynm} across {Xnm} with levels "
         f"{X1nm} and {X2nm}",
         f"Grouping Variable:  {Xnm}",
         f"Response Variable:  {Ynm}", ""]

    L.append("------ Describe ------" if not brief
             else " --- Describe ---")
    L.append("")
    miss1 = f"n.miss = {n1m},  " if from_data else ""
    miss2 = f"n.miss = {n2m},  " if from_data else ""
    L.append(f"{Ynm} for {Xnm} {X1nm}:  {miss1}n = {n1}, "
             f" mean = {_F(m1, sd_d)},  sd = {_F(s1, sd_d)}")
    L.append(f"{Ynm} for {Xnm} {X2nm}:  {miss2}n = {n2}, "
             f" mean = {_F(m2, sd_d)},  sd = {_F(s2, sd_d)}")
    L += ["", f"Mean Difference of {Ynm}:  {_F(m1 - m2, d)}"]

    df1, df2 = n1 - 1, n2 - 1
    swsq = (df1 * v1 + df2 * v2) / (df1 + df2)
    sw = np.sqrt(swsq)
    smd = (m1 - m2) / sw
    L += ["", f"Weighted Average Standard Deviation:  {_F(sw, d)}"]
    if brief:
        L.append(f"Standardized Mean Difference of {Ynm}: "
                 f"{_F(smd, d)}")

    if not brief:
        L += _assumptions(YA, YB, X1nm, X2nm, Ynm, n1, n2, v1,
                          v2, df1, df2, from_data, d)

    # ----- Infer: equal variances (pooled) -----
    L.append("")
    L.append("------ Infer ------" if not brief
             else " --- Infer ---")
    L.append("")
    if not brief:
        L.append("--- Assume equal population variances of "
                 f"{Ynm} for each {Xnm}")
        L.append("")
    sterr = sw * np.sqrt(1 / n1 + 1 / n2)
    df = df1 + df2
    tcut = _tcrit(conf_level, df, alt)
    tvalue = (m1 - m2) / sterr
    pvalue = _pval(tvalue, df, alt)
    E = tcut * sterr
    lb, ub = (m1 - m2) - E, (m1 - m2) + E
    if alt != "two_sided":
        L.append("Alternative hypothesis: Population mean "
                 f"difference is {_alt_word(alt)} than 0")
    L += [f"t-cutoff for {clpct} range of variation: "
          f"tcut = {_F(tcut, 3)}",
          f"Standard Error of Mean Difference: SE = {_F(sterr, d)}",
          "",
          f"Hypothesis Test of 0 Mean Diff:  t-value = "
          f"{_F(tvalue, 3)},  df = {df},  p-value = "
          f"{_F(pvalue, 3)}", "",
          f"Margin of Error for {clpct} Confidence Level:  "
          f"{_F(E, d)}",
          f"{clpct} Confidence Interval for Mean Difference:  "
          f"{_F(lb, d)} to {_F(ub, d)}"]

    welch = None
    if not brief:
        welch = _welch(m1, m2, v1, v2, n1, n2, conf_level, alt,
                       YA, YB)
        L += ["", "--- Do not assume equal population variances "
              f"of {Ynm} for each {Xnm}", "",
              f"t-cutoff: tcut = {_F(welch['tcut'], 3)}",
              f"Standard Error of Mean Difference: SE = "
              f"{_F(welch['sterr'], d)}", "",
              f"Hypothesis Test of 0 Mean Diff:  t = "
              f"{_F(welch['t'], 3)},  df = {_F(welch['df'], 3)}, "
              f"p-value = {_F(welch['p'], 3)}", "",
              f"Margin of Error for {clpct} Confidence Level:  "
              f"{_F(welch['E'], d)}",
              f"{clpct} Confidence Interval for Mean Difference:  "
              f"{_F(welch['lb'], d)} to {_F(welch['ub'], d)}"]

        L += ["", "------ Effect Size ------", "",
              "--- Assume equal population variances of "
              f"{Ynm} for each {Xnm}", "",
              f"Standardized Mean Difference of {Ynm}, "
              f"Cohen's d:  {_F(smd, d)}"]
        L += _practical(mmd, msmd, sw, m1 - m2, smd, lb, ub, d)
        L += _needed_2(Edesired, conf_level, sw, E, n1, n2, d)

    print("\n".join(L))
    plots = {}
    if graph and from_data:
        plots["two_group"] = _two_group_plot(
            YA, YB, Ynm, X1nm, X2nm, m1, m2, d)
    return ttestResults(
        kind="two-group", n_total=n1 + n2, digits_d=d,
        group1={"name": X1nm, "n": n1, "mean": m1, "sd": s1},
        group2={"name": X2nm, "n": n2, "mean": m2, "sd": s2},
        mean_diff=m1 - m2, pooled_sd=sw, cohen_d=smd,
        equal_var={"t": tvalue, "df": df, "p_value": pvalue,
                   "se": sterr, "lb": lb, "ub": ub},
        welch=welch, plots=plots)


def _assumptions(YA, YB, X1nm, X2nm, Ynm, n1, n2, v1, v2, df1,
                 df2, from_data, d):
    from scipy.stats import f as fdist, shapiro
    L = ["", "", "------ Assumptions ------", "",
         "Note: These hypothesis tests can perform poorly, and "
         "the",
         "      t-test is typically robust to violations of "
         "assumptions.",
         "      Use as heuristic guides instead of interpreting "
         "literally.", ""]
    if from_data:
        L.append("Null hypothesis, for each group, is a normal "
                 f"distribution of {Ynm}.")
        for nm, Y, ni in ((X1nm, YA, n1), (X2nm, YB, n2)):
            if ni > 30:
                L.append(f"Group {nm}: Sample mean assumed "
                         "normal because n > 30, so no test "
                         "needed.")
            elif 2 < ni < 5000:
                W, p = shapiro(Y)
                L.append(f"Group {nm}  Shapiro-Wilk normality "
                         f"test:  W = {_F(W, 3)},  p-value = "
                         f"{_F(p, 3)}")
            else:
                L.append(f"Group {nm}  Sample size out of range "
                         "for Shapiro-Wilk normality test.")
        L.append("")
    # variance ratio F test
    if v1 >= v2:
        vr, dfn, dfd = v1 / v2, df1, df2
        vrs = f"{_F(v1, d)}/{_F(v2, d)}"
    else:
        vr, dfn, dfd = v2 / v1, df2, df1
        vrs = f"{_F(v2, d)}/{_F(v1, d)}"
    pv = float(fdist.cdf(vr, dfn, dfd))
    pv = 2 * min(pv, 1 - pv)
    L.append(f"Null hypothesis is equal variances of {Ynm}, "
             "homogeneous.")
    L.append(f"Variance Ratio test:  F = {vrs} = {_F(vr, d)}, "
             f" df = {dfn};{dfd},  p-value = {_F(pv, 3)}")
    if from_data:
        # Levene, Brown-Forsythe: pooled t on |Y - median|
        a = np.abs(YA - np.median(YA))
        b = np.abs(YB - np.median(YB))
        t_bf, df_bf, p_bf = _pooled_t(a, b)
        L.append(f"Levene's test, Brown-Forsythe:  t = "
                 f"{_F(t_bf, 3)},  df = {df_bf},  p-value = "
                 f"{_F(p_bf, 3)}")
    return L


def _pooled_t(a, b):
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    dfa, dfb = na - 1, nb - 1
    sw = np.sqrt((dfa * va + dfb * vb) / (dfa + dfb))
    se = sw * np.sqrt(1 / na + 1 / nb)
    t = (a.mean() - b.mean()) / se
    df = dfa + dfb
    return t, df, _pval(t, df, "two_sided")


def _welch(m1, m2, v1, v2, n1, n2, conf_level, alt, YA, YB):
    k1, k2 = v1 / n1, v2 / n2
    df = (k1 + k2) ** 2 / (k1 ** 2 / (n1 - 1) + k2 ** 2 / (n2 - 1))
    sterr = np.sqrt(k1 + k2)
    tcut = _tcrit(conf_level, df, alt)
    t = (m1 - m2) / sterr
    p = _pval(t, df, alt)
    E = tcut * sterr
    return {"t": t, "df": df, "p": p, "se": sterr, "tcut": tcut,
            "E": E, "lb": (m1 - m2) - E, "ub": (m1 - m2) + E,
            "sterr": sterr}


def _practical(mmd, msmd, sw, mdiff, smd, lb, ub, d):
    L = ["", "", "------ Practical Importance ------", "",
         "Minimum Mean Difference of practical importance: mmd"]
    if mmd is not None or msmd is not None:
        if mmd is not None:
            msmd = mmd / sw
        else:
            mmd = msmd * sw
        L += [f"Compare mmd = {_F(mmd, d)} to the obtained value "
              f"of md = {_F(mdiff, d)}",
              f"Compare mmd to the confidence interval for md: "
              f"{_F(lb, d)} to {_F(ub, d)}", "",
              "Minimum Standardized Mean Difference of practical "
              "importance: msmd",
              f"Compare msmd = {_F(msmd, d)} to the obtained "
              f"value of smd = {_F(smd, d)}"]
    else:
        L += ["Minimum Standardized Mean Difference of "
              "practical importance: msmd",
              "Neither value specified, so no analysis"]
    return L


def _needed_2(Edesired, conf_level, sw, E, n1, n2, d):
    if Edesired is None:
        return []
    from scipy.stats import norm
    zcut = norm.ppf((1 - conf_level) / 2)
    ns = 2 * ((zcut * sw) / Edesired) ** 2
    needed = int(np.ceil(1.099 * ns + 4.863))
    L = ["", "", "------ Needed Sample Size ------", ""]
    if Edesired > E:
        L.append(f"Note: Desired margin of error, {_F(Edesired, d)}"
                 f" is worse than what was obtained, {_F(E, d)}")
        L.append("")
    L += [f"Desired Margin of Error: {_F(Edesired, d)}", "",
          "For the following sample size there is a 0.9 "
          "probability of obtaining",
          f"the desired margin of error for the resulting "
          "confidence interval.",
          f"Needed sample size per group: {needed}", "",
          f"Additional data values needed Group 1: {needed - n1}",
          f"Additional data values needed Group 2: {needed - n2}"]
    return L


# ------------------------------------------------------------
# one group (and paired, on the differences)
# ------------------------------------------------------------

def _one_group(Y, n, m, s, mu, Ynm, brief, conf_level,
               alternative, digits_d, mmd, msmd, Edesired,
               paired, graph):
    from scipy.stats import shapiro
    from_data = Y is not None
    if from_data:
        Y0 = np.asarray(Y, dtype=float)
        n_miss = int(np.isnan(Y0).sum())
        Y = Y0[~np.isnan(Y0)]
        n, m, s = len(Y), Y.mean(), Y.std(ddof=1)
        d = digits_d if digits_d is not None else _getdigits(Y, 3)
    else:
        n_miss = 0
        d = digits_d if digits_d is not None else 3
    sd_d = d if from_data else d - 1
    clpct = f"{round(conf_level * 100, 2):g}%"
    alt = alternative

    L = ["", f"------ Describe ------" if not brief
         else " --- Describe ---", ""]
    lead = f"{Ynm}:  " if Ynm not in ("Y",) else ""
    miss = f"n.miss = {n_miss},  " if from_data else ""
    L.append(f"{lead}{miss}n = {n},   mean = {_F(m, sd_d)}, "
             f" sd = {_F(s, sd_d)}")

    if not brief and from_data:
        L += ["", "", "------ Normality Assumption ------", ""]
        if n > 30:
            L.append("Sample mean assumed normal because n > 30, "
                     "so no test needed.")
        elif 2 < n < 5000:
            W, p = shapiro(Y)
            L += [f"Null hypothesis is a normal distribution of "
                  f"{Ynm}.",
                  f"Shapiro-Wilk normality test:  W = {_F(W, 3)}, "
                  f" p-value = {_F(p, 3)}"]
        else:
            L.append("Sample size out of range for Shapiro-Wilk "
                     "normality test.")

    L += ["", "", "------ Infer ------" if not brief
          else " --- Infer ---", ""]
    df = n - 1
    sterr = s * np.sqrt(1 / n)
    tcut = _tcrit(conf_level, df, alt)
    E = tcut * sterr
    lb, ub = m - E, m + E
    tvalue = pvalue = None
    if mu is not None:
        tvalue = (m - mu) / sterr
        pvalue = _pval(tvalue, df, alt)
    L += [f"t-cutoff for {clpct} range of variation: "
          f"tcut = {_F(tcut, 3)}",
          f"Standard Error of Mean: SE = {_F(sterr, d)}", ""]
    if mu is not None:
        if alt != "two_sided":
            L.append("Alternative hypothesis: Population mean is "
                     f"{_alt_word(alt)} than {mu}")
        L += [f"Hypothesized Value H0: mu = {mu}",
              f"Hypothesis Test of Mean:  t-value = "
              f"{_F(tvalue, 3)},  df = {df},  p-value = "
              f"{_F(pvalue, 3)}", ""]
    L += [f"Margin of Error for {clpct} Confidence Level:  "
          f"{_F(E, d)}",
          f"{clpct} Confidence Interval for Mean:  {_F(lb, d)} "
          f"to {_F(ub, d)}"]

    cohen = None
    if mu is not None:
        mdiff = m - mu
        cohen = abs(mdiff / s)
        L += ["", "", "------ Effect Size ------", "",
              f"Distance of sample mean from hypothesized:  "
              f"{_F(mdiff, d)}",
              f"Standardized Distance, Cohen's d:  "
              f"{_F(cohen, d)}"]

    if Edesired is not None and from_data:
        L += _needed_1(Edesired, conf_level, s, E, n, d)

    print("\n".join(L))
    plots = {}
    if graph and from_data:
        plots["one_group"] = _one_group_plot(
            Y, Ynm, m, lb, ub, d, paired)
    return ttestResults(
        kind="paired" if paired else "one-group", n_total=n,
        digits_d=d, n=n, mean=m, sd=s, mu=mu,
        infer={"t": tvalue, "df": df, "p_value": pvalue,
               "se": sterr, "lb": lb, "ub": ub},
        cohen_d=cohen, plots=plots)


def _needed_1(Edesired, conf_level, s, E, n, d):
    from scipy.stats import norm
    zcut = norm.ppf((1 - conf_level) / 2)
    ns = ((zcut * s) / Edesired) ** 2
    needed = int(np.ceil(1.132 * ns + 7.368))
    L = ["", "", "------ Needed Sample Size ------", ""]
    if Edesired > E:
        L += [f"Note: Desired margin of error, {_F(Edesired, d)}"
              f" is worse than what was obtained, {_F(E, d)}", ""]
    L += [f"Desired Margin of Error: {_F(Edesired, d)}", "",
          "For the following sample size there is a 0.9 "
          "probability of obtaining",
          "the desired margin of error for the confidence "
          "interval.",
          f"Needed sample size: {needed}", "",
          f"Additional data values needed: {needed - n}"]
    return L


# ------------------------------------------------------------
# density plots
# ------------------------------------------------------------

def _kde(v):
    from scipy.stats import gaussian_kde
    k = gaussian_kde(v)
    lo, hi = v.min(), v.max()
    pad = 0.15 * (hi - lo if hi > lo else 1)
    xs = np.linspace(lo - pad, hi + pad, 200)
    return xs, k(xs)


def _two_group_plot(YA, YB, Ynm, X1nm, X2nm, m1, m2, d):
    import plotly.graph_objects as go
    from .plotly_utils import (
        BASE_COLORS, axis_format, axis_num, make_trans,
        plot_border, plotly_style, to_hex, x_grid)
    from .utils import pretty
    style = plotly_style()
    fig = go.Figure()
    allx, ally = [], []
    for (nm, v, col) in ((X1nm, YA, BASE_COLORS[1]),
                         (X2nm, YB, BASE_COLORS[0])):
        xs, ys = _kde(v)
        allx += [xs.min(), xs.max()]
        ally.append(ys.max())
        c = to_hex(col)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=str(nm),
            line=dict(color=c, width=2), fill="tozeroy",
            fillcolor=make_trans(col, 0.8), hoverinfo="x+name"))
    axT1 = pretty(min(allx), max(allx))
    ax_x = axis_num(Ynm, axT1, axis_format(axT1, d))
    ax_y = axis_num("Density", pretty(0, max(ally)),
                    axis_format(pretty(0, max(ally)), d))
    fig.update_layout(
        xaxis=ax_x, yaxis=ax_y,
        shapes=x_grid(axT1) + plot_border(), template=None,
        plot_bgcolor=to_hex(style["panel_fill"]),
        paper_bgcolor=to_hex(style["window_fill"]),
        legend=dict(title=dict(text=str(X1nm) + " / " + str(X2nm))),
        title=dict(text="Two-Group Density Plot", x=0.5,
                   xanchor="center",
                   font=dict(size=round(
                       16 * get_option("main_size", 1)))))
    return fig


def _one_group_plot(Y, Ynm, m, lb, ub, d, paired):
    import plotly.graph_objects as go
    from .plotly_utils import (
        BASE_COLORS, axis_format, axis_num, make_trans,
        plot_border, plotly_style, to_hex, x_grid)
    from .utils import pretty
    style = plotly_style()
    xs, ys = _kde(Y)
    c = BASE_COLORS[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color=to_hex(c),
                                            width=2),
        fill="tozeroy", fillcolor=make_trans(c, 0.85),
        hoverinfo="x", showlegend=False))
    ym = float(np.interp(m, xs, ys))
    fig.add_shape(type="line", x0=m, x1=m, y0=0, y1=ym,
                  line=dict(color=to_hex("gray50"), width=1))
    axT1 = pretty(float(xs.min()), float(xs.max()))
    xlab = ("Differences of Matched Pairs" if paired else Ynm)
    ax_x = axis_num(xlab, axT1, axis_format(axT1, d))
    ax_y = axis_num("Density", pretty(0, float(ys.max())),
                    axis_format(pretty(0, float(ys.max())), d))
    fig.update_layout(
        xaxis=ax_x, yaxis=ax_y,
        shapes=x_grid(axT1) + plot_border(), template=None,
        plot_bgcolor=to_hex(style["panel_fill"]),
        paper_bgcolor=to_hex(style["window_fill"]),
        title=dict(
            text="One-Group Density Plot", x=0.5,
            xanchor="center",
            font=dict(size=round(16 * get_option("main_size",
                                                 1)))))
    return fig
