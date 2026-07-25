# Prop_test.py — analog of Prop_test.R.
#
# Prop_test(): four proportion analyses, chosen from the
# arguments —
#   one proportion   variable + success (data) | n_succ + n_tot
#                     exact binomial test vs pi (default 0.5)
#   many proportions  variable + success + by | n_succ,n_tot lists
#                     chi-square test of equal proportions
#   goodness-of-fit  variable (data) | n_tot vector
#                     chi-square vs equal proportions
#   independence     variable + by (data) | n_table
#                     chi-square cross-tab test, with Cramer's V
# Numerics through scipy.stats (binomtest, chi2_contingency,
# chisquare). Prints as R; returns a PropResults object.

import numpy as np
import pandas as pd

from .utils import fmt, get_column


class PropResults:
    """Results of Prop_test(): the analysis kind and its
    statistics (proportion/p-value/CI, or chi-square/df/p and
    the tables)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return f"<lessPy Prop_test: {self.kind}>"


def Prop_test(variable=None, success=None, by=None, data=None,
              n_succ=None, n_fail=None, n_tot=None, n_table=None,
              Yates=False, pi=None, digits_d=3,
              alternative="two_sided"):
    """Test of one or more proportions, goodness-of-fit, or
    cross-tab independence, from data columns or from summary
    counts. R analog: Prop_test()"""
    if alternative not in ("two_sided", "less", "greater"):
        raise ValueError('alternative: "two_sided", "less", '
                         '"greater"')
    do_data = (n_succ is None and n_tot is None
               and n_table is None)
    d = digits_d

    if do_data:
        if variable is None:
            raise ValueError("variable= is required with data=")
        v = get_column(data, variable, "variable")
        g = get_column(data, by, "by") if by is not None else None
        if success is not None and g is None:
            return _one_prop_data(v, success, variable, pi, d,
                                  alternative)
        if success is not None and g is not None:
            return _many_prop_data(v, g, success, variable, by,
                                   Yates, d)
        if success is None and g is None:
            return _gof_data(v, variable, d)
        return _crosstab_data(v, g, variable, by, Yates, d)

    # summary counts
    if n_fail is not None and n_tot is None:
        n_tot = (np.add(n_succ, n_fail) if np.ndim(n_succ)
                 else n_succ + n_fail)
    if n_table is not None:
        return _crosstab_table(n_table, Yates, d)
    if n_succ is None:
        return _gof_counts(np.asarray(n_tot, dtype=float), d)
    if np.ndim(n_succ) == 0:
        return _one_prop(int(n_succ), int(n_tot), pi, d,
                         alternative)
    return _many_prop(np.asarray(n_succ), np.asarray(n_tot), d)


# ------------------------------------------------------------
# one proportion (exact binomial)
# ------------------------------------------------------------

def _one_prop_data(v, success, name, pi, d, alt):
    s = v.astype(str)
    n_na = int(v.isna().sum())
    n_tot = int(s.notna().sum() if False else (~v.isna()).sum())
    n_succ = int((s == str(success)).sum())
    return _one_prop(n_succ, n_tot, pi, d, alt, name, success,
                     n_na)


def _one_prop(n_succ, n_tot, pi, d, alt, name=None, success=None,
              n_na=None):
    from scipy.stats import binomtest
    if pi is None:
        pi = 0.5
    sci_alt = {"two_sided": "two-sided", "less": "less",
               "greater": "greater"}[alt]
    r = binomtest(n_succ, n_tot, pi, alternative=sci_alt)
    lo, hi = r.proportion_ci(0.95, method="exact")
    prop = n_succ / n_tot

    L = ["", "<<< Exact binomial test of a proportion", ""]
    if name is not None:
        L += [f"variable: {name}", f"success: {success}", ""]
    L += ["------ Describe ------", ""]
    if n_na is not None:
        L.append(f"Number of missing values: {n_na}")
    L += [f"Number of successes: {n_succ}",
          f"Number of failures: {n_tot - n_succ}",
          f"Number of trials: {n_tot}",
          f"Sample proportion: {fmt(prop, d)}", "",
          "------ Infer ------", ""]
    if alt != "two_sided":
        L.append(f"Alternative hypothesis: Population proportion "
                 f"is {sci_alt} than {pi}")
    L += [f"Hypothesis test for null of {pi}, p-value: "
          f"{fmt(r.pvalue, d)}",
          f"95% Confidence interval: {fmt(lo, d)} to "
          f"{fmt(hi, d)}"]
    print("\n".join(L))
    return PropResults(kind="one-proportion", proportion=prop,
                       n_succ=n_succ, n_tot=n_tot,
                       p_value=float(r.pvalue),
                       conf_int=(float(lo), float(hi)), pi=pi)


# ------------------------------------------------------------
# many proportions (chi-square of equal proportions)
# ------------------------------------------------------------

def _many_prop_data(v, g, success, name, by, Yates, d):
    outcome = np.where(v.astype(str) == str(success),
                       str(success), "fail")
    tab = pd.crosstab(g.astype(str), outcome)
    cols = [str(success), "fail"]
    tab = tab.reindex(columns=[c for c in cols if c in
                               tab.columns], fill_value=0)
    return _many_prop_table(tab.to_numpy(), list(tab.index), d,
                            name, by, success)


def _many_prop(n_succ, n_tot, d):
    tab = np.column_stack([n_succ, n_tot - n_succ])
    return _many_prop_table(tab, [str(i + 1) for i in
                                  range(len(n_succ))], d)


def _many_prop_table(tab, groups, d, name=None, by=None,
                     success=None):
    from scipy.stats import chi2_contingency
    chi2, p, dof, _ = chi2_contingency(tab, correction=False)
    props = tab[:, 0] / tab.sum(axis=1)
    L = ["", "<<< Chi-square test of equal proportions", ""]
    if name is not None:
        L += [f"variable: {name}", f"success: {success}",
              f"by: {by}", ""]
    L += ["--- Description", ""]
    desc = pd.DataFrame(
        {g: [int(tab[i, 0]), int(tab[i].sum()),
             round(props[i], d)]
         for i, g in enumerate(groups)},
        index=[f"n_{success}" if success else "n_success",
               "n_total", "proportion"])
    L.append(desc.to_string())
    L += ["", "--- Inference", "",
          f"Chi-square statistic: {fmt(chi2, d)}",
          f"Degrees of freedom: {dof}",
          "Hypothesis test of equal population proportions: "
          f"p-value = {fmt(p, d)}"]
    print("\n".join(L))
    return PropResults(kind="many-proportions",
                       proportions=props, chi2=float(chi2),
                       df=int(dof), p_value=float(p))


# ------------------------------------------------------------
# goodness-of-fit
# ------------------------------------------------------------

def _gof_data(v, name, d):
    tab = v.astype(str).value_counts().sort_index()
    return _gof(tab.to_numpy(dtype=float), list(tab.index), d,
               name)


def _gof_counts(n_tot, d):
    return _gof(n_tot, [str(i + 1) for i in range(len(n_tot))], d)


def _gof(obs, names, d, name=None):
    from scipy.stats import chisquare
    n = obs.sum()
    exp = np.full(len(obs), n / len(obs))
    chi2, p = chisquare(obs, exp)
    dof = len(obs) - 1
    resid = (obs - exp) / np.sqrt(exp)
    stdres = (obs - exp) / np.sqrt(exp * (1 - exp / n))

    L = ["", "<<< Chi-squared test for given probabilities", ""]
    if name is not None:
        L.append(f"variable: {name}")
    L += ["", "--- Description", ""]
    tbl = pd.DataFrame(
        {nm: [int(obs[i]), round(exp[i], d), round(resid[i], d),
              round(stdres[i], d)] for i, nm in enumerate(names)},
        index=["observed", "expected", "residual", "stdn res"])
    L += [tbl.to_string(), "", "--- Inference", "",
          f"Chi-square statistic: {fmt(chi2, d)}",
          f"Degrees of freedom: {dof}",
          "Hypothesis test of equal population proportions: "
          f"p-value = {fmt(p, d)}"]
    print("\n".join(L))
    return PropResults(kind="goodness-of-fit", observed=obs,
                       expected=exp, residuals=resid,
                       stdres=stdres, chi2=float(chi2), df=dof,
                       p_value=float(p))


# ------------------------------------------------------------
# cross-tab independence
# ------------------------------------------------------------

def _crosstab_data(v, g, name, by, Yates, d):
    tab = pd.crosstab(g.astype(str), v.astype(str))
    return _crosstab(tab, Yates, d, name, by)


def _crosstab_table(n_table, Yates, d):
    if isinstance(n_table, (pd.DataFrame, np.ndarray)):
        tab = pd.DataFrame(n_table)
    else:                                  # a file path
        tab = pd.read_csv(n_table, header=None)
    return _crosstab(tab, Yates, d, None, None)


def _crosstab(tab, Yates, d, name, by):
    from scipy.stats import chi2_contingency
    O = tab.to_numpy(dtype=float)
    chi2, p, dof, E = chi2_contingency(O, correction=Yates)
    n = O.sum()
    rs = O.sum(axis=1, keepdims=True)
    cs = O.sum(axis=0, keepdims=True)
    stdres = (O - E) / np.sqrt(E * (1 - rs / n) * (1 - cs / n))
    V = np.sqrt(chi2 / (min(O.shape[0] - 1, O.shape[1] - 1) * n))

    L = ["", "<<< Pearson's Chi-squared test", ""]
    if name is not None:
        L += [f"variable: {name}", f"by: {by}"]
    L += ["", "--- Description", "", tab.to_string(),
          "",
          f"Cramer's V{' (phi)' if dof == 1 else ''}: "
          f"{fmt(V, 3)}", ""]
    cells = pd.DataFrame(
        [[i + 1, j + 1, int(O[i, j]), round(E[i, j], d),
          round(O[i, j] - E[i, j], d), round(stdres[i, j], d)]
         for i in range(O.shape[0]) for j in range(O.shape[1])],
        columns=["Row", "Col", "Observed", "Expected",
                 "Residual", "Stnd Res"])
    L += [cells.to_string(index=False), "", "--- Inference", "",
          f"Chi-square statistic: {fmt(chi2, d)}",
          f"Degrees of freedom: {dof}",
          f"Hypothesis test of independence: p-value = "
          f"{fmt(p, d)}"]
    print("\n".join(L))
    return PropResults(kind="independence", observed=O,
                       expected=E, stdres=stdres,
                       cramers_v=float(V), chi2=float(chi2),
                       df=int(dof), p_value=float(p))
