# Regression.py — analog of Regression.R (core numeric OLS)
#
# Regression(): least-squares regression with the lessR output
# pipeline — background, estimated model with confidence
# intervals, model fit (with the PRESS R-squared), sequential
# ANOVA with the aggregate Model row, collinearity, the
# residuals-and-influence listing, and prediction intervals —
# plus the plotly graphics: the simple-regression scatterplot
# with confidence and prediction bands (~ .reg5Plot), the
# distribution of residuals (~ .reg3dnResidual), and residuals
# vs fitted values with Cook's-distance flagging
# (~ .reg3resfitResidual). As with the views, the pipeline is
# ported, not the lines.
#
# The model is a formula string, "Y ~ X1 + X2" ("Y ~ ." takes
# every other numeric column), the Python analog of the R
# formula. Expression terms such as log(Years) or I(Years^2)
# are materialized as data columns (~ .formula_expr); top-level
# interaction/crossing operators (: * ^) are not ported.
# Categorical predictors become treatment-coded
# indicator variables (VarLevel columns, first level the
# reference, ~ model.matrix); with exactly one covariate and
# one factor, the ANOVA reports term-level Type II sums of
# squares — each term adjusted for the other, matching R's
# .reg1ancova as corrected July 2026 (its earlier row
# replacement mislabeled the factor's SS). The
# the full Regression.R analysis is ported. Rmd= generates a
# Quarto (.qmd) report rather than R Markdown (see reg_rmd.py).
# quiet= does not exist, as in R: Regression() always prints.
#
# Numerics through statsmodels OLS (imported lazily, as
# plt_forecast does); influence measures from OLSInfluence.
# Returns a RegressionResults object holding the tables as
# DataFrames and the plotly figures in .plots — figures are
# not auto-shown (no R graphics device to open).

import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as sps

from .dn_plotly import dn_plotly
from .plt_mat_plotly import scatter_matrix
from .reg_rmd import reg_rmd
from .plotly_utils import (
    BASE_COLORS, as_plotly_color, axis_format, axis_num,
    make_trans, plot_border, plotly_style, to_hex, x_grid)
from .utils import (
    category_order, fmt, get_column, get_option, pretty)
from .X import _breaks_from_args


def _dash(n):
    return "-" * n


def _getdigits(values, min_digits=3):
    """Decimal digits for output: one more than the largest
    number of decimal places in the response, at least
    min_digits. R analog: .getdigits()"""
    dmax = 0
    for v in np.asarray(values, dtype=float)[:500]:
        s = f"{v:.10f}".rstrip("0")
        if "." in s:
            dmax = max(dmax, len(s.split(".")[1]))
        if dmax >= 8:
            break
    return max(min_digits, min(dmax + 1, 8))


def _rescalable(s):
    """A variable is rescaled only if numeric with more than two
    distinct values, so binary and indicator columns pass through
    unchanged. ~ Regression.R  unq.x > 2 && is.numeric"""
    return (pd.api.types.is_numeric_dtype(s)
            and s.dropna().nunique() > 2)


def _rescale(v, kind, digits_d):
    """Rescale a numeric vector, missing values ignored, rounded
    to digits_d. R analog: rescale()
      z       (x - mean) / sd        (sd with n-1)
      center   x - mean
      0to1    (x - min) / (max - min)
      robust  (x - median) / IQR"""
    if kind == "z":
        out = (v - np.nanmean(v)) / np.nanstd(v, ddof=1)
    elif kind == "center":
        out = v - np.nanmean(v)
    elif kind == "0to1":
        lo, hi = np.nanmin(v), np.nanmax(v)
        out = (v - lo) / (hi - lo)
    else:                              # robust
        q1, q3 = np.nanpercentile(v, [25, 75])
        out = (v - np.nanmedian(v)) / (q3 - q1)
    return np.round(out, digits_d)


def _split_top(s, sep):
    """Split s on sep at parenthesis depth 0, so a separator
    inside a function call is not a break."""
    out, depth, start = [], 0, 0
    for i, ch in enumerate(s):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == sep and depth == 0:
            out.append(s[start:i])
            start = i + 1
    out.append(s[start:])
    return [t.strip() for t in out]


def _has_top(s, chars):
    """True if any char in chars appears at parenthesis depth 0."""
    depth = 0
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch in chars and depth == 0:
            return True
    return False


_BARE = re.compile(r"^[A-Za-z_.][\w.]*$")
_EXPR_FUN = {"log": np.log, "log10": np.log10, "log2": np.log2,
             "sqrt": np.sqrt, "exp": np.exp, "abs": np.abs,
             "sin": np.sin, "cos": np.cos, "tan": np.tan}


def _materialize(term, data):
    """Evaluate a formula expression term (log(Years), sqrt(x),
    I(Years^2), x/100 ...) into a numeric column named by the
    term's text, as R's .formula_expr does with terms()
    variables that are calls. Returns the (name, data) with the
    new column added to a copy of data."""
    py = re.sub(r"\bI\(", "(", term.replace("^", "**"))
    ns = {c: data[c].to_numpy() for c in data.columns}
    ns.update(_EXPR_FUN)
    try:
        val = eval(py, {"__builtins__": {}}, ns)  # user's own model
    except Exception as e:
        raise ValueError(
            f'cannot evaluate the formula term "{term}": {e}. '
            "Name a data column, or compute the transformed "
            "column first.")
    data = data.copy()
    data[term] = np.asarray(val, dtype=float)
    return term, data


def _resolve_term(term, data):
    """A bare column name stays; an expression is materialized
    (~ .formula_expr). A top-level interaction/crossing operator
    (: * ^) is not ported."""
    if _BARE.match(term) and term in data.columns:
        return term, data
    if _has_top(term, ":*^"):
        raise NotImplementedError(
            f'formula operator in "{term}" is not ported: list '
            "predictors with +, and compute any interaction "
            "column first (I(a*b) materializes a product term)")
    return _materialize(term, data)


def _parse_formula(my_formula, data):
    """ "Y ~ X1 + X2" -> (response, [predictors], data). "Y ~ ."
    takes every other numeric column; "Y ~ 1" is the null model.
    Expression terms such as log(Years) or I(Years^2) are
    materialized as data columns (~ .formula_expr); the returned
    data carries those columns."""
    if not isinstance(my_formula, str):
        raise TypeError(
            "the model is a formula string, such as "
            '"Salary ~ Years + Pre"')
    if my_formula.count("~") != 1:
        raise ValueError(
            'the formula has one "~":  "Y ~ X1 + X2"')
    lhs, rhs = (s.strip() for s in my_formula.split("~"))
    if not lhs:
        raise ValueError("the formula names the response "
                         'before the "~"')
    y_name, data = _resolve_term(lhs, data)
    if rhs == ".":
        preds = [c for c in data.columns
                 if c != y_name
                 and pd.api.types.is_numeric_dtype(data[c])]
    elif rhs in ("1", ""):
        preds = []
    else:
        terms = _split_top(rhs, "+")
        if any(not t for t in terms):
            raise ValueError(f'cannot parse "{rhs}": list the '
                             "predictors separated by +")
        preds = []
        for t in terms:
            nm, data = _resolve_term(t, data)
            preds.append(nm)
    return y_name, preds, data


def _expand_indicators(pred_names, pred_sers, note_fmt):
    """Treatment-coded indicator variables for the categorical
    predictors, in formula position: the first level (declared
    Categorical order, else sorted) is the reference, and each
    other level becomes a 0/1 column named VarLevel, the naming
    of R's model.matrix(). Returns (names, series, notes,
    cat_names). R analog: the "construct the indicator
    variables" block of Regression.R / Logit.R"""
    from .utils import category_order
    names, sers, notes, cat_names = [], [], [], []
    term_map = {}          # original predictor -> its columns
    for nm, s in zip(pred_names, pred_sers):
        if pd.api.types.is_numeric_dtype(s):
            names.append(nm)
            sers.append(s)
            term_map[nm] = [nm]
            continue
        cat_names.append(nm)
        notes.append(note_fmt.format(nm))
        term_map[nm] = []
        for lv in category_order(
                s if isinstance(s.dtype, pd.CategoricalDtype)
                else s.astype(str))[1:]:
            dnm = f"{nm}{lv}"
            names.append(dnm)
            term_map[nm].append(dnm)
            sers.append((s.astype(str) == str(lv))
                        .astype(float).rename(dnm))
    return names, sers, notes, cat_names, term_map


def _best_subsets(Xd, yv, tot_ss, MSW, best_sub, nbest=10):
    """Best subset regressions over all predictor subsets,
    keeping the nbest best of each size (the leaps() default),
    scored by adjusted R-squared or Mallows' Cp on the full
    model's error variance. Engine deviation: an exhaustive
    all-subsets search replaces the leaps branch-and-bound —
    same results, no size limit beyond practicality.
    R analog: leaps::leaps() in .reg2Relations"""
    from itertools import combinations
    names = list(Xd.columns)
    p = len(names)
    if p > 15:
        return None                    # 2^15 solves is the cap
    n = len(yv)
    Xc = Xd.to_numpy(dtype=float)
    rows = []
    for k in range(1, p + 1):
        size_rows = []
        for cols in combinations(range(p), k):
            Xs = np.column_stack(
                [np.ones(n), Xc[:, list(cols)]])
            rss = float(((yv - Xs @ np.linalg.lstsq(
                Xs, yv, rcond=None)[0]) ** 2).sum())
            if best_sub == "adjr2":
                crit = 1 - (rss / (n - k - 1)) \
                    / (tot_ss / (n - 1))
            else:                      # Mallows' Cp
                crit = rss / MSW - (n - 2 * (k + 1))
            size_rows.append((cols, crit))
        size_rows.sort(key=lambda r: r[1],
                       reverse=best_sub == "adjr2")
        rows += size_rows[:nbest]
    lbl = "R2adj" if best_sub == "adjr2" else "Cp"
    out = pd.DataFrame(
        [[int(j in cols) for j in range(p)] + [crit, len(cols)]
         for cols, crit in rows],
        columns=names + [lbl, "X's"])
    return out.sort_values(
        lbl, ascending=best_sub != "adjr2",
        kind="stable").reset_index(drop=True)


def _prntbl(df, digits_d, int_cols=()):
    """Aligned text table with row labels, floats at digits_d.
    Columns named in int_cols print as integers. R analog:
    .prntbl()"""
    show = pd.DataFrame(index=df.index.astype(str))
    for c in df.columns:
        col = df[c]
        if c in int_cols:
            show[c] = ["" if pd.isna(v) else str(int(round(v)))
                       for v in col]
        elif pd.api.types.is_float_dtype(col):
            show[c] = [fmt(v, digits_d) if pd.notna(v) else ""
                       for v in col]
        else:
            show[c] = col.astype(str)
    return show.to_string()


def _int_vars(df, names):
    """Of the given data columns, those whose values are all whole
    numbers -- shown without decimals, as the integer variables
    they are (an integer column with a missing value elsewhere is
    stored as float, so this is decided by value, not dtype)."""
    out = []
    for nm in names:
        if nm not in df.columns:
            continue
        col = df[nm]
        if not pd.api.types.is_numeric_dtype(col):
            continue
        v = col.to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) and np.all(v == np.floor(v)):
            out.append(nm)
    return out


class RegressionResults:
    """Numeric results and figures of Regression(): estimates,
    fit, anova (DataFrames/dict), residuals and predictions
    listings, and the plotly figures in .plots."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return (f"<lessPy Regression: {self.formula}, "
                f"n={self.n_keep}>")


def Regression(my_formula, data=None, filter=None, digits_d=None,
               brief=False,
               n_res_rows=None, res_sort="cooks",
               n_pred_rows=None, pred_sort="predint",
               subsets=None, best_sub="adjr2",
               cooks_cut=1,
               X1_new=None, X2_new=None, X3_new=None,
               X4_new=None, X5_new=None, X6_new=None,
               kfold=0, seed=None,
               new_scale="none", scale_response=False,
               mod=None, mod_transf="center",
               Rmd=None, Rmd_data=None, Rmd_format="html",
               Rmd_browser=True,
               results=True, explain=True, interpret=True,
               code=True,
               graphics=True):
    """Least-squares regression of a formula string,
    "Y ~ X1 + X2", with the lessR analysis pipeline: estimates,
    fit, ANOVA, collinearity, residuals and influence,
    prediction intervals, and the regression graphics. Always
    prints, as in R; returns a RegressionResults object with
    the figures in .plots."""
    import statsmodels.api as smapi

    if data is None:
        raise ValueError(
            "data= is required: a pandas DataFrame containing "
            "the model's variables")
    if res_sort not in ("cooks", "rstudent", "dffits", "off"):
        raise ValueError(
            'res_sort: "cooks", "rstudent", "dffits", or "off"')
    if pred_sort not in ("predint", "off"):
        raise ValueError('pred_sort: "predint" or "off"')
    if best_sub not in ("adjr2", "Cp"):
        raise ValueError('best_sub: "adjr2" or "Cp"')
    if new_scale not in ("none", "z", "center", "0to1", "robust"):
        raise ValueError('new_scale: "none", "z", "center", '
                         '"0to1", or "robust"')
    if mod_transf not in ("center", "z", "none"):
        raise ValueError('mod_transf: "center", "z", or "none"')
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

    y_name, pred_names, data = _parse_formula(my_formula, data)
    formula = (f"{y_name} ~ "
               + (" + ".join(pred_names) if pred_names else "1"))
    n_pred = len(pred_names)

    y_ser = get_column(data, y_name, "response")
    if not pd.api.types.is_numeric_dtype(y_ser):
        raise TypeError(
            f"'{y_name}' is {y_ser.dtype}: the response of "
            "Regression() is numeric. For a two-level "
            "categorical response use Logit().")
    pred_sers = [get_column(data, nm, "predictor")
                 for nm in pred_names]

    # digits from the response, before any rescaling (~ R sets
    # digits_d here, ahead of the new_scale block)
    if digits_d is None:
        digits_d = _getdigits(
            y_ser.dropna().to_numpy(dtype=float))
    d = digits_d

    # new_scale: rescale the numeric model variables in place --
    # predictors always, the response only when scale_response;
    # binary and non-numeric variables pass through unchanged
    # (~ Regression.R rescale block, rescale.R). For kfold the
    # rescaling is per fold, done in _reg_kfold.
    transf = None
    rescale_lines = []
    if new_scale != "none":
        transf = {"z": "Standardized", "center": "Centered",
                  "0to1": "Min-Max (0 to 1)",
                  "robust": "Robust Version of Standardized"
                  }[new_scale]
        if kfold == 0:
            if scale_response and _rescalable(y_ser):
                y_ser = pd.Series(
                    _rescale(y_ser.to_numpy(dtype=float),
                             new_scale, d),
                    index=y_ser.index, name=y_name)
            pred_sers = [
                pd.Series(_rescale(s.to_numpy(dtype=float),
                                   new_scale, d),
                          index=s.index, name=s.name)
                if _rescalable(s) else s
                for s in pred_sers]
            head = pd.concat(
                [y_ser] + pred_sers, axis=1).head(6)
            rescale_lines = (
                ["Rescaled Data, First Six Rows", ""]
                + head.to_string().split("\n") + [""])

    # mod: moderation. Add the X*W interaction and refit
    # Y ~ X + W + X*W. The two predictors are first centered (or
    # standardized) per mod_transf, which curbs the collinearity
    # the product term would otherwise introduce. ~ Regression.R
    # mod block, .reg6mod
    mod_info = None
    if mod is not None:
        if n_pred != 2:
            raise ValueError(
                "mod moderation currently needs exactly 2 "
                "predictors")
        if mod not in pred_names:
            raise ValueError(
                f"mod variable '{mod}' must be one of the two "
                "predictors")
        if any(not pd.api.types.is_numeric_dtype(s)
               for s in pred_sers):
            raise TypeError(
                "both predictors of a moderation must be numeric")
        cc = pd.concat([y_ser] + pred_sers,
                       axis=1).notna().all(axis=1)
        if mod_transf != "none":
            is_z = mod_transf == "z"
            scaled = []
            for s in pred_sers:
                v = s.astype(float)
                mu = v[cc].mean()
                v = ((v - mu) / v[cc].std(ddof=1) if is_z
                     else v - mu)
                scaled.append(pd.Series(v, index=s.index,
                                        name=s.name))
            pred_sers = scaled
        w_name = mod
        x_name = next(p for p in pred_names if p != mod)
        xw_name = f"{w_name}.{x_name}"
        inter = pd.Series(
            pred_sers[0].to_numpy(dtype=float)
            * pred_sers[1].to_numpy(dtype=float),
            index=pred_sers[0].index, name=xw_name)
        pred_names = pred_names + [xw_name]
        pred_sers = pred_sers + [inter]
        mod_info = {"x": x_name, "w": w_name, "xw": xw_name}

    used = pd.concat([y_ser] + pred_sers, axis=1)
    keep = ~used.isna().any(axis=1)
    n_obs = len(data)
    n_keep = int(keep.sum())

    # categorical predictors become indicator variables; with
    # exactly one covariate and one factor, the ANOVA reports
    # Type II sums of squares (the ANCOVA table)
    (pred_names, pred_sers_x, ind_notes, cat_names,
     term_map) = _expand_indicators(
        pred_names, [s[keep] for s in pred_sers],
        ">>>  {0} is not numeric. "
        "Converted to indicator variables.")
    ancova = (n_pred == 2 and len(cat_names) == 1)
    ancova_terms = ([nm for nm in term_map
                     if nm not in cat_names] + cat_names
                    if ancova else None)
    ancova_info = None
    if ancova:
        # the covariate is the numeric term (maps to itself), the
        # factor is the one categorical predictor; keep its
        # original values, in level order, for the group plot
        cov_name = next(k for k, v in term_map.items()
                        if v == [k])
        fac_name = cat_names[0]
        fac_series = next(s for s in pred_sers
                          if s.name == fac_name)[keep]
        fac_series = (fac_series
                      if isinstance(fac_series.dtype,
                                    pd.CategoricalDtype)
                      else fac_series.astype(str))
        ancova_info = {
            "cov": cov_name, "fac": fac_name,
            "preds": list(term_map.keys()),
            "levels": [str(lv) for lv in
                       category_order(fac_series)],
            "fac_vals": fac_series.astype(str).to_numpy()}
    # collinearity and subsets follow the count of listed
    # predictors, before indicator expansion, as R
    n_pred_orig = n_pred
    n_pred = len(pred_names)

    if n_keep < n_pred + 2:
        raise ValueError(
            f"only {n_keep} complete rows: too few to estimate "
            f"{n_pred + 1} coefficients")
    yv = y_ser[keep].to_numpy(dtype=float)
    Xd = pd.DataFrame(
        {nm: s.to_numpy(dtype=float)
         for nm, s in zip(pred_names, pred_sers_x)},
        index=y_ser[keep].index)
    row_labels = Xd.index.astype(str)

    if kfold and kfold > 0:
        # cross-validation replaces the single-model analysis;
        # graphics and the residual/prediction listings are off,
        # exactly as R turns them off for kfold > 0
        if n_pred == 0:
            raise ValueError(
                "kfold cross-validation needs at least one "
                "predictor")
        return _reg_kfold(yv, Xd, y_name, pred_names, n_keep,
                          formula, kfold, seed, d, smapi,
                          new_scale, scale_response)

    X = smapi.add_constant(Xd, has_constant="add") \
        if n_pred > 0 else pd.DataFrame(
            {"const": np.ones(n_keep)}, index=Xd.index)
    fit = smapi.OLS(yv, X).fit()
    infl = fit.get_influence()
    hat = infl.hat_matrix_diag
    rstudent = infl.resid_studentized_external
    dffits = infl.dffits[0]
    cooks = infl.cooks_distance[0]
    resid = np.asarray(fit.resid, dtype=float)
    fitted = np.asarray(fit.fittedvalues, dtype=float)
    df_res = int(fit.df_resid)

    lines = list(rescale_lines)        # console output
    for note in ind_notes:
        lines += [note, ""]

    # ----------  BACKGROUND  -------------------------------------
    lines += ["", "  BACKGROUND", ""]
    for i, nm in enumerate([y_name] + pred_names):
        if i == 0:
            lbl = "Response Variable: "
        elif n_pred > 1:
            lbl = f"Predictor Variable {i}: "
        else:
            lbl = "Predictor Variable: "
        lines.append(lbl + nm)
    if transf is not None:
        lines += ["", f"Data are {transf}"]
    lines += ["",
              f"Number of cases (rows) of data:  {n_obs}",
              f"Number of cases retained for analysis:  "
              f"{n_keep}"]

    # ----------  BASIC ANALYSIS  ---------------------------------
    lines += ["", "", "  BASIC ANALYSIS", ""]

    # estimates with 95% confidence intervals, ~ .reg1modelBasic
    ci = fit.conf_int(alpha=0.05)
    est = pd.DataFrame({
        "Estimate": fit.params,
        "Std Err": fit.bse,
        "t-value": fit.tvalues,
        "p-value": fit.pvalues,
        "Lower 95%": ci[0],
        "Upper 95%": ci[1],
    })
    est.index = ["(Intercept)"] + pred_names
    lines += [f"-- Estimated Model for {y_name}", ""]
    buf = max(len(s) for s in est.index)
    w = [max(9, max(len(fmt(v, d)) for v in est[c]) + 1)
         for c in est.columns]
    lines.append(" " * buf
                 + f"{'Estimate':>{w[0] + 1}}"
                 + f"{'Std Err':>{w[1] + 2}}"
                 + f"{'t-value':>9}{'p-value':>9}"
                 + f"{'Lower 95%':>{w[4] + 3}}"
                 + f"{'Upper 95%':>{w[5] + 3}}")
    for lbl, r in est.iterrows():
        lines.append(
            f"{lbl:<{buf}}"
            + f"{fmt(r['Estimate'], d):>{w[0] + 1}}"
            + f"{fmt(r['Std Err'], d):>{w[1] + 2}}"
            + f"{fmt(r['t-value'], 3):>9}"
            + f"{fmt(r['p-value'], 3):>9}"
            + f"{fmt(r['Lower 95%'], d):>{w[4] + 3}}"
            + f"{fmt(r['Upper 95%'], d):>{w[5] + 3}}")

    # model fit, ~ .reg1fitBasic
    tot_ss = float(((yv - yv.mean()) ** 2).sum())
    sy = math.sqrt(tot_ss / (n_keep - 1))
    se = math.sqrt(fit.scale)
    tcut = -sps.t.ppf(0.025, df=df_res)
    res_range = 2 * tcut * se
    prs = resid / (1 - hat)
    prs = np.where(hat >= 1 - 1e-13, np.nan, prs)
    PRESS = float(np.nansum(prs ** 2))
    Rsq_press = (np.nan if tot_ss == 0
                 else 1 - PRESS / tot_ss)
    lines += ["", "-- Model Fit", "",
              f"Standard deviation of {y_name}: {fmt(sy, d)}",
              "",
              f"Standard deviation of residuals:  {fmt(se, d)}"
              f" for df={df_res}",
              f"95% range of residuals:  {fmt(res_range, d)}"
              f" = 2 * ({fmt(tcut, 3)} * {fmt(se, d)})"]
    if n_pred > 0:
        lines += ["",
                  f"R-squared: {fmt(fit.rsquared, 3)}    "
                  f"Adjusted R-squared: "
                  f"{fmt(fit.rsquared_adj, 3)}    "
                  f"PRESS R-squared: {fmt(Rsq_press, 3)}",
                  "",
                  "Null hypothesis of all 0 population slope "
                  "coefficients:",
                  f"  F-statistic: {fmt(fit.fvalue, 3)}     "
                  f"df: {int(fit.df_model)} and {df_res}     "
                  f"p-value: {fmt(fit.f_pvalue, 3)}"]

    # ANOVA: sequential (Type I) with the Model row,
    # ~ .reg1anvBasic; for the ANCOVA case (one covariate, one
    # factor) term-level Type II sums of squares, each term
    # adjusted for the other
    anova = None
    MSW = fit.scale
    res_ss = float(fit.ssr)
    if n_pred > 0 and ancova:
        rows = []
        for t in ancova_terms:
            others = [c for c in pred_names
                      if c not in term_map[t]]
            Xr = smapi.add_constant(Xd[others],
                                    has_constant="add")
            ss = float(smapi.OLS(yv, Xr).fit().ssr) - res_ss
            dft = len(term_map[t])
            f_v = (ss / dft) / MSW
            rows.append([t, dft, ss, ss / dft, f_v,
                         float(sps.f.sf(f_v, dft, df_res))])
        anova = pd.DataFrame(
            rows + [["Residuals", df_res, res_ss, MSW,
                     np.nan, np.nan]],
            columns=["term", "df", "Sum Sq", "Mean Sq",
                     "F-value", "p-value"]).set_index("term")
        lines += ["", "-- Analysis of Variance from Type II "
                      "Sums of Squares", ""]
        anv_names = ancova_terms
    elif n_pred > 0:
        seq_ss = []
        rss_prev = tot_ss
        for i in range(1, n_pred + 1):
            Xi = smapi.add_constant(Xd.iloc[:, :i],
                                    has_constant="add")
            rss_i = float(smapi.OLS(yv, Xi).fit().ssr)
            seq_ss.append(rss_prev - rss_i)
            rss_prev = rss_i
        rows = []
        for nm, ss in zip(pred_names, seq_ss):
            f_v = ss / MSW
            rows.append([nm, 1, ss, ss, f_v,
                         float(sps.f.sf(f_v, 1, df_res))])
        mod_df = n_pred
        mod_ss = sum(seq_ss)
        mod_ms = mod_ss / mod_df
        mod_f = mod_ms / MSW
        mod_p = float(sps.f.sf(mod_f, mod_df, df_res))
        anova = pd.DataFrame(
            rows + [["Model", mod_df, mod_ss, mod_ms, mod_f,
                     mod_p],
                    ["Residuals", df_res, res_ss, MSW,
                     np.nan, np.nan],
                    [y_name, n_keep - 1, tot_ss,
                     tot_ss / (n_keep - 1), np.nan, np.nan]],
            columns=["term", "df", "Sum Sq", "Mean Sq",
                     "F-value", "p-value"]).set_index("term")
        lines += ["", "-- Analysis of Variance", ""]
        anv_names = pred_names
    if n_pred > 0:
        c1 = max(len(s) for s in anova.index)
        wn = [max(9, max(len(fmt(v, d))
                         for v in anova[c].dropna()) + 1)
              for c in ("Sum Sq", "Mean Sq", "F-value")]
        lines.append(" " * c1 + f"{'df':>7}"
                     + f"{'Sum Sq':>{wn[0]}}"
                     + f"{'Mean Sq':>{wn[1]}}"
                     + f"{'F-value':>{wn[2]}}"
                     + f"{'p-value':>9}")

        def anv_line(lbl, r, with_test=True):
            t = (f"{lbl:<{c1}}{int(r['df']):>7}"
                 + f"{fmt(r['Sum Sq'], d):>{wn[0]}}"
                 + f"{fmt(r['Mean Sq'], d):>{wn[1]}}")
            if with_test:
                t += (f"{fmt(r['F-value'], d):>{wn[2]}}"
                      + f"{fmt(r['p-value'], 3):>9}")
            return t

        for nm in anv_names:
            lines.append(anv_line(nm, anova.loc[nm]))
        if not ancova:                 # Type II has no total
            lines.append("")
            lines.append(anv_line("Model",
                                  anova.loc["Model"]))
        lines.append(anv_line("Residuals",
                              anova.loc["Residuals"],
                              with_test=False))
        if not ancova:
            lines.append(anv_line(y_name, anova.loc[y_name],
                                  with_test=False))

    # ----------  MODERATION ANALYSIS  ----------------------------
    # the simple slopes at the moderator's mean and +/-1 SD; as R,
    # this section is generated with the graphics (~ .reg6mod)
    if mod_info is not None and graphics:
        lines += _moderation_lines(fit, Xd, mod_info, d)

    # ----------  ANCOVA GROUP MODELS  ----------------------------
    # the interaction test and the per-level parallel-line
    # equations; as R, generated with the graphics (~ .reg5ancova)
    if ancova and graphics:
        lines += _reg_ancova_models(fit, Xd, yv, y_name,
                                    ancova_info, d, smapi)

    # ----------  RELATIONS AMONG THE VARIABLES  ------------------
    # collinearity, ~ .reg2Relations (tolerance and VIF from the
    # coefficient standard errors, the R computation), and the
    # best-subset models
    tol = vif = None
    sub_df = None
    if not brief and n_pred_orig > 1:
        vif = np.array([
            (Xd[nm].var(ddof=1) * (n_keep - 1)
             * est.loc[nm, "Std Err"] ** 2) / MSW
            for nm in pred_names])
        tol = 1 / vif
        lines += ["", "", "  RELATIONS AMONG THE VARIABLES",
                  "", "-- Collinearity", ""]
        c1 = max(len(s) for s in pred_names)
        lines.append(" " * c1 + f"{'Tolerance':>11}"
                     + f"{'VIF':>9}")
        for nm, t_i, v_i in zip(pred_names, tol, vif):
            lines.append(f"{nm:<{c1}}{fmt(t_i, 3):>11}"
                         + f"{fmt(v_i, 3):>9}")

        # best subsets: default on, subsets=n caps the listing
        max_sublns = 50
        do_subsets = True if subsets is None else subsets
        if not isinstance(do_subsets, bool) \
                and isinstance(do_subsets, (int, float)) \
                and do_subsets > 1:
            max_sublns = int(do_subsets)
            do_subsets = True
        if do_subsets:
            sub_df = _best_subsets(Xd, yv, tot_ss, MSW,
                                   best_sub)
        if sub_df is not None:
            lines += ["", "-- Best Subset Regression Models"]
            if n_pred > 5:
                lines.append("up to 10 subsets of each "
                             "number of predictors")
            crit_lbl = sub_df.columns[-2]
            xs_lbl = "X's"
            wids = [max(4, len(nm) + 1) for nm in pred_names]
            hdr = ("".join(f"{nm:>{w}}" for nm, w in
                           zip(pred_names, wids))
                   + f"{crit_lbl:>9}{xs_lbl:>7}")
            lines += ["", hdr]
            shown = min(max_sublns, len(sub_df))
            for i in range(shown):
                if shown > 40 and (i + 1) % 30 == 0:
                    lines.append(hdr)
                r = sub_df.iloc[i]
                lines.append(
                    "".join(f"{int(r[nm]):>{w}}"
                            for nm, w in zip(pred_names,
                                             wids))
                    + f" {r[crit_lbl]:>8.3f}"
                    + f" {int(r[xs_lbl]):>6}")
            if len(sub_df) > max_sublns:
                lines += ["",
                          f">>> Only first {shown} of "
                          f"{len(sub_df)} rows printed",
                          "    To indicate more, add "
                          "subsets=n, where n is the number "
                          "of lines"]
            lines += ["",
                      "[exhaustive search of all predictor "
                      "subsets]"]

    # ----------  RESIDUALS AND INFLUENCE  ------------------------
    res_tbl = None
    if brief and n_res_rows is None:
        n_res_rows = 0
    if n_res_rows is None:
        n_res_rows = n_keep if n_keep < 20 else 20
    if n_res_rows == "all":
        n_res_rows = n_keep
    n_res_rows = min(int(n_res_rows), n_keep)

    if n_res_rows > 0:
        res_tbl = pd.DataFrame(index=row_labels)
        for nm in pred_names:
            res_tbl[nm] = Xd[nm].to_numpy()
        res_tbl[y_name] = yv
        res_tbl["fitted"] = fitted
        res_tbl["resid"] = resid
        res_tbl["rstdnt"] = rstudent
        res_tbl["dffits"] = dffits
        res_tbl["cooks"] = np.round(cooks, 5)
        if res_sort == "cooks":
            res_tbl = res_tbl.sort_values(
                "cooks", ascending=False)
        elif res_sort == "rstudent":
            res_tbl = res_tbl.reindex(
                res_tbl["rstdnt"].abs().sort_values(
                    ascending=False).index)
        elif res_sort == "dffits":
            res_tbl = res_tbl.reindex(
                res_tbl["dffits"].abs().sort_values(
                    ascending=False).index)
        lines += ["", "", "  RESIDUALS AND INFLUENCE", "",
                  "-- Data, Fitted, Residual, Studentized "
                  "Residual, Dffits, Cook's Distance"]
        if res_sort == "cooks":
            lines.append("   [sorted by Cook's Distance]")
        elif res_sort == "rstudent":
            lines.append("   [sorted by Studentized Residual,"
                         " ignoring + or - sign]")
        elif res_sort == "dffits":
            lines.append("   [sorted by dffits, ignoring + or"
                         " - sign]")
        more = (" rows of data, or do n_res_rows=\"all\"]"
                if n_res_rows < n_keep else "]")
        lines.append(f"   [n_res_rows = {n_res_rows}, out of "
                     f"{n_keep}{more}")
        res_ints = _int_vars(res_tbl, list(pred_names) + [y_name])
        lines += _prntbl(res_tbl.head(n_res_rows), d,
                         int_cols=res_ints).split("\n")

    # ----------  PREDICTION ERROR  -------------------------------
    pred_tbl = None
    new_data = X1_new is not None
    if brief and n_pred_rows is None and not new_data:
        n_pred_rows = 0
    if n_pred_rows is None:
        n_pred_rows = n_keep if n_keep < 25 else 10
    if n_pred_rows == "all":
        n_pred_rows = n_keep

    if n_pred_rows > 0 or new_data:
        if new_data:                   # X1_new..X6_new grid
            grids = [np.atleast_1d(g) for g in
                     (X1_new, X2_new, X3_new, X4_new,
                      X5_new, X6_new)[:n_pred]
                     if g is not None]
            if len(grids) != n_pred:
                raise ValueError(
                    "specify new values for every predictor: "
                    f"X1_new ... X{n_pred}_new")
            mesh = np.meshgrid(*grids, indexing="ij")
            Xnew = pd.DataFrame(
                {nm: m.ravel() for nm, m in
                 zip(pred_names, mesh)})
            Xn = smapi.add_constant(Xnew, has_constant="add")
            pr = fit.get_prediction(Xn)
            base = Xnew.copy()
            base[y_name] = ""
            base.index = [""] * len(base)
        else:
            pr = fit.get_prediction(X)
            base = pd.DataFrame(index=row_labels)
            for nm in pred_names:
                base[nm] = Xd[nm].to_numpy()
            base[y_name] = yv
        sf = pr.summary_frame(alpha=0.05)
        s_pred = np.sqrt(fit.scale + pr.se_mean ** 2)
        pred_tbl = base
        pred_tbl["pred"] = sf["mean"].to_numpy()
        pred_tbl["s_pred"] = s_pred
        pred_tbl["pi.lwr"] = sf["obs_ci_lower"].to_numpy()
        pred_tbl["pi.upr"] = sf["obs_ci_upper"].to_numpy()
        pred_tbl["width"] = (pred_tbl["pi.upr"]
                             - pred_tbl["pi.lwr"])
        if pred_sort == "predint":
            pred_tbl = pred_tbl.sort_values("pi.lwr")

        hdr = ["", "", "  PREDICTION ERROR", "",
               "-- Data, Predicted, Standard Error of "
               "Prediction, 95% Prediction Intervals",
               "   [sorted by lower bound of prediction "
               "interval]"]
        if n_pred_rows < n_keep and not new_data:
            hdr.append('   [to see all intervals add '
                       'n_pred_rows="all"]')
        hdr.append(_dash(46))
        lines += hdr
        pred_ints = _int_vars(pred_tbl, list(pred_names)
                              + [y_name])
        if new_data or n_pred_rows >= len(pred_tbl):
            lines += _prntbl(pred_tbl, d,
                             int_cols=pred_ints).split("\n")
        else:
            # three pieces, as R: around the widest interval
            # when it falls in that half (else the start/end),
            # and around the narrowest interval in the middle
            widths = pred_tbl["width"].to_numpy()
            nr = len(pred_tbl)
            min_row = int(widths.argmin())
            max_row = int(widths.argmax())
            max_side = max_row < n_keep / 2
            piece = max(1, round(n_pred_rows / 3))
            pr2 = piece // 2
            if max_side:
                r1 = list(range(max(max_row - pr2, 0),
                                min(max_row + pr2 + 1, nr)))
                r3 = list(range(max(nr - piece, 0), nr))
            else:
                r1 = list(range(0, min(piece, nr)))
                r3 = list(range(max(max_row - pr2, 0),
                                min(max_row + pr2 + 1, nr)))
            r2 = list(range(max(min_row - pr2, 0),
                            min(min_row + pr2 + 1, nr)))
            body = _prntbl(pred_tbl, d,
                           int_cols=pred_ints).split("\n")
            head_ln, rows_ln = body[0], body[1:]
            lines.append(head_ln)
            for k, rr in enumerate((r1, r2, r3)):
                if k > 0:
                    lines.append("...")
                for i in rr:
                    lines.append(rows_ln[i])

    # ----------  graphics  ---------------------------------------
    plots = {}
    if graphics and n_pred > 0 and n_res_rows > 0:
        plots["residuals_density"] = _reg_dn_residual(resid)
        plots["residuals_fitted"] = _reg_resfit(
            fitted, resid, cooks, cooks_cut, row_labels, d)
    if graphics and n_pred == 1:
        pi_df = (pred_tbl if (pred_tbl is not None
                              and not new_data) else None)
        plots["scatter"] = _reg_scatter(
            Xd.iloc[:, 0].to_numpy(), yv, pred_names[0],
            y_name, fit, smapi, d,
            with_bands=pi_df is not None)
    if graphics and n_pred > 1 and not ancova:
        # scatterplot matrix of the model variables, response
        # first, with the least-squares fit; ~ .reg5Plot .plt.mat
        mat_df = pd.DataFrame(
            {y_name: yv}, index=Xd.index).join(Xd)
        plots["scatter_matrix"] = scatter_matrix(
            mat_df, fit="lm", digits_d=d)
    if graphics and ancova:
        # ANCOVA: grouped scatterplot with a parallel
        # least-squares line per factor level; ~ .reg5ancova
        plots["ancova"] = _reg_ancova_plot(
            fit, Xd, yv, y_name, ancova_info, d)
    if graphics and mod_info is not None:
        plots["moderation"] = _moderation_plot(
            fit, Xd, y_name, mod_info, d)

    print("\n".join(lines))

    out = RegressionResults(
        formula=formula, n_obs=n_obs, n_keep=n_keep,
        digits_d=d,
        estimates=est, anova=anova,
        fit={"se": se, "resid_range": res_range,
             "Rsq": fit.rsquared if n_pred else np.nan,
             "Rsq_adj": (fit.rsquared_adj if n_pred
                         else np.nan),
             "PRESS": PRESS, "Rsq_PRESS": Rsq_press,
             "sy": sy, "MSW": MSW},
        tolerance=tol, vif=vif, subsets=sub_df,
        residuals=res_tbl, predictions=pred_tbl,
        plots=plots)

    if Rmd is not None:
        reg_rmd(out, y_name, pred_names, formula,
                list(data.columns), Rmd, Rmd_data, Rmd_format,
                Rmd_browser, results, explain, interpret, code,
                n_res_rows, n_pred_rows, res_sort, d)

    return out


def _reg_kfold(yv, Xd, y_name, pred_names, n_keep, formula,
               kfold, seed, digits_d, smapi,
               new_scale="none", scale_response=False):
    """K-fold cross-validation. Partition the complete cases into
    kfold folds; for each, fit on the other kfold-1 folds
    (training) and evaluate the fitted model on the held-out fold
    (testing). Report per-fold and mean se/MSE/Rsq for both, the
    training se as the residual sigma and the testing sp from the
    held-out prediction errors. ~ .regKfold

    With new_scale, each fold's training and testing subsets are
    rescaled separately (predictors always, the response only
    when scale_response), the continuous variables only.

    Folds are random; seed= makes them reproducible within Python
    but, because the RNG differs from R's, not identical to R for
    the same seed."""
    if kfold < 2:
        raise ValueError("kfold must be 2 or larger")
    d = 3 if digits_d is None else digits_d

    # each row's fold, from a scrambled 1..n mod kfold, as R
    rng = np.random.default_rng(seed)
    nk = rng.permutation(np.arange(1, n_keep + 1)) % kfold

    Xmat = Xd.to_numpy(dtype=float)
    # continuous columns (>2 distinct) are the ones rescaled;
    # the decision is global, the rescaling is per fold, as R
    do_scale = new_scale != "none"
    cont = ([c for c in range(Xmat.shape[1])
             if len(np.unique(Xmat[:, c])) > 2] if do_scale
            else [])
    scale_y = (do_scale and scale_response
               and len(np.unique(yv)) > 2)
    tr = {"n": [], "se": [], "MSE": [], "Rsq": []}
    te = {"n": [], "se": [], "MSE": [], "Rsq": []}
    for f in range(kfold):
        train, test = nk != f, nk == f
        ytr, yte = yv[train].copy(), yv[test].copy()
        Xtr_m, Xte_m = Xmat[train].copy(), Xmat[test].copy()
        if do_scale:                   # separate scaling per side
            for c in cont:
                Xtr_m[:, c] = _rescale(Xtr_m[:, c], new_scale, d)
                Xte_m[:, c] = _rescale(Xte_m[:, c], new_scale, d)
            if scale_y:
                ytr = _rescale(ytr, new_scale, d)
                yte = _rescale(yte, new_scale, d)

        # training fit
        Xtr = smapi.add_constant(
            pd.DataFrame(Xtr_m), has_constant="add")
        fit = smapi.OLS(ytr, Xtr).fit()
        coefs = np.asarray(fit.params, dtype=float)
        tr["n"].append(int(train.sum()))
        tr["MSE"].append(float(fit.scale))   # residual mean sq
        tr["se"].append(math.sqrt(float(fit.scale)))
        tr["Rsq"].append(float(fit.rsquared))

        # apply the training model to the held-out test fold
        n_te = int(test.sum())
        Xte = np.column_stack([np.ones(n_te), Xte_m])
        if Xte.shape[1] != coefs.shape[0]:
            raise ValueError(
                "a fold has more variables than estimated "
                "coefficients: at least one variable has no "
                "variation in a test fold")
        sse = float(((yte - Xte @ coefs) ** 2).sum())
        denom = n_te - coefs.shape[0]
        mse = sse / denom if denom > 0 else math.nan
        ssy = float(((yte - yte.mean()) ** 2).sum())
        te["n"].append(n_te)
        te["MSE"].append(mse)
        te["se"].append(math.sqrt(mse) if denom > 0 else math.nan)
        te["Rsq"].append(1 - sse / ssy if ssy > 0 else math.nan)

    cv = pd.DataFrame({
        "fold": range(1, kfold + 1),
        "train_n": tr["n"], "train_se": tr["se"],
        "train_MSE": tr["MSE"], "train_Rsq": tr["Rsq"],
        "test_n": te["n"], "test_se": te["se"],
        "test_MSE": te["MSE"], "test_Rsq": te["Rsq"]})
    means = {"train_se": float(np.mean(tr["se"])),
             "train_MSE": float(np.mean(tr["MSE"])),
             "train_Rsq": float(np.mean(tr["Rsq"])),
             "test_se": float(np.mean(te["se"])),
             "test_MSE": float(np.mean(te["MSE"])),
             "test_Rsq": float(np.mean(te["Rsq"]))}

    print("\n".join(_kfold_lines(cv, means, kfold, d)))
    return RegressionResults(
        formula=formula, n_keep=n_keep, kfold=kfold,
        digits_d=d, cv=cv, cv_means=means)


def _kfold_lines(cv, means, kfold, d):
    """The cross-validation table: a training block (se, MSE, Rsq)
    and a testing block (sp, MSE, Rsq), then the column means.
    ~ .regKfold display."""
    def w(col):                        # width of a numeric column
        vals = [v for v in cv[col] if not math.isnan(v)]
        return max(6, max((len(fmt(v, d)) for v in vals),
                          default=6))

    def num(v):
        return "NA" if math.isnan(v) else fmt(v, d)

    ws = {c: w(c) for c in ("train_se", "train_MSE", "train_Rsq",
                            "test_se", "test_MSE", "test_Rsq")}
    wn = max(3, max(len(str(n)) for n in cv["train_n"]))
    wk = max(3, max(len(str(n)) for n in cv["test_n"]))
    gap = "   "

    def block_w(pre):
        return (ws[pre + "_se"] + ws[pre + "_MSE"]
                + ws[pre + "_Rsq"] + 2 * len(gap))
    tw, kw = block_w("train"), block_w("test")

    lines = [f"\n  {kfold}-FOLD CROSS-VALIDATION", ""]
    # group titles, centered over each block (past the n column)
    tt, kt = "Model from Training Data", "Applied to Testing Data"
    lead = 5 + 1 + wn                   # "fold " + "|" area + n
    lines.append(" " * lead + tt.center(tw)
                 + " " * (3 + wk) + kt.center(kw))
    lines.append(" " * lead + "-" * tw
                 + " " * (3 + wk) + "-" * kw)
    lines.append(
        "fold" + " " + "n".rjust(wn)
        + gap + "se".rjust(ws["train_se"])
        + gap + "MSE".rjust(ws["train_MSE"])
        + gap + "Rsq".rjust(ws["train_Rsq"])
        + "   " + "n".rjust(wk)
        + gap + "sp".rjust(ws["test_se"])
        + gap + "MSE".rjust(ws["test_MSE"])
        + gap + "Rsq".rjust(ws["test_Rsq"]))
    for _, r in cv.iterrows():
        lines.append(
            f"{int(r['fold']):>3} |" + " "
            + str(int(r["train_n"])).rjust(wn)
            + gap + num(r["train_se"]).rjust(ws["train_se"])
            + gap + num(r["train_MSE"]).rjust(ws["train_MSE"])
            + gap + num(r["train_Rsq"]).rjust(ws["train_Rsq"])
            + "   " + str(int(r["test_n"])).rjust(wk)
            + gap + num(r["test_se"]).rjust(ws["test_se"])
            + gap + num(r["test_MSE"]).rjust(ws["test_MSE"])
            + gap + num(r["test_Rsq"]).rjust(ws["test_Rsq"]))
    lines.append(" " * lead + "-" * tw
                 + " " * (3 + wk) + "-" * kw)
    lines.append(
        "Mean" + " " + " " * wn
        + gap + num(means["train_se"]).rjust(ws["train_se"])
        + gap + num(means["train_MSE"]).rjust(ws["train_MSE"])
        + gap + num(means["train_Rsq"]).rjust(ws["train_Rsq"])
        + "   " + " " * wk
        + gap + num(means["test_se"]).rjust(ws["test_se"])
        + gap + num(means["test_MSE"]).rjust(ws["test_MSE"])
        + gap + num(means["test_Rsq"]).rjust(ws["test_Rsq"]))
    return lines


def _mod_slopes(fit, mod_info, wv):
    """The simple-slope intercept and slope of the response on the
    focal predictor at the moderator W = wc, from the fitted
    Y = b0 + bx X + bw W + bxw X*W: at fixed W, intercept
    b0 + bw*wc and slope bx + bxw*wc. Returns the coefficients and
    the three (label, wc) levels, mean and +/-1 SD."""
    b0 = float(fit.params["const"])
    bx = float(fit.params[mod_info["x"]])
    bw = float(fit.params[mod_info["w"]])
    bxw = float(fit.params[mod_info["xw"]])
    m_w, s_w = float(wv.mean()), float(wv.std(ddof=1))
    levels = [("+1SD", m_w + s_w), ("Mean", m_w),
              ("-1SD", m_w - s_w)]
    return b0, bx, bw, bxw, m_w, s_w, levels


def _moderation_lines(fit, Xd, mod_info, d):
    """The moderation text: the moderator's mean and SD, then the
    simple-slope intercept b0 and slope b1 at mean and +/-1 SD.
    ~ .reg6mod out_mod"""
    w_name = mod_info["w"]
    b0, bx, bw, bxw, m_w, s_w, levels = _mod_slopes(
        fit, mod_info, Xd[w_name].to_numpy(dtype=float))
    out = ["", "", "  MODERATION ANALYSIS", "",
           f"Mean of {w_name}: {fmt(m_w, d)}",
           f"SD of   {w_name}: {fmt(s_w, d)}", ""]
    for tag, wc in (("mean+1SD", levels[0][1]),
                    ("mean    ", levels[1][1]),
                    ("mean-1SD", levels[2][1])):
        out.append(f"{tag} for {w_name}:  "
                   f"b0={fmt(b0 + bw * wc, d)}  "
                   f"b1={fmt(bx + bxw * wc, d)}")
    return out


def _moderation_plot(fit, Xd, y_name, mod_info, d):
    """Interaction plot: the simple regression line of the
    response on the focal predictor at the moderator's mean and
    +/-1 SD, one line each. ~ .reg6mod plot"""
    x_name, w_name = mod_info["x"], mod_info["w"]
    b0, bx, bw, bxw, _, _, levels = _mod_slopes(
        fit, mod_info, Xd[w_name].to_numpy(dtype=float))
    xv = Xd[x_name].to_numpy(dtype=float)
    xs = np.array([float(xv.min()), float(xv.max())])
    style_opts = plotly_style()
    colors = {"+1SD": to_hex(BASE_COLORS[0]),
              "Mean": to_hex("gray20"),
              "-1SD": to_hex(BASE_COLORS[1])}
    widths = {"+1SD": 2, "Mean": 1.3, "-1SD": 2}
    fig = go.Figure()
    ys_all = []
    for lbl, wc in levels:
        ys = (b0 + bw * wc) + (bx + bxw * wc) * xs
        ys_all += list(ys)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=lbl,
            line=dict(color=colors[lbl], width=widths[lbl]),
            hoverinfo="name+x+y"))
    axT1 = pretty(float(xs[0]), float(xs[1]))
    axT2 = pretty(min(ys_all), max(ys_all))
    ax_x = axis_num(x_name, axT1, axis_format(axT1, d))
    ax_y = axis_num(y_name, axT2, axis_format(axT2, d))
    ax_y.update(showgrid=True,
                gridcolor=to_hex(style_opts["grid_col"]),
                gridwidth=1, griddash="dot")
    fig.update_layout(
        xaxis=ax_x, yaxis=ax_y,
        shapes=x_grid(axT1) + plot_border(), template=None,
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]),
        legend=dict(title=dict(text=w_name)),
        title=dict(text="Moderator Variable Interaction Plot",
                   x=0.5, xanchor="center",
                   font=dict(size=round(
                       16 * get_option("main_size", 1)))))
    return fig


def _reg_ancova_models(fit, Xd, yv, y_name, info, d, smapi):
    """The ANCOVA group models text: the test of the covariate x
    factor interaction (the parallel-lines assumption), the
    per-level equations of the no-interaction model, and the
    Plot suggestion. ~ .reg5ancova txmdl"""
    cov, fac = info["cov"], info["fac"]
    p1, p2 = info["preds"]
    ind_cols = [c for c in Xd.columns if c != cov]

    # interaction model: add covariate x indicator columns, and
    # test them jointly against the no-interaction model (fit).
    # The sequential interaction SS = SSE(reduced) - SSE(full),
    # over the full model's residual mean square. ~ anova(lm(
    # y ~ cov * factor))[interaction row]
    inter = pd.DataFrame(
        {f"{cov}:{c}": Xd[cov].to_numpy() * Xd[c].to_numpy()
         for c in ind_cols}, index=Xd.index)
    Xf = pd.concat(
        [smapi.add_constant(Xd, has_constant="add"), inter],
        axis=1)
    full = smapi.OLS(yv, Xf).fit()
    df_int = len(ind_cols)
    df_res = int(full.df_resid)
    ss_int = float(fit.ssr) - float(full.ssr)
    F = (ss_int / df_int) / (float(full.ssr) / df_res)
    p = float(sps.f.sf(F, df_int, df_res))

    b0 = float(fit.params["const"])
    b_slope = float(fit.params[cov])
    out = ["", "",
           f"  MODELS OF {y_name} FOR LEVELS OF {fac}", "",
           "-- Test of Interaction", "",
           f"{p1}:{p2}  df: {df_int}  df resid: {df_res}  "
           f"SS: {fmt(ss_int, 3)}  F: {fmt(F, 3)}  "
           f"p-value: {fmt(p, 3)}", "",
           "-- Assume parallel lines, no interaction of "
           f"{fac} with {cov}", ""]
    for i, lv in enumerate(info["levels"]):
        eff = 0.0 if i == 0 else float(
            fit.params.get(f"{fac}{lv}", 0.0))
        out.append(
            f"Level {lv}: y^_{y_name} = {fmt(b0 + eff, d)} + "
            f"{fmt(b_slope, d)}(x_{cov})")
    out += ["",
            "-- Visualize Separately Computed Regression Lines",
            "",
            f'XY("{cov}", "{y_name}", data=d, by="{fac}", '
            'fit="lm")']
    return out


def _reg_ancova_plot(fit, Xd, yv, y_name, info, d):
    """Grouped scatterplot with one parallel least-squares line
    per factor level: shared slope on the covariate, a separate
    intercept per level (the reference plus its group effect).
    ~ .reg5ancova plot"""
    cov, fac, levels = info["cov"], info["fac"], info["levels"]
    fac_vals = info["fac_vals"]
    xv = Xd[cov].to_numpy(dtype=float)
    b0 = float(fit.params["const"])
    b_slope = float(fit.params[cov])
    xs = np.array([float(xv.min()), float(xv.max())])
    style_opts = plotly_style()
    fig = go.Figure()
    ys_all = list(yv)
    for i, lv in enumerate(levels):
        col = to_hex(BASE_COLORS[i % len(BASE_COLORS)])
        m = fac_vals == lv
        fig.add_trace(go.Scatter(
            x=xv[m], y=yv[m], mode="markers", name=lv,
            legendgroup=lv,
            marker=dict(symbol="circle", size=7,
                        color=make_trans(col, 0.9), opacity=1,
                        line=dict(color=col, width=1)),
            hoverinfo="x+y+name"))
        eff = 0.0 if i == 0 else float(
            fit.params.get(f"{fac}{lv}", 0.0))
        ys = (b0 + eff) + b_slope * xs
        ys_all += list(ys)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", legendgroup=lv,
            line=dict(color=col, width=2),
            hoverinfo="skip", showlegend=False))
    axT1 = pretty(float(xs[0]), float(xs[1]))
    axT2 = pretty(min(ys_all), max(ys_all))
    ax_x = axis_num(cov, axT1, axis_format(axT1, d))
    ax_y = axis_num(y_name, axT2, axis_format(axT2, d))
    ax_y.update(showgrid=True,
                gridcolor=to_hex(style_opts["grid_col"]),
                gridwidth=1, griddash="dot")
    fig.update_layout(
        xaxis=ax_x, yaxis=ax_y,
        shapes=x_grid(axT1) + plot_border(), template=None,
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]),
        legend=dict(title=dict(text=fac)),
        title=dict(text="Scatterplot and Least-Squares Lines",
                   x=0.5, xanchor="center",
                   font=dict(size=round(
                       16 * get_option("main_size", 1)))))
    return fig


def _reg_scatter(xv, yv, x_name, y_name, fit, smapi, digits_d,
                 with_bands=True):
    """Scatterplot with the least-squares line and, by default,
    the 95% confidence and prediction bands. ~ .reg5Plot"""
    od = np.argsort(xv, kind="stable")
    xs = xv[od]
    Xl = smapi.add_constant(
        pd.DataFrame({x_name: xs}), has_constant="add")
    pr = fit.get_prediction(Xl)
    sf = pr.summary_frame(alpha=0.05)
    style_opts = plotly_style()
    fig = go.Figure()
    if with_bands:
        fig.add_trace(go.Scatter(     # prediction band
            x=np.concatenate([xs, xs[::-1]]),
            y=np.concatenate(
                [sf["obs_ci_upper"].to_numpy(),
                 sf["obs_ci_lower"].to_numpy()[::-1]]),
            mode="none", fill="toself",
            fillcolor=as_plotly_color(
                get_option("se_fill", "#1A1A1A19")),
            hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(     # confidence band
            x=np.concatenate([xs, xs[::-1]]),
            y=np.concatenate(
                [sf["mean_ci_upper"].to_numpy(),
                 sf["mean_ci_lower"].to_numpy()[::-1]]),
            mode="none", fill="toself",
            fillcolor=as_plotly_color(
                get_option("se_fill", "#1A1A1A19")),
            hoverinfo="skip", showlegend=False))
    pt_fill = get_option("pt_color", "#324E5C")
    fig.add_trace(go.Scatter(
        x=xv, y=yv, mode="markers",
        marker=dict(symbol="circle", size=7.25,
                    sizemode="diameter",
                    color=make_trans(pt_fill, 0.9),
                    opacity=1,
                    line=dict(color=to_hex(pt_fill), width=1)),
        hoverinfo="x+y", showlegend=False))
    fig.add_trace(go.Scatter(
        x=xs, y=sf["mean"].to_numpy(), mode="lines",
        line=dict(color=to_hex(get_option("fit_color",
                                          "#5C4032")),
                  width=get_option("fit_lwd", 2)),
        hoverinfo="skip", showlegend=False))
    axT1 = pretty(float(xv.min()), float(xv.max()))
    ylo = min(float(yv.min()),
              float(sf["obs_ci_lower"].min())
              if with_bands else float(yv.min()))
    yhi = max(float(yv.max()),
              float(sf["obs_ci_upper"].max())
              if with_bands else float(yv.max()))
    axT2 = pretty(ylo, yhi)
    ax_x = axis_num(x_name, axT1,
                    axis_format(axT1, digits_d))
    ax_y = axis_num(y_name, axT2,
                    axis_format(axT2, digits_d))
    ax_y.update(showgrid=True,
                gridcolor=to_hex(style_opts["grid_col"]),
                gridwidth=1, griddash="dot")
    title = ("Reg Line, Confidence & Prediction Intervals"
             if with_bands
             else "Scatterplot and Least-Squares Line")
    fig.update_layout(
        xaxis=ax_x, yaxis=ax_y,
        shapes=x_grid(axT1) + plot_border(), template=None,
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]),
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=round(
                       16 * get_option("main_size", 1)))))
    return fig


def _reg_resfit(fitted, resid, cooks, cooks_cut, labels,
                digits_d):
    """Residuals vs fitted values, zero line, points at or
    above cooks_cut labeled (or the single largest when none
    reach it). ~ .reg3resfitResidual"""
    max_cook = float(np.nanmax(cooks))
    if max_cook < cooks_cut:
        cut = math.floor(max_cook * 100) / 100
        sub = ("Point with largest Cook's Distance of "
               f"{fmt(max_cook, 2)} is labeled")
    else:
        cut = cooks_cut
        sub = (f"Points with Cook's Distance > {cooks_cut} "
               "are labeled")
    style_opts = plotly_style()
    pt_fill = get_option("pt_color", "#324E5C")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fitted, y=resid, mode="markers",
        marker=dict(symbol="circle", size=5,
                    sizemode="diameter",
                    color=make_trans(pt_fill, 0.9),
                    opacity=1,
                    line=dict(color=to_hex(pt_fill), width=0.5)),
        hoverinfo="x+y", showlegend=False))
    flag = np.where(cooks >= cut)[0]
    for i in flag:
        fig.add_annotation(
            x=float(fitted[i]), y=float(resid[i]),
            text=str(labels[i]), showarrow=False, yshift=12,
            font=dict(size=11, color=to_hex("gray30")))
    axT1 = pretty(float(fitted.min()), float(fitted.max()))
    axT2 = pretty(float(resid.min()), float(resid.max()))
    ax_x = axis_num("Fitted Values", axT1,
                    axis_format(axT1, digits_d))
    ax_y = axis_num("Residuals", axT2,
                    axis_format(axT2, digits_d))
    ax_y.update(showgrid=True,
                gridcolor=to_hex(style_opts["grid_col"]),
                gridwidth=1, griddash="dot")
    shapes = x_grid(axT1) + plot_border()
    shapes.append(dict(                # zero residual line
        type="line", xref="paper", yref="y",
        x0=0, x1=1, y0=0, y1=0,
        line=dict(color=to_hex("gray50"), width=1,
                  dash="dash")))
    fig.update_layout(
        xaxis=ax_x, yaxis=ax_y, shapes=shapes, template=None,
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]),
        title=dict(
            text=("Residuals vs Fitted Values<br>"
                  f"<sup>{sub}</sup>"),
            x=0.5, xanchor="center",
            font=dict(size=round(
                16 * get_option("main_size", 1)))))
    return fig


def _reg_dn_residual(resid):
    """Distribution of the residuals: the density display with
    its histogram backdrop. ~ .reg3dnResidual"""
    edges = _breaks_from_args(np.asarray(resid, dtype=float),
                              None, None, None, "Sturges")
    fig = dn_plotly(np.asarray(resid, dtype=float),
                    x_name="Residuals",
                    x_lab="Residuals",
                    show_histogram=True, hist_edges=edges,
                    main="Distribution of Residuals")
    return fig
