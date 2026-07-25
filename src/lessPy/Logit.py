# Logit.py — analog of Logit.R (core numeric predictors)
#
# Logit(): logistic regression with the lessR output pipeline —
# the estimated model on the logit scale with Wald confidence
# intervals, odds ratios, model fit (deviances, AIC),
# collinearity via the auxiliary linear model, the residuals and
# influence listing (with R's glm formulas for the studentized
# residual, dffits, and Cook's distance), the classification
# table sorted by fitted probability, confusion matrices per
# prob_cut threshold with accuracy/sensitivity/precision, and
# the fitted-sigmoid plot for a single predictor. As elsewhere,
# the pipeline is ported, not the lines.
#
# The model is a formula string, "Y ~ X1 + X2", as Regression().
# The response is numeric 0/1 or a two-level categorical
# (second level = the reference group predicted as 1, re-ordered
# by ref_group=). Categorical predictors become treatment-coded
# indicator variables (VarLevel columns, ~ model.matrix), with
# R's ">>> Note" announcement. Expression terms such as
# log(Years) or I(Years^2) are materialized as columns (shared
# _parse_formula, ~ .formula_expr). A multiple logit model draws
# the symmetric scatterplot matrix with loess smooths
# (plt_mat_plotly, ~ logit.4Pred). Rmd= writes a Quarto (.qmd)
# classification report (logit_rmd.py) — designed on the
# Regression report, as R's Logit has no Rmd. quiet= does not
# exist; brief= trims residuals and prediction, as R.
#
# Numerics through statsmodels GLM (Binomial), imported lazily.
# Returns a LogitResults object; figures in .plots are not
# auto-shown.

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .plotly_utils import (
    axis_format, axis_num, make_trans, plot_border,
    plotly_style, to_hex, x_grid)
from .plt_mat_plotly import scatter_matrix
from .logit_rmd import logit_rmd
from .Regression import (
    _expand_indicators, _parse_formula, _prntbl)
from .utils import fmt, get_column, get_option, pretty


class LogitResults:
    """Numeric results and figures of Logit(): estimates,
    odds_ratios, fit, residuals and predictions listings,
    confusion matrices, and the plotly figures in .plots."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return (f"<lessPy Logit: {self.formula}, "
                f"n={self.n_keep}>")


def _glm_influence(y01, mu, hat, k_params):
    """R's glm influence measures from the hat values and the
    Pearson residuals, dispersion 1 (binomial): rstudent, as R
    returns it for a glm (the standardized Pearson residual,
    pearson / sqrt(1 - hat) — verified against rstudent() on
    this model), dffits = rstudent * sqrt(hat / (1 - hat)),
    and Cook's distance. R analogs: rstudent(), dffits(),
    cooks.distance.glm()"""
    with np.errstate(divide="ignore", invalid="ignore"):
        pear = (y01 - mu) / np.sqrt(mu * (1 - mu))
        rstud = pear / np.sqrt(1 - hat)
        dffits = rstud * np.sqrt(hat / (1 - hat))
        cooks = (pear / (1 - hat)) ** 2 * hat / k_params
    return rstud, dffits, cooks


def Logit(my_formula, data=None, filter=None, ref_group=None,
          digits_d=4, brief=False,
          res_rows=None, res_sort="cooks",
          pred=True, pred_all=False, prob_cut=0.5, cooks_cut=1,
          X1_new=None, X2_new=None, X3_new=None,
          X4_new=None, X5_new=None, X6_new=None,
          pt_size=0.9, transparency=0.8,
          Rmd=None, Rmd_data=None, Rmd_format="html",
          Rmd_browser=True,
          results=True, explain=True, interpret=True, code=True,
          xlab=None, ylab=None, graphics=True):
    """Logistic regression of a formula string, "Y ~ X1 + X2",
    with the lessR analysis pipeline: estimates and odds ratios,
    fit, collinearity, residuals and influence, classification
    with confusion matrices, and the fitted-sigmoid plot. Always
    prints, as in R; returns a LogitResults object with the
    figures in .plots."""
    import statsmodels.api as smapi

    if data is None:
        raise ValueError(
            "data= is required: a pandas DataFrame containing "
            "the model's variables")
    if res_sort not in ("cooks", "rstudent", "dffits", "off"):
        raise ValueError(
            'res_sort: "cooks", "rstudent", "dffits", or "off"')
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
    if brief:
        if res_rows is None:
            res_rows = 0
        pred = False

    y_name, pred_names, data = _parse_formula(my_formula, data)
    formula = (f"{y_name} ~ "
               + (" + ".join(pred_names) if pred_names else "1"))
    n_pred = len(pred_names)
    if n_pred == 0:
        raise ValueError("Logit() requires at least one "
                         "predictor")

    y_ser = get_column(data, y_name, "response")
    pred_sers = [get_column(data, nm, "predictor")
                 for nm in pred_names]

    used = pd.concat([y_ser] + pred_sers, axis=1)
    keep = ~used.isna().any(axis=1)
    n_obs = len(data)
    n_keep = int(keep.sum())
    yk = y_ser[keep]

    # categorical predictors become indicator variables, as R
    (pred_names, pred_sers_x, ind_notes, _cat_names,
     _term_map) = _expand_indicators(
        pred_names, [s[keep] for s in pred_sers],
        ">>> Note:  {0} is not a numeric variable.\n"
        "           Indicator variables are created and "
        "analyzed.")
    n_pred = len(pred_names)
    Xd = pd.DataFrame(
        {nm: s.to_numpy(dtype=float)
         for nm, s in zip(pred_names, pred_sers_x)},
        index=yk.index)

    # response: numeric 0/1, or two-level categorical with the
    # second level the reference group (predicted as 1)
    y_is_factor = not pd.api.types.is_numeric_dtype(yk)
    if y_is_factor:
        if isinstance(yk.dtype, pd.CategoricalDtype):
            levels = [lv for lv in yk.cat.categories
                      if lv in set(yk)]
        else:
            levels = sorted(yk.astype(str).unique())
        if len(levels) != 2:
            raise ValueError(
                f"Response variable: {y_name}\n"
                "If numeric, can only have values of 0 or 1.\n"
                "If a factor, can only have two levels.")
        if ref_group is not None:
            if ref_group not in levels:
                raise ValueError(
                    f"Values of response {y_name}: "
                    f"{levels[0]} {levels[1]}\n"
                    "You specified a non-existent value, "
                    f"ref_group = {ref_group}")
            if levels[1] != ref_group:
                levels = [levels[1], levels[0]]
        # single numeric predictor: order the levels so the
        # slope is positive, as R
        if n_pred == 1:
            avg = Xd.iloc[:, 0].groupby(
                yk.astype(str).to_numpy()).mean()
            if avg[str(levels[0])] > avg[str(levels[1])]:
                levels = [levels[1], levels[0]]
        y01 = (yk.astype(str) ==
               str(levels[1])).to_numpy(dtype=float)
    else:
        if ref_group is not None:
            raise ValueError(
                "Parameter ref_group only applies when the "
                "response is a factor.")
        vals = set(yk.unique())
        if not vals <= {0, 1}:
            raise ValueError(
                f"Response variable: {y_name}\n"
                "If numeric, can only have values of 0 or 1.\n"
                "If a factor, can only have two levels.")
        levels = None
        y01 = yk.to_numpy(dtype=float)

    new_data = X1_new is not None
    row_labels = Xd.index.astype(str)
    d = digits_d

    X = smapi.add_constant(Xd, has_constant="add")
    glm = smapi.GLM(y01, X,
                    family=smapi.families.Binomial()).fit(
                        tol=1e-12)   # match R glm precision
    if glm.params.isna().any():
        bad = ", ".join(glm.params.index[glm.params.isna()])
        raise ValueError(
            "Variable redundant with a prior predictor in the "
            f"model: {bad}")
    mu = np.asarray(glm.fittedvalues, dtype=float)
    hat = glm.get_influence().hat_matrix_diag
    rstud, dffits, cooks = _glm_influence(
        y01, mu, hat, len(glm.params))

    lines = []
    for note in ind_notes:
        lines += [note, ""]

    # ----------  variables and cases  ----------------------------
    for i, nm in enumerate([y_name] + pred_names):
        if i == 0:
            lbl = "Response Variable: "
        elif n_pred > 1:
            lbl = f"Predictor Variable {i}: "
        else:
            lbl = "Predictor Variable: "
        lines.append(lbl + nm)
    lines += ["",
              f"Number of cases (rows) of data:  {n_obs}",
              f"Number of cases retained for analysis:  "
              f"{n_keep}"]

    # ----------  BASIC ANALYSIS  ---------------------------------
    lines += ["", "", "  BASIC ANALYSIS", "",
              f"-- Estimated Model of {y_name} for the Logit "
              "of Reference Group Membership", ""]
    ci = glm.conf_int(alpha=0.05)     # Wald, ~ confint.default
    est = pd.DataFrame({
        "Estimate": glm.params,
        "Std Err": glm.bse,
        "z-value": glm.tvalues,
        "p-value": glm.pvalues,
        "Lower 95%": ci[0],
        "Upper 95%": ci[1],
    })
    est.index = ["(Intercept)"] + pred_names
    buf = max(len(s) for s in est.index)
    w = [max(9, max(len(fmt(v, d)) for v in est[c]) + 1)
         for c in est.columns]
    lines.append(" " * buf
                 + f"{'Estimate':>{w[0] + 1}}"
                 + f"{'Std Err':>{w[1] + 2}}"
                 + f"{'z-value':>9}{'p-value':>9}"
                 + f"{'Lower 95%':>{w[4] + 3}}"
                 + f"{'Upper 95%':>{w[5] + 3}}")
    for lbl, r in est.iterrows():
        lines.append(
            f"{lbl:<{buf}}"
            + f"{fmt(r['Estimate'], d):>{w[0] + 1}}"
            + f"{fmt(r['Std Err'], d):>{w[1] + 2}}"
            + f"{fmt(r['z-value'], 3):>9}"
            + f"{fmt(r['p-value'], 3):>9}"
            + f"{fmt(r['Lower 95%'], d):>{w[4] + 3}}"
            + f"{fmt(r['Upper 95%'], d):>{w[5] + 3}}")

    # odds ratios and 95% CI
    orci = pd.DataFrame({
        "Odds Ratio": np.exp(est["Estimate"]),
        "Lower 95%": np.exp(est["Lower 95%"]),
        "Upper 95%": np.exp(est["Upper 95%"]),
    }, index=est.index)
    lines += ["", "",
              "-- Odds Ratios and Confidence Intervals", ""]
    wo = [max(10, max(len(fmt(v, d)) for v in orci[c]) + 1)
          for c in orci.columns]
    lines.append(" " * (buf + 2)
                 + f"{'Odds Ratio':>{wo[0] + 1}}"
                 + f"{'Lower 95%':>{wo[1] + 3}}"
                 + f"{'Upper 95%':>{wo[2] + 3}}")
    for lbl, r in orci.iterrows():
        lines.append(
            f"{lbl:<{buf}}  "
            + f"{fmt(r['Odds Ratio'], d):>{wo[0] + 1}}"
            + f"{fmt(r['Lower 95%'], d):>{wo[1] + 3}}"
            + f"{fmt(r['Upper 95%'], d):>{wo[2] + 3}}")

    # model fit
    n_iter = len(glm.fit_history["deviance"]) - 1
    lines += ["", "", "-- Model Fit", "",
              f"    Null deviance: {fmt(glm.null_deviance, 3)}"
              f" on {int(glm.df_resid + n_pred)} degrees of "
              "freedom",
              f"Residual deviance: {fmt(glm.deviance, 3)} on "
              f"{int(glm.df_resid)} degrees of freedom", "",
              f"AIC: {fmt(glm.aic, 3)}", "",
              f"Number of iterations to convergence: {n_iter}"]

    # collinearity: R computes tolerance/VIF from the auxiliary
    # linear model of the (numeric) response on the predictors
    tol = vif = None
    if n_pred > 1:
        ols = smapi.OLS(y01, X).fit()
        MSW = ols.scale
        vif = np.array([
            (Xd[nm].var(ddof=1) * (n_keep - 1)
             * ols.bse[nm] ** 2) / MSW
            for nm in pred_names])
        tol = 1 / vif
        lines += ["", "", "Collinearity", ""]
        c1 = max(len(s) for s in pred_names)
        lines.append(" " * c1 + f"{'Tolerance':>11}"
                     + f"{'VIF':>9}")
        for nm, t_i, v_i in zip(pred_names, tol, vif):
            lines.append(f"{nm:<{c1}}{fmt(t_i, 3):>11}"
                         + f"{fmt(v_i, 3):>9}")

    # ----------  ANALYSIS OF RESIDUALS AND INFLUENCE  ------------
    res_tbl = None
    if res_rows is None:
        res_rows = n_keep if n_keep < 20 else 20
    if res_rows == "all":
        res_rows = n_keep
    res_rows = min(int(res_rows), n_keep)
    if res_rows > 0:
        res_tbl = pd.DataFrame(index=row_labels)
        for nm in pred_names:
            res_tbl[nm] = Xd[nm].to_numpy()
        res_tbl[y_name] = yk.to_numpy()
        res_tbl["P(Y=1)"] = mu
        res_tbl["residual"] = y01 - mu
        res_tbl["rstudent"] = rstud
        res_tbl["dffits"] = dffits
        res_tbl["cooks"] = cooks
        if res_sort == "cooks":
            res_tbl = res_tbl.sort_values("cooks",
                                          ascending=False)
        elif res_sort == "rstudent":
            res_tbl = res_tbl.reindex(
                res_tbl["rstudent"].abs().sort_values(
                    ascending=False).index)
        elif res_sort == "dffits":
            res_tbl = res_tbl.reindex(
                res_tbl["dffits"].abs().sort_values(
                    ascending=False).index)
        lines += ["", "",
                  "  ANALYSIS OF RESIDUALS AND INFLUENCE", "",
                  "Data, Fitted, Residual, Standardized "
                  "Pearson Residual, Dffits, Cook's Distance"]
        if res_sort == "cooks":
            lines.append("   [sorted by Cook's Distance]")
        elif res_sort == "rstudent":
            lines.append("   [sorted by Standardized Pearson "
                         "Residual, ignoring + or - sign]")
        elif res_sort == "dffits":
            lines.append("   [sorted by dffits, ignoring + or"
                         " - sign]")
        lines.append(f"   [res_rows = {res_rows} out of "
                     f"{n_keep} cases (rows) of data]")
        lines.append("-" * 68)
        lines += _prntbl(res_tbl.head(res_rows), d).split("\n")
        lines.append("-" * 68)

    # ----------  PREDICTION  -------------------------------------
    pred_tbl = None
    confusion = []
    plots = {}
    prob_cuts = [float(p) for p in np.atleast_1d(prob_cut)]
    p_cut = prob_cuts[0] if len(prob_cuts) == 1 else 0.5
    lv2_txt = str(levels[1]) if levels is not None else ""

    if pred:
        if new_data:
            grids = [np.atleast_1d(g) for g in
                     (X1_new, X2_new, X3_new, X4_new,
                      X5_new, X6_new)[:n_pred]
                     if g is not None]
            if len(grids) != n_pred:
                raise ValueError(
                    "Specified new data values for one "
                    "predictor variable, so do for all.")
            mesh = np.meshgrid(*grids, indexing="ij")
            Xnew = pd.DataFrame(
                {nm: m.ravel() for nm, m in
                 zip(pred_names, mesh)})
            Xn = smapi.add_constant(Xnew, has_constant="add")
            prd = glm.get_prediction(Xn)
            fitv = np.asarray(prd.predicted)
            sev = np.asarray(prd.se)
            pred_tbl = Xnew.copy()
            pred_tbl[y_name] = ""
            pred_tbl.index = [""] * len(pred_tbl)
        else:
            prd = glm.get_prediction(X)
            fitv = np.asarray(prd.predicted)
            sev = np.asarray(prd.se)
            pred_tbl = pd.DataFrame(index=row_labels)
            for nm in pred_names:
                pred_tbl[nm] = Xd[nm].to_numpy()
            pred_tbl[y_name] = yk.to_numpy()
        pred_tbl["label"] = (fitv >= p_cut).astype(int)
        pred_tbl["fitted"] = fitv
        pred_tbl["std.err"] = sev
        pred_tbl = pred_tbl.sort_values("fitted")

        lines += ["", "", "  PREDICTION", "",
                  "Probability threshold for classification "
                  f"{lv2_txt}: {p_cut}", ""]
        if levels is not None:
            lines += [f" 0: {levels[0]}",
                      f" 1: {levels[1]}", ""]
        lines += ["Data, Fitted Values, Standard Errors",
                  "   [sorted by fitted value]"]
        if n_keep > 50 and not pred_all and not new_data:
            lines.append("   [pred_all=TRUE to see all "
                         "intervals displayed]")
        lines.append("-" * 68)
        body = _prntbl(pred_tbl, d).split("\n")
        head_ln, rows_ln = body[0], body[1:]
        if n_keep < 25 or pred_all or new_data:
            lines += body
        else:
            fits_sorted = pred_tbl["fitted"].to_numpy()
            i_mid = int(np.abs(0.5 - fits_sorted).argmin())
            i_mid = min(max(i_mid, 2), n_keep - 3)
            lines.append(head_ln)
            lines += rows_ln[:4]
            lines += ["", "... for the rows of data where "
                      "fitted is close to 0.5 ...", ""]
            lines += rows_ln[i_mid - 2:i_mid + 3]
            lines += ["", "... for the last 4 rows of sorted "
                      "data ...", ""]
            lines += rows_ln[n_keep - 4:]
        lines.append("-" * 68)

        # confusion matrix per prob_cut threshold
        if not new_data:
            lines += ["", "", "-" * 28,
                      "Specified confusion matrices",
                      "-" * 28, ""]
            for pc in prob_cuts:
                confusion.append(_logit_confuse(
                    lines, y01, fitv, pc, y_name, lv2_txt,
                    glm.params, pred_names, d))
        else:
            lines += ["", "",
                      "With X1_new, etc., no confusion "
                      "matrix."]

        # sigmoid plot for a single numeric predictor
        if graphics and n_pred == 1 and not new_data:
            plots["logit_fit"] = _logit_plot(
                Xd.iloc[:, 0].to_numpy(), y01, mu, levels,
                y_name, pred_names[0],
                prob_cuts[0] if len(prob_cuts) == 1 else None,
                glm.params, xlab, ylab, pt_size, transparency,
                d)

        # scatterplot matrix of the model variables for a
        # multiple logit model, symmetric with loess smooths and
        # no correlations, ~ logit.4Pred pairs(panel=smooth)
        if graphics and n_pred > 1 and not new_data:
            mat_df = pd.DataFrame(
                {y_name: y01}, index=Xd.index).join(Xd)
            plots["scatter_matrix"] = scatter_matrix(
                mat_df, fit="loess", cor_coef=False, band=False,
                digits_d=d)

    print("\n".join(lines))

    out = LogitResults(
        formula=formula, n_obs=n_obs, n_keep=n_keep,
        digits_d=d, levels=levels,
        estimates=est, odds_ratios=orci,
        fit={"null_deviance": glm.null_deviance,
             "deviance": glm.deviance,
             "df_null": int(glm.df_resid + n_pred),
             "df_residual": int(glm.df_resid),
             "aic": glm.aic, "iterations": n_iter},
        tolerance=tol, vif=vif,
        residuals=res_tbl, predictions=pred_tbl,
        confusion=confusion, plots=plots)

    if Rmd is not None:
        logit_rmd(out, y_name, pred_names, formula,
                  list(data.columns), Rmd, Rmd_data, Rmd_format,
                  Rmd_browser, results, explain, interpret, code)

    return out


def _logit_confuse(lines, y01, fitv, pc, y_name, lv2_txt,
                   params, pred_names, digits_d):
    """One confusion matrix at threshold pc, with the accuracy,
    sensitivity, and precision. Appends the printed block to
    lines and returns the counts. R analog: .logit5Confuse()"""
    label = (fitv >= pc).astype(int)
    hit0 = int(((y01 == 0) & (label == 0)).sum())
    mis0 = int(((y01 == 0) & (label == 1)).sum())
    hit1 = int(((y01 == 1) & (label == 1)).sum())
    mis1 = int(((y01 == 1) & (label == 0)).sum())
    tot0, tot1 = hit0 + mis0, hit1 + mis1
    totG = tot0 + tot1
    per0 = hit0 / tot0 if tot0 else np.nan
    per1 = hit1 / tot1 if tot1 else np.nan
    perT = (hit0 + hit1) / totG

    lines.append("Probability threshold for predicting "
                 f"{lv2_txt}: {pc}")
    if len(pred_names) == 1:
        x_cut = ((math.log(pc / (1 - pc)) - params.iloc[0])
                 / params.iloc[1])
        lines.append("Corresponding cutoff threshold for "
                     f"{pred_names[0]}: {round(x_cut, 3)}")
    lines.append("")
    ln = len(y_name)
    pad = " " * ln
    lines.append(f"{pad}        Baseline         Predicted")
    lines.append("-" * 51)
    lines.append(f"{pad}       Total  %Tot        0      1"
                 "  %Correct")
    lines.append("-" * 51)
    lines.append(f"{pad}  1  {tot1:6d} {100 * tot1 / totG:5.1f}"
                 f"  {mis1:6d} {hit1:6d}    "
                 f"{100 * per1:.1f}")
    lines.append(f"{y_name}  0  {tot0:6d} "
                 f"{100 * tot0 / totG:5.1f}  {hit0:6d} "
                 f"{mis0:6d}    {100 * per0:.1f}")
    lines.append("-" * 51)
    lines.append(f"{pad} Total {totG:6d}" + " " * 26
                 + f"{100 * perT:.1f}")
    lines.append("")
    accuracy = 100 * (hit0 + hit1) / totG
    recall = 100 * hit1 / (hit1 + mis1) if tot1 else np.nan
    precision = (100 * hit1 / (hit1 + mis0)
                 if (hit1 + mis0) else np.nan)
    lines += [f"Accuracy: {fmt(accuracy, 2)}",
              f"Sensitivity: {fmt(recall, 2)}",
              f"Precision: {fmt(precision, 2)}", ""]
    return {"prob_cut": pc, "hit0": hit0, "mis0": mis0,
            "hit1": hit1, "mis1": mis1,
            "accuracy": accuracy, "sensitivity": recall,
            "precision": precision}


def _logit_plot(xv, y01, mu, levels, y_name, x_name, pc,
                params, xlab, ylab, pt_size, transparency,
                digits_d):
    """Scatterplot of the 0/1 response on the predictor with
    the fitted logistic curve; dashed crosshairs at the
    probability threshold and its predictor cutoff; the two
    response levels labeled on the right axis.
    R analog: the sigmoid plot of .logit4Pred()"""
    style_opts = plotly_style()
    x_lab = x_name if xlab is None else xlab
    lv2 = str(levels[1]) if levels is not None else "1"
    y_lab = (f"Probability {y_name} = {lv2}"
             if ylab is None else ylab)
    pt_fill = get_option("pt_color", "#324E5C")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xv, y=y01, mode="markers",
        marker=dict(symbol="circle",
                    size=max(1.0, pt_size * 7.25),
                    sizemode="diameter",
                    color=make_trans(pt_fill,
                                     1 - transparency),
                    opacity=1,
                    line=dict(color=to_hex(pt_fill),
                              width=1)),
        hoverinfo="x+y", showlegend=False))
    od = np.argsort(xv, kind="stable")
    fig.add_trace(go.Scatter(
        x=xv[od], y=mu[od], mode="lines",
        line=dict(color=to_hex(pt_fill), width=2),
        hoverinfo="skip", showlegend=False))
    axT1 = pretty(float(xv.min()), float(xv.max()))
    axT2 = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ax_x = axis_num(x_lab, axT1,
                    axis_format(axT1, digits_d))
    ax_y = axis_num(y_lab, axT2,
                    [f"{v:g}" for v in axT2])
    ax_y.update(range=[-0.10, 1.10], showgrid=True,
                gridcolor=to_hex(style_opts["grid_col"]),
                gridwidth=1, griddash="dot")
    shapes = x_grid(axT1) + plot_border()
    if pc is not None:                 # threshold crosshairs
        x_cut = ((math.log(pc / (1 - pc)) - params.iloc[0])
                 / params.iloc[1])
        shapes.append(dict(
            type="line", xref="paper", yref="y",
            x0=0, x1=1, y0=pc, y1=pc,
            line=dict(color=to_hex("gray35"), width=0.75,
                      dash="dash")))
        if float(xv.min()) <= x_cut <= float(xv.max()):
            shapes.append(dict(
                type="line", xref="x", yref="paper",
                x0=x_cut, x1=x_cut, y0=0, y1=1,
                line=dict(color=to_hex("gray35"), width=0.75,
                          dash="dash")))
    anns = []
    if levels is not None:             # right-axis level labels
        for yv_i, lv in ((0, levels[0]), (1, levels[1])):
            anns.append(dict(
                xref="paper", yref="y", x=1.01, y=yv_i,
                text=str(lv), showarrow=False,
                xanchor="left",
                font=dict(size=round(
                    14 * get_option("axis_size", 0.9)),
                    color=to_hex(get_option(
                        "axis_color", "black")))))
    fig.update_layout(
        xaxis=ax_x, yaxis=ax_y, shapes=shapes,
        annotations=anns, template=None,
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]))
    return fig
