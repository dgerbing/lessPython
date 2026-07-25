# logit_rmd.py — Quarto (.qmd) report for Logit(), the
# classification analog of reg_rmd.py.
#
# R's Logit() has NO Rmd= feature (no logit.Rmd, inst/Rmd holds
# only reg/), so this report is not a port: it is designed on the
# reg_rmd pattern, adapted to logistic regression — the log-odds
# model and odds ratios, the deviance fit with the likelihood-
# ratio test and McFadden pseudo-R^2, and the classification
# (confusion) table. Same design choices as the Regression
# report: Quarto target, direct per-section emitters, live code
# chunks for tables/plots, narrative numbers baked at generation
# time, and the explain/interpret/results/code toggles.

from pathlib import Path

import numpy as np
from scipy import stats as sps

from .reg_rmd import _chunk, _maybe_render, _xAnd, _xNum, _xP, _xU


def logit_rmd(result, y_name, pred_names, formula_str, data_cols,
              rmd, rmd_data, rmd_format, rmd_browser,
              results, explain, interpret, code):
    """Write a Quarto report reproducing the logistic-regression
    analysis. Returns the path of the .qmd file written."""
    n_pred = len(pred_names)
    d = result.digits_d
    Y = y_name
    X = _xAnd(pred_names)
    pl = "s" if n_pred > 1 else ""
    event = (f"{Y} = {result.levels[1]}" if result.levels
             else f"{Y} = 1")

    data_name = "d"
    read_expr = (f'pd.read_csv("{rmd_data}")' if rmd_data
                 else 'pd.read_csv("your_data.csv")')
    call = f'Logit("{formula_str}", data={data_name})'

    tx = []
    tx += _front_matter(Y, X, n_pred, results, explain,
                        interpret, code)
    tx += _setup_chunk(read_expr, data_name, call, code)
    tx += _sec_intro(Y, X, pl, event)
    tx += _sec_data(read_expr, data_name, data_cols, code)
    tx += _sec_model(Y, X, pred_names, pl, event, result, d,
                     n_pred, results, explain, interpret, code)
    tx += _sec_fit(Y, result, d, n_pred, results, explain)
    tx += _sec_relations(Y, X, pl, result, n_pred, code)
    tx += _sec_classification(Y, event, result, d, results,
                              explain, code)
    if result.residuals is not None:
        tx += _sec_influence(Y, result, code)
    if result.predictions is not None:
        tx += _sec_prediction(Y, event, result, code)

    path = Path(rmd)
    if path.suffix.lower() != ".qmd":
        path = path.with_suffix(".qmd")
    path.write_text("\n".join(tx))
    print(f"\nQuarto report written:  {path}")
    _maybe_render(path, rmd_format, rmd_browser)
    return path


# ------------------------------------------------------------
# front matter and setup
# ------------------------------------------------------------

def _front_matter(Y, X, n_pred, results, explain, interpret, code):
    title = (f"Logistic Regression of {Y}" if n_pred > 1
             else f"Logistic Regression of {Y} on {X}")
    return [
        "---",
        f'title: "{title}"',
        "format:",
        "  html:",
        "    toc: true",
        "    toc-depth: 4",
        "    embed-resources: true",
        "engine: jupyter",
        "---",
        "",
        "------",
        "",
        f"_Output Options: explain={explain}, "
        f"interpret={interpret}, results={results}, "
        f"code={code}_",
        "",
        "------",
        ""]


def _setup_chunk(read_expr, data_name, call, code):
    return _chunk(
        ["import pandas as pd",
         "from lessPy import Logit",
         f"{data_name} = {read_expr}",
         f"r = {call}"],
        echo=code, output=False)


# ------------------------------------------------------------
# 1 - Introduction
# ------------------------------------------------------------

def _sec_intro(Y, X, pl, event):
    return [
        "## Introduction",
        "",
        f"The _response variable_ {Y} is binary. This logistic "
        "regression models the probability of the outcome "
        f"_{event}_ from the _predictor variable{pl}_ {X}. It is "
        "an example of _supervised machine learning_: a "
        "classifier that learns, from the training data, how the "
        f"feature{pl} {X} relate to the target {Y}.",
        "",
        "Rather than the value of the response directly, the "
        "model estimates its probability, and from a probability "
        "threshold assigns each case to a predicted class.",
        ""]


# ------------------------------------------------------------
# 2 - Data
# ------------------------------------------------------------

def _sec_data(read_expr, data_name, data_cols, code):
    out = ["## Data", "",
           "Read the data into a pandas data frame.", ""]
    out += _chunk([f"{data_name} = {read_expr}", data_name],
                  echo=code)
    out += [
        "The rows are the _training data_, from which the model "
        "is estimated.",
        "",
        "Data from the following variables are available for "
        f"analysis: {_xAnd(data_cols)}.",
        ""]
    return out


# ------------------------------------------------------------
# 3 - Model
# ------------------------------------------------------------

def _sec_model(Y, X, pred_names, pl, event, result, d, n_pred,
               results, explain, interpret, code):
    coefs = result.estimates["Estimate"]
    out = ["## Model", "", "### Specified Model", "",
           f"The response {Y} is not modeled directly. Instead "
           "the model is linear in the _log odds_ (the logit) of "
           f"the probability $\\hat\\pi$ of the outcome "
           f"_{event}_:",
           "",
           _eq_logit_b(pred_names),
           ""]
    if explain:
        out += [
            "The _odds_ are $\\hat\\pi / (1 - \\hat\\pi)$, and "
            "the logit is their natural logarithm. Solving for "
            "the probability gives the S-shaped logistic curve:",
            "",
            "$$\\hat\\pi = \\frac{1}{1 + e^{-(b_0 + b_1 X_1 + "
            "\\cdots)}}$$",
            "",
            "A slope coefficient $b_j$ is the change in the log "
            "odds for a one-unit increase in its predictor; "
            f"exponentiated, $e^{{b_j}}$ is the _odds ratio_, "
            "the multiplicative change in the odds.",
            ""]
    out += ["Estimate the coefficients by maximum likelihood "
            "with the _lessPy_ function _Logit()_. The analysis "
            "ran in the setup chunk above.",
            "", "### Estimated Model", ""]
    if results:
        out += _chunk(["r.estimates"], echo=code)
    out += [
        "The estimated logit model, with maximum-likelihood "
        "coefficients:",
        "",
        _eq_logit_est(pred_names, coefs, d),
        "",
        "### Odds Ratios", "",
        "Exponentiating each coefficient gives its odds ratio "
        "and 95% confidence interval, the more interpretable "
        "scale for the size of an effect.",
        ""]
    if results:
        out += _chunk(["r.odds_ratios"], echo=code)
    if interpret:
        out += _intr_or(Y, event, pred_names, result, d, n_pred)
    return out


def _eq_logit_b(pred_names):
    s = "$$\\ln\\!\\left(\\frac{\\hat\\pi}{1-\\hat\\pi}\\right) " \
        "= b_0 + b_1 X_{" + pred_names[0] + "}"
    for i, nm in enumerate(pred_names[1:], start=2):
        s += f" + b_{i} X_{{{nm}}}"
    return s + "$$"


def _eq_logit_est(pred_names, coefs, d):
    s = ("$$\\ln\\!\\left(\\frac{\\hat\\pi}{1-\\hat\\pi}\\right) "
         f"= {_xP(coefs.iloc[0], d)}")
    for i, nm in enumerate(pred_names, start=1):
        b = coefs.iloc[i]
        sign = "+" if b >= 0 else "-"
        s += f" {sign} {_xP(abs(b), d)} X_{{{nm}}}"
    return s + "$$"


def _intr_or(Y, event, pred_names, result, d, n_pred):
    pvals = result.estimates["p-value"]
    orr = result.odds_ratios
    sig = [pred_names[i] for i in range(n_pred)
           if pvals.iloc[i + 1] <= 0.05]
    ns = [pred_names[i] for i in range(n_pred)
          if pvals.iloc[i + 1] > 0.05]
    out = []
    if sig:
        out.append(
            f"At $\\alpha$ = 0.05, {_xAnd([f'_{s}_' for s in sig])} "
            f"significantly predict{'s' if len(sig) == 1 else ''} "
            f"the odds of {event}.")
        for nm in sig:
            o = float(orr.loc[nm, "Odds Ratio"])
            lo = float(orr.loc[nm, "Lower 95%"])
            hi = float(orr.loc[nm, "Upper 95%"])
            direction = ("multiplies" if o >= 1
                         else "multiplies (reduces)")
            bullet = f"- _{nm}_: " if n_pred > 1 else ""
            out.append(
                f"{bullet}each one-unit increase {direction} the "
                f"odds of {event} by {_xP(o, d)} "
                f"(95% CI {_xP(lo, d)} to {_xP(hi, d)}), the "
                "other predictors held constant."
                if n_pred > 1 else
                f"{bullet}each one-unit increase in {nm} "
                f"{direction} the odds of {event} by {_xP(o, d)} "
                f"(95% CI {_xP(lo, d)} to {_xP(hi, d)}).")
        out.append("")
    if ns:
        pl2 = "s" if len(ns) > 1 else ""
        out += [
            f"{_xU(_xNum(len(ns)))} predictor{pl2} "
            f"({_xAnd(ns)}) did not reach significance, so "
            "there is a reasonable possibility of no effect on "
            f"the odds of {event}.",
            ""]
    return out


# ------------------------------------------------------------
# 4 - Fit
# ------------------------------------------------------------

def _sec_fit(Y, result, d, n_pred, results, explain):
    f = result.fit
    g2 = f["null_deviance"] - f["deviance"]
    df_lr = f["df_null"] - f["df_residual"]
    p = float(sps.chi2.sf(g2, df_lr))
    mcf = 1 - f["deviance"] / f["null_deviance"]
    out = ["## Fit", "", "### Deviance", ""]
    if explain:
        out += [
            "Least squares does not apply to a binary outcome; "
            "fit is assessed with _deviance_, minus twice the "
            "maximized log-likelihood. The _null deviance_ is "
            "the deviance of the intercept-only model (no "
            "predictors); the _residual deviance_ is that of the "
            "fitted model. The drop from null to residual "
            "deviance is the variation the predictors explain.",
            ""]
    if results:
        out += _chunk(["r.fit"], echo=False)
        out += [
            f"Null deviance {_xP(f['null_deviance'], d)} on "
            f"{f['df_null']} df; residual deviance "
            f"{_xP(f['deviance'], d)} on {f['df_residual']} df; "
            f"AIC {_xP(f['aic'], d)}.",
            "", "### Likelihood-Ratio Test", "",
            "The overall test of whether the predictors as a set "
            "relate to the response compares the two deviances. "
            "The likelihood-ratio statistic is their difference, "
            "a chi-square on the difference in degrees of "
            "freedom.",
            "",
            f"$$G^2 = D_{{null}} - D_{{residual}} = "
            f"{_xP(f['null_deviance'], d)} - "
            f"{_xP(f['deviance'], d)} = {_xP(g2, d)}$$",
            "",
            f"With {df_lr} degrees of freedom, $G^2$ = "
            f"{_xP(g2, d)} has a _p_-value of {_xP(p, 4)}, so "
            + ("the predictors are related to the response."
               if p < 0.05 else
               "the predictors are not detectably related to "
               "the response."),
            "", "### Pseudo $R^2$", "",
            "McFadden's pseudo-$R^2$ rescales the deviance drop "
            "to a 0-to-1 fit index (not the proportion of "
            "variance of an OLS $R^2$).",
            "",
            f"$$R^2_{{McF}} = 1 - \\frac{{D_{{residual}}}}"
            f"{{D_{{null}}}} = 1 - "
            f"\\frac{{{_xP(f['deviance'], d)}}}"
            f"{{{_xP(f['null_deviance'], d)}}} = {_xP(mcf, 3)}$$",
            ""]
    return out


# ------------------------------------------------------------
# 5 - Relations
# ------------------------------------------------------------

def _sec_relations(Y, X, pl, result, n_pred, code):
    if n_pred == 1:
        out = ["## Relations", "", "### Fitted Probabilities", "",
               f"The fitted logistic curve shows how the "
               f"probability of the modeled outcome varies with "
               f"{X}, with the 0/1 data points.",
               ""]
        if "logit_fit" in result.plots:
            out += _chunk(['r.plots["logit_fit"]'], echo=code)
        return out
    out = ["## Relations", "", "### Scatter Plot Matrix", "",
           f"The pairwise relations among the response {Y} and "
           f"the predictors {X} appear in a _scatterplot "
           "matrix_, each off-diagonal cell with a loess smooth. "
           f"The {Y} row and column, being 0/1, show how the "
           "outcome rate shifts across each predictor.",
           ""]
    if "scatter_matrix" in result.plots:
        out += _chunk(['r.plots["scatter_matrix"]'], echo=code)
    return out


# ------------------------------------------------------------
# 6 - Classification
# ------------------------------------------------------------

def _sec_classification(Y, event, result, d, results, explain,
                        code):
    if not result.confusion:
        return []
    cm = result.confusion[0]
    out = ["## Classification", ""]
    if explain:
        out += [
            "Applying a probability threshold to the fitted "
            "probabilities assigns each case a predicted class. "
            "The _confusion matrix_ cross-tabulates the "
            "predicted class against the actual class.",
            ""]
    out += [
        "At the probability threshold "
        f"{_xP(cm['prob_cut'], 2)} for predicting _{event}_:",
        ""]
    if results:
        out += _chunk(
            ["import pandas as pd",
             "cm = r.confusion[0]",
             'pd.DataFrame('
             '{"predicted 0": [cm["hit0"], cm["mis1"]], '
             '"predicted 1": [cm["mis0"], cm["hit1"]]}, '
             'index=["actual 0", "actual 1"])'],
            echo=code)
    out += [
        f"Accuracy is {_xP(cm['accuracy'], 2)}% of cases "
        f"classified correctly. Sensitivity (recall) is "
        f"{_xP(cm['sensitivity'], 2)}%, the share of actual "
        f"_{event}_ cases caught; precision is "
        f"{_xP(cm['precision'], 2)}%, the share of predicted "
        f"_{event}_ cases that were correct.",
        ""]
    return out


# ------------------------------------------------------------
# 7 - Influence
# ------------------------------------------------------------

def _sec_influence(Y, result, code):
    out = ["## Influence", "",
           "Which cases contribute most to the lack of fit? For "
           "each case the analysis reports the fitted "
           "probability, the residual, the externally "
           "Studentized residual (_rstudent_), the standardized "
           "change in fit (_dffits_), and Cook's Distance "
           "(_cooks_).",
           ""]
    out += _chunk(["r.residuals"], echo=code)
    return out


# ------------------------------------------------------------
# 8 - Prediction
# ------------------------------------------------------------

def _sec_prediction(Y, event, result, code):
    out = ["## Prediction", "",
           "For each case the model gives the predicted "
           f"probability of _{event}_ and its standard error. "
           "Applied to new predictor values, the same model "
           "predicts the probability of the outcome for cases "
           "not in the training data.",
           ""]
    out += _chunk(["r.predictions"], echo=code)
    return out
