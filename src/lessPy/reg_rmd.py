# reg_rmd.py — generate a Quarto (.qmd) reproducible report for
# Regression(), the Python analog of R's Rmd= (reg.Rmd.R + the
# inst/Rmd/reg/*.txt prose templates + reg.RmdParse.R).
#
# The document is a full tutorial-style narrative with the same
# section structure as lessR: Introduction, Data, Model, Fit,
# Relations, Influence, Prediction, Validity. The explain,
# interpret, and results toggles gate the prose exactly as in R.
#
# Deviations from the R generator, by design:
#  - Target is Quarto (.qmd) with Python chunks calling lessPy,
#    not R Markdown (.Rmd) with R chunks.
#  - Sections are emitted directly in Python rather than through
#    the .txt-file + .RmdParse backtick-symbol engine; the output
#    prose is the same, the indirection is not reproduced. So
#    Rmd_custom / Rmd_dir / Rmd_labels (custom .txt templates) are
#    not ported.
#  - Tables and plots render live from the result object in code
#    chunks (r.estimates, r.plots[...]); narrative numbers are
#    baked at generation time from the same object. The analysis
#    (tables, figures) is fully reproducible; the narrative
#    numbers are fixed to the generating run.
#  - The "Tests of Multiple Coefficients" subsection (R's
#    explain + n.pred>2 block) uses Nest(), which is not ported,
#    so it is omitted.

import shutil
import subprocess
import webbrowser
from pathlib import Path

import numpy as np

from .utils import fmt


# ------------------------------------------------------------
# narrative helpers, ports of xP / xNum / xAnd / xU (xP.R etc.)
# ------------------------------------------------------------

def _xP(x, d):
    """Format a number, thousands separated, d decimals. ~ xP()"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{float(x):,.{d}f}"


_WORDS = ["none", "one", "two", "three", "four", "five", "six",
          "seven", "eight", "nine", "ten", "eleven", "twelve"]


def _xNum(x):
    """Small counts as words, larger as integers. ~ xNum()"""
    x = int(round(x))
    return _WORDS[x] if 0 <= x <= 12 else str(x)


def _xAnd(seq):
    """Join with commas and a final "and". ~ xAnd()"""
    seq = list(seq)
    if not seq:
        return ""
    if len(seq) == 1:
        return str(seq[0])
    if len(seq) == 2:
        return f"{seq[0]} and {seq[1]}"
    return ", ".join(str(s) for s in seq[:-1]) + f", and {seq[-1]}"


def _xU(s):
    """Capitalize the first letter. ~ xU()"""
    return s[:1].upper() + s[1:] if s else s


# ------------------------------------------------------------
# code-chunk emitter
# ------------------------------------------------------------

def _chunk(lines, echo, output=True):
    """A Quarto python code chunk. echo shows the code (the code=
    toggle); output=False runs the chunk but hides its output."""
    out = ["```{python}"]
    if not echo:
        out.append("#| echo: false")
    if not output:
        out.append("#| output: false")
    out += lines
    out.append("```")
    out.append("")
    return out


def _hr():
    return ["", "------", ""]


# ============================================================
# the generator
# ============================================================

def reg_rmd(result, y_name, pred_names, formula_str, data_cols,
            rmd, rmd_data, rmd_format, rmd_browser,
            results, explain, interpret, code,
            n_res_rows, n_pred_rows, res_sort, digits_d):
    """Write a Quarto report reproducing the regression analysis.
    Returns the path of the .qmd file written."""
    n_pred = len(pred_names)
    d = digits_d
    est = result.estimates
    coefs = est["Estimate"]
    pvals = est["p-value"]
    cilb, ciub = est["Lower 95%"], est["Upper 95%"]
    anova = result.anova
    fitd = result.fit

    # context strings, ~ reg.Rmd.R symbol setup
    Y = y_name
    X = _xAnd(pred_names)
    pl = "s" if n_pred > 1 else ""
    et_c = "Each " if n_pred > 1 else "The"
    cnst = (", with the values of all remaining predictor "
            "variables held constant" if n_pred > 1 else "")
    mult = f"through $b_{n_pred}$" if n_pred > 1 else ""

    data_name = "d"
    read_expr = (f'pd.read_csv("{rmd_data}")' if rmd_data
                 else 'pd.read_csv("your_data.csv")')
    call = f'Regression("{formula_str}", data={data_name})'

    tx = []
    tx += _front_matter(Y, X, n_pred, results, explain,
                        interpret, code)
    tx += _setup_chunk(read_expr, data_name, call, code)
    tx += _sec_intro(Y, X, pl)
    tx += _sec_data(read_expr, data_name, data_cols, code)
    tx += _sec_model(Y, X, pred_names, pl, et_c, cnst, mult,
                     coefs, pvals, cilb, ciub, result, d, n_pred,
                     results, explain, interpret, code)
    tx += _sec_fit(Y, anova, fitd, d, n_pred, results, explain)
    tx += _sec_relations(Y, X, pred_names, pl, result, fitd, d,
                         n_pred, results, explain, interpret,
                         code)
    if n_res_rows and n_res_rows > 0 and result.residuals is not None:
        tx += _sec_influence(Y, result, res_sort, d, code)
    if (n_pred_rows and n_pred_rows > 0 and n_pred <= 6
            and result.predictions is not None):
        tx += _sec_prediction(Y, result, d, n_pred, code)
    tx += _sec_validity(code)

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
    title = (f"Multiple Regression of {Y}" if n_pred > 1
             else f"Regression of {Y} on {X}")
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
    # runs the analysis once; output hidden, r reused per section
    return _chunk(
        ["import pandas as pd",
         "from lessPy import Regression",
         f"{data_name} = {read_expr}",
         f"r = {call}"],
        echo=code, output=False)


# ------------------------------------------------------------
# 1 - Introduction
# ------------------------------------------------------------

def _sec_intro(Y, X, pl):
    return [
        "## Introduction",
        "",
        f"The variable of primary interest is the _response "
        f"variable_ {Y}. The purpose of this analysis is to "
        f"account for the values of {Y} from the information "
        f"provided by the values of the _predictor "
        f"variable{pl}_, {X}.",
        "",
        "This regression analysis is an example of _supervised "
        "machine learning_. In the language of machine learning, "
        "refer to the response variable as the _target_ and each "
        "predictor variable as a _feature_. The machine learns "
        f"the relationship between the target {Y} and the "
        f"feature{pl} {X} from the analysis of the training "
        "data. Express this learning in the form of a regression "
        "model.",
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
        "The corresponding data values for the variables in the "
        "model comprise the _training data_, from which the "
        "model is estimated.",
        "",
        "Data from the following variables are available for "
        f"analysis: {_xAnd(data_cols)}.",
        ""]
    return out


# ------------------------------------------------------------
# 3 - Model
# ------------------------------------------------------------

def _sec_model(Y, X, pred_names, pl, et_c, cnst, mult, coefs,
               pvals, cilb, ciub, result, d, n_pred, results,
               explain, interpret, code):
    out = ["## Model", "", "### Specified Model", "",
           f"Express {Y} as a linear function of {_xNum(n_pred)} "
           f"predictor variable{pl}: {X}. Within the context of "
           "the model, indicate the response variable with a Y "
           f"subscripted by the variable name, $Y_{{{Y}}}$. "
           "Identify each predictor variable according to a "
           "subscript to an X. From the training data compute "
           f"$\\hat Y_{{{Y}}}$, the _fitted value_ of the "
           "response variable from the model for a specific set "
           f"of values for {X}.",
           "",
           _eq_model_b(Y, pred_names),
           "",
           f"The _intercept_, $b_0$, indicates the fitted value "
           f"of {Y} for values of {X} all equal to zero. "
           f"{et_c} _slope coefficient_ $b_1$ {mult} is the "
           "average change in the value of response variable, "
           f"{Y}, for a one-unit increase in the value of the "
           f"corresponding predictor variable{cnst}.",
           "",
           "Estimate the coefficients with ordinary least "
           "squares (OLS), which minimizes the sum of the "
           "squared residuals, $\\sum e^2_i$, across all the "
           "rows of the training data.",
           "",
           _eq_resid(),
           "",
           "Accomplish the estimation with the _lessPy_ function "
           "_Regression()_. The analysis was run in the setup "
           "chunk above; its output is examined section by "
           "section below.",
           ""]

    # background: cases presented / retained
    out += [
        f"Of the {result.n_obs} cases presented for analysis, "
        f"{result.n_keep} are retained. The number of deleted "
        f"cases due to missing data is "
        f"{result.n_obs - result.n_keep}.",
        "", "### Estimated Model", "",
        "The analysis begins with the estimation of each sample "
        "regression coefficient, $b_j$, what the estimation "
        "algorithm learns from the training data. Of greater "
        "interest is each corresponding population value, "
        "$\\beta_j$, in the _population model_.",
        "",
        _eq_model_beta(Y, pred_names),
        "",
        "Each _t_-test evaluates the _null hypothesis_ that the "
        "corresponding population regression coefficient is 0.",
        "",
        "$$H_0: \\beta_j=0$$",
        "$$H_1: \\beta_j \\ne 0$$",
        ""]
    if results:
        out += _chunk(["r.estimates"], echo=code)
    out += [
        "This estimated model is the linear function with "
        "estimated numeric coefficients that yield a fitted "
        f"value of {Y}, from the provided data value{pl} of {X}.",
        "",
        _eq_model_est(Y, pred_names, coefs, d),
        ""]
    if interpret:
        out += _intr_ci(Y, pred_names, pvals, cilb, ciub, d,
                        n_pred, cnst)
    return out


def _eq_model_b(Y, pred_names):
    s = f"$$\\hat Y_{{{Y}}} = b_0 + b_1 X_{{{pred_names[0]}}}"
    for i, nm in enumerate(pred_names[1:], start=2):
        s += f" + b_{i} X_{{{nm}}}"
    return s + "$$"


def _eq_model_beta(Y, pred_names):
    s = (f"$$\\hat Y_{{{Y}}} = \\beta_0 + \\beta_1 "
         f"X_{{{pred_names[0]}}}")
    for i, nm in enumerate(pred_names[1:], start=2):
        s += f" + \\beta_{i} X_{{{nm}}}"
    return s + "$$"


def _eq_resid():
    return "$$e_i = Y_i - \\hat Y_i$$"


def _eq_model_est(Y, pred_names, coefs, d):
    s = f"$$\\hat Y_{{{Y}}} = {_xP(coefs.iloc[0], d)}"
    for i, nm in enumerate(pred_names, start=1):
        b = coefs.iloc[i]
        sign = "+" if b >= 0 else "-"
        s += f" {sign} {_xP(abs(b), d)} X_{{{nm}}}"
    return s + "$$"


def _intr_ci(Y, pred_names, pvals, cilb, ciub, d, n_pred, cnst):
    out = []
    ns = [pred_names[i] for i in range(n_pred)
          if pvals.iloc[i + 1] > 0.05]
    sig = [pred_names[i] for i in range(n_pred)
           if pvals.iloc[i + 1] <= 0.05]
    if ns:
        pl2 = "s" if len(ns) > 1 else ""
        has = "have _p_-values" if len(ns) > 1 else "has a _p_-value"
        the = "each " if len(ns) > 1 else "the "
        out += [
            f"{_xU(_xNum(len(ns)))} predictor variable{pl2} "
            f"{has} larger than $\\alpha$ = 0.05: "
            f"_{_xAnd(ns)}_. {_xU(the)}null hypothesis of no "
            "relationship could not be rejected, so there is a "
            f"reasonable possibility that {the}predictor "
            f"variable may not contribute to explaining the "
            f"values of {Y}.",
            ""]
    if sig:
        pl3 = "s" if len(sig) > 1 else ""
        t1 = ("These predictor variables each have "
              if len(sig) > 1 else "This predictor variable has ")
        t2 = "their" if len(sig) > 1 else "its"
        t3 = ("these corresponding slope coefficients"
              if len(sig) > 1 else "this corresponding slope "
              "coefficient")
        out += [
            f"{t1}a _p_-value less than or equal to $\\alpha$ = "
            f"0.05: _{_xAnd(sig)}_. To extend the results beyond "
            "this sample to the population, interpret the "
            f"meaning of {t3} in terms of {t2} confidence "
            f"interval{pl3}.",
            ""]
        for nm in sig:
            j = pred_names.index(nm)
            remain = [p for p in pred_names if p != nm]
            hold = (f", with the values of {_xAnd(remain)} held "
                    "constant" if n_pred > 1 else "")
            bullet = f"- _{nm}_: " if n_pred > 1 else ""
            out += [
                f"{bullet}With 95% confidence, for each "
                f"additional unit of {nm}, on average, {Y} "
                "changes somewhere between "
                f"{_xP(cilb.iloc[j + 1], d)} to "
                f"{_xP(ciub.iloc[j + 1], d)}{hold}.",
                ""]
    return out


# ------------------------------------------------------------
# 4 - Fit
# ------------------------------------------------------------

def _sec_fit(Y, anova, fitd, d, n_pred, results, explain):
    m_ss = anova.loc["Model", "Sum Sq"]
    r_ss = anova.loc["Residuals", "Sum Sq"]
    t_ss = anova.loc[Y, "Sum Sq"]
    r_ms = anova.loc["Residuals", "Mean Sq"]
    t_ms = anova.loc[Y, "Mean Sq"]
    r_df = int(anova.loc["Residuals", "df"])
    t_df = int(anova.loc[Y, "df"])
    tcut = float(-_tppf(r_df))
    out = ["## Fit", "", "### Partitioning Variance", ""]
    if explain:
        out += [
            "The analysis of fit evaluates how well the model "
            f"accounts for the variability of {Y}. The core "
            "measure of variability is the _sum of squares_, "
            "_SS_. The analysis of variance (ANOVA) partitions "
            f"the Total sum of squares of {Y} into the Residual "
            "variability, $\\sum e^2_i$, and the Model sum of "
            "squares. The larger the explained variability "
            "relative to the unexplained, the better the model "
            "fits the data.",
            ""]
    if results:
        out += _chunk(["r.anova"], echo=False)
        out += [
            _eq_decomp(Y, m_ss, r_ss, t_ss, d),
            "",
            "The total variation is that which is explained by "
            "the model, and that which is _not_ explained.",
            ""]
    out += ["### Fit Indices", ""]
    if results:
        out += _chunk(["r.fit"], echo=False)
        out += [
            "#### Standard Deviation of Residuals", "",
            "The _standard deviation of the residuals_, $s_e$, "
            f"the square root of the mean square of the "
            f"residuals, assesses the variability of {Y} about "
            "the fitted values.",
            "",
            _eq_se(r_ms, fitd["se"], d),
            "",
            f"To interpret $s_e$ = {_xP(fitd['se'], d)}, consider "
            "the estimated range of 95% of the values of a "
            "normally distributed variable, from the 2.5% cutoff "
            f"of the _t_-distribution for df={r_df}: "
            f"{_xP(tcut, 3)}.",
            "",
            _eq_se_range(tcut, fitd["se"], fitd["resid_range"], d),
            "",
            "#### $R^2$ Family", "",
            f"$R^2$ is the proportion of the variability of {Y} "
            "accounted for by the model.",
            "",
            _eq_r2(Y, r_ss, t_ss, fitd["Rsq"], d),
            "",
            "Use the adjusted version, $R^2_{adj}$, to compare "
            "models with different numbers of predictors, which "
            "guards against overfitting.",
            "",
            _eq_r2adj(Y, r_ms, t_ms, fitd["Rsq_adj"], d),
            "",
            f"Compare $R^2$ = {_xP(fitd['Rsq'], 3)} to the "
            f"adjusted $R^2_{{adj}}$ = {_xP(fitd['Rsq_adj'], 3)}, "
            f"a difference of {_xP(fitd['Rsq'] - fitd['Rsq_adj'], 3)}. "
            "A large difference indicates too many predictors "
            "for the available data.",
            "",
            "To generalize to prediction accuracy on _new_ data, "
            "evaluate the model with the _predictive residual_ "
            "(PRESS): estimate the model with each case deleted, "
            "predict that case, and sum the squared predictive "
            "residuals.",
            "",
            _eq_r2press(Y, fitd["PRESS"], t_ss, fitd["Rsq_PRESS"],
                        d),
            "",
            "Because an estimated model overfits its training "
            f"data, $R^2_{{PRESS}}$ = {_xP(fitd['Rsq_PRESS'], 3)} "
            "is lower than both $R^2$ and $R^2_{adj}$, and is "
            "the more appropriate value for how well the model "
            "predicts beyond the training data.",
            ""]
    return out


def _eq_decomp(Y, m, r, t, d):
    return (f"$$SS_{{{Y}}} = SS_{{Model}} + SS_{{Residual}} = "
            f"{_xP(m, d)} + {_xP(r, d)} = {_xP(t, d)}$$")


def _eq_se(r_ms, se, d):
    return (f"$$s_e = \\sqrt{{MS_{{Residual}}}} = "
            f"\\sqrt{{{_xP(r_ms, d)}}} = {_xP(se, d)}$$")


def _eq_se_range(tcut, se, rng, d):
    return ("$$95\\% \\; Range: 2 * t_{cutoff} * s_e = "
            f"2 * {_xP(tcut, 3)} * {_xP(se, d)} = {_xP(rng, d)}$$")


def _eq_r2(Y, r_ss, t_ss, rsq, d):
    return (f"$$R^2 = 1 - \\frac{{SS_{{Residual}}}}{{SS_{{{Y}}}}} "
            f"= 1 - \\frac{{{_xP(r_ss, d)}}}{{{_xP(t_ss, d)}}} = "
            f"{_xP(rsq, 3)}$$")


def _eq_r2adj(Y, r_ms, t_ms, rsqadj, d):
    return (f"$$R^2_{{adj}} = 1 - "
            f"\\frac{{MS_{{Residual}}}}{{MS_{{{Y}}}}} = 1 - "
            f"\\frac{{{_xP(r_ms, d)}}}{{{_xP(t_ms, d)}}} = "
            f"{_xP(rsqadj, 3)}$$")


def _eq_r2press(Y, press, t_ss, rsqpress, d):
    return (f"$$R^2_{{PRESS}} = 1 - "
            f"\\frac{{SS_{{PRE}}}}{{SS_{{{Y}}}}} = 1 - "
            f"\\frac{{{_xP(press, d)}}}{{{_xP(t_ss, d)}}} = "
            f"{_xP(rsqpress, 3)}$$")


# ------------------------------------------------------------
# 5 - Relations
# ------------------------------------------------------------

def _sec_relations(Y, X, pred_names, pl, result, fitd, d, n_pred,
                   results, explain, interpret, code):
    out = ["## Relations", ""]
    if n_pred == 1:
        out += ["### Scatter Plot", ""]
        cor = float(np.sign(result.estimates["Estimate"].iloc[1])
                    * np.sqrt(fitd["Rsq"]))
        out += [
            f"How do the variables relate? The correlation of "
            f"the response variable {Y} with the predictor "
            f"{X} should be relatively high. In the training "
            f"data, $r$ = {_xP(cor, 3)}.",
            "",
            "Visually summarize the relation with a scatterplot "
            "and its least-squares line.",
            ""]
        out += _chunk(['r.plots["scatter"]'], echo=code)
    else:
        out += ["### Scatter Plot Matrix", ""]
        out += [
            f"How do the variables relate? The correlations of "
            f"the response variable {Y} with the predictor "
            f"variables {X} should be relatively high, and the "
            "correlations of the predictors with each other "
            "relatively small.",
            "",
            "The scatterplots between each pair of variables "
            "appear in a _scatterplot matrix_, each with its "
            "best-fitting line below the diagonal and the "
            "corresponding correlation above the diagonal.",
            ""]
        out += _chunk(['r.plots["scatter_matrix"]'], echo=code)
        out += _relations_collinear(X, result, d, code, results,
                                    interpret)
    return out


def _relations_collinear(X, result, d, code, results, interpret):
    out = ["### Collinearity", "",
           "The collinearity analysis assesses the extent that "
           f"the predictor variables -- {X} -- linearly depend "
           "upon each other. Collinear predictors have large "
           "standard errors and unstable estimates. The "
           "_tolerance_ of a predictor is $1 - R^2_j$ from "
           "regressing it on the other predictors; it should be "
           "high, at least above about 0.20. The _variance "
           "inflation factor_ (VIF) is its reciprocal, and "
           "should be below about 5.",
           ""]
    if results:
        out += _chunk(
            ["import pandas as pd",
             'pd.DataFrame({"Tolerance": r.tolerance, '
             '"VIF": r.vif}, index=r.estimates.index[1:])'],
            echo=code)
    if interpret:
        out += _intr_tolerance(result, d)
    out += ["### Subset Models", "",
            "Especially when collinearity is present, can a "
            "simpler model be about as effective? Each row below "
            "is a different model: a 1 means the predictor is in "
            "the model, a 0 means it is excluded.",
            ""]
    if results:
        out += _chunk(["r.subsets"], echo=code)
    out += [
        "The goal is _parsimony_: the most explanatory power, "
        "assessed with $R^2_{adj}$, from the fewest predictors. "
        "This subset analysis is descriptive only; its inferential "
        "statistics are no longer valid, so a revised model "
        "requires cross-validation on new data.",
        ""]
    return out


def _intr_tolerance(result, d):
    tol = np.asarray(result.tolerance, dtype=float)
    names = list(result.estimates.index[1:])
    lo = [names[i] for i in range(len(tol)) if tol[i] < 0.20]
    out = []
    if lo:
        hh = ("s have tolerances" if len(lo) > 1
              else " has a tolerance")
        out.append(
            f"Collinearity is indicated. {_xU(_xNum(len(lo)))} "
            f"variable{hh} less than the cutoff of 0.20: "
            f"{_xAnd(lo)}.")
    if len(lo) == 0:
        out.append("No collinearity exists according to the "
                   "tolerance cutoff of 0.20.")
    j = int(np.argmin(tol))
    out.append(
        f"The predictor variable with the lowest tolerance is "
        f"{names[j]} at {_xP(tol[j], 3)}.")
    return out + [""]


# ------------------------------------------------------------
# 6 - Influence
# ------------------------------------------------------------

_SORTCOL = {"cooks": "cooks", "rstudent": "rstdnt",
            "dffits": "dffits"}
_SORTLBL = {"cooks": "Cook's distances",
            "rstudent": "Studentized residuals",
            "dffits": "dffits values"}


def _sec_influence(Y, result, res_sort, d, code):
    out = ["## Influence", "",
           "Which cases (rows of data) contribute the most to "
           "the lack of fit? For each case the analysis reports "
           "the residual $e = Y - \\hat Y$, the externally "
           "Studentized residual (_rstudent_), the standardized "
           "change in fit (_dffits_), and Cook's Distance "
           "(_cooks_), the aggregate influence of the case on "
           "all the fitted values.",
           ""]
    out += _chunk(["r.residuals"], echo=code)
    col = _SORTCOL.get(res_sort, "cooks")
    if col in result.residuals.columns:
        vals = result.residuals[col].to_numpy(dtype=float)[:5]
        top = ", ".join(_xP(v, 3) for v in vals)
        out += [
            f"The largest {_SORTLBL.get(res_sort, 'values')}: "
            f"{top}.",
            ""]
        if res_sort == "cooks":
            labels = result.residuals.index[
                result.residuals["cooks"] > 1]
            if len(labels):
                out.append(
                    "The following cases exceed the informal "
                    "Cook's Distance cutoff of 1: "
                    f"{_xAnd([f'Row {i}' for i in labels])}.")
            else:
                out.append("No cases have a Cook's Distance "
                           "larger than 1 in this analysis.")
            out.append("")
    return out


# ------------------------------------------------------------
# 7 - Prediction
# ------------------------------------------------------------

def _sec_prediction(Y, result, d, n_pred, code):
    out = ["## Prediction", "", "### Prediction Error", "",
           "Prediction moves beyond the training sample to "
           f"predict {Y} from new values of the predictors. "
           "Prediction error combines the modeling error $s_e$ "
           "with the sampling error of a value on the regression "
           "line, $s_{\\hat Y}$, into the _standard error of "
           "prediction_.",
           "",
           "$$s_{\\hat Y_{p,i}} = \\sqrt{s^2_e + "
           "s^2_{\\hat Y_{p,i}}}$$",
           "",
           "### From Training Data", "",
           "Each row of data is treated _as if_ it were new, "
           "with a predicted value, a standard error of "
           "prediction, and a 95% prediction interval.",
           ""]
    out += _chunk(["r.predictions"], echo=code)
    w = result.predictions["width"]
    imin, imax = w.idxmin(), w.idxmax()
    out += [
        "The width of the prediction intervals ranges from a "
        f"minimum of {_xP(w.loc[imin], d)} for Row {imin} to a "
        f"maximum of {_xP(w.loc[imax], d)} for Row {imax}.",
        ""]
    return out


# ------------------------------------------------------------
# 8 - Validity
# ------------------------------------------------------------

def _sec_validity(code):
    out = ["## Validity", "",
           "The residuals should be independent, normal random "
           "variables with a mean of zero and constant variance.",
           "", "### Distribution of Residuals", "",
           "For the inferential tests to be valid, the residuals "
           "should be normally distributed.",
           ""]
    out += _chunk(['r.plots["residuals_density"]'], echo=code)
    out += ["### Fitted Values vs Residuals", "",
            "The residuals should scatter randomly about 0 with "
            "roughly constant variability across the fitted "
            "values (_homoscedasticity_), free of any pattern.",
            ""]
    out += _chunk(['r.plots["residuals_fitted"]'], echo=code)
    return out


# ------------------------------------------------------------
# rendering
# ------------------------------------------------------------

def _tppf(df):
    from scipy import stats as sps
    return sps.t.ppf(0.025, df)


def _maybe_render(path, rmd_format, rmd_browser):
    """Render the .qmd with the quarto CLI if it is installed and
    a format other than "none" is requested; otherwise leave the
    .qmd for the user to render."""
    if rmd_format in (None, "none"):
        return
    if shutil.which("quarto") is None:
        print("   [quarto CLI not found on PATH: render the "
              f".qmd yourself with  quarto render {path.name}]")
        return
    fmt_map = {"html": "html", "pdf": "pdf", "docx": "docx",
               "word": "docx"}
    to = fmt_map.get(rmd_format, "html")
    try:
        subprocess.run(["quarto", "render", str(path), "--to", to],
                       check=True)
    except subprocess.CalledProcessError as e:
        print(f"   [quarto render failed: {e}]")
        return
    rendered = path.with_suffix("." + ("html" if to == "html"
                                       else to))
    print(f"   rendered:  {rendered}")
    if rmd_browser and to == "html" and rendered.exists():
        webbrowser.open(rendered.resolve().as_uri())
