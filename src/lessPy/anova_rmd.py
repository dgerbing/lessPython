# anova_rmd.py — Quarto (.qmd) report for ANOVA(), the analog of
# R's Rmd= for ANOVA (av.Rmd.R). Unlike the Regression report,
# av.Rmd.R uses no external prose templates and no toggles
# (explain/results are hardcoded on), so this generator is
# correspondingly simple. Same design choices as the other lessPy
# reports: Quarto target, direct per-section emitters, live code
# chunks for tables and plots (recomputed from the result object
# / data), narrative from R's prose. Deviation from av.Rmd.R: the
# cell-means plot is included (R's ANOVA report has no graphics).

from pathlib import Path

from .reg_rmd import _chunk, _maybe_render, _xAnd
from .utils import fmt


def anova_rmd(result, y_name, facs, formula_str, design,
              data_cols, rmd, rmd_data, rmd_format, rmd_browser):
    """Write a Quarto report reproducing the ANOVA. Returns the
    path of the .qmd file written."""
    Y = y_name
    X = _xAnd(facs)
    d = result.digits_d
    data_name = "d"
    read_expr = (f'pd.read_csv("{rmd_data}")' if rmd_data
                 else 'pd.read_csv("your_data.csv")')
    call = f'ANOVA("{formula_str}", data={data_name})'

    tx = _front_matter(Y)
    tx += [
        "The purpose of this analysis is the analysis of "
        f"variance of the values of {Y} for the different "
        f"levels of {X}.", ""]

    # data
    tx += ["## The Data", "",
           "Read the data into a pandas data frame.", ""]
    tx += _chunk(["import pandas as pd",
                  f"{data_name} = {read_expr}", data_name],
                 echo=True)
    tx += ["Data from the following variables are available for "
           f"analysis: {_xAnd(data_cols)}.", ""]

    # analysis
    tx += ["## Analysis of Variance", "",
           "Obtain the analysis with the _lessPy_ function "
           "_ANOVA()_.", "", _design_sentence(design, Y, facs),
           ""]
    tx += _chunk(["import pandas as pd",
                  "from lessPy import ANOVA",
                  f"r = {call}"], echo=True, output=False)

    # background
    tx += ["### Background", "",
           "The output begins with a specification of the "
           "variables in the model and a brief description of "
           "the data.", "",
           f"The response variable is {Y}. The "
           f"{'factor' if len(facs) == 1 else 'factors'} "
           f"{X} define the groups. Of {result.n_obs} cases, "
           f"{result.n_keep} are retained for analysis.", ""]

    tx += _sec_descriptive(result, Y, facs, design, d)
    tx += _sec_summary(Y, X)
    tx += _sec_effects(result, facs, design, d)
    tx += _sec_pairwise(result, X, design, facs)
    tx += _sec_residuals()

    path = Path(rmd)
    if path.suffix.lower() != ".qmd":
        path = path.with_suffix(".qmd")
    path.write_text("\n".join(tx))
    print(f"\nQuarto report written:  {path}")
    _maybe_render(path, rmd_format, rmd_browser)
    return path


def _front_matter(Y):
    return [
        "---",
        f'title: "ANOVA of {Y}"',
        "format:",
        "  html:",
        "    toc: true",
        "    toc-depth: 4",
        "    embed-resources: true",
        "engine: jupyter",
        "---",
        "",
        "------",
        ""]


def _design_sentence(design, Y, facs):
    if design == "oneway":
        return (f"This is a one-way ANOVA of {Y} with treatment "
                f"factor {facs[0]}.")
    if design == "two-between":
        return (f"This is a two-way between groups ANOVA of {Y} "
                f"with treatment factors {_xAnd(facs)}.")
    return (f"This is a randomized blocks ANOVA of {Y} with one "
            f"treatment factor, {facs[0]}, and one blocking "
            f"factor, {facs[1]}.")


def _sec_descriptive(result, Y, facs, design, d):
    out = ["### Descriptive Statistics", "",
           "The descriptive statistics of "
           f"{Y} for the different levels of {_xAnd(facs)} "
           "begin the analysis.", ""]
    if design == "oneway":
        out += _chunk(["r.descriptive"], echo=False)
        out += [f"Grand Mean: {fmt(result.grand_mean, d + 1)}",
                ""]
        out += _plot_chunk(result, "means")
        return out

    f1, f2 = facs
    if design == "two-between":
        out += ["First, the number of data values in each "
                "cell.", ""]
        out += _chunk(
            [f'd.pivot_table(index="{f2}", columns="{f1}", '
             f'values="{Y}", aggfunc="size")'], echo=False)
        out += ["The cell means follow.", ""]
        out += _chunk(
            [f'd.pivot_table(index="{f2}", columns="{f1}", '
             f'values="{Y}", aggfunc="mean")'], echo=False)
    out += ["The marginal means of each factor assist in "
            "interpreting any main effects.", ""]
    out += _chunk(
        [f'd.groupby("{f1}")["{Y}"].mean()'], echo=False)
    out += _chunk(
        [f'd.groupby("{f2}")["{Y}"].mean()'], echo=False)
    out += [f"The grand mean of all the data is "
            f"{fmt(result.grand_mean, d + 1)}.", ""]
    if design == "two-between":
        out += ["The variation in each cell, its standard "
                "deviation, follows.", ""]
        out += _chunk(
            [f'd.pivot_table(index="{f2}", columns="{f1}", '
             f'values="{Y}", aggfunc="std")'], echo=False)
        out += _plot_chunk(result, "interaction")
    else:
        out += _plot_chunk(result, "data")
        out += _plot_chunk(result, "fitted")
    return out


def _sec_summary(Y, X):
    return ["### Summary Table", "",
            "The analysis of variance (ANOVA) partitions the "
            f"total sum of squares for {Y} into the residual "
            "variability, $\\sum e^2_i$, and the sum of squares "
            f"for {X}. The ANOVA table displays these sources "
            "of variation.", "",
            *_chunk(["r.anova"], echo=False)]


def _sec_effects(result, facs, design, d):
    out = ["### Effects", "",
           "The ANOVA table gives the significance test for the "
           "factor; also of interest is the size of the effect.",
           ""]
    e = result.effects
    if design == "oneway":
        out += [
            f"R Squared: {fmt(e['R_squared'], 3)}  \n"
            f"R Sq Adjusted: {fmt(e['R_sq_adjusted'], 3)}  \n"
            f"Omega Squared: {fmt(e['omega_squared'], 3)}  \n"
            f"Cohen's f: {fmt(e['cohen_f'], 3)}", ""]
    elif design == "two-between":
        f1, f2 = facs
        out += [
            f"Partial Omega Squared for {f1}: "
            f"{fmt(e['omega_sq_' + f1], 3)}  \n"
            f"Partial Omega Squared for {f2}: "
            f"{fmt(e['omega_sq_' + f2], 3)}  \n"
            f"Partial Omega Squared for {f1} & {f2}: "
            f"{fmt(e['omega_sq_interaction'], 3)}", ""]
    else:
        f1, f2 = facs
        out += [
            f"Partial Omega Squared for {f1}: "
            f"{fmt(e['omega_sq_' + f1], 3)}  \n"
            f"Partial Intraclass Correlation for {f2}: "
            f"{fmt(e['intraclass_' + f2], 3)}", ""]
    return out


def _sec_pairwise(result, X, design, facs):
    out = ["### Pairwise Differences", "",
           f"A significant effect shows that {X} impacts the "
           "response, but do all levels differ or just some? "
           "The Tukey pairwise comparisons, adjusted for the "
           "overall significance level, are the search for "
           "Honestly Significant Differences (HSD).", ""]
    if result.tukey is None:
        return out
    if design == "oneway":
        out += _chunk(["r.tukey"], echo=False)
    else:
        f1 = facs[0]
        out += [f"Factor {f1}:", ""]
        out += _chunk([f'r.tukey["{f1}"]'], echo=False)
        if design == "two-between":
            f2 = facs[1]
            out += [f"Factor {f2}:", ""]
            out += _chunk([f'r.tukey["{f2}"]'], echo=False)
            out += ["Cell means:", ""]
            out += _chunk(['r.tukey["cells"]'], echo=False)
    return out


def _sec_residuals():
    return ["### Residuals", "",
            "ANOVA is a form of regression: each data value has "
            "a fitted value, its cell mean, and the difference "
            "is the residual. The cases with the largest "
            "standardized residuals are listed first.", "",
            *_chunk(["r.residuals"], echo=False)]


def _plot_chunk(result, key):
    if key not in result.plots:
        return []
    return _chunk([f'r.plots["{key}"]'], echo=False)
