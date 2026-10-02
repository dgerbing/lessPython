import numpy as np
import pandas as pd
import pytest

from lessPy import X, XY, Chart


@pytest.fixture
def d():
    rng = np.random.default_rng(11)
    n = 60
    yrs = rng.uniform(0, 20, n)
    return pd.DataFrame({
        "Years": yrs,
        "Salary": 35000 + 2600 * yrs + rng.normal(0, 8000, n),
        "Dept": rng.choice(["Ops", "Sales", "IT"], n),
        "Gender": rng.choice(["F", "M"], n)})


def test_xy_correlation(d, capsys):
    XY("Years", "Salary", data=d)
    out = capsys.readouterr().out
    assert "correlation" in out
    assert "95% CI" in out
    r = np.corrcoef(d["Years"], d["Salary"])[0, 1]
    assert f"{r:.3f}" in out


def test_xy_fit_stats(d, capsys):
    XY("Years", "Salary", data=d, fit="lm")
    out = capsys.readouterr().out
    assert 'fit="lm"' in out
    assert "R-squared" in out


def test_xy_quiet(d, capsys):
    XY("Years", "Salary", data=d, quiet=True)
    assert capsys.readouterr().out == ""


def test_x_summary(d, capsys):
    X("Salary", data=d)
    out = capsys.readouterr().out
    assert "mean:" in out
    assert "median:" in out
    assert "outliers (1.5 IQR):" in out
    assert f"n: {len(d)}" in out


def test_x_quiet(d, capsys):
    X("Salary", data=d, quiet=True)
    assert capsys.readouterr().out == ""


# Chart() console output, golden against lessR 4.5.7 on Employee
# (trailing spaces of R's print.out_all stripped)

def _out(capsys, *a, **k):
    from lessPy import read_data
    Chart(*a, data=read_data("Employee"), **k)
    return [ln.rstrip() for ln in capsys.readouterr().out.split("\n")]


def _has_block(out, block):
    want = block.strip("\n").split("\n")
    n = len(want)
    return any(out[i:i + n] == want for i in range(len(out)))


def test_chart_frequencies(capsys):
    assert _has_block(_out(capsys, "Dept"), """
--- Dept ---

Missing Values: 1

                ACCT   ADMN   FINC   MKTG   SALE    Total
Frequencies:       5      6      4      6     15       36
Proportions:   0.139  0.167  0.111  0.167  0.417    1.000

Chi-squared test of null hypothesis of equal probabilities
  Chisq = 10.944, df = 4, p-value = 0.027
""")


def test_chart_frequencies_vertical(capsys):
    out = _out(capsys, "Years")
    assert _has_block(out, """
Years Count   Prop
------------------
    1    1   0.028
    2    3   0.083
""")
    assert _has_block(out, """
------------------
Total   36   1.000

Chi-squared test of null hypothesis of equal probabilities
  Chisq = 8.444, df = 15, p-value = 0.905
>>> Low cell expected frequencies, so chi-squared approximation may not be accurate
""")


def test_chart_crosstab_chisq(capsys):
    assert _has_block(_out(capsys, "Gender", by="Plan"), """
Joint and Marginal Frequencies
------------------------------

   Gender
Plan     M   W Sum
  1     10   4  14
  2      3  14  17
  3      5   1   6
  Sum   18  19  37

Cramer's V: 0.577

Chi-square Test of Independence:
     Chisq = 12.338, df = 2, p-value = 0.002
>>> Low cell expected frequencies, chi-squared approximation may not be accurate
""")


def test_chart_crosstab_2x2_uncorrected():
    # R's summary.table() applies no continuity correction, nor does
    # the port, whatever scipy's 2 x 2 default
    from lessPy.stats_out import _counts_2d
    x = pd.Series(list("aabbbbab"))
    b = pd.Series(list("ccddcdcc"))
    lines = _counts_2d(x, b, "x", "b")
    assert any(ln.startswith("Cramer's V (phi):") for ln in lines)
    chi = [ln for ln in lines if "Chisq =" in ln][0]
    obs = pd.crosstab(b, x).to_numpy()
    want = __import__("scipy.stats").stats.chi2_contingency(
        obs, correction=False)[0]
    assert f"Chisq = {want:.3f}" in chi


def test_chart_stat_table(capsys):
    out = _out(capsys, "Dept", y="Salary", stat="mean")
    assert _has_block(out, """
Salary
  - by levels of -
Dept

       n   miss       mean         sd        min        mdn        max
ACCT    5      0   71792.78   12774.61   56124.97   79547.60   82502.50
ADMN    6      0   91277.12   27585.15   63788.26   81058.60  132563.38
FINC    4      0   79010.68   17852.50   67139.90   71937.62  105027.55
MKTG    6      0   80257.13   19869.81   61036.85   71658.99  109062.66
SALE   15      0   88830.06   23476.84   59188.96   87714.85  144419.23
""")
    assert _has_block(out, """
 Plotted Values
 --------------
      ACCT      ADMN      FINC      MKTG      SALE
  71792.78  91277.12  79010.68  80257.13  88830.06
""")
    # plotted values belong to the bar chart alone
    out = _out(capsys, "Dept", y="Salary", stat="mean", form="radar")
    assert " Plotted Values" not in out


def test_chart_summary_table(capsys):
    assert _has_block(
        _out(capsys, "Dept", y="Salary", stat="mean", by="Gender"), """
Summary Table for Mean of Salary

       ACCT      ADMN      FINC      MKTG      SALE
M  69626.20  90963.35  82967.60 109062.66  96150.97
W  73237.16  91434.00  67139.90  74496.02  74188.25
""")


def test_chart_quiet(d, capsys):
    Chart("Dept", data=d, quiet=True)
    assert capsys.readouterr().out == ""
