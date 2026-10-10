# XY() console output, golden against lessR 4.5.7 on Employee:
# R's text line for line (trailing blanks aside, blank-line runs
# collapsed), less the suggestions, which R draws at random.

import contextlib
import io
import re

import pytest

import lessPy as lp


@pytest.fixture(scope="module")
def emp():
    return lp.read_data("Employee")


def _report(emp, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lp.XY(data=emp, **kw)
    out, skip = [], False
    for ln in buf.getvalue().split("\n"):
        ln = ln.rstrip()
        if ln.startswith(">>> Suggestions"):
            skip = True
            continue
        if skip and re.match(r"^(lp\.)?XY\(", ln):
            continue
        skip = False
        if ln == "" and out and out[-1] == "":
            continue
        out.append(ln)
    return "\n".join(out).strip("\n")


GOLDEN = {    "sc": (dict(x="Years", y="Salary"), """\
>>> Pearson's product-moment correlation

Number of paired values with neither missing, n = 36
Sample Correlation of Years and Salary: r = 0.852

Hypothesis Test of 0 Correlation:  t = 9.501,  df = 34,  p-value = 0.000
95% Confidence Interval for Correlation:  0.727 to 0.923"""),
    "sc_lm": (dict(x="Years", y="Salary", fit="lm"), """\
>>> Pearson's product-moment correlation

Number of paired values with neither missing, n = 36
Sample Correlation of Years and Salary: r = 0.852

Hypothesis Test of 0 Correlation:  t = 9.501,  df = 34,  p-value = 0.000
95% Confidence Interval for Correlation:  0.727 to 0.923

  Line: b0 = 52710.898    b1 = 3249.552
  Linear Model MSE = 134,129,397.124   Rsq = 0.726"""),
    "sc_loess": (dict(x="Years", y="Salary", fit="loess"), """\
   Loess Model MSE = 100,834,065.368"""),
    "sc_by_lm": (dict(x="Years", y="Salary", by="Gender", fit="lm"), """\
Gender: M  Line: b0 = 40842.335    b1 = 4047.307
  Linear Model MSE = 107,647,877.258   Rsq = 0.819

Gender: W  Line: b0 = 57109.787    b1 = 2882.272
  Linear Model MSE = 144,700,624.695   Rsq = 0.598"""),
    "sc_md": (dict(x="Years", y="Salary", MD_cut=4), """\
>>> Outlier analysis with squared Mahalanobis Distance

  MD                   ID
-----                -----
8.34      Correll, Trevon
7.73        Capelle, Adam
5.83   Korhalkar, Jessica
5.77        James, Leslie

3.92          Hoang, Binh
3.10       Billing, Susan
3.01       Skrotzki, Sara
...                   ...

>>> Pearson's product-moment correlation

Number of paired values with neither missing, n = 36
Sample Correlation of Years and Salary: r = 0.852

Hypothesis Test of 0 Correlation:  t = 9.501,  df = 34,  p-value = 0.000
95% Confidence Interval for Correlation:  0.727 to 0.923"""),
    "run": (dict(x=".Index", y="Salary"), """\
    n   miss         mean           sd          min          mdn          max
     37      0    83795.557    21799.533    56124.970    79547.600   144419.230

------------
Run Analysis
------------

Total number of runs: 21
Total number of values that do not equal the median: 36"""),
}


@pytest.mark.parametrize("name", list(GOLDEN))
def test_xy_report_matches_lessR(emp, name):
    kw, want = GOLDEN[name]
    assert _report(emp, **kw) == want


def test_multi_x_one_correlation_each(emp, capsys):
    # R kept only the last variable's; each is reported, by name
    lp.XY(["Pre", "Post"], "Salary", data=emp)
    out = capsys.readouterr().out
    assert "Variable: Pre with Salary" in out
    assert "Variable: Post with Salary" in out
    assert out.count(">>> Pearson") == 2


def test_series_suggestions_python(emp, capsys):
    lp.XY(".Index", "Salary", data=emp)
    out = capsys.readouterr().out
    assert 'XY(".Index", "Salary", data=emp' in out
