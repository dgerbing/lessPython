# X() console output, golden against lessR 4.5.7 on Employee: each
# report is R's text line for line (trailing blanks aside), less the
# suggestions (Python syntax here) and two deliberate differences:
# jitter_y is lessPy's own scale, and the t-test pointer is a Python
# call. S3 = Salary/7 to 3 decimals tests the digits rules.

import contextlib
import io
import re

import pytest

import lessPy as lp


@pytest.fixture(scope="module")
def emp():
    d = lp.read_data("Employee")
    d["S3"] = (d["Salary"] / 7).round(3)
    return d


def _report(emp, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lp.X(data=emp, **kw)
    out, skip = [], False
    for ln in buf.getvalue().split("\n"):
        if ln.startswith(">>> Suggestions"):
            skip = True
            continue
        if skip:
            skip = ln != ""
            continue
        if re.match(r"(jitter_y:|> (lp\.)?ttest\()", ln):
            continue
        out.append(ln.rstrip())
    return "\n".join(out).strip("\n")


GOLDEN = {    "hist": (dict(x="Salary"), """\
--- Salary ---

    n   miss         mean           sd          min          mdn          max
     37      0    83795.557    21799.533    56124.970    79547.600   144419.230



--- Outliers ---     from the boxplot: 1

Small      Large
-----      -----
            144419.2


Bin Width: 10000
Number of Bins: 10

             Bin  Midpnt  Count    Prop  Cumul.c  Cumul.p
---------------------------------------------------------
  50000 >  60000   55000      4    0.11        4     0.11
  60000 >  70000   65000      8    0.22       12     0.32
  70000 >  80000   75000      8    0.22       20     0.54
  80000 >  90000   85000      5    0.14       25     0.68
  90000 > 100000   95000      3    0.08       28     0.76
 100000 > 110000  105000      5    0.14       33     0.89
 110000 > 120000  115000      1    0.03       34     0.92
 120000 > 130000  125000      1    0.03       35     0.95
 130000 > 140000  135000      1    0.03       36     0.97
 140000 > 150000  145000      1    0.03       37     1.00"""),
    "hist_by": (dict(x="Salary", by="Gender"), """\
Salary by Gender

 Gender  n na     Mean   Median       SD      IQR      Min      Max
      M 18  0 91147.46 89792.95 23128.44 27772.74 59188.96 144419.2
      W 19  0 76830.60 71356.69 18438.46 15766.48 56124.97 132563.4

For inferential analysis of the mean difference:"""),
    "hist_facet": (dict(x="Salary", facet="Gender"), """\
Salary
  - by levels of -
Gender

    n   miss       mean         sd        min        mdn        max
M   18      0   91147.46   23128.44   59188.96   89792.95  144419.23
W   19      0   76830.60   18438.46   56124.97   71356.69  132563.38"""),
    "hist_by_facet": (dict(x="Salary", by="Gender", facet="Plan"), """\
Salary
  - by levels of -
Plan, Gender

       n   miss         mean           sd          min          mdn          max
1, M   10      0    88357.151    23247.675    63788.260    80973.115   144419.230
1, W    4      0    68107.152     5623.641    63772.580    66159.100    76337.830
2, M    3      0   101007.970    14846.425    91871.050    93014.430   118138.430
2, W   14      0    75342.098    14096.231    56124.970    71658.990   102681.190
3, M    5      0    90811.764    29234.758    59188.960   105027.550   121074.860
3, W    1      0   132563.380"""),
    "dens_normal": (dict(x="Salary", form="density", kind="normal"), """\
--- Bandwidth ---      for general curve: 9529.0447


Null hypothesis is a normal population
Shapiro-Wilk normality test:  W = 0.9117,  p-value = 0.0063

--- Salary ---

    n   miss         mean           sd          min          mdn          max
     37      0    83795.557    21799.533    56124.970    79547.600   144419.230



--- Outliers ---     from the boxplot: 1

Small      Large
-----      -----
            144419.2"""),
    "vbs": (dict(x="Salary", form="vbs"), """\
--- Salary ---
Present: 37
Missing: 0
Total  : 37

Mean         : 83795.557
Stnd Dev     : 21799.533
IQR          : 31012.560
Skew         : 0.190   [medcouple, -1 to 1]

Minimum      : 56124.970
Lower Whisker: 56124.970
1st Quartile : 66772.950
Median       : 79547.600
3rd Quartile : 97785.510
Upper Whisker: 132563.380
Maximum      : 144419.230


--- Outliers ---     from the boxplot: 1

Small      Large
-----      -----
            144419.23

Number of duplicated values: 0


---------- Parameter values (can be manually set)

pt_size: 0.61   size of plotted points
out_size: 0.82  size of plotted outlier points
jitter_x: 0.00  random horizontal movement of points
bw: 9529.04       set bandwidth higher for smoother edges


---------- Summary Statistics for Salary

Salary   n      Mean    Median        SD       IQR       Min        Max
Salary  37  83795.56  79547.60  21799.53  31012.56  56124.97  144419.23"""),
    "y_vbs_by": (dict(x="Years", form="vbs", by="Gender"), """\
       Max Dupli-
Level   cations   Values
------------------------------
M           3     13
W           3     2 10


---------- Parameter values (can be manually set)

pt_size: 0.58   size of plotted points
out_size: 0.81  size of plotted outlier points
jitter_x: 0.50  random horizontal movement of points
bw: 2.52       set bandwidth higher for smoother edges


---------- Summary Statistics for Years

Gender   n   Mean  Median    SD   IQR   Min    Max
     M  17  12.24   13.00  5.27  5.00  5.00  24.00
     W  19   6.84    6.00  4.95  6.50  1.00  18.00"""),
    "years_by": (dict(x="Years", by="Gender"), """\
Years by Gender

 Gender  n na   Mean Median    SD IQR Min Max
      M 17  0 12.235     13 5.274 5.0   5  24
      W 19  0  6.842      6 4.947 6.5   1  18

For inferential analysis of the mean difference:"""),
    "s3_hist": (dict(x="S3"), """\
--- S3 ---

    n   miss           mean             sd            min            mdn            max
     37      0     11970.7939      3114.2191      8017.8530     11363.9430     20631.3190



--- Outliers ---     from the boxplot: 1

Small      Large
-----      -----
            20631.3


Bin Width: 2000
Number of Bins: 7

           Bin  Midpnt  Count    Prop  Cumul.c  Cumul.p
-------------------------------------------------------
  8000 > 10000    9000     12    0.32       12     0.32
 10000 > 12000   11000     12    0.32       24     0.65
 12000 > 14000   13000      4    0.11       28     0.76
 14000 > 16000   15000      5    0.14       33     0.89
 16000 > 18000   17000      2    0.05       35     0.95
 18000 > 20000   19000      1    0.03       36     0.97
 20000 > 22000   21000      1    0.03       37     1.00"""),
    "fp_by": (dict(x="Salary", form="freq_poly", by="Gender"), """\
--- Salary ---

    n   miss         mean           sd          min          mdn          max
     37      0    83795.557    21799.533    56124.970    79547.600   144419.230



--- Outliers ---     from the boxplot: 1

Small      Large
-----      -----
            144419.2"""),
}


@pytest.mark.parametrize("name", list(GOLDEN))
def test_x_report_matches_lessR(emp, name):
    kw, want = GOLDEN[name]
    assert _report(emp, **kw) == want


def test_suggestions_python_syntax(emp, capsys):
    lp.X("Salary", data=emp)
    out = capsys.readouterr().out
    assert 'X("Salary", data=emp, form="density")' in out
    lp.X("Salary", data=emp, bin_width=5000)
    assert "bin_width:" not in capsys.readouterr().out
    lp.style(suggest=False)
    lp.X("Salary", data=emp)
    assert ">>> Suggestions" not in capsys.readouterr().out
    lp.style()


def test_list_x_one_figure_each(emp, capsys):
    figs = lp.X(["Salary", "Years"], data=emp)
    assert isinstance(figs, list) and len(figs) == 2
    assert figs[1].stats["n"] == emp["Years"].notna().sum()
    out = capsys.readouterr().out
    assert "--- Salary ---" in out and "--- Years ---" in out
    assert ">>> Suggestions" not in out     # none for several, as R


def test_list_by_refused(emp):
    with pytest.raises(ValueError, match="Only one by variable"):
        lp.X("Salary", by=["Gender", "Dept"], data=emp)
