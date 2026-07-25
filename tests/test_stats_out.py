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


def test_chart_frequencies(d, capsys):
    Chart("Dept", data=d)
    out = capsys.readouterr().out
    assert "Total" in out
    for lvl in ("Ops", "Sales", "IT"):
        assert lvl in out


def test_chart_crosstab_chisq(d, capsys):
    Chart("Dept", by="Gender", data=d)
    out = capsys.readouterr().out
    assert "chi-square" in out
    assert "p-value" in out


def test_chart_stat_table(d, capsys):
    Chart("Dept", y="Salary", stat="mean", data=d)
    out = capsys.readouterr().out
    assert "mean" in out
    m = d.groupby("Dept")["Salary"].mean()["IT"]
    assert f"{m:.2f}" in out


def test_chart_quiet(d, capsys):
    Chart("Dept", data=d, quiet=True)
    assert capsys.readouterr().out == ""
