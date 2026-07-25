import numpy as np
import pandas as pd
import pytest

from lessPy import Logit


@pytest.fixture
def d():
    # the data of the R cross-validation run: hard-coded
    # expectations below verified against lessR Logit() on this
    # exact frame (seed 5, July 2026)
    rng = np.random.default_rng(5)
    n = 98
    d = pd.DataFrame({
        "Years": rng.uniform(1, 25, n).round(1),
        "Pre": rng.uniform(50, 100, n).round(0),
    })
    eta = -5.5 + 0.32 * d.Years + 0.02 * d.Pre
    d["Purchase"] = (rng.uniform(size=n)
                     < 1 / (1 + np.exp(-eta))).astype(int)
    return d


def test_estimates_match_R(d, capsys):
    r = Logit("Purchase ~ Years + Pre", data=d)
    capsys.readouterr()
    est = r.estimates
    assert est.loc["(Intercept)", "Estimate"] == \
        pytest.approx(-7.8683, abs=0.0002)
    assert est.loc["Years", "Estimate"] == \
        pytest.approx(0.3328, abs=0.0002)
    assert est.loc["Years", "Std Err"] == \
        pytest.approx(0.0622, abs=0.0002)
    assert est.loc["Pre", "p-value"] == \
        pytest.approx(0.045, abs=0.001)


def test_odds_ratios_match_R(d, capsys):
    r = Logit("Purchase ~ Years + Pre", data=d)
    capsys.readouterr()
    orci = r.odds_ratios
    assert orci.loc["Years", "Odds Ratio"] == \
        pytest.approx(1.3949, abs=0.0005)
    assert orci.loc["Years", "Lower 95%"] == \
        pytest.approx(1.2348, abs=0.0005)
    assert orci.loc["Years", "Upper 95%"] == \
        pytest.approx(1.5757, abs=0.0005)


def test_fit_match_R(d, capsys):
    r = Logit("Purchase ~ Years + Pre", data=d)
    capsys.readouterr()
    assert r.fit["null_deviance"] == pytest.approx(135.489,
                                                   abs=0.001)
    assert r.fit["deviance"] == pytest.approx(69.907,
                                              abs=0.001)
    assert r.fit["df_null"] == 97
    assert r.fit["df_residual"] == 95
    assert r.fit["aic"] == pytest.approx(75.907, abs=0.001)
    assert r.vif[0] == pytest.approx(1.02, abs=0.001)


def test_influence_match_R(d, capsys):
    r = Logit("Purchase ~ Years + Pre", data=d)
    capsys.readouterr()
    res = r.residuals
    # R's top Cook's row (its row 14 is pandas label 13)
    assert res.index[0] == "13"
    assert res.iloc[0]["cooks"] == pytest.approx(0.215327,
                                                 abs=1e-4)
    assert res.iloc[0]["rstudent"] == pytest.approx(-4.1319,
                                                    abs=0.001)
    assert res.iloc[0]["dffits"] == pytest.approx(-0.8038,
                                                  abs=0.001)


def test_confusion_match_R(d, capsys):
    r = Logit("Purchase ~ Years + Pre", data=d)
    capsys.readouterr()
    cm = r.confusion[0]
    assert (cm["hit0"], cm["hit1"]) == (40, 45)
    assert (cm["mis0"], cm["mis1"]) == (6, 7)
    assert cm["accuracy"] == pytest.approx(86.73, abs=0.01)
    assert cm["sensitivity"] == pytest.approx(86.54, abs=0.01)
    assert cm["precision"] == pytest.approx(88.24, abs=0.01)


def test_prob_cut_list(d, capsys):
    r = Logit("Purchase ~ Years", data=d,
              prob_cut=[0.3, 0.5, 0.7])
    out = capsys.readouterr().out
    assert len(r.confusion) == 3
    # single predictor reports the x cutoff per threshold
    assert out.count("Corresponding cutoff threshold") == 3


def test_factor_response_ref_group(d, capsys):
    d2 = d.assign(Purchase=d.Purchase.map({0: "No",
                                           1: "Yes"}))
    r = Logit("Purchase ~ Years + Pre", data=d2)
    capsys.readouterr()
    assert r.levels == ["No", "Yes"]     # Yes = group 1
    assert r.estimates.loc["Years", "Estimate"] == \
        pytest.approx(0.3328, abs=0.0002)
    r2 = Logit("Purchase ~ Years + Pre", data=d2,
               ref_group="No")
    capsys.readouterr()
    assert r2.levels == ["Yes", "No"]    # No = group 1
    assert r2.estimates.loc["Years", "Estimate"] == \
        pytest.approx(-0.3328, abs=0.0002)


def test_x_new(d, capsys):
    r = Logit("Purchase ~ Years + Pre", data=d,
              X1_new=[5, 20], X2_new=[60, 90])
    out = capsys.readouterr().out
    assert len(r.predictions) == 4
    assert "no confusion matrix" in out
    assert len(r.confusion) == 0
    with pytest.raises(ValueError, match="so do for all"):
        Logit("Purchase ~ Years + Pre", data=d,
              X1_new=[5, 20])


def test_brief_and_plots(d, capsys):
    Logit("Purchase ~ Years + Pre", data=d, brief=True)
    out = capsys.readouterr().out
    assert "Estimated Model" in out
    assert "RESIDUALS" not in out
    assert "PREDICTION" not in out
    r1 = Logit("Purchase ~ Years", data=d)
    r2 = Logit("Purchase ~ Years + Pre", data=d)
    capsys.readouterr()
    assert "logit_fit" in r1.plots
    assert "scatter_matrix" in r2.plots  # multiple-predictor


def test_logit_errors(d):
    d2 = d.assign(Purchase=d.Purchase + 1)   # values 1, 2
    with pytest.raises(ValueError, match="0 or 1"):
        Logit("Purchase ~ Years", data=d2)
    with pytest.raises(ValueError, match="ref_group"):
        Logit("Purchase ~ Years", data=d, ref_group="Yes")


def test_indicator_predictor_match_R(capsys):
    # the frame of the R indicator cross-check (seed 9)
    rng = np.random.default_rng(9)
    n = 90
    d = pd.DataFrame({
        "Years": rng.uniform(1, 25, n).round(1),
        "Dept": rng.choice(["ACCT", "MKTG", "SALE"], n),
    })
    eff = d.Dept.map({"ACCT": 0, "MKTG": 6000, "SALE": 11000})
    d["Salary"] = (38000 + 1700 * d.Years + eff
                   + rng.normal(0, 6500, n)).round(2)
    eta = (-4.2 + 0.30 * d.Years
           + d.Dept.map({"ACCT": 0, "MKTG": 0.8,
                         "SALE": 1.6}))
    d["Purchase"] = (rng.uniform(size=n)
                     < 1 / (1 + np.exp(-eta))).astype(int)
    r = Logit("Purchase ~ Years + Dept", data=d)
    out = capsys.readouterr().out
    assert ">>> Note:  Dept is not a numeric variable." in out
    assert "Indicator variables are created" in out
    est = r.estimates
    assert list(est.index) == ["(Intercept)", "Years",
                               "DeptMKTG", "DeptSALE"]
    assert est.loc["DeptMKTG", "Estimate"] == \
        pytest.approx(2.1670, abs=0.0002)
    assert est.loc["DeptSALE", "Estimate"] == \
        pytest.approx(2.7782, abs=0.0002)
    assert r.odds_ratios.loc["DeptMKTG", "Odds Ratio"] == \
        pytest.approx(8.7317, abs=0.001)
    assert r.fit["deviance"] == pytest.approx(66.216,
                                              abs=0.001)


def test_formula_expression(d, capsys):
    # log(Years) materialized as a column via the shared
    # _parse_formula (~ .formula_expr)
    r = Logit("Purchase ~ log(Years) + Pre", data=d)
    capsys.readouterr()
    assert list(r.estimates.index) == \
        ["(Intercept)", "log(Years)", "Pre"]


def test_rmd_report(d, tmp_path, capsys):
    out = tmp_path / "rep"
    Logit("Purchase ~ Years + Pre", data=d, Rmd=str(out),
          Rmd_format="none")
    capsys.readouterr()
    qmd = out.with_suffix(".qmd")
    assert qmd.exists()
    text = qmd.read_text()
    for piece in ("## Model", "## Fit", "## Classification",
                  "## Relations", "Odds Ratios", "R^2_{McF}",
                  'r.plots["scatter_matrix"]'):
        assert piece in text
    with pytest.raises(ValueError, match="brief"):
        Logit("Purchase ~ Years", data=d, Rmd=str(out),
              brief=True)
