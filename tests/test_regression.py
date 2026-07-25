import numpy as np
import pandas as pd
import pytest

from lessPy import Regression


@pytest.fixture
def d():
    # the data of the R cross-validation run: every hard-coded
    # expectation below was verified against lessR Regression()
    # on this exact frame (seed 3, July 2026)
    rng = np.random.default_rng(3)
    n = 98
    d = pd.DataFrame({
        "Years": rng.uniform(1, 25, n).round(1),
        "Pre": rng.uniform(50, 100, n).round(0),
    })
    d["Salary"] = (35000 + 1800 * d.Years + 120 * d.Pre
                   + rng.normal(0, 7000, n)).round(2)
    return d


def test_estimates_match_R(d, capsys):
    r = Regression("Salary ~ Years + Pre", data=d)
    capsys.readouterr()
    est = r.estimates
    assert est.loc["(Intercept)", "Estimate"] == \
        pytest.approx(38499.802, abs=0.001)
    assert est.loc["Years", "Estimate"] == \
        pytest.approx(1753.446, abs=0.001)
    assert est.loc["Years", "Std Err"] == \
        pytest.approx(99.790, abs=0.001)
    assert est.loc["Pre", "p-value"] == \
        pytest.approx(0.088, abs=0.001)
    assert est.loc["Pre", "Lower 95%"] == \
        pytest.approx(-13.014, abs=0.001)


def test_fit_match_R(d, capsys):
    r = Regression("Salary ~ Years + Pre", data=d)
    capsys.readouterr()
    assert r.fit["se"] == pytest.approx(6537.546, abs=0.001)
    assert r.fit["Rsq"] == pytest.approx(0.765, abs=0.001)
    assert r.fit["Rsq_adj"] == pytest.approx(0.760, abs=0.001)
    assert r.fit["Rsq_PRESS"] == pytest.approx(0.749,
                                               abs=0.001)


def test_anova_match_R(d, capsys):
    r = Regression("Salary ~ Years + Pre", data=d)
    capsys.readouterr()
    a = r.anova
    # sequential (Type I) sums of squares, and the total row
    assert a.loc["Years", "Sum Sq"] == \
        pytest.approx(13084480023.049, rel=1e-9)
    assert a.loc["Pre", "F-value"] == \
        pytest.approx(2.975, abs=0.001)
    assert a.loc["Model", "df"] == 2
    assert a.loc["Salary", "df"] == 97
    assert a.loc["Salary", "Sum Sq"] == \
        pytest.approx(17271870245.260, rel=1e-9)


def test_collinearity_match_R(d, capsys):
    r = Regression("Salary ~ Years + Pre", data=d)
    capsys.readouterr()
    assert r.vif[0] == pytest.approx(1.004, abs=0.001)
    assert r.tolerance[0] == pytest.approx(1 / r.vif[0])


def test_residual_table(d, capsys):
    r = Regression("Salary ~ Years + Pre", data=d)
    capsys.readouterr()
    res = r.residuals
    # sorted by Cook's distance, R's top row (its row 29 is
    # pandas label 28)
    assert res.index[0] == "28"
    assert res.iloc[0]["cooks"] == pytest.approx(0.090,
                                                 abs=0.001)
    assert res.iloc[0]["rstdnt"] == pytest.approx(-2.553,
                                                  abs=0.001)
    assert (res["cooks"].to_numpy()[:-1]
            >= res["cooks"].to_numpy()[1:]).all()


def test_integer_predictors_display_without_decimals():
    from lessPy.Regression import _int_vars, _prntbl
    df = pd.DataFrame({
        "Years": [12.3, 4.5, 7.0],      # has decimals -> not int
        "Pre":   [83.0, 74.0, 90.0],    # whole -> integer
        "Salary": [61036.85, 82502.50, 67562.36],
        "fitted": [56434.741, 60722.164, 86046.054],
    }, index=["a", "b", "c"])
    # only the whole-valued data column is treated as integer
    assert _int_vars(df, ["Years", "Pre", "Salary"]) == ["Pre"]
    out = _prntbl(df, 3, int_cols=["Pre"])
    assert "83.000" not in out          # Pre: no trailing zeros
    assert "12.300" in out              # Years: decimals kept
    assert "61036.850" in out           # Salary: decimals kept
    assert "56434.741" in out           # fitted: decimals kept
    # NaN in an integer column prints blank, not a crash
    df.loc["a", "Pre"] = np.nan
    assert "nan" not in _prntbl(df, 3, int_cols=["Pre"]).lower()


def test_integer_predictor_in_output(d, capsys):
    # Pre is whole-valued in the fixture -> integer in the tables;
    # Years has a decimal -> keeps decimals
    Regression("Salary ~ Years + Pre", data=d, n_res_rows=5)
    out = capsys.readouterr().out
    res = out.split("RESIDUALS AND INFLUENCE")[1]
    assert "Pre" in res
    # no whole-number Pre value shows a decimal in the table
    assert ".000" in res                # Salary/fitted still do
    assert not any(f"{v}.000" in res for v in range(50, 101))


def test_prediction_intervals(d, capsys):
    r = Regression("Salary ~ Years + Pre", data=d)
    capsys.readouterr()
    p = r.predictions
    assert ((p["pi.lwr"] < p["pred"])
            & (p["pred"] < p["pi.upr"])).all()
    # R row 21: the smallest lower bound
    assert p.iloc[0]["pi.lwr"] == pytest.approx(33702.819,
                                                abs=0.001)
    assert (p["s_pred"] > r.fit["se"]).all()


def test_x_new_grid(d, capsys):
    r = Regression("Salary ~ Years + Pre", data=d,
                   X1_new=[5, 15], X2_new=[60, 90])
    capsys.readouterr()
    p = r.predictions
    assert len(p) == 4                 # 2 x 2 grid
    assert p.iloc[0]["pred"] == pytest.approx(52436.337,
                                              abs=0.001)
    with pytest.raises(ValueError, match="X2_new"):
        Regression("Salary ~ Years + Pre", data=d,
                   X1_new=[5, 15])


def test_brief_and_sections(d, capsys):
    Regression("Salary ~ Years + Pre", data=d, brief=True)
    out = capsys.readouterr().out
    assert "Estimated Model" in out
    assert "Analysis of Variance" in out
    assert "Collinearity" not in out
    assert "RESIDUALS" not in out
    assert "PREDICTION ERROR" not in out


def test_plots(d, capsys):
    r1 = Regression("Salary ~ Years", data=d)
    r2 = Regression("Salary ~ Years + Pre", data=d)
    capsys.readouterr()
    assert sorted(r1.plots) == ["residuals_density",
                                "residuals_fitted", "scatter"]
    # multiple regression: no scatterplot (matrix unported)
    assert "scatter" not in r2.plots
    fit_lines = [t for t in r1.plots["scatter"].data
                 if getattr(t, "mode", None) == "lines"]
    assert len(fit_lines) == 1


def test_formula_errors(d):
    with pytest.raises(NotImplementedError, match="operator"):
        Regression("Salary ~ Years * Pre", data=d)
    with pytest.raises(ValueError, match="~"):
        Regression("Salary Years", data=d)


def test_dot_formula(d, capsys):
    r = Regression("Salary ~ .", data=d)
    capsys.readouterr()
    assert list(r.estimates.index) == \
        ["(Intercept)", "Years", "Pre"]


@pytest.fixture
def d_cat():
    # the frame of the R indicator/ANCOVA cross-check (seed 9)
    rng = np.random.default_rng(9)
    n = 90
    d = pd.DataFrame({
        "Years": rng.uniform(1, 25, n).round(1),
        "Dept": rng.choice(["ACCT", "MKTG", "SALE"], n),
    })
    eff = d.Dept.map({"ACCT": 0, "MKTG": 6000, "SALE": 11000})
    d["Salary"] = (38000 + 1700 * d.Years + eff
                   + rng.normal(0, 6500, n)).round(2)
    return d


def test_indicator_estimates_match_R(d_cat, capsys):
    r = Regression("Salary ~ Years + Dept", data=d_cat)
    out = capsys.readouterr().out
    assert ">>>  Dept is not numeric. Converted to indicator" \
        in out
    est = r.estimates
    assert list(est.index) == ["(Intercept)", "Years",
                               "DeptMKTG", "DeptSALE"]
    assert est.loc["DeptMKTG", "Estimate"] == \
        pytest.approx(5779.142, abs=0.001)
    assert est.loc["DeptSALE", "Estimate"] == \
        pytest.approx(11634.878, abs=0.001)
    assert r.fit["Rsq"] == pytest.approx(0.826, abs=0.001)


def test_ancova_type2_anova(d_cat, capsys):
    # term-level Type II, each term adjusted for the other --
    # values verified against direct R anova() drop-term runs
    r = Regression("Salary ~ Years + Dept", data=d_cat)
    out = capsys.readouterr().out
    assert "Type II Sums of Squares" in out
    a = r.anova
    assert list(a.index) == ["Years", "Dept", "Residuals"]
    assert a.loc["Years", "df"] == 1
    assert a.loc["Years", "Sum Sq"] == \
        pytest.approx(12945045670, rel=1e-6)
    assert a.loc["Dept", "df"] == 2
    assert a.loc["Dept", "Sum Sq"] == \
        pytest.approx(1889969610, rel=1e-6)
    assert "Model" not in a.index       # no Model/total rows


def test_indicator_not_ancova(d_cat, capsys):
    # a lone factor predictor: indicators with the standard
    # sequential ANOVA, not the ANCOVA table
    r = Regression("Salary ~ Dept", data=d_cat)
    out = capsys.readouterr().out
    assert "Type II" not in out
    assert "Model" in r.anova.index
    assert list(r.estimates.index) == \
        ["(Intercept)", "DeptMKTG", "DeptSALE"]


def test_best_subsets_adjr2_match_R(d, capsys):
    r = Regression("Salary ~ Years + Pre", data=d)
    out = capsys.readouterr().out
    assert "Best Subset Regression Models" in out
    s = r.subsets
    # every non-empty subset of 2 predictors: 3 rows, sorted
    # by adjusted R-squared descending (R-verified values)
    assert list(s.columns) == ["Years", "Pre", "R2adj", "X's"]
    assert len(s) == 3
    assert list(s.iloc[0][["Years", "Pre"]]) == [1, 1]
    assert s.iloc[0]["R2adj"] == pytest.approx(0.760,
                                               abs=0.001)
    assert s.iloc[1]["R2adj"] == pytest.approx(0.755,
                                               abs=0.001)
    assert s.iloc[2]["R2adj"] == pytest.approx(-0.009,
                                               abs=0.001)


def test_best_subsets_Cp_match_R(d, capsys):
    r = Regression("Salary ~ Years + Pre", data=d,
                   best_sub="Cp")
    capsys.readouterr()
    s = r.subsets
    assert list(s.columns) == ["Years", "Pre", "Cp", "X's"]
    # sorted ascending by Cp; full model Cp = n_params
    assert list(s.iloc[0][["Years", "Pre"]]) == [1, 1]
    assert s.iloc[0]["Cp"] == pytest.approx(3.000, abs=0.001)
    assert s.iloc[1]["Cp"] == pytest.approx(3.975, abs=0.001)
    assert s.iloc[2]["Cp"] == pytest.approx(309.749, abs=0.01)


def test_best_subsets_indicator_columns(d_cat, capsys):
    r = Regression("Salary ~ Years + Dept", data=d_cat)
    capsys.readouterr()
    s = r.subsets
    # subsets operate on the expanded indicator columns
    assert list(s.columns) == ["Years", "DeptMKTG",
                               "DeptSALE", "R2adj", "X's"]
    assert len(s) == 7                 # 2^3 - 1 subsets
    assert list(s.iloc[0][["Years", "DeptMKTG",
                           "DeptSALE"]]) == [1, 1, 1]
    assert s.iloc[0]["R2adj"] == pytest.approx(0.820,
                                               abs=0.001)


def test_best_subsets_off_and_single(d, capsys):
    # one predictor: no relations section, no subsets
    r1 = Regression("Salary ~ Years", data=d)
    capsys.readouterr()
    assert r1.subsets is None
    # brief suppresses the relations/subsets block
    r2 = Regression("Salary ~ Years + Pre", data=d,
                    brief=True)
    out = capsys.readouterr().out
    assert r2.subsets is None
    assert "Best Subset" not in out
    with pytest.raises(ValueError, match="best_sub"):
        Regression("Salary ~ Years + Pre", data=d,
                   best_sub="aic")


def test_categorical_response_redirect(d_cat):
    d2 = d_cat.assign(Salary=np.where(d_cat.Salary
                                      > d_cat.Salary.median(),
                                      "hi", "lo"))
    with pytest.raises(TypeError, match="Logit"):
        Regression("Salary ~ Years", data=d2)


def test_formula_expression_match_R(d):
    # log(Years) materialized as a column, ~ .formula_expr;
    # coefficients verified against lessR on the fixture frame
    r = Regression("Salary ~ log(Years) + Pre", data=d,
                   graphics=False)
    est = r.estimates["Estimate"]
    assert list(r.estimates.index) == \
        ["(Intercept)", "log(Years)", "Pre"]
    assert round(est["(Intercept)"], 3) == 24965.626
    assert round(est["log(Years)"], 3) == 15959.338
    assert round(est["Pre"], 3) == 66.123


def test_formula_interaction_rejected(d):
    with pytest.raises(NotImplementedError, match="not ported"):
        Regression("Salary ~ Years : Pre", data=d)
    with pytest.raises(NotImplementedError, match="not ported"):
        Regression("Salary ~ Years * Pre", data=d)


def test_rmd_report(d, tmp_path, capsys):
    out = tmp_path / "rep"
    Regression("Salary ~ Years + Pre", data=d, Rmd=str(out),
               Rmd_format="none")
    capsys.readouterr()
    qmd = out.with_suffix(".qmd")
    assert qmd.exists()
    text = qmd.read_text()
    for piece in ("## Model", "## Fit", "## Relations",
                  "## Prediction", "R^2_{PRESS}",
                  'r.plots["scatter_matrix"]'):
        assert piece in text
    with pytest.raises(ValueError, match="brief"):
        Regression("Salary ~ Years", data=d, Rmd=str(out),
                   brief=True)
