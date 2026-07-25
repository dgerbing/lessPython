import numpy as np
import pandas as pd
import pytest

from lessPy import ANOVA


# every hard-coded expectation below was verified against lessR
# ANOVA() on the same frame (July 2026)

@pytest.fixture
def d1():                                   # one-way, seed 4
    rng = np.random.default_rng(4)
    n = 60
    g = rng.choice(["Low", "Med", "High"], n)
    base = {"Low": 50, "Med": 58, "High": 65}
    df = pd.DataFrame({"Group": g})
    df["Score"] = (np.array([base[k] for k in g])
                   + rng.normal(0, 8, n)).round(2)
    return df


@pytest.fixture
def d2():                                   # two-way, seed 8
    rng = np.random.default_rng(8)
    n = 72
    A = rng.choice(["a1", "a2", "a3"], n)
    B = rng.choice(["b1", "b2"], n)
    base = {"a1": 50, "a2": 58, "a3": 64}
    badd = {"b1": 0, "b2": 6}
    y = np.array([base[a] + badd[b] for a, b in zip(A, B)])
    return pd.DataFrame({"A": A, "B": B,
                         "Y": (y + rng.normal(0, 7, n)).round(2)})


@pytest.fixture
def db():                                   # blocks, seed 15
    rng = np.random.default_rng(15)
    treats = ["T1", "T2", "T3", "T4"]
    teff = {"T1": 0, "T2": 4, "T3": 7, "T4": 3}
    rows = []
    for i in range(1, 11):
        beff = rng.normal(0, 6)
        for t in treats:
            rows.append({"Treat": t, "Block": f"B{i}",
                         "Y": round(50 + teff[t] + beff
                                    + rng.normal(0, 3), 2)})
    return pd.DataFrame(rows)


def test_oneway_match_R(d1, capsys):
    r = ANOVA("Score ~ Group", data=d1)
    capsys.readouterr()
    assert r.design == "oneway"
    a = r.anova
    assert a.loc["Group", "Sum Sq"] == pytest.approx(3210.643,
                                                     abs=0.001)
    assert a.loc["Group", "F-value"] == pytest.approx(25.972,
                                                      abs=0.001)
    assert a.loc["Residuals", "df"] == 57
    e = r.effects
    assert e["R_squared"] == pytest.approx(0.4768, abs=0.0005)
    assert e["omega_squared"] == pytest.approx(0.4543, abs=0.0005)
    assert e["cohen_f"] == pytest.approx(0.9124, abs=0.0005)
    tk = r.tukey.loc["Low-High"]
    assert tk["diff"] == pytest.approx(-18.5785, abs=0.001)
    assert tk["p adj"] == pytest.approx(0.0, abs=0.001)
    assert "means" in r.plots


def test_twoway_type2_and_effects(d2, capsys):
    r = ANOVA("Y ~ A * B", data=d2)
    capsys.readouterr()
    assert r.design == "two-between"
    a = r.anova
    # unbalanced -> Type II sums of squares
    assert a.loc["A", "Sum Sq"] == pytest.approx(1172.484,
                                                 abs=0.01)
    assert a.loc["A", "F-value"] == pytest.approx(12.640,
                                                  abs=0.001)
    assert a.loc["A:B", "F-value"] == pytest.approx(0.232,
                                                    abs=0.001)
    # effect sizes from the Type I (sequential) F-values
    assert r.effects["omega_sq_A"] == pytest.approx(0.249,
                                                    abs=0.001)
    assert r.effects["omega_sq_B"] == pytest.approx(0.092,
                                                    abs=0.001)
    # Factor B Tukey uses model.tables sequential marginal means
    assert r.tukey["B"].loc["b2-b1", "diff"] == \
        pytest.approx(4.396, abs=0.001)
    assert r.tukey["A"].loc["a2-a1", "diff"] == \
        pytest.approx(5.876, abs=0.001)
    assert "cells" in r.tukey
    assert "interaction" in r.plots


def test_blocks_match_R(db, capsys):
    r = ANOVA("Y ~ Treat + Block", data=db)
    capsys.readouterr()
    assert r.design == "blocks"
    a = r.anova
    assert a.loc["Treat", "F-value"] == pytest.approx(18.793,
                                                      abs=0.001)
    assert a.loc["Block", "F-value"] == pytest.approx(39.294,
                                                      abs=0.001)
    assert r.effects["omega_sq_Treat"] == pytest.approx(
        0.572, abs=0.001)
    assert r.effects["intraclass_Block"] == pytest.approx(
        0.905, abs=0.001)
    # blocks: Tukey for the factor of interest only
    assert "Treat" in r.tukey and "Block" not in r.tukey
    assert set(r.plots) == {"data", "fitted"}


def test_errors(d1):
    with pytest.raises(ValueError, match="one or two factors"):
        ANOVA("Score ~ A * B * C", data=d1)
    with pytest.raises(TypeError, match="numeric"):
        ANOVA("Group ~ Score", data=d1)
    with pytest.raises(ValueError, match="res_sort"):
        ANOVA("Score ~ Group", data=d1, res_sort="bad")


def test_brief_trims(d1, capsys):
    ANOVA("Score ~ Group", data=d1, brief=True)
    out = capsys.readouterr().out
    assert "ANOVA" in out
    assert "TUKEY" in out            # heading stays as a signpost
    assert "RESIDUALS" not in out


def test_rmd_reports(d1, d2, db, tmp_path, capsys):
    for name, formula, data, plot in (
            ("a1", "Score ~ Group", d1, 'r.plots["means"]'),
            ("a2", "Y ~ A * B", d2, 'r.plots["interaction"]'),
            ("ab", "Y ~ Treat + Block", db, 'r.plots["data"]')):
        out = tmp_path / name
        ANOVA(formula, data=data, Rmd=str(out), Rmd_format="none")
        capsys.readouterr()
        text = out.with_suffix(".qmd").read_text()
        for piece in ("## The Data", "### Summary Table",
                      "### Effects", "### Pairwise Differences",
                      "r.anova", "r.residuals", plot):
            assert piece in text
    with pytest.raises(ValueError, match="brief"):
        ANOVA("Score ~ Group", data=d1, Rmd=str(tmp_path / "x"),
              brief=True)
