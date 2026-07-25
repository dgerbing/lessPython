import numpy as np
import pandas as pd
import pytest

from lessPy import ttest


# every hard-coded expectation below was verified against lessR
# ttest() on the same frame (July 2026)

@pytest.fixture
def dg():                                   # two-group, seed 21
    rng = np.random.default_rng(21)
    n = 45
    g = rng.choice(["M", "F"], n)
    y = np.where(g == "M", 78, 72) + rng.normal(0, 9, n)
    return pd.DataFrame({"Gender": g, "Score": y.round(2)})


@pytest.fixture
def dp():                                   # paired/one, seed 30
    rng = np.random.default_rng(30)
    n = 28
    return pd.DataFrame({"Pre": rng.normal(70, 8, n).round(2),
                         "Post": rng.normal(74, 8, n).round(2)})


def test_two_group_formula_match_R(dg, capsys):
    r = ttest("Score ~ Gender", data=dg, graph=False)
    capsys.readouterr()
    assert r.kind == "two-group"
    # larger-mean group (M) reported first
    assert r.group1["name"] == "M"
    assert r.group1["mean"] == pytest.approx(77.855, abs=0.001)
    assert r.group2["mean"] == pytest.approx(71.601, abs=0.001)
    assert r.mean_diff == pytest.approx(6.253, abs=0.001)
    assert r.pooled_sd == pytest.approx(7.251, abs=0.001)
    assert r.cohen_d == pytest.approx(0.862, abs=0.001)
    ev = r.equal_var
    assert ev["t"] == pytest.approx(2.875, abs=0.001)
    assert ev["df"] == 43
    assert ev["p_value"] == pytest.approx(0.006, abs=0.001)
    assert ev["lb"] == pytest.approx(1.866, abs=0.001)
    assert ev["ub"] == pytest.approx(10.640, abs=0.001)
    assert r.welch["t"] == pytest.approx(2.910, abs=0.001)
    assert r.welch["df"] == pytest.approx(42.392, abs=0.001)


def test_one_group_mu_match_R(dp, capsys):
    r = ttest("Pre", data=dp, mu=72, graph=False)
    capsys.readouterr()
    assert r.kind == "one-group"
    assert r.mean == pytest.approx(71.705, abs=0.001)
    assert r.sd == pytest.approx(7.668, abs=0.001)
    assert r.infer["t"] == pytest.approx(-0.203, abs=0.001)
    assert r.infer["df"] == 27
    assert r.infer["p_value"] == pytest.approx(0.840, abs=0.001)
    assert r.infer["lb"] == pytest.approx(68.732, abs=0.001)
    assert r.cohen_d == pytest.approx(0.038, abs=0.001)


def test_paired_match_R(dp, capsys):
    r = ttest("Post", "Pre", data=dp, paired=True, graph=False)
    capsys.readouterr()
    assert r.kind == "paired"
    assert r.mean == pytest.approx(-2.182, abs=0.002)  # Pre-Post
    assert r.infer["df"] == 27
    assert r.infer["p_value"] == pytest.approx(0.280, abs=0.001)
    assert r.cohen_d == pytest.approx(0.208, abs=0.001)


def test_two_group_stats_match_R(capsys):
    r = ttest(n1=30, m1=52.4, s1=8.1, n2=35, m2=48.2, s2=7.3)
    capsys.readouterr()
    assert r.mean_diff == pytest.approx(4.200, abs=0.001)
    assert r.pooled_sd == pytest.approx(7.679, abs=0.001)
    assert r.equal_var["t"] == pytest.approx(2.198, abs=0.001)
    assert r.equal_var["df"] == 63
    assert r.equal_var["p_value"] == pytest.approx(0.032,
                                                   abs=0.001)
    assert r.cohen_d == pytest.approx(0.547, abs=0.001)


def test_one_group_stats_match_R(capsys):
    r = ttest(n=40, m=52.6, s=9.2, mu=50)
    capsys.readouterr()
    assert r.infer["t"] == pytest.approx(1.787, abs=0.001)
    assert r.infer["df"] == 39
    assert r.infer["p_value"] == pytest.approx(0.082, abs=0.001)
    assert r.cohen_d == pytest.approx(0.283, abs=0.001)


def test_plots(dg, dp, capsys):
    r = ttest("Score ~ Gender", data=dg)
    ro = ttest("Pre", data=dp, mu=72)
    capsys.readouterr()
    assert "two_group" in r.plots
    assert "one_group" in ro.plots


def test_errors(dg):
    with pytest.raises(ValueError, match="alternative"):
        ttest("Score ~ Gender", data=dg, alternative="x")
    with pytest.raises(ValueError, match="two values"):
        ttest("Score ~ Score", data=dg)   # 3+ unique -> ANOVA
    with pytest.raises(TypeError, match="numeric"):
        ttest("Gender ~ Score", data=dg)
