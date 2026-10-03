import pytest

from lessPy import Chart, read_data


@pytest.fixture
def emp():
    return read_data("Employee")


def test_stats_counts_match_R(emp):
    # values as lessR's Chart() returns them (freq, prop, p_value,
    # n_miss), kept even when quiet suppresses the report
    st = Chart("Dept", data=emp, quiet=True).stats
    assert st.freq.tolist() == [5, 6, 4, 6, 15]
    assert st.p_value == pytest.approx(0.02719553, abs=1e-8)
    assert st.n_miss == 1
    assert st["prop"].sum() == pytest.approx(1.0)
    assert "Frequencies:" in st.text


def test_stats_crosstab(emp):
    st = Chart("Dept", by="Gender", data=emp, quiet=True).stats
    assert st.n_dim == 2
    assert st.p_value == pytest.approx(0.1847017, abs=1e-7)
    assert st.cramer_v == pytest.approx(0.415, abs=5e-4)
    assert st.prop_row.sum(axis=1).tolist() == pytest.approx([1, 1])


def test_stats_values_not_shadowed(emp):
    # an entry named values reads as the plotted values, not a method
    st = Chart("Dept", y="Salary", stat="mean", data=emp,
               quiet=True).stats
    assert round(st.values["ACCT"], 2) == 71792.78
    assert list(st.summary.columns) == ["n", "miss", "mean", "sd",
                                        "min", "mdn", "max"]


def test_stats_survive_layout_update_not_json(emp):
    fig = Chart("Dept", data=emp, quiet=True)
    fig.update_layout(title="x")
    assert fig.stats.freq.sum() == 36
    assert "stats" not in fig.to_dict()


def test_stats_other_forms(emp):
    st = Chart("Dept", y="Salary", stat="mean", by="Gender",
               form="dot", data=emp, quiet=True).stats
    assert len(st.diff) == 5                     # W - M per Dept
    st = Chart(["Gender", "Dept"], one_plot=False, data=emp,
               quiet=True).stats
    assert set(st.variables) == {"Gender", "Dept"}
    st = Chart("Dept", by=["Gender", "Plan"], form="treemap",
               data=emp, quiet=True).stats
    assert st.freq.sum() == 36


# ----- X() and XY() -------------------------------------------------

def test_x_stats_match_R(emp, capsys):
    from lessPy import X
    st = X("Salary", data=emp, quiet=True).stats
    assert capsys.readouterr().out == ""        # quiet shows nothing
    assert round(st.mean, 3) == 83795.557            # R's values
    assert round(st.sd, 3) == 21799.533
    assert st.bin_width == 10000 and st.n_bins == 10
    assert st.counts == [4, 8, 8, 5, 3, 5, 1, 1, 1, 1]
    assert "--- Salary ---" in st.text               # report kept
    st = X("Salary", form="density", data=emp, quiet=True).stats
    assert round(st.bw, 4) == 9529.0447


def test_xy_stats_match_R(emp):
    from lessPy import XY
    st = XY("Years", "Salary", fit="lm", data=emp, quiet=True).stats
    assert round(st.r, 3) == 0.852
    assert round(st.fit["rsq"], 3) == 0.726
    # error variance on n - 2 df, as R's .plt.fit()
    assert round(st.fit["mse"], 3) == 134129397.124
    assert round(st.fit["b0"], 3) == 52710.898       # R's line
    assert round(st.fit["b1"], 3) == 3249.552
    st = XY("Years", "Salary", by="Gender", data=emp, quiet=True).stats
    assert set(st.cor) == {"M", "W"}


def test_advisory_shown_when_quiet(emp, capsys):
    # an advisory is not part of the report, so quiet does not hide it
    from lessPy import X
    X("Salary", data=emp, facet=["Dept", "Gender", "Dept"], quiet=True)
    assert "uses the first two" in capsys.readouterr().out
