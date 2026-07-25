import pytest

from lessPy import pivot, read_data

# expectations verified against lessR pivot() on the Employee data


@pytest.fixture
def d():
    return read_data("Employee")


def test_agg_mean_by_one(d):
    p = pivot(d, "mean", "Salary", by="Dept")
    assert list(p.columns) == ["Dept", "n", "na", "Salary_mean"]
    row = p.set_index("Dept").loc["ADMN"]
    assert row["n"] == 6
    assert round(row["Salary_mean"], 2) == 91277.12
    assert p["Dept"].iloc[0] == "ACCT"      # sorted, NA last


def test_agg_two_stats_two_by(d):
    p = pivot(d, ["mean", "sd"], "Salary", by=["Dept", "Gender"])
    assert list(p.columns) == ["Dept", "Gender", "n", "na",
                               "Salary_mean", "Salary_sd"]
    assert len(p) == 12                     # all combos incl NA
    # first var (Dept) varies fastest: all M, then all W
    assert list(p["Gender"])[:6] == ["M"] * 6
    r = p[(p.Dept == "SALE") & (p.Gender == "M")].iloc[0]
    assert round(r["Salary_mean"], 2) == 96150.97
    assert round(r["Salary_sd"], 2) == 23459.65


def test_show_n_and_sort(d):
    assert list(pivot(d, "mean", "Salary", by="Dept",
                      show_n=False).columns) == ["Dept",
                                                 "Salary_mean"]
    p = pivot(d, "mean", "Salary", by="Dept", sort="-")
    assert p["Salary_mean"].is_monotonic_decreasing


def test_table_one_and_two(d):
    t1 = pivot(d, "table", by="Dept")
    assert list(t1.columns) == ["Dept", "n", "Prop"]
    assert t1.set_index("Dept").loc["SALE", "n"] == 15
    assert t1.set_index("Dept").loc["SALE", "Prop"] == 0.41
    t2 = pivot(d, "table", by=["Dept", "Gender"])
    assert list(t2.columns) == ["Gender", "Dept", "n"]
    assert len(t2) == 12


def test_errors(d):
    with pytest.raises(TypeError, match="numeric"):
        pivot(d, "mean", "Dept", by="Gender")
    with pytest.raises(ValueError, match="unknown compute"):
        pivot(d, "nope", "Salary", by="Dept")
    with pytest.raises(ValueError, match="sort"):
        pivot(d, "mean", "Salary", by="Dept", sort="x")
