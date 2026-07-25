import numpy as np
import pandas as pd
import pytest

from lessPy import Chart


@pytest.fixture
def d():
    rng = np.random.default_rng(7)
    n = 60
    return pd.DataFrame({
        "Dept": rng.choice(["ACCT", "ADMN", "FINC", "MKTG"], n),
        "Gender": rng.choice(["F", "M"], n),
        "Salary": rng.normal(60000, 12000, n).round(2),
    })


def test_counts_1d(d):
    fig = Chart("Dept", data=d)
    assert len(fig.data) == 1
    got = dict(zip(fig.data[0].x, fig.data[0].y))
    want = d["Dept"].value_counts().to_dict()
    assert got == want
    assert fig.layout.yaxis.title.text == "Count of Dept"
    # alphabetical category order, as with an R factor
    assert list(fig.data[0].x) == sorted(d["Dept"].unique())


def test_counts_sort_descending(d):
    fig = Chart("Dept", data=d, sort="-")
    ys = list(fig.data[0].y)
    assert ys == sorted(ys, reverse=True)


def test_counts_by(d):
    fig = Chart("Dept", by="Gender", data=d)
    assert len(fig.data) == 2
    assert fig.layout.barmode == "stack"
    assert fig.data[0].name == "F"
    fig2 = Chart("Dept", by="Gender", data=d, beside=True)
    assert fig2.layout.barmode == "group"


def test_proportion(d):
    fig = Chart("Dept", data=d, stat_x="proportion")
    assert abs(sum(fig.data[0].y) - 1.0) < 1e-9
    assert fig.layout.yaxis.title.text == "Proportion of Dept"


def test_stat_mean(d):
    fig = Chart("Dept", y="Salary", stat="mean", data=d)
    want = d.groupby("Dept")["Salary"].mean()
    got = dict(zip(fig.data[0].x, fig.data[0].y))
    assert got == pytest.approx(want.to_dict())
    assert fig.layout.yaxis.title.text == "Mean of Salary"
    # explicit stat labels: input values, no % hover
    assert "% of total" not in fig.data[0].hovertemplate


def test_stat_mean_by(d):
    fig = Chart("Dept", by="Gender", y="Salary", stat="mean",
                data=d)
    assert len(fig.data) == 2
    want = (d[d["Gender"] == "F"]
            .groupby("Dept")["Salary"].mean())
    got = dict(zip(fig.data[0].x, fig.data[0].y))
    assert got == pytest.approx(want.to_dict())


def test_deviation_negative_ticks(d):
    fig = Chart("Dept", y="Salary", stat="deviation", data=d)
    means = d.groupby("Dept")["Salary"].mean()
    dev = means - means.mean()
    got = dict(zip(fig.data[0].x, fig.data[0].y))
    assert got == pytest.approx(dev.to_dict())
    # axis must extend below zero for the negative deviations
    assert min(fig.layout.yaxis.tickvals) < 0


def test_filter(d):
    fig = Chart("Dept", data=d, filter="Salary > 60000")
    assert sum(fig.data[0].y) == (d["Salary"] > 60000).sum()


def test_preaggregated_y(d):
    pivot = d.groupby("Dept", as_index=False)["Salary"].mean()
    fig = Chart("Dept", y="Salary", data=pivot)
    assert len(fig.data[0].x) == pivot.shape[0]


def test_errors(d):
    with pytest.raises(ValueError, match="stat"):
        Chart("Dept", y="Salary", data=d)      # raw y, no stat
    pivot = d.groupby("Dept", as_index=False)["Salary"].mean()
    with pytest.raises(ValueError, match="summary table"):
        Chart("Dept", y="Salary", stat="mean", data=pivot)
    with pytest.raises(ValueError, match="deviation"):
        Chart("Dept", by="Gender", y="Salary", stat="deviation",
              data=d)
    with pytest.raises(KeyError, match="not a column"):
        Chart("Dpet", data=d)
    with pytest.raises(ValueError, match="form"):
        Chart("Dept", data=d, form="histogram")
    with pytest.raises(ValueError, match="data="):
        Chart("Dept")


def test_categorical_dtype_order(d):
    d2 = d.copy()
    d2["Dept"] = pd.Categorical(
        d2["Dept"], categories=["MKTG", "FINC", "ADMN", "ACCT"])
    fig = Chart("Dept", data=d2)
    assert list(fig.data[0].x) == ["MKTG", "FINC", "ADMN", "ACCT"]


# ----- facet= ------------------------------------------------------
# larger n so every (x, by, facet) cell is occupied (radar requires
# no empty cells)

@pytest.fixture
def dfac():
    rng = np.random.default_rng(7)
    n = 300
    return pd.DataFrame({
        "Dept": rng.choice(["ACCT", "ADMN", "FINC", "MKTG"], n),
        "Gender": rng.choice(["F", "M"], n),
        "Site": rng.choice(["East", "West"], n),
        "Salary": rng.normal(60000, 12000, n).round(2),
    })


def test_facet_bar_trellis(dfac):
    fig = Chart("Dept", facet="Gender", data=dfac)
    assert len(fig.data) == 2          # one trace per panel
    want = pd.crosstab(dfac["Gender"], dfac["Dept"])
    got = dict(zip(fig.data[0].y, fig.data[0].x))  # horizontal bars
    assert got == want.loc["F"].to_dict()
    txts = " ".join(a.text or "" for a in fig.layout.annotations)
    assert "F" in txts and "M" in txts


def test_facet_bar_two_vars(dfac):
    # two facet variables flatten to one " / " interaction grid
    fig = Chart("Dept", facet=["Gender", "Site"], data=dfac)
    assert len(fig.data) == 4
    txts = " ".join(a.text or "" for a in fig.layout.annotations)
    assert "F / East" in txts


def test_facet_bar_proportion(dfac):
    fig = Chart("Dept", facet="Gender", data=dfac,
                stat_x="proportion")
    for tr in fig.data:                # each panel sums to 1
        assert sum(tr.x) == pytest.approx(1.0)


def test_facet_radar_counts(dfac):
    fig = Chart("Dept", facet="Site", data=dfac, form="radar")
    assert len(fig.data) == 2
    m = dfac["Site"] == "East"
    want = (dfac.loc[m, "Dept"].value_counts().sort_index()
            .tolist())
    assert list(fig.data[0].r)[:4] == want


def test_facet_radar_by(dfac):
    fig = Chart("Dept", by="Gender", facet="Site", data=dfac,
                form="radar")
    assert len(fig.data) == 4          # 2 by-groups x 2 panels


def test_facet_radar_stat(dfac):
    fig = Chart("Dept", y="Salary", stat="mean", facet="Gender",
                data=dfac, form="radar")
    m = dfac["Gender"] == "F"
    want = (dfac[m].groupby("Dept")["Salary"].mean().sort_index()
            .to_numpy())
    assert list(fig.data[0].r)[:4] == pytest.approx(want)


def test_facet_bubble(dfac):
    fig = Chart("Dept", facet="Gender", data=dfac, form="bubble")
    assert len(fig.data) == 2
    m = dfac["Gender"] == "F"
    want = (dfac.loc[m, "Dept"].value_counts(normalize=True)
            .sort_index())
    got = [cd["pct_x"] for cd in fig.data[0].customdata]
    assert got == pytest.approx(want.tolist())


def test_facet_dot_stat_sort(dfac):
    fig = Chart("Dept", y="Salary", stat="mean", facet="Gender",
                data=dfac, form="dot", sort="-")
    # marker traces hold the panel values (lines traces are the
    # dropped segments); values descend within each panel
    marks = [tr for tr in fig.data if tr.mode == "markers"]
    assert len(marks) == 2
    for tr in marks:
        ys = list(tr.y)
        assert ys == sorted(ys, reverse=True)


def test_facet_hier(dfac):
    for form in ("treemap", "icicle"):
        fig = Chart("Dept", facet="Gender", data=dfac, form=form)
        assert len(fig.data) == 2      # one trace per panel
    # pie with by= is a sunburst; facet= panels it
    fig = Chart("Dept", by="Gender", facet="Site", data=dfac,
                form="pie")
    assert len(fig.data) == 2


def test_facet_pie_grid(dfac):
    # facet= on a plain pie: the facet becomes the grouping and
    # the pie grid renders, one pie per level (as in R)
    fig = Chart("Dept", facet="Gender", data=dfac, form="pie")
    assert len(fig.data) == 2
    assert all(tr.type == "pie" for tr in fig.data)


def test_facet_errors(dfac):
    with pytest.raises(ValueError, match="Sort"):
        Chart("Dept", facet="Gender", data=dfac, sort="-")
    with pytest.raises(ValueError, match="Trellis"):
        Chart("Dept", y="Salary", stat="mean", facet="Gender",
              data=dfac)
    with pytest.raises(ValueError, match="facet"):
        Chart("Dept", by="Site", facet="Gender", data=dfac)
    with pytest.raises(ValueError, match="single"):
        Chart("Dept", by="Gender", facet="Site", data=dfac,
              form="bubble")
    with pytest.raises(ValueError, match="Multiple y"):
        Chart("Dept", y=["Salary", "Salary"], facet="Gender",
              data=dfac, form="dot")
    with pytest.raises(ValueError, match="deviation"):
        Chart("Dept", y="Salary", stat="deviation",
              facet="Gender", data=dfac, form="dot")


# ----- add= annotations -----------------------------------------

def test_add_bar_hline():
    d = pd.DataFrame({"Dept": ["A", "A", "B", "B", "B", "C"]})
    fig = Chart("Dept", data=d, add="h_line", y1=2, quiet=True)
    hl = [s for s in fig.layout.shapes
          if s.xref == "paper" and s.y0 == 2
          and s.layer != "below"]
    assert len(hl) == 1


def test_add_bar_only():
    d = pd.DataFrame({"Dept": ["A", "B", "C"]})
    with pytest.raises(ValueError, match="bar"):
        Chart("Dept", data=d, form="pie", add="h_line", y1=1)


# ----- axis_fmt / prefixes / rotate_y ---------------------------

def test_axis_fmt_K_default(d):
    # value axis renders thousands as "60K" by default, as in R
    fig = Chart("Dept", y="Salary", stat="mean", data=d, quiet=True)
    txt = list(fig.layout.yaxis.ticktext)
    assert any(t.endswith("K") for t in txt)
    assert "60000" not in txt


def test_axis_fmt_comma_and_prefix(d):
    fig = Chart("Dept", y="Salary", stat="mean", data=d,
                axis_fmt=",", axis_y_pre="$", quiet=True)
    txt = list(fig.layout.yaxis.ticktext)
    assert any("," in t for t in txt)
    assert all(t.startswith("$") for t in txt)


def test_rotate_y_value_axis(d):
    fig = Chart("Dept", y="Salary", stat="mean", data=d,
                rotate_y=45, quiet=True)
    assert fig.layout.yaxis.tickangle == -45


# ----- auto labels_position -------------------------------------

def test_labels_position_auto(d):
    # default: plotly places each label in/out per bar (auto)
    fig = Chart("Dept", data=d)
    assert fig.data[0].textposition == "auto"


def test_labels_position_out(d):
    fig = Chart("Dept", data=d, labels_position="out")
    assert fig.data[0].textposition == "outside"


def test_labels_position_bad(d):
    with pytest.raises(ValueError, match="labels_position"):
        Chart("Dept", data=d, labels_position="middle")


def test_labels_position_out_stacked_rejected(d):
    # out is meaningless for a stacked (by=, not beside) bar, as in R
    with pytest.raises(ValueError, match="stacked"):
        Chart("Dept", by="Gender", data=d, labels_position="out")
    # but fine for grouped (beside) bars
    fig = Chart("Dept", by="Gender", data=d, beside=True,
                labels_position="out")
    assert fig.data[0].textposition == "outside"


# ----- stat_x="proportion" with facet= --------------------------

def test_facet_proportion_dot(dfac):
    fig = Chart("Dept", facet="Gender", data=dfac, form="dot",
                stat_x="proportion", quiet=True)
    assert fig.layout.yaxis.title.text == "Proportion of Dept"
    for tr in fig.data:
        if tr.mode == "markers":
            vals = [v for v in tr.y if v is not None]
            assert sum(vals) == pytest.approx(1.0)


def test_facet_proportion_radar(dfac):
    fig = Chart("Dept", facet="Gender", data=dfac, form="radar",
                stat_x="proportion", quiet=True)
    # each polygon's open vertices (drop the closing repeat) sum to 1
    assert sum(list(fig.data[0].r)[:4]) == pytest.approx(1.0)


def test_facet_proportion_bubble(dfac):
    fig = Chart("Dept", facet="Gender", data=dfac, form="bubble",
                stat_x="proportion", quiet=True)
    # first panel's displayed values are proportions summing to 1
    # (hovertext is rounded to digits_d=2, so allow for rounding)
    vals = [float(t) for t in fig.data[0].hovertext]
    assert sum(vals) == pytest.approx(1.0, abs=0.02)


def test_facet_proportion_rejects_y(dfac):
    with pytest.raises(ValueError, match="proportion"):
        Chart("Dept", y="Salary", facet="Gender", data=dfac,
              form="dot", stat_x="proportion")


# ----- n_row / n_col facet layout -------------------------------

def _y_domain_starts(fig):
    j = fig.layout.to_plotly_json()
    return {round(j[k]["domain"][0], 3)
            for k in j if k.startswith("yaxis")}


def test_n_col_single_row(dfac):
    # three panels in one row -> one distinct y-domain band
    fig = Chart("Dept", facet="Gender", data=dfac, form="dot",
                n_col=2, quiet=True)
    assert len(_y_domain_starts(fig)) == 1


def test_n_row_stacks(dfac):
    fig = Chart("Dept", facet="Gender", data=dfac, form="dot",
                n_row=2, quiet=True)
    assert len(_y_domain_starts(fig)) == 2


def test_n_col_requires_facet(d):
    with pytest.raises(ValueError, match="facet"):
        Chart("Dept", data=d, n_col=2)


def test_n_col_bar_trellis(dfac):
    # Trellis bar honors n_col: two panels side by side, one row
    fig = Chart("Dept", facet=["Gender", "Site"], data=dfac,
                n_col=2, quiet=True)
    assert len(_y_domain_starts(fig)) == 2   # 4 panels, 2 cols


def test_facet_series_input(dfac):
    # computed facet values: a Series aligned with data, the
    # Python analog of R's facet expression
    lvl = (dfac["Dept"] == "ACCT").map(
        {True: "acct", False: "other"}).rename("IsAcct")
    fig = Chart("Dept", facet=lvl, data=dfac, quiet=True)
    assert fig is not None
    with pytest.raises(ValueError, match="values"):
        Chart("Dept", facet=lvl.to_numpy()[:5], data=dfac,
              quiet=True)


def test_font_size_scales_text(d):
    base = Chart("Dept", data=d)
    big = Chart("Dept", font_size=1.4, data=d)
    bx = list(base.select_xaxes())[0]
    gx = list(big.select_xaxes())[0]
    assert gx.tickfont.size > bx.tickfont.size


def test_fill_split_two_colors(d):
    # a deviation bar chart split at 0: below -> one color, above
    # -> another; exactly two distinct fills, split matching sign
    fig = Chart("Dept", y="Salary", stat="deviation", sort="+",
                fill_split=0, data=d)
    bar = [t for t in fig.data if t.type == "bar"][0]
    cols = list(bar.marker.color)
    vals = list(bar.y) if bar.orientation != "h" else list(bar.x)
    assert len(set(cols)) == 2
    below = {c for c, v in zip(cols, vals) if v <= 0}
    above = {c for c, v in zip(cols, vals) if v > 0}
    assert below and above and below.isdisjoint(above)


def test_fill_scaled_gradient(d):
    # fill_scaled maps values to a per-bar gradient (many colors),
    # unlike a plain uniform fill
    fig = Chart("Dept", y="Salary", stat="mean", sort="+",
                fill_scaled=True, fill=["black", "red"], data=d)
    bar = [t for t in fig.data if t.type == "bar"][0]
    assert len(set(bar.marker.color)) > 2


def test_fill_scaled_rejects_by(d):
    import pytest
    with pytest.raises(ValueError, match="fill_scaled"):
        Chart("Dept", by="Gender", y="Salary", stat="mean",
              fill_scaled=True, data=d)


def test_theme_sets_palette(d):
    # theme= colors the bars with the theme's HCL palette family
    fig = Chart("Dept", theme="green", data=d)
    bar = [t for t in fig.data if t.type == "bar"][0]
    cols = list(bar.marker.color)
    # a sequential green palette: several distinct greenish fills
    assert len(set(cols)) == len(d["Dept"].unique())
    # greens: green channel dominant in the light end
    r, g, b = int(cols[0][1:3], 16), int(cols[0][3:5], 16), \
        int(cols[0][5:7], 16)
    assert g > r and g > b


def test_theme_unknown_rejected(d):
    import pytest
    with pytest.raises(ValueError, match="unknown theme"):
        Chart("Dept", theme="chartreuse", data=d)


# ----- stack100 / gap / scale_y / break_x ------------------------


def test_stack100_bars_sum_to_one(d):
    # R analog: prop.table(table(by, x), 2) -- within each column
    fig = Chart("Dept", by="Gender", stack100=True, data=d)
    tot = np.zeros(len(fig.data[0].y))
    for t in fig.data:
        tot = tot + np.asarray(t.y, dtype=float)
    assert np.allclose(tot, 1.0)
    want = pd.crosstab(d["Gender"], d["Dept"])
    want = want / want.sum(axis=0)
    for t in fig.data:
        assert np.allclose(t.y, want.loc[t.name].to_numpy())


def test_stack100_labels_show_counts(d):
    # R: labels="input" over proportions displays x.count
    fig = Chart("Dept", by="Gender", stack100=True, data=d)
    cnt = pd.crosstab(d["Gender"], d["Dept"])
    for t in fig.data:
        assert list(t.text) == [str(v) for v in cnt.loc[t.name]]
    assert (fig.layout.yaxis.title.text
            == "Cell % within Dept by Gender")


def test_stack100_percent_labels_are_within_x(d):
    fig = Chart("Dept", by="Gender", stack100=True, labels="%",
                data=d)
    for t in fig.data:
        want = [f"{100 * v:.0f}%" for v in t.y]
        assert list(t.text) == want


def test_stack100_beside_labels_axis(d):
    fig = Chart("Dept", by="Gender", stack100=True, beside=True,
                data=d)
    assert fig.layout.yaxis.title.text == "Percentage"


def test_stack100_without_by_is_proportion(d):
    fig = Chart("Dept", stack100=True, data=d)
    assert np.isclose(np.sum(fig.data[0].y), 1.0)
    assert fig.layout.yaxis.title.text == "Proportion of Dept"


def test_stack100_rejects_non_bar_and_facet(d):
    with pytest.raises(ValueError, match="bar"):
        Chart("Dept", by="Gender", form="pie", stack100=True,
              data=d)
    with pytest.raises(ValueError, match="by"):
        Chart("Dept", facet="Gender", stack100=True, data=d)


def test_scale_y_gives_n_intervals_plus_one_ticks(d):
    # R analog: axTicks(2, axp=scale_y)
    fig = Chart("Dept", scale_y=(0, 20, 4), data=d)
    assert list(fig.layout.yaxis.tickvals) == [0, 5, 10, 15, 20]
    assert tuple(fig.layout.yaxis.range) == (0.0, 20.0)


def test_scale_y_applies_to_value_axis_when_horiz(d):
    fig = Chart("Dept", scale_y=(0, 20, 4), horiz=True, data=d)
    assert tuple(fig.layout.xaxis.range) == (0.0, 20.0)
    assert fig.layout.yaxis.range is None


def test_scale_y_validated(d):
    with pytest.raises(ValueError, match="three values"):
        Chart("Dept", scale_y=(0, 20), data=d)
    with pytest.raises(ValueError, match="exceed"):
        Chart("Dept", scale_y=(20, 0, 4), data=d)


def test_gap_default_leaves_plotly_alone(d):
    fig = Chart("Dept", data=d)
    assert fig.layout.bargap is None
    assert fig.layout.bargroupgap is None


def test_gap_converts_bar_widths_to_fractions(d):
    # R's space=g is g bar widths, so g / (1 + g) of the slot
    fig = Chart("Dept", gap=1.0, data=d)
    assert np.isclose(fig.layout.bargap, 0.5)


def test_gap_pair_with_beside(d):
    fig = Chart("Dept", by="Gender", beside=True, gap=(0.1, 1.0),
                data=d)
    k = 2                       # two by groups
    within, between = 0.1, 1.0
    slot = k + (k - 1) * within + between
    assert np.isclose(fig.layout.bargap, between / slot)
    assert np.isclose(fig.layout.bargroupgap,
                      (k - 1) * within / (k + (k - 1) * within))


@pytest.fixture
def dwide(d):
    x = d.copy()
    x["Unit"] = x["Dept"].map({
        "ACCT": "Corporate Accounting",
        "ADMN": "General Administration",
        "FINC": "Finance",
        "MKTG": "Field Sales~Group"})
    return x


def test_break_x_wraps_labels_by_default(dwide):
    fig = Chart("Unit", data=dwide)
    assert "Corporate<br>Accounting" in fig.layout.xaxis.ticktext
    # "~" is the non-breaking space
    assert "Field<br>Sales Group" in fig.layout.xaxis.ticktext


def test_break_x_off_for_horiz_or_rotated(dwide):
    for kw in (dict(horiz=True), dict(rotate_x=45)):
        fig = Chart("Unit", data=dwide, **kw)
        ax = (fig.layout.yaxis if kw.get("horiz")
              else fig.layout.xaxis)
        assert "Corporate Accounting" in ax.ticktext
    fig = Chart("Unit", break_x=False, data=dwide)
    assert "Corporate Accounting" in fig.layout.xaxis.ticktext



# ----- parameter order ------------------------------------------


def test_second_positional_is_y(d):
    # R's order: Chart(x, y, data, filter, by, facet)
    pos = Chart("Dept", "Salary", stat="mean", data=d)
    kw = Chart("Dept", y="Salary", stat="mean", data=d)
    assert list(pos.data[0].y) == list(kw.data[0].y)
    assert pos.layout.yaxis.title.text == "Mean of Salary"


def test_signature_order_matches_r():
    import inspect
    names = list(inspect.signature(Chart).parameters)[:6]
    assert names == ["x", "y", "data", "filter", "by", "facet"]


def test_categorical_y_points_at_by(d):
    # the misbinding the R order invites: a second categorical
    # variable passed positionally lands on y, not by
    with pytest.raises(ValueError, match="by='Gender'"):
        Chart("Dept", "Gender", data=d)


# ----- labels_decimals ------------------------------------------


def test_labels_decimals_bar(d):
    fig = Chart("Dept", labels="prop", labels_decimals=3, data=d)
    assert all(len(t.split(".")[1]) == 3 for t in fig.data[0].text)
    fig = Chart("Dept", labels="%", labels_decimals=1, data=d)
    assert all(t.endswith("%") and len(t.split(".")[1]) == 2
               for t in fig.data[0].text)          # "13.9%"


def test_labels_decimals_pie_and_bubble(d):
    pie = Chart("Dept", form="pie", labels="prop",
                labels_decimals=3, data=d)
    assert all(len(t.split("<br>")[1].split(".")[1]) == 3
               for t in pie.data[0].text)
    bub = Chart("Dept", form="bubble", labels="prop",
                labels_decimals=3, data=d)
    assert all(len(t.split(".")[1]) == 3 for t in bub.data[0].text)


def test_prop_labels_are_not_flattened_to_zero(d):
    # digits_d is 0 for counts, which would render every
    # proportion as "0" -- each renderer defaults "prop" to 2
    for kw in (dict(), dict(form="pie"), dict(form="bubble")):
        fig = Chart("Dept", labels="prop", data=d, **kw)
        txt = [t.split("<br>")[-1] for t in fig.data[0].text]
        assert all(t.startswith("0.") for t in txt), (kw, txt)


def test_labels_decimals_defaults_unchanged(d):
    # no labels_decimals: each renderer keeps its own default,
    # matching what R's plotly path has always emitted
    counts = [str(v) for v in
              d["Dept"].value_counts().sort_index().tolist()]
    assert list(Chart("Dept", data=d).data[0].text) == counts
    means = Chart("Dept", "Salary", stat="mean", data=d)
    assert all("." in t for t in means.data[0].text)


# ----- legend_* -------------------------------------------------


def test_legend_title_and_labels(d):
    fig = Chart("Dept", by="Gender", legend_title="Sex",
                legend_labels=["Men", "Women"], data=d)
    assert fig.layout.legend.title.text == "Sex"
    assert [t.name for t in fig.data] == ["Men", "Women"]
    # legendgroup keeps the underlying level, not the new label
    assert [t.legendgroup for t in fig.data] == ["F", "M"]


def test_legend_labels_length_checked(d):
    with pytest.raises(ValueError, match="legend_labels"):
        Chart("Dept", by="Gender", legend_labels=["only one"],
              data=d)


def test_legend_horiz_and_position(d):
    fig = Chart("Dept", by="Gender", legend_horiz=True,
                legend_position="bottom", data=d)
    L = fig.layout.legend
    assert L.orientation == "h"
    assert (L.x, L.y) == (0.50, 0.02)
    assert (L.xanchor, L.yanchor) == ("center", "bottom")


def test_legend_adjust_shifts_x(d):
    base = Chart("Dept", by="Gender", legend_position="topright",
                 data=d).layout.legend.x
    moved = Chart("Dept", by="Gender", legend_position="topright",
                  legend_adjust=0.1, data=d).layout.legend.x
    assert moved == pytest.approx(base + 0.1)


def test_legend_size_scales_text(d):
    small = Chart("Dept", by="Gender", data=d).layout.legend
    big = Chart("Dept", by="Gender", legend_size=2.0,
                data=d).layout.legend
    assert big.font.size > small.font.size
    assert big.title.font.size > small.title.font.size


def test_legend_abbrev_truncates(d):
    fig = Chart("Dept", by="Gender", legend_title="Department",
                legend_labels=["Female", "Male"], legend_abbrev=3,
                data=d)
    assert fig.layout.legend.title.text == "Dep"
    assert [t.name for t in fig.data] == ["Fem", "Mal"]


def test_legend_defaults_leave_placement_to_plotly(d):
    L = Chart("Dept", by="Gender", data=d).layout.legend
    assert L.x is None and L.y is None
    assert L.orientation is None
    assert L.title.text == "Gender"
    # "right_margin" IS plotly's own placement, so also a no-op
    L2 = Chart("Dept", by="Gender", legend_position="right_margin",
               data=d).layout.legend
    assert L2.x is None and L2.y is None


def test_legend_params_need_a_bar_with_by(d):
    with pytest.raises(ValueError, match='form="bar"'):
        Chart("Dept", by="Gender", form="pie", legend_title="Sex",
              data=d)
    with pytest.raises(ValueError, match="by="):
        Chart("Dept", legend_title="Sex", data=d)


def test_legend_position_validated(d):
    with pytest.raises(ValueError, match="legend_position"):
        Chart("Dept", by="Gender", legend_position="nowhere",
              data=d)


def test_faceted_pie_title_says_across(dfac):
    # the pie grid re-uses by= internally to group its panels, but
    # the variable is a facet, so the title must not say "by"
    grid = Chart("Dept", facet="Gender", form="pie", data=dfac)
    assert grid.layout.title.text == "Count of Dept across Gender"
    # a real by= is still "by": that one is a sunburst
    sun = Chart("Dept", by="Gender", form="pie", data=dfac)
    assert sun.layout.title.text == "Count of Dept by Gender"


def test_facet_titles_use_across_for_every_form(dfac):
    for form in ("pie", "treemap", "icicle", "radar", "bubble"):
        ttl = Chart("Dept", facet="Gender", form=form,
                    data=dfac).layout.title.text
        assert ttl == "Count of Dept across Gender", (form, ttl)
    # by= and facet= together name both, each with its own word
    both = Chart("Dept", by="Gender", facet="Site", form="radar",
                 data=dfac).layout.title.text
    assert both == "Count of Dept by Gender across Site"
