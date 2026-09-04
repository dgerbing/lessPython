import numpy as np
import pandas as pd
import pytest

from lessPy import X
from lessPy.utils import bw_nrd0


@pytest.fixture
def d():
    rng = np.random.default_rng(7)
    n = 120
    return pd.DataFrame({
        "Salary": rng.normal(60000, 12000, n).round(2),
        "Gender": rng.choice(["F", "M"], n),
        "Dept": rng.choice(["ACCT", "MKTG"], n),
    })


def test_histogram_counts(d):
    fig = X("Salary", data=d)
    tr = fig.data[0]
    assert tr.type == "bar"
    assert sum(tr.y) == len(d)               # every case binned
    assert fig.layout.bargap == 0
    assert fig.layout.yaxis.title.text == "Count of Salary"
    # grid lines sit at the bin boundaries
    n_bins = len(tr.x)
    grid_x = [s for s in fig.layout.shapes
              if s.xref == "x" and s.type == "line"]
    assert len(grid_x) == n_bins + 1


def test_histogram_matches_r_binning(d):
    # right-closed bins on pretty (Sturges) edges, as in R hist()
    fig = X("Salary", data=d)
    tr = fig.data[0]
    widths = np.array(tr.width) / 0.98
    edges = [tr.x[0] - widths[0] / 2] + \
            [m + w / 2 for m, w in zip(tr.x, widths)]
    want = (pd.cut(d["Salary"], edges, right=True,
                   include_lowest=True)
            .value_counts(sort=False).tolist())
    assert list(tr.y) == want


def test_histogram_proportion(d):
    fig = X("Salary", data=d, stat="proportion")
    assert sum(fig.data[0].y) == pytest.approx(1.0)
    assert fig.layout.yaxis.title.text == "Proportion of Salary"


def test_histogram_bin_width(d):
    fig = X("Salary", data=d, bin_start=20000, bin_width=10000)
    tr = fig.data[0]
    assert np.allclose(np.array(tr.width) / 0.98, 10000)
    assert tr.x[0] == pytest.approx(25000)   # first mid
    assert sum(tr.y) == len(d)


def test_histogram_by(d):
    fig = X("Salary", by="Gender", data=d)
    assert len(fig.data) == 2
    assert [t.name for t in fig.data] == ["F", "M"]
    assert fig.layout.barmode == "overlay"
    # translucent fills so overlapped bars stay readable
    assert "rgba" in fig.data[0].marker.color
    fig2 = X("Salary", by="Gender", data=d, position="stack")
    assert fig2.layout.barmode == "stack"


def test_density(d):
    fig = X("Salary", data=d, form="density")
    # faint histogram behind the curve (R show_histogram=TRUE)
    assert fig.data[0].type == "bar"
    curve = fig.data[1]
    assert curve.mode == "lines"
    assert curve.fill == "tozeroy"
    # curve integrates to ~1
    area = np.trapezoid(curve.y, curve.x)
    assert area == pytest.approx(1.0, abs=0.01)
    # next trace: vertical line at the mean
    mline = fig.data[2]
    assert mline.x[0] == mline.x[1] == \
        pytest.approx(d["Salary"].mean())
    assert fig.layout.yaxis.title.text == "Density"
    # show_histogram=False: curve first, no bar trace
    f2 = X("Salary", data=d, form="density",
           show_histogram=False)
    assert f2.data[0].type == "scatter"


def test_density_by(d):
    fig = X("Salary", by="Gender", data=d, form="density")
    assert len(fig.data) == 4          # 2 curves + 2 mean lines
    assert fig.layout.legend.title.text == "Gender"


def test_bw_nrd0_matches_r_formula(d):
    x = d["Salary"]
    sd = x.std(ddof=1)
    iqr = x.quantile(0.75) - x.quantile(0.25)
    want = 0.9 * min(sd, iqr / 1.34) * len(x) ** -0.2
    assert bw_nrd0(x) == pytest.approx(want)


def test_stat_density_switches_form(d):
    fig = X("Salary", data=d, stat="density")
    assert fig.data[1].mode == "lines"     # density curve after
    assert fig.layout.yaxis.title.text == "Density"  # the faint
    # background histogram (bar trace drawn first)
    assert fig.data[0].type == "bar"


def test_facet_histogram(d):
    fig = X("Salary", facet="Dept", data=d)
    bars = [t for t in fig.data if t.type == "bar"]
    assert len(bars) == 2
    # shared bins: same mids in both panels
    assert list(bars[0].x) == list(bars[1].x)
    # per-panel counts sum to the panel n; first level (ACCT)
    # draws in the BOTTOM panel, which is the last subplot row
    n_acct = (d["Dept"] == "ACCT").sum()
    bottom = [t for t in bars if t.yaxis == "y2"][0]
    assert sum(bottom.y) == n_acct
    assert sum(sum(t.y) for t in bars) == len(d)
    # strip labels bottom-up
    labels = [a.text for a in fig.layout.annotations]
    assert "ACCT" in labels and "MKTG" in labels
    # both panels share the count scale
    assert (fig.layout.yaxis.range[1]
            == fig.layout.yaxis2.range[1])


def test_facet_histogram_proportion(d):
    fig = X("Salary", facet="Dept", data=d, stat="proportion")
    bars = [t for t in fig.data if t.type == "bar"]
    for t in bars:                    # per-panel proportions
        assert sum(t.y) == pytest.approx(1.0)


def test_facet_density(d):
    fig = X("Salary", facet="Dept", data=d, form="density")
    curves = [t for t in fig.data
              if getattr(t, "mode", None) == "lines"
              and len(t.x) > 100]
    assert len(curves) == 2           # one curve per panel
    assert curves[0].yaxis != curves[1].yaxis
    # common support grid across panels
    assert list(curves[0].x) == list(curves[1].x)


def test_x_errors(d):
    with pytest.raises(TypeError, match="Chart"):
        X("Gender", data=d)                # categorical -> Chart()
    with pytest.raises(ValueError, match="data="):
        X("Salary")


# ----- cumulate / counts / kind / rug ---------------------------

def test_cumulate(d):
    fig = X("Salary", data=d, cumulate="on")
    bars = [t for t in fig.data if t.type == "bar"]
    assert len(bars) == 1
    y = np.asarray(bars[0].y)
    assert (np.diff(y) >= 0).all()     # nondecreasing
    assert y[-1] == len(d)             # last bar: total count
    assert fig.layout.yaxis.title.text.startswith("Cumulative")


def test_cumulate_both(d):
    fig = X("Salary", data=d, cumulate="both")
    bars = [t for t in fig.data if t.type == "bar"]
    assert len(bars) == 2
    y_cum = np.asarray(bars[0].y)
    y_reg = np.asarray(bars[1].y)
    assert np.allclose(np.cumsum(y_reg), y_cum)
    assert bars[1].marker.color == "#EEE9E9"   # reg="snow2"


def test_cumulate_no_by(d):
    with pytest.raises(ValueError, match="cumulate"):
        X("Salary", by="Gender", data=d, cumulate="on")


def test_counts_labels(d):
    fig = X("Salary", data=d, counts=True)
    bar = [t for t in fig.data if t.type == "bar"][0]
    assert bar.text is not None
    assert sum(float(v) for v in bar.text) == len(d)


def test_kind_normal(d):
    fig = X("Salary", data=d, form="density", kind="normal",
            show_histogram=False)
    lines = [t for t in fig.data if t.type == "scatter"
             and t.mode == "lines"]
    assert len(lines) == 2             # normal curve + mean line
    curve = max(lines, key=lambda t: len(t.x))
    area = np.trapezoid(curve.y, curve.x)
    assert area == pytest.approx(1.0, abs=0.02)


def test_kind_both(d):
    fig = X("Salary", data=d, form="density", kind="both",
            show_histogram=False)
    # general curve + mean line + normal curve
    lines = [t for t in fig.data if t.type == "scatter"
             and t.mode == "lines"]
    assert len(lines) == 3


def test_kind_normal_default_fill(d):
    # the normal curve carries R's own fill, X.R density
    # fill_normal = rgb(250,210,230, alpha=80)
    fig = X("Salary", data=d, form="density", kind="normal",
            show_histogram=False)
    curve = max((t for t in fig.data if t.type == "scatter"
                 and t.mode == "lines"), key=lambda t: len(t.x))
    assert curve.fill == "tozeroy"
    assert curve.fillcolor == "rgba(250,210,230,0.314)"

    # "transparent" leaves the curve unfilled
    f2 = X("Salary", data=d, form="density", kind="normal",
           fill_normal="transparent", show_histogram=False)
    c2 = max((t for t in f2.data if t.type == "scatter"
              and t.mode == "lines"), key=lambda t: len(t.x))
    assert c2.fill == "none"


def test_kind_shapiro_console(d, capsys):
    X("Salary", data=d, form="density", kind="both")
    out = capsys.readouterr().out
    assert "Shapiro-Wilk" in out
    assert "normal population" in out


def test_rug_implies_density(d):
    fig = X("Salary", data=d, rug=True)   # form was histogram
    assert fig.layout.yaxis.title.text == "Density"
    # rug trace: 3 points (x, x, None) per observation
    rug_tr = fig.data[-1]
    assert len(rug_tr.x) == 3 * len(d)
    ys = np.asarray([v for v in rug_tr.y if v is not None],
                    dtype=float)
    assert ys.min() < 0                # ticks below the axis


def test_rug_no_by(d):
    with pytest.raises(ValueError, match="no by="):
        X("Salary", by="Gender", data=d, rug=True)


def _curves(fig):
    """The density curves: line traces on the full grid, so the
    two-point mean lines are left out."""
    return [t for t in fig.data if t.type == "scatter"
            and t.mode == "lines" and len(t.x) > 2]


def test_kind_by_group_normals(d):
    # one normal per by group, each fit to that group, drawn dashed
    # in the group's color. R analog: dn.plotly.R kind with groups
    fig = X("Salary", by="Gender", data=d, form="density",
            kind="both")
    curves = _curves(fig)
    assert len(curves) == 4            # 2 general + 2 normal
    dashed = [t for t in curves if t.line.dash == "dash"]
    assert len(dashed) == 2
    assert {t.name for t in dashed} == {"F normal", "M normal"}

    for g in ("F", "M"):
        sub = d.loc[d["Gender"] == g, "Salary"]
        nrm = next(t for t in dashed if t.name.startswith(g))
        peak = np.asarray(nrm.x)[int(np.argmax(nrm.y))]
        # the normal peaks at its own group's mean
        assert peak == pytest.approx(sub.mean(),
                                     abs=0.05 * sub.std())


def test_kind_normal_by_legend(d):
    # with no general curve to key them, the normals carry the legend
    fig = X("Salary", by="Gender", data=d, form="density",
            kind="normal")
    curves = _curves(fig)
    assert len(curves) == 2
    assert all(t.showlegend for t in curves)
    assert {t.name for t in curves} == {"F", "M"}


# ----- VBS refinements: box_adj / bw_iter / out_cut -------------

def test_medcouple_properties():
    from lessPy.vbs_plotly import _medcouple
    rng = np.random.default_rng(3)
    sym = np.sort(rng.normal(0, 1, 200))
    x = np.concatenate([sym, -sym])    # exactly symmetric
    assert _medcouple(x) == pytest.approx(0, abs=1e-12)
    skew = rng.lognormal(3, 0.6, 151)  # odd n
    m1 = _medcouple(skew)
    assert m1 > 0.1                    # right skew
    assert _medcouple(-skew) == pytest.approx(-m1)


def test_box_adj_widens_skewed_fence():
    from lessPy.vbs_plotly import (_medcouple, _adj_fences,
                                   five_num)
    rng = np.random.default_rng(4)
    x = rng.lognormal(3, 0.6, 150)
    _, q1, _, q3, _ = five_num(x)
    iqr = q3 - q1
    plain_in, _ = _adj_fences(q1, q3, iqr, 1.5, 0.0, -4, 3)
    adj_in, _ = _adj_fences(q1, q3, iqr, 1.5, _medcouple(x),
                            -4, 3)
    # right skew: upper fence extends, lower fence tightens
    assert adj_in[1] > plain_in[1]
    assert adj_in[0] > plain_in[0]


def test_box_adj_flags_fewer_outliers(d):
    rng = np.random.default_rng(4)
    ds = pd.DataFrame({"v": rng.lognormal(3, 0.6, 150)})

    def n_out(fig):
        return sum(len(t.x) for t in fig.data
                   if t.mode == "markers"
                   and t.marker.symbol == "circle"
                   and "outlier" in (t.hovertemplate or ""))
    f0 = X("v", data=ds, form="vbs", quiet=True)
    f1 = X("v", data=ds, form="vbs", box_adj=True, quiet=True)
    assert n_out(f1) < n_out(f0)


def test_bw_iter_widens_multimodal():
    from lessPy.utils import band_width, bw_nrd0
    rng = np.random.default_rng(5)
    x = np.r_[rng.normal(0, 1, 60), rng.normal(8, 1, 60)]
    assert band_width(x, 10) > bw_nrd0(x)
    assert band_width(x, 0) == pytest.approx(bw_nrd0(x))


def test_out_cut_labels_vbs(d):
    fig = X("Salary", data=d, form="vbs", out_cut=2, quiet=True)
    anns = [a for a in fig.layout.annotations
            if a.textangle == -90]
    assert len(anns) <= 2
    # labels are DataFrame indices of the flagged rows
    for a in anns:
        assert a.text.isdigit()


def test_out_cut_id_column(d):
    d2 = d.assign(Name=[f"p{i}" for i in range(len(d))])
    fig = X("Salary", data=d2, form="vbs", out_cut=1,
            ID="Name", quiet=True)
    anns = [a for a in fig.layout.annotations
            if a.textangle == -90]
    assert len(anns) == 1 and anns[0].text.startswith("p")


def test_out_cut_non_vbs_error(d):
    with pytest.raises(ValueError, match="VBS"):
        X("Salary", data=d, out_cut=2)
    with pytest.raises(ValueError, match="VBS"):
        X("Salary", data=d, form="density", box_adj=True)


# ----- n_row / n_col facet grid ---------------------------------

@pytest.fixture
def d4():
    rng = np.random.default_rng(9)
    n = 200
    return pd.DataFrame({
        "Score": rng.normal(50, 10, n),
        "Site": rng.choice(["A", "B", "C", "D", "E"], n)})


def test_facet_grid_2col(d4):
    fig = X("Score", facet="Site", data=d4,
            filter="Site != 'E'", n_col=2)
    bars = [t for t in fig.data if t.type == "bar"]
    assert len(bars) == 4
    # 2x2 grid: bottom row = subplots 3 and 4 (x3, x4);
    # first level (A) bottom-left
    axes = sorted(t.xaxis for t in bars)
    assert axes == ["x", "x2", "x3", "x4"]
    a_bar = [t for t in bars if t.xaxis == "x3"][0]
    n_a = ((d4["Site"] == "A")).sum()
    assert sum(a_bar.y) == n_a


def test_facet_grid_n_row(d4):
    # n_row=1 with 5 levels -> 5 columns
    fig = X("Score", facet="Site", data=d4, n_row=1)
    bars = [t for t in fig.data if t.type == "bar"]
    assert sorted(t.yaxis for t in bars) == \
        ["y", "y2", "y3", "y4", "y5"]


def test_facet_grid_ragged(d4):
    # 5 levels, 2 columns -> 3 rows, empty top-right cell hidden
    fig = X("Score", facet="Site", data=d4, n_col=2)
    bars = [t for t in fig.data if t.type == "bar"]
    assert len(bars) == 5
    assert fig.layout.xaxis2.visible is False
    assert fig.layout.yaxis2.visible is False


def test_facet_grid_requires_facet(d4):
    with pytest.raises(ValueError, match="facet="):
        X("Score", data=d4, n_col=2)


def test_facet_grid_vbs_rejected(d4):
    with pytest.raises(ValueError, match="bands"):
        X("Score", facet="Site", data=d4, form="vbs", n_col=2)


# ----- add= annotations -----------------------------------------

def test_add_hist_vline(d):
    fig = X("Salary", data=d, add=["v_line", "h_line"],
            x1=60000, y1=10, quiet=True)
    vl = [s for s in fig.layout.shapes
          if s.yref == "paper" and s.x0 == 60000
          and s.layer != "below"]
    hl = [s for s in fig.layout.shapes
          if s.xref == "paper" and s.y0 == 10
          and s.layer != "below"]
    assert len(vl) == 1 and len(hl) == 1


def test_add_hist_only(d):
    with pytest.raises(ValueError, match="histogram"):
        X("Salary", data=d, form="density", add="v_line",
          x1=60000)
    with pytest.raises(ValueError, match="histogram"):
        X("Salary", by="Gender", data=d, add="v_line", x1=6e4)


# ----- axis format family ---------------------------------------

def test_axis_fmt_K_default(d):
    # R's default axis_fmt="K": 40000 -> "40K"
    fig = X("Salary", data=d, quiet=True)
    assert any(str(t).endswith("K")
               for t in fig.layout.xaxis.ticktext)


def test_axis_fmt_comma_prefix(d):
    fig = X("Salary", data=d, axis_fmt=",", axis_x_pre="$",
            quiet=True)
    tt = [str(t) for t in fig.layout.xaxis.ticktext]
    assert all(t.startswith("$") for t in tt)
    assert any("," in t for t in tt)


def test_rotate_and_scale_x(d):
    fig = X("Salary", data=d, rotate_x=45,
            scale_x=(20000, 100000, 5), quiet=True)
    assert fig.layout.xaxis.tickangle == -45
    assert list(fig.layout.xaxis.range) == [20000, 100000]
    assert len(fig.layout.xaxis.tickvals) == 5


# ----- facet orthogonality (R July 2026 parity) -----------------

def test_two_facet_histogram_grid(d):
    fig = X("Salary", data=d, facet=["Dept", "Gender"],
            quiet=True)
    bars = [t for t in fig.data if t.type == "bar"]
    assert len(bars) == 4              # 2 Dept x 2 Gender cells
    assert sum(sum(t.y) for t in bars) == len(d)
    labs = [a.text for a in fig.layout.annotations]
    assert 'Dept = "ACCT", Gender = "F"' in labs


def test_two_facet_density_embellished(d):
    # per-panel normal curve and rug, as R's .plt.dist.facet
    fig = X("Salary", data=d, form="density",
            facet=["Dept", "Gender"], kind="both", rug=True,
            quiet=True)
    lines = [t for t in fig.data
             if getattr(t, "mode", None) == "lines"]
    assert len(lines) >= 8             # general + normal per cell
    bars = [t for t in fig.data if t.type == "bar"]
    assert len(bars) == 4              # histogram backdrop


def test_facet_series_input(d):
    lvl = (d["Salary"] > d["Salary"].median()).map(
        {True: "high", False: "low"}).rename("Level")
    fig = X("Salary", data=d, facet=lvl, quiet=True)
    bars = [t for t in fig.data if t.type == "bar"]
    assert len(bars) == 2
    with pytest.raises(ValueError, match="values"):
        X("Salary", data=d, facet=lvl[:10], quiet=True)


def test_facet_three_names_message(d, capsys):
    X("Salary", data=d,
      facet=["Dept", "Gender", "Dept"], quiet=True)
    out = capsys.readouterr().out
    assert "uses the first two" in out


def test_vbs_two_facet_sections(d):
    # hybrid: facet1 bands within a section per facet2 level,
    # all sections on the one shared x axis
    fig = X("Salary", data=d, form="vbs",
            facet=["Dept", "Gender"], quiet=True)
    labs = [a.text for a in fig.layout.annotations]
    assert 'Gender = "F"' in labs and 'Gender = "M"' in labs
    assert labs.count("ACCT") == 2     # band label per section


def test_font_size_scales_text(d):
    base = X("Salary", data=d)
    big = X("Salary", font_size=1.4, data=d)
    bx = list(base.select_xaxes())[0]
    gx = list(big.select_xaxes())[0]
    assert gx.tickfont.size > bx.tickfont.size
