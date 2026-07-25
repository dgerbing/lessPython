import numpy as np
import pandas as pd
import pytest

from lessPy import XY


@pytest.fixture
def d():
    rng = np.random.default_rng(11)
    n = 80
    yrs = rng.uniform(0, 20, n).round(1)
    sal = (35000 + 2600 * yrs
           + rng.normal(0, 8000, n)).round(2)
    return pd.DataFrame({
        "Years": yrs,
        "Salary": sal,
        "Gender": rng.choice(["F", "M"], n),
        "Dept": rng.choice(["ACCT", "MKTG"], n),
        "Pre": (sal / 1000 + rng.normal(0, 6, n)).round(2),
        "Post": (sal / 1000 + rng.normal(0, 4, n) + 5).round(2),
    })


def scatter_traces(fig):
    return [t for t in fig.data
            if t.mode is not None and "markers" in t.mode]


def test_scatter_basic(d):
    fig = XY("Years", "Salary", data=d)
    pts = scatter_traces(fig)
    assert len(pts) == 1
    assert len(pts[0].x) == len(d)
    assert fig.layout.xaxis.title.text == "Years"
    assert fig.layout.yaxis.title.text == "Salary"
    # single group: no legend entry for the points
    assert pts[0].showlegend is False


def test_y_required(d):
    with pytest.raises(ValueError, match="use X\\(\\)"):
        XY("Years", data=d)


def test_categorical_rejected(d):
    with pytest.raises(TypeError, match="Chart\\(\\)"):
        XY("Years", "Gender", data=d)


def test_filter(d):
    fig = XY("Years", "Salary", data=d, filter="Years > 10")
    n_want = (d["Years"] > 10).sum()
    assert len(scatter_traces(fig)[0].x) == n_want


def test_by_groups(d):
    fig = XY("Years", "Salary", by="Gender", data=d)
    pts = scatter_traces(fig)
    assert [t.name for t in pts] == ["F", "M"]   # factor order
    assert sum(len(t.x) for t in pts) == len(d)
    assert fig.layout.legend.title.text == "Gender"


def test_fit_lm(d):
    fig = XY("Years", "Salary", data=d, fit="lm")
    fit = [t for t in fig.data if t.name == "Fit"]
    assert len(fit) == 1
    # fitted endpoints match least squares on the data
    b1, b0 = np.polyfit(d["Years"], d["Salary"], 1)
    xs = np.asarray(fit[0].x)
    assert fit[0].y[0] == pytest.approx(b0 + b1 * xs[0])
    assert fit[0].y[-1] == pytest.approx(b0 + b1 * xs[-1])
    # default 95% SE band drawn beneath the line
    bands = [t for t in fig.data if t.fill == "toself"
             and t.mode == "none"]
    assert len(bands) == 1


def test_fit_ls_alias(d):
    y1 = XY("Years", "Salary", data=d, fit="ls")
    y2 = XY("Years", "Salary", data=d, fit="lm")
    f1 = [t for t in y1.data if t.name == "Fit"][0]
    f2 = [t for t in y2.data if t.name == "Fit"][0]
    assert np.allclose(f1.y, f2.y)


def test_fit_se_off(d):
    fig = XY("Years", "Salary", data=d, fit="lm", fit_se=0)
    bands = [t for t in fig.data if t.fill == "toself"
             and t.mode == "none"]
    assert bands == []


def test_fit_by_no_default_band(d):
    # with by=, SE bands default off (R: fit_se <- 0)
    fig = XY("Years", "Salary", by="Gender", data=d, fit="lm")
    fits = [t for t in fig.data
            if t.name and t.name.startswith("Fit")]
    assert len(fits) == 2
    bands = [t for t in fig.data if t.fill == "toself"
             and t.mode == "none"]
    assert bands == []


def test_fit_exp(d):
    fig = XY("Years", "Salary", data=d, fit="exp")
    fit = [t for t in fig.data if t.name == "Fit"][0]
    b1, b0 = np.polyfit(d["Years"], np.log(d["Salary"]), 1)
    xs = np.asarray(fit.x)
    assert fit.y[0] == pytest.approx(np.exp(b0 + b1 * xs[0]))


def test_fit_loess(d):
    fig = XY("Years", "Salary", data=d, fit="loess")
    fit = [t for t in fig.data if t.name == "Fit"]
    assert len(fit) == 1
    assert len(fit[0].x) == len(d)
    # default 95% SE band, as for lm (R: bands for lm and loess)
    bands = [t for t in fig.data if t.fill == "toself"
             and t.mode == "none"]
    assert len(bands) == 1


def test_loess_reproduces_quadratic():
    # local quadratic regression recovers a noiseless quadratic
    x = np.linspace(0, 10, 40)
    y = 2 + 3 * x - 0.5 * x ** 2
    df = pd.DataFrame({"x": x, "y": y})
    fig = XY("x", "y", data=df, fit="loess")
    fit = [t for t in fig.data if t.name == "Fit"][0]
    xs = np.asarray(fit.x)
    assert np.allclose(fit.y, 2 + 3 * xs - 0.5 * xs ** 2)


def test_loess_span(d):
    f1 = [t for t in XY("Years", "Salary", data=d,
                        fit="loess").data if t.name == "Fit"][0]
    f2 = [t for t in XY("Years", "Salary", data=d, fit="loess",
                        span=0.3).data if t.name == "Fit"][0]
    assert not np.allclose(f1.y, f2.y)


def test_loess_by_groups(d):
    fig = XY("Years", "Salary", by="Gender", data=d, fit="loess")
    fits = [t for t in fig.data
            if t.name and t.name.startswith("Fit")]
    assert len(fits) == 2
    bands = [t for t in fig.data if t.fill == "toself"
             and t.mode == "none"]
    assert bands == []                 # by=: bands default off


def test_form_hexbin_dropped(d):
    with pytest.raises(ValueError, match="form must be one of"):
        XY("Years", "Salary", data=d, form="hexbin")


def test_ellipse(d):
    fig = XY("Years", "Salary", data=d, ellipse=0.95)
    ell = [t for t in fig.data if t.fill == "toself"
           and t.mode == "lines"]
    assert len(ell) == 1
    # centered on the means (midrange of a cosine is its center)
    ex, ey = np.asarray(ell[0].x), np.asarray(ell[0].y)
    assert (ex.max() + ex.min()) / 2 == \
        pytest.approx(d["Years"].mean(), abs=0.05)
    assert (ey.max() + ey.min()) / 2 == \
        pytest.approx(d["Salary"].mean(), abs=100)


def test_ellipse_true_is_95(d):
    f1 = XY("Years", "Salary", data=d, ellipse=True)
    f2 = XY("Years", "Salary", data=d, ellipse=0.95)
    e1 = [t for t in f1.data if t.fill == "toself"][0]
    e2 = [t for t in f2.data if t.fill == "toself"][0]
    assert np.allclose(e1.x, e2.x)


def test_time_series():
    dates = pd.date_range("2024-01-31", periods=24, freq="ME")
    dts = pd.DataFrame({"Month": dates,
                        "Sales": np.linspace(100, 240, 24)})
    fig = XY("Month", "Sales", data=dts)
    tr = fig.data[0]
    assert tr.mode == "lines+markers"        # connected series
    assert len(tr.x) == 24
    with pytest.raises(NotImplementedError):
        XY("Month", "Sales", data=dts, fit="lm")


@pytest.fixture
def dts():
    dates = pd.date_range("2023-01-01", periods=10, freq="MS")
    return pd.DataFrame({
        "Month": dates,
        "Sales": [10.0, 12, 14, 13, 15, 16, 18, 17, 19, 20]})


def test_forecast_es_golden(dts, capsys):
    # golden values from R:
    # HoltWinters(ts(y, frequency=12), alpha=.5, beta=FALSE,
    #             gamma=FALSE); predict(n.ahead=3, PI, .95)
    from lessPy.plt_forecast import plt_forecast
    p = plt_forecast(dts["Month"].to_numpy(),
                     dts["Sales"].to_numpy(float),
                     "Month", "Sales", ts_ahead=3, ts_alpha=0.5,
                     ts_source="classic")
    f = p["forecast"]
    assert np.allclose(f["predicted"], 18.93359375)
    assert np.allclose(f["lower"], [17.1398661142,
                                    16.9281452866, 16.7367350273])
    assert np.allclose(f["upper"], [20.7273213858,
                                    20.9390422134, 21.1304524727])
    assert list(f["Month"]) == ["Nov 2023", "Dec 2023",
                                "Jan 2024"]


def test_forecast_lm_golden(dts):
    # golden values from R: lm(y ~ 1:n) with the classic-path
    # prediction interval (plt.forecast.R lines 543-556)
    from lessPy.plt_forecast import plt_forecast
    p = plt_forecast(dts["Month"].to_numpy(),
                     dts["Sales"].to_numpy(float),
                     "Month", "Sales", ts_ahead=3,
                     ts_method="lm", ts_trend="A", ts_seasons="N",
                     ts_source="classic")
    f = p["forecast"]
    assert np.allclose(f["predicted"], [21.0666666667,
                                        22.0969696970,
                                        23.1272727273])
    assert np.allclose(f["lower"], [18.8979872331,
                                    19.8232941229, 20.7369235682])
    assert np.allclose(f["upper"], [23.2353461002,
                                    24.3706452711, 25.5176218863])


def test_forecast_traces(dts, capsys):
    fig = XY("Month", "Sales", data=dts, ts_ahead=3,
             ts_alpha=0.5, ts_source="classic")
    names = [t.name for t in fig.data]
    assert "Model fit" in names
    assert "Forecast" in names
    assert "95% PI" in names
    assert len(fig.data) == 7          # series + 6 overlays
    out = capsys.readouterr().out      # console report
    assert "Smoothing Parameters" in out
    assert "Forecast" in out


def test_forecast_requires_date(d):
    with pytest.raises(ValueError, match="date"):
        XY("Years", "Salary", data=d, ts_ahead=5)


def test_forecast_fable_es(dts, capsys):
    # default path: statsmodels ETS with fixed components
    fig = XY("Month", "Sales", data=dts, ts_ahead=3,
             ts_error="A", ts_trend="A", ts_seasons="N")
    fore = [t for t in fig.data if t.name == "Forecast"][0]
    assert len(fore.y) == 3
    # trending data: forecast continues upward
    assert fore.y[2] > fore.y[0] > 15
    out = capsys.readouterr().out
    assert "ETS(A,A,N)" in out
    band = [t for t in fig.data
            if (t.name or "").endswith("PI")]
    assert len(band) == 1


def test_forecast_fable_lm(dts, capsys):
    fig = XY("Month", "Sales", data=dts, ts_ahead=3,
             ts_method="lm", ts_trend="A", ts_seasons="N")
    fore = [t for t in fig.data if t.name == "Forecast"][0]
    # TSLM with trend only: same point forecasts as classic lm
    assert np.allclose(fore.y, [21.0666666667, 22.0969696970,
                                23.1272727273])
    out = capsys.readouterr().out
    assert "TSLM" in out
    assert "R-squared" in out


def test_forecast_no_by(dts):
    dts = dts.assign(G=["a", "b"] * 5)
    with pytest.raises(ValueError, match="by"):
        XY("Month", "Sales", by="G", data=dts, ts_ahead=3)


def test_facet_panels(d):
    d2 = d.assign(Dept=np.where(d.index % 3 == 0, "Ops",
                  np.where(d.index % 3 == 1, "Sales", "IT")))
    fig = XY("Years", "Salary", data=d2, facet="Dept")
    pts = scatter_traces(fig)
    assert len(pts) == 3               # one panel per level
    assert sum(len(t.x) for t in pts) == len(d2)
    # panels on distinct y axes (stacked subplots)
    assert len({t.yaxis for t in pts}) == 3
    # strip label per panel
    txt = [a.text for a in fig.layout.annotations]
    assert set(txt) >= {"Ops", "Sales", "IT"}


def test_facet_fit_per_panel(d):
    d2 = d.assign(Dept=np.where(d.index % 2 == 0, "A", "B"))
    fig = XY("Years", "Salary", data=d2, facet="Dept", fit="lm")
    fits = [t for t in fig.data if t.name == "Fit"]
    assert len(fits) == 2              # a line in each panel
    bands = [t for t in fig.data if t.fill == "toself"
             and t.mode == "none"]
    assert len(bands) == 2             # default band per panel
    # each panel's fit is that panel's least squares line
    for lvl, tr in zip(["A", "B"], fits):
        sub = d2[d2["Dept"] == lvl]
        b1, b0 = np.polyfit(sub["Years"], sub["Salary"], 1)
        xs = np.asarray(tr.x)
        assert tr.y[0] == pytest.approx(b0 + b1 * xs[0])


def test_facet_with_by(d):
    d2 = d.assign(Dept=np.where(d.index % 2 == 0, "A", "B"))
    fig = XY("Years", "Salary", by="Gender", data=d2,
             facet="Dept")
    pts = scatter_traces(fig)
    assert len(pts) == 4               # 2 groups x 2 panels
    shown = [t for t in pts if t.showlegend]
    assert len(shown) == 2             # legend from first panel


def test_facet_forecast_error(dts):
    dts = dts.assign(G=["a", "b"] * 5)
    with pytest.raises(ValueError, match="facet"):
        XY("Month", "Sales", data=dts, facet="G", ts_ahead=3)


@pytest.fixture
def dout():
    rng = np.random.default_rng(2)
    n = 40
    x = rng.uniform(0, 20, n)
    y = 3 * x + rng.normal(0, 5, n)
    y[5] += 60                          # two clear bivariate
    x[12] += 25                         # outliers
    return pd.DataFrame({"x": x, "y": y})


def test_md_flagging(dout, capsys):
    fig = XY("x", "y", data=dout, MD_cut=6)
    out = capsys.readouterr().out
    assert "Mahalanobis" in out
    # planted outliers flagged: open-symbol overdraw trace
    otl = [t for t in fig.data
           if t.marker is not None
           and t.marker.symbol == "circle-open"]
    assert len(otl) == 1
    assert len(otl[0].x) >= 2
    # ID labels annotate the flagged points
    txt = {a.text for a in fig.layout.annotations}
    assert {"5", "12"} <= txt


def test_out_cut_count(dout):
    fig = XY("x", "y", data=dout, out_cut=3, quiet=True)
    otl = [t for t in fig.data
           if t.marker is not None
           and t.marker.symbol == "circle-open"][0]
    assert len(otl.x) == 3


def test_md_second_fit(dout):
    fig = XY("x", "y", data=dout, MD_cut=6, fit="lm",
             quiet=True)
    dashed = [t for t in fig.data
              if t.name == "Fit (no outliers)"]
    assert len(dashed) == 1
    assert dashed[0].line.dash == "dash"
    # refit excludes the planted outliers: slope near 3
    xs = np.asarray(dashed[0].x)
    b1 = (dashed[0].y[-1] - dashed[0].y[0]) / (xs[-1] - xs[0])
    assert b1 == pytest.approx(3, abs=0.3)


def test_md_no_by(dout):
    dout = dout.assign(G=["a", "b"] * 20)
    with pytest.raises(ValueError, match="MD_cut"):
        XY("x", "y", by="G", data=dout, MD_cut=6)


def test_casewise_deletion(d):
    d2 = d.copy()
    d2.loc[d2.index[:5], "Salary"] = np.nan
    fig = XY("Years", "Salary", data=d2)
    assert len(scatter_traces(fig)[0].x) == len(d) - 5


# ----- form="contour" -------------------------------------------

def contour_traces(fig):
    return [t for t in fig.data if t.type == "contour"]


def test_contour_basic(d):
    fig = XY("Years", "Salary", data=d, form="contour",
             quiet=True)
    ct = contour_traces(fig)
    assert len(ct) == 1
    z = np.asarray(ct[0].z)
    assert z.shape == (50, 50)         # contour_nbins default
    # density: integrates to ~1 over the grid
    dx = ct[0].x[1] - ct[0].x[0]
    dy = ct[0].y[1] - ct[0].y[0]
    assert z.sum() * dx * dy == pytest.approx(1, abs=0.05)
    assert ct[0].showscale is False    # colorbar off by default
    assert fig.layout.xaxis.title.text == "Years"
    assert fig.layout.yaxis.title.text == "Salary"


def test_contour_param_selects_form(d):
    # a contour_ parameter switches the form, as in R
    fig = XY("Years", "Salary", data=d, contour_nbins=30,
             quiet=True)
    ct = contour_traces(fig)
    assert len(ct) == 1
    assert np.asarray(ct[0].z).shape == (30, 30)


def test_contour_levels(d):
    fig = XY("Years", "Salary", data=d, contour_n=10, quiet=True)
    c = contour_traces(fig)[0].contours
    # bands: one below the first level + one per interior step
    n_bands = round((c.end - c.start) / c.size) + 1
    assert n_bands == 10


def test_contour_points_overlay(d):
    fig = XY("Years", "Salary", data=d, form="contour",
             contour_points=True, quiet=True)
    pts = [t for t in fig.data if t.type == "scatter"
           and "markers" in t.mode]
    assert len(pts) == 1
    assert len(pts[0].x) == len(d)


def test_contour_legend(d):
    fig = XY("Years", "Salary", data=d, contour_legend=True,
             quiet=True)
    assert contour_traces(fig)[0].showscale is True


def test_contour_fit_ellipse(d):
    fig = XY("Years", "Salary", data=d, form="contour",
             fit="lm", ellipse=0.95, quiet=True)
    lines = [t for t in fig.data
             if t.type == "scatter" and t.mode == "lines"]
    assert len(lines) == 2             # ellipse + fit line
    # no SE band and no fill on the contour ellipse, as in R
    assert not [t for t in fig.data
                if t.type == "scatter" and t.mode == "none"]
    assert all(t.fill is None for t in lines)


def test_contour_no_by(d):
    with pytest.raises(ValueError, match='form="scatter"'):
        XY("Years", "Salary", by="Gender", data=d,
           form="contour")


def test_contour_no_date():
    dts = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=24,
                              freq="MS"),
        "y": np.arange(24.0)})
    with pytest.raises(ValueError, match="time series"):
        XY("date", "y", data=dts, form="contour")


# ----- form="smooth" --------------------------------------------

def heatmap_traces(fig):
    return [t for t in fig.data if t.type == "heatmap"]


def test_smooth_basic(d):
    fig = XY("Years", "Salary", data=d, form="smooth",
             quiet=True)
    hm = heatmap_traces(fig)
    assert len(hm) == 1
    z = np.asarray(hm[0].z)
    assert z.shape == (128, 128)       # smooth_bins default
    assert hm[0].showscale is False
    # all n < smooth_points (100) data points overplotted
    pts = [t for t in fig.data if t.type == "scatter"
           and t.mode == "markers"]
    assert len(pts) == 1
    assert len(pts[0].x) == len(d)
    assert pts[0].marker.color == "#000000"


def test_smooth_density_integrates(d):
    # untransformed density (smooth_power=1) integrates to ~1
    fig = XY("Years", "Salary", data=d, form="smooth",
             smooth_power=1, quiet=True)
    hm = heatmap_traces(fig)[0]
    z = np.asarray(hm.z)
    dx = hm.x[1] - hm.x[0]
    dy = hm.y[1] - hm.y[0]
    assert z.sum() * dx * dy == pytest.approx(1, abs=0.05)


def test_smooth_power_transform(d):
    # displayed z is density^smooth_power (default 0.25)
    f1 = XY("Years", "Salary", data=d, form="smooth",
            smooth_power=1, quiet=True)
    f4 = XY("Years", "Salary", data=d, form="smooth",
            quiet=True)
    z1 = np.asarray(heatmap_traces(f1)[0].z)
    z4 = np.asarray(heatmap_traces(f4)[0].z)
    assert np.allclose(z4 ** 4, z1)


def test_smooth_points_selection(d):
    fig = XY("Years", "Salary", data=d, form="smooth",
             smooth_points=10, quiet=True)
    pts = [t for t in fig.data if t.type == "scatter"
           and t.mode == "markers"]
    assert len(pts[0].x) == 10
    fig0 = XY("Years", "Salary", data=d, form="smooth",
              smooth_points=0, quiet=True)
    assert not [t for t in fig0.data if t.type == "scatter"
                and t.mode == "markers"]


def test_smooth_bins(d):
    fig = XY("Years", "Salary", data=d, form="smooth",
             smooth_bins=64, quiet=True)
    assert np.asarray(heatmap_traces(fig)[0].z).shape == (64, 64)


def test_smooth_fit_band(d):
    # unlike contour, smooth keeps the fit SE band, as in R
    fig = XY("Years", "Salary", data=d, form="smooth",
             fit="lm", quiet=True)
    assert [t for t in fig.data if t.name == "Fit"]
    bands = [t for t in fig.data if t.type == "scatter"
             and t.mode == "none" and t.fill == "toself"]
    assert len(bands) == 1


def test_smooth_no_by(d):
    with pytest.raises(ValueError, match='form="scatter"'):
        XY("Years", "Salary", by="Gender", data=d, form="smooth")


def test_smooth_no_outlier_flagging(d):
    with pytest.raises(ValueError, match="MD_cut"):
        XY("Years", "Salary", data=d, form="smooth", MD_cut=6)


# ----- jitter ---------------------------------------------------

@pytest.fixture
def ddisc():
    # discrete scales: few unique values, n > 14 -> auto-jitter
    rng = np.random.default_rng(7)
    n = 60
    return pd.DataFrame({
        "Rating": rng.integers(1, 6, n).astype(float),
        "Quality": rng.integers(1, 8, n).astype(float)})


def test_auto_jitter_discrete(ddisc):
    fig = XY("Rating", "Quality", data=ddisc, quiet=True)
    pts = scatter_traces(fig)[0]
    xs, ys = np.asarray(pts.x), np.asarray(pts.y)
    xo = ddisc["Rating"].to_numpy()
    yo = ddisc["Quality"].to_numpy()
    # both axes move, each within +/- range/32
    assert not np.allclose(xs, xo)
    assert not np.allclose(ys, yo)
    assert np.abs(xs - xo).max() <= np.ptp(xo) / 32 + 1e-9
    assert np.abs(ys - yo).max() <= np.ptp(yo) / 32 + 1e-9


def test_jitter_off(ddisc):
    fig = XY("Rating", "Quality", data=ddisc, jitter_x=0,
             jitter_y=0, quiet=True)
    pts = scatter_traces(fig)[0]
    assert np.allclose(pts.x, ddisc["Rating"])
    assert np.allclose(pts.y, ddisc["Quality"])


def test_no_auto_jitter_continuous(d):
    fig = XY("Years", "Salary", data=d, quiet=True)
    pts = scatter_traces(fig)[0]
    assert np.allclose(pts.x, d["Years"])
    assert np.allclose(pts.y, d["Salary"])


def test_jitter_explicit_bounds(d):
    fig = XY("Years", "Salary", data=d, jitter_x=0.5,
             quiet=True)
    pts = scatter_traces(fig)[0]
    dx = np.abs(np.asarray(pts.x) - d["Years"].to_numpy())
    assert 0 < dx.max() <= 0.5
    assert np.allclose(pts.y, d["Salary"])   # y untouched


def test_jitter_fit_uses_original(d):
    # display moves, but the fit is on the original data
    f1 = XY("Years", "Salary", data=d, fit="lm", quiet=True)
    f2 = XY("Years", "Salary", data=d, fit="lm", jitter_x=1,
            jitter_y=1000, quiet=True)
    l1 = [t for t in f1.data if t.name == "Fit"][0]
    l2 = [t for t in f2.data if t.name == "Fit"][0]
    assert np.allclose(l1.x, l2.x)
    assert np.allclose(l1.y, l2.y)


def test_jitter_console_note(ddisc, capsys):
    XY("Rating", "Quality", data=ddisc)
    out = capsys.readouterr().out
    assert "can be manually set" in out
    assert "jitter_x:" in out
    assert "jitter_y:" in out


# ----- enhance= -------------------------------------------------

def mean_shapes(fig):
    return [s for s in fig.layout.shapes
            if (s.xref, s.yref) in (("x", "paper"),
                                    ("paper", "y"))
            and s.line.color == "#1A1A1A"]


def test_enhance_bundle(dout, capsys):
    fig = XY("x", "y", data=dout, enhance=True)
    out = capsys.readouterr().out
    # fit "lm" with its SE band
    assert [t for t in fig.data if t.name == "Fit"]
    # 95% data ellipse
    assert [t for t in fig.data if t.fill == "toself"
            and t.mode == "lines"]
    # MD_cut 6: the planted outliers flag and report
    otl = [t for t in fig.data if t.marker is not None
           and t.marker.symbol == "circle-open"]
    assert len(otl) == 1 and len(otl[0].x) >= 2
    assert "Mahalanobis" in out or "MD" in out
    # mean crosshair: one vertical, one horizontal line
    ms = mean_shapes(fig)
    assert len(ms) == 2
    vline = [s for s in ms if s.xref == "x"][0]
    assert vline.x0 == pytest.approx(dout["x"].mean())


def test_enhance_by_skips_md(d):
    # with by=, flagging is unsupported: enhance omits MD_cut
    # but keeps fit, ellipses, and the crosshair
    fig = XY("Years", "Salary", by="Gender", data=d,
             enhance=True, quiet=True)
    fits = [t for t in fig.data
            if t.name and t.name.startswith("Fit")]
    assert len(fits) == 2
    ells = [t for t in fig.data if t.fill == "toself"
            and t.mode == "lines"]
    assert len(ells) == 2
    assert not [t for t in fig.data if t.marker is not None
                and t.marker.symbol == "circle-open"]
    assert len(mean_shapes(fig)) == 2


def test_enhance_keeps_explicit(dout):
    # an explicitly set component is not overridden
    f1 = XY("x", "y", data=dout, enhance=True, ellipse=0.5,
            quiet=True)
    f2 = XY("x", "y", data=dout, ellipse=0.5, quiet=True)
    e1 = [t for t in f1.data if t.fill == "toself"
          and t.mode == "lines"][0]
    e2 = [t for t in f2.data if t.fill == "toself"
          and t.mode == "lines"][0]
    assert np.allclose(e1.x, e2.x)
    assert np.allclose(e1.y, e2.y)


def test_enhance_smooth_no_md(d):
    # smooth: fit, ellipse, crosshair, but no outlier flagging
    fig = XY("Years", "Salary", data=d, form="smooth",
             enhance=True, quiet=True)
    assert [t for t in fig.data if t.name == "Fit"]
    assert len(mean_shapes(fig)) == 2


def test_enhance_no_date():
    dts = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=24,
                              freq="MS"),
        "y": np.arange(24.0)})
    with pytest.raises(ValueError, match="time series"):
        XY("date", "y", data=dts, enhance=True)


# ----- facet= for a date x (time-series panels) -----------------

@pytest.fixture
def dtsf():
    dates = pd.date_range("2023-01-01", periods=12, freq="MS")
    return pd.DataFrame({
        "Month": list(dates) * 2,
        "Sales": np.r_[np.linspace(10, 21, 12),
                       np.linspace(30, 41, 12)],
        "Region": ["East"] * 12 + ["West"] * 12})


def test_ts_facet_panels(dtsf):
    fig = XY("Month", "Sales", data=dtsf, facet="Region")
    lines = [t for t in fig.data if "lines" in t.mode]
    assert len(lines) == 2
    assert all(len(t.x) == 12 for t in lines)
    strips = [a.text for a in fig.layout.annotations]
    assert "East" in strips and "West" in strips
    # first level (East) on the bottom panel (row 2 -> y2)
    bottom = [t for t in lines if t.yaxis == "y2"][0]
    assert bottom.y[0] == pytest.approx(10)


def test_ts_facet_sorted(dtsf):
    shuffled = dtsf.sample(frac=1, random_state=3)
    fig = XY("Month", "Sales", data=shuffled, facet="Region")
    for t in fig.data:
        xs = np.asarray(t.x, dtype="datetime64[ns]")
        assert (np.diff(xs.astype(np.int64)) > 0).all()


def test_ts_facet_by_not_ported(dtsf):
    dtsf = dtsf.assign(G=["a", "b"] * 12)
    with pytest.raises(NotImplementedError, match="by="):
        XY("Month", "Sales", data=dtsf, by="G", facet="Region")


def test_ts_facet_no_ts_unit(dtsf):
    with pytest.raises(ValueError, match="ts_unit"):
        XY("Month", "Sales", data=dtsf, facet="Region",
           ts_unit="years")


# ----- ts_ aggregation / stack / area ---------------------------

@pytest.fixture
def ddaily():
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "date": pd.date_range("2023-01-05", periods=200,
                              freq="D"),
        "y": rng.normal(50, 10, 200)})


def test_ts_unit_aggregates(ddaily):
    fig = XY("date", "y", data=ddaily, ts_unit="months")
    tr = fig.data[0]
    assert len(tr.x) == 6         # 200 days: 6 complete months,
    m1 = ddaily[ddaily["date"].dt.month == 1]["y"].sum()
    assert tr.y[0] == pytest.approx(m1)   # sum is the default
    # aggregated dates move to the period start
    assert pd.Timestamp(tr.x[0]) == pd.Timestamp("2023-01-01")


def test_ts_agg_mean(ddaily):
    fig = XY("date", "y", data=ddaily, ts_unit="months",
             ts_agg="mean")
    m1 = ddaily[ddaily["date"].dt.month == 1]["y"].mean()
    assert fig.data[0].y[0] == pytest.approx(m1)


def test_ts_unit_finer_than_data():
    dm = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=24,
                              freq="MS"),
        "y": np.arange(24.0)})
    with pytest.raises(ValueError, match="Resolution"):
        XY("date", "y", data=dm, ts_unit="days")


def test_ts_area_fill(ddaily):
    fig = XY("date", "y", data=ddaily, ts_area_fill="on")
    fills = [t for t in fig.data if t.mode == "none"
             and t.fill == "toself"]
    assert len(fills) == 1
    # points default off with an area: line-only trace
    assert [t for t in fig.data if t.mode == "lines"]


def test_ts_area_requires_date(d):
    with pytest.raises(ValueError, match="time series"):
        XY("Years", "Salary", data=d, ts_area_fill="on")


def test_ts_stack():
    dates = pd.date_range("2023-01-01", periods=12, freq="MS")
    d2 = pd.DataFrame({
        "date": list(dates) * 2,
        "y": np.r_[np.full(12, 10.0), np.full(12, 5.0)],
        "g": ["a"] * 12 + ["b"] * 12})
    fig = XY("date", "y", data=d2, by="g", ts_stack=True)
    polys = [t for t in fig.data if t.mode == "none"
             and t.fill == "toself"]
    assert len(polys) == 2
    # second level draws cumulative: 10 + 5 = 15
    lines = [t for t in fig.data if t.mode == "lines"
             and len(t.x) == 12]
    tops = [t for t in lines
            if np.allclose(np.asarray(t.y, float), 15)]
    assert len(tops) == 1


def test_ts_stack_needs_by(ddaily):
    with pytest.raises(ValueError, match="by="):
        XY("date", "y", data=ddaily, ts_stack=True)


def test_ts_stack_needs_aligned_dates():
    dates = pd.date_range("2023-01-01", periods=12, freq="MS")
    d2 = pd.DataFrame({
        "date": list(dates) + list(dates[:6]),
        "y": np.arange(18.0),
        "g": ["a"] * 12 + ["b"] * 6})
    with pytest.raises(ValueError, match="same dates"):
        XY("date", "y", data=d2, by="g", ts_stack=True)


def test_ts_area_by_needs_stack():
    dates = pd.date_range("2023-01-01", periods=12, freq="MS")
    d2 = pd.DataFrame({
        "date": list(dates) * 2, "y": np.arange(24.0),
        "g": ["a"] * 12 + ["b"] * 12})
    with pytest.raises(ValueError, match="ts_stack=True"):
        XY("date", "y", data=d2, by="g", ts_area_fill="on")


def test_ts_facet_area(dtsf):
    fig = XY("Month", "Sales", data=dtsf, facet="Region",
             ts_area_fill="on")
    fills = [t for t in fig.data if t.mode == "none"
             and t.fill == "toself"]
    assert len(fills) == 2             # one per panel


# ----- add= annotations -----------------------------------------

def test_add_vocabulary(d):
    fig = XY("Years", "Salary", data=d, quiet=True,
             add=["rect", "line", "arrow", "point", "Note"],
             x1=[2, 4, 6, 8, 10],
             y1=[4e4, 5e4, 6e4, 7e4, 8e4],
             x2=[3, 5, 7], y2=[4.5e4, 5.5e4, 6.5e4])
    types = [s.type for s in fig.layout.shapes]
    assert "rect" in types and "line" in types
    anns = fig.layout.annotations
    assert any(a.text == "Note" for a in anns)
    assert any(a.showarrow for a in anns)          # arrow
    pts = [t for t in fig.data if t.mode == "markers"
           and t.hoverinfo == "skip"]
    assert len(pts) == 1                           # point


def test_add_means_and_mean_coords(d):
    fig = XY("Years", "Salary", data=d, add="means",
             quiet=True)
    cross = [s for s in fig.layout.shapes
             if "paper" in (s.xref, s.yref)
             and s.line.color == "#1A1A1A"]
    assert len(cross) == 2
    f2 = XY("Years", "Salary", data=d, add="v_line",
            x1="mean_x", quiet=True)
    vl = [s for s in f2.layout.shapes
          if s.xref == "x" and s.yref == "paper"
          and s.x0 == s.x1 and s.layer != "below"]
    assert vl[0].x0 == pytest.approx(d["Years"].mean())


def test_add_one_object_many_locations(d):
    fig = XY("Years", "Salary", data=d, add="v_line",
             x1=[5, 10, 15], quiet=True)
    vl = [s for s in fig.layout.shapes
          if s.yref == "paper" and s.x0 == s.x1
          and s.layer != "below" and s.x0 in (5, 10, 15)]
    assert len(vl) == 3


def test_add_guards(d):
    with pytest.raises(ValueError, match="contour"):
        XY("Years", "Salary", data=d, form="contour",
           add="v_line", x1=5)
    with pytest.raises(ValueError, match="facet"):
        XY("Years", "Salary", data=d, facet="Gender",
           add="v_line", x1=5)


# ----- axis format family ---------------------------------------

def test_axis_family(d):
    fig = XY("Years", "Salary", data=d, quiet=True,
             axis_y_pre="$", rotate_x=30, scale_x=(0, 20, 5))
    yt = [str(t) for t in fig.layout.yaxis.ticktext]
    assert all(t.startswith("$") for t in yt)
    assert fig.layout.xaxis.tickangle == -30
    assert list(fig.layout.xaxis.range) == [0, 20]
    # default "K" format on the Salary axis
    f2 = XY("Years", "Salary", data=d, quiet=True)
    assert any(str(t).endswith("K")
               for t in f2.layout.yaxis.ticktext)


# ----- facet orthogonality (R July 2026 parity) -----------------

def test_two_facet_scatter_grid(d):
    fig = XY("Years", "Salary", data=d,
             facet=["Dept", "Gender"], quiet=True)
    pts = scatter_traces(fig)
    assert len(pts) == 4               # 2 Dept x 2 Gender cells
    # cell labels name both variables
    labs = [a.text for a in fig.layout.annotations]
    assert 'Dept = "ACCT", Gender = "F"' in labs


def test_facet_series_input(d):
    lvl = (d["Salary"] > d["Salary"].median()).map(
        {True: "high", False: "low"}).rename("Level")
    fig = XY("Years", "Salary", data=d, facet=lvl, quiet=True)
    assert len(scatter_traces(fig)) == 2
    with pytest.raises(ValueError, match="values"):
        XY("Years", "Salary", data=d, facet=lvl[:10],
           quiet=True)


def test_contour_facet(d):
    import plotly.graph_objects as go
    fig = XY("Years", "Salary", data=d, form="contour",
             facet="Gender", quiet=True)
    cts = [t for t in fig.data if isinstance(t, go.Contour)]
    assert len(cts) == 2
    # shared levels: identical contour band starts across panels
    assert cts[0].contours.start == cts[1].contours.start


def test_smooth_two_facet(d):
    import plotly.graph_objects as go
    fig = XY("Years", "Salary", data=d, form="smooth",
             facet=["Dept", "Gender"], quiet=True)
    hms = [t for t in fig.data if isinstance(t, go.Heatmap)]
    assert len(hms) == 4
    # shared color range on the transformed density
    assert len({(t.zmin, t.zmax) for t in hms}) == 1


def test_series_overlay(d):
    fig = XY(["Pre", "Post"], "Salary", data=d, quiet=True)
    pts = scatter_traces(fig)
    assert [t.name for t in pts] == ["Pre", "Post"]
    assert fig.layout.xaxis.title.text == "Pre, Post"
    assert fig.layout.yaxis.title.text == "Salary"


def test_facet_series(d):
    fig = XY(["Pre", "Post"], "Salary", data=d,
             facet="Gender", fit="lm", quiet=True)
    pts = scatter_traces(fig)
    assert len(pts) == 4               # 2 panels x 2 series
    fits = [t for t in fig.data
            if t.mode == "lines" and t.legendgroup]
    assert len(fits) == 4              # per-series fit per panel


def test_series_by_redirect(d):
    with pytest.raises(ValueError, match="use facet="):
        XY(["Pre", "Post"], "Salary", data=d, by="Gender")
    # different x and y vectors are not a valid scatterplot matrix
    with pytest.raises(ValueError, match="same variables"):
        XY(["Pre", "Post"], ["Years", "Salary"], data=d)
    # the same vector for both x and y builds the matrix
    assert XY(["Pre", "Post"], ["Pre", "Post"], data=d) is not None


def test_cat_cont_facet_redirect(d):
    with pytest.raises(TypeError, match="use X\\(\\) with a by"):
        XY("Gender", "Salary", data=d, facet="Dept")


def test_n_row_col_need_facet(d):
    with pytest.raises(ValueError, match="specify facet="):
        XY("Years", "Salary", data=d, n_col=2)


def test_facet_summary_stats(d, capsys):
    XY("Years", "Salary", data=d, facet=["Dept", "Gender"])
    out = capsys.readouterr().out
    assert "Summary Statistics for Years" in out
    assert "Median" in out and "IQR" in out
    assert "ACCT" in out and "MKTG" in out   # Dept table
    assert "F" in out and "M" in out         # Gender table


def test_index_run_chart(d):
    # ".Index" x is the row-number pseudo-variable 1..n
    fig = XY(".Index", "Salary", data=d)
    # a run chart connects adjacent points by default
    pts = [t for t in fig.data if "markers" in (t.mode or "")][0]
    assert pts.mode == "lines+markers"
    assert list(pts.x) == list(range(1, len(d) + 1))
    ax = list(fig.select_xaxes())[0]
    assert ax.title.text == "Index"
    # a run chart keeps the wide aspect (not squared)
    assert fig.layout.width is None and fig.layout.height is None


def test_font_size_scales_text(d):
    base = XY("Years", "Salary", data=d)
    big = XY("Years", "Salary", font_size=1.5, data=d)
    bx = list(base.select_xaxes())[0]
    gx = list(big.select_xaxes())[0]
    assert gx.title.font.size > bx.title.font.size
    assert gx.tickfont.size > bx.tickfont.size
    # font_size=1 is a no-op
    same = XY("Years", "Salary", font_size=1, data=d)
    sx = list(same.select_xaxes())[0]
    assert sx.title.font.size == bx.title.font.size


def test_stat_cleveland_dot(d):
    # XY with stat= aggregates a numeric by a categorical -> a
    # Cleveland dot plot (delegates to Chart form="dot")
    fig = XY("Dept", "Salary", stat="mean", data=d)
    marks = [t for t in fig.data if t.mode and "markers" in t.mode]
    assert marks
    assert set(marks[0].x) == set(d["Dept"].unique())


def test_stat_sort_orders_by_value(d):
    fig = XY("Dept", "Salary", stat="mean", sort="+", data=d)
    marks = [t for t in fig.data if t.mode and "markers" in t.mode]
    cats = list(marks[0].x)
    means = d.groupby("Dept")["Salary"].mean()
    assert cats == list(means.sort_values().index)


def test_stat_needs_one_categorical(d):
    with pytest.raises(ValueError, match="stat="):
        XY("Years", "Salary", stat="mean", data=d)


def test_show_runs(d, capsys):
    fig = XY(".Index", "Salary", show_runs=True, data=d)
    # points are connected into a run chart
    line = [t for t in fig.data if t.mode and "lines" in t.mode][0]
    assert "markers" in line.mode
    # a median center line is drawn
    assert any(getattr(s, "type", None) == "line"
               for s in fig.layout.shapes)
    # the run analysis is printed
    out = capsys.readouterr().out
    assert "Run Analysis" in out
    assert "Total number of runs:" in out


def test_fit_new_predictions(d, capsys):
    # fit_new prints predicted y at the new x values, ascending
    XY("Years", "Salary", fit="lm", fit_new=[5, 1, 20], data=d)
    out = capsys.readouterr().out
    assert "Salary_Fit" in out
    # linear model: predictions match b0 + b1*x_new at sorted x
    import numpy as np
    b1, b0 = np.polyfit(d["Years"], d["Salary"], 1)
    for xn in (1, 5, 20):
        assert f"{b0 + b1 * xn:.2f}" in out or \
               f"{b0 + b1 * xn:.3f}" in out
    # x values appear ascending (1 before 20)
    assert out.index(" 1 ") < out.index(" 20 ") \
        or out.index("  1 ") < out.index(" 20 ")


def test_row_names_dot_plot(d):
    # row_names -> the index as a Cleveland dot plot (via Chart dot)
    fig = XY("Salary", "row_names", data=d)
    marks = [t for t in fig.data if t.mode and "markers" in t.mode]
    assert marks
    assert len(marks[0].y) == len(d)           # one point per row
    # segments_y=False removes the lollipop stems (axis-name swap)
    stems = lambda **k: len([t for t in XY("Salary", "row_names",
                             data=d, **k).data if t.mode == "lines"])
    assert stems() >= 1
    assert stems(segments_y=False) == 0
    # a categorical partner is rejected
    import pytest
    with pytest.raises(ValueError, match="numerical"):
        XY("Gender", "row_names", data=d)
