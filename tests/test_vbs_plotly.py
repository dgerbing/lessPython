import numpy as np
import pandas as pd
import pytest

from lessPy import X
from lessPy.vbs_plotly import five_num


@pytest.fixture
def d():
    rng = np.random.default_rng(21)
    vals = rng.normal(50, 10, 150)
    vals[:3] = [95.0, 99.0, 120.0]          # outliers, one extreme
    return pd.DataFrame({
        "Score": vals.round(2),
        "Grp": rng.choice(["A", "B"], 150),
    })


def test_fivenum_matches_r():
    # R: fivenum(1:7) -> 1.0 2.5 4.0 5.5 7.0
    x = np.arange(1.0, 8.0)
    assert list(five_num(x)) == [1, 2.5, 4, 5.5, 7]
    # R: fivenum(1:6) -> 1.0 2.0 3.5 5.0 6.0
    x = np.arange(1.0, 7.0)
    assert list(five_num(x)) == [1, 2, 3.5, 5, 6]


def trace_kinds(fig):
    fills = [t for t in fig.data if t.fill == "toself"]
    markers = [t for t in fig.data
               if t.mode == "markers"]
    return fills, markers


def test_vbs_components(d):
    fig = X("Score", data=d, form="vbs")
    fills, markers = trace_kinds(fig)
    # violin + box are the two filled polygons
    assert len(fills) == 2
    # strip points + outliers + extreme outliers
    assert len(markers) == 3
    n_pts = sum(len(t.x) for t in markers)
    assert n_pts == len(d)                   # every case plotted
    assert fig.layout.yaxis.visible is False


def test_box_stats(d):
    fig = X("Score", data=d, form="box")
    fills, markers = trace_kinds(fig)
    assert len(fills) == 1                   # box only, no violin
    box = fills[0]
    mn, q1, md, q3, mx = five_num(d["Score"].to_numpy())
    assert min(box.x) == pytest.approx(q1)
    assert max(box.x) == pytest.approx(q3)
    # median segment present at the median value
    med = [t for t in fig.data
           if t.mode == "lines" and len(t.x) == 2
           and t.x[0] == t.x[1]
           and t.x[0] == pytest.approx(md)]
    assert len(med) == 1
    # no strip points in a pure box plot
    strip = [t for t in markers if len(t.x) > 10]
    assert strip == []


def test_outlier_split(d):
    fig = X("Score", data=d, form="vbs")
    x = d["Score"].to_numpy()
    _, q1, _, q3, _ = five_num(x)
    iqr = q3 - q1
    n_reg = (((x >= q1 - 3 * iqr) & (x < q1 - 1.5 * iqr))
             | ((x > q3 + 1.5 * iqr) & (x <= q3 + 3 * iqr))).sum()
    n_ext = ((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).sum()
    assert n_reg > 0 and n_ext > 0           # fixture provides both
    _, markers = trace_kinds(fig)
    sizes = sorted(len(t.x) for t in markers)
    assert n_reg in sizes and n_ext in sizes


def test_strip_only(d):
    fig = X("Score", data=d, form="strip")
    fills, markers = trace_kinds(fig)
    assert fills == []                       # no violin, no box
    assert sum(len(t.x) for t in markers) == len(d)


def test_vbs_mean_and_fences(d):
    # jitter_x=0: the fixture has ties, which otherwise trigger
    # the automatic tie-breaking jitter and shift the statistics
    fig = X("Score", data=d, form="vbs", vbs_mean=True,
            fences=True, jitter_x=0)
    m = d["Score"].mean()
    mean_seg = [t for t in fig.data
                if t.mode == "lines" and len(t.x) == 2
                and t.x[0] == t.x[1]
                and t.x[0] == pytest.approx(m)]
    assert len(mean_seg) == 1
    x = d["Score"].to_numpy()
    _, q1, _, q3, _ = five_num(x)
    iqr = q3 - q1
    fence_lo = [t for t in fig.data
                if t.mode == "lines" and len(t.x) == 2
                and t.x[0] == t.x[1]
                and t.x[0] == pytest.approx(q1 - 1.5 * iqr)]
    assert len(fence_lo) == 1


def test_by_groups(d):
    fig = X("Score", by="Grp", data=d, form="vbs", jitter_x=0)
    strip = [t for t in fig.data
             if t.mode == "markers" and t.showlegend]
    assert [t.name for t in strip] == ["A", "B"]
    # by groups differ by COLOR ONLY: same symbol for every group
    assert strip[0].marker.symbol == "circle"
    assert strip[1].marker.symbol == "circle"
    # one violin and one box over all the data, box translucent
    fills = [t for t in fig.data if t.fill == "toself"]
    assert len(fills) == 2
    box = [t for t in fills if t.hoverinfo == "text"][0]
    assert box.fillcolor.startswith("rgba")
    assert box.fillcolor.endswith("0.250)")
    # outliers take group colors, not firebrick
    outs = [t for t in fig.data
            if t.mode == "markers" and not t.showlegend]
    assert all("8B1A1A" not in str(t.marker.color).upper()
               for t in outs)
    # every case appears exactly once across strip + outliers
    n_pts = sum(len(t.x) for t in fig.data
                if t.mode == "markers")
    assert n_pts == len(d)
    assert fig.layout.legend.orientation == "h"


def test_shape_vector_overrides(d):
    # pt_shape= as a per-group vector varies the strip symbols
    fig = X("Score", by="Grp", data=d, form="vbs", jitter_x=0,
            pt_shape=["circle", "square"])
    strip = [t for t in fig.data
             if t.mode == "markers" and t.showlegend]
    assert [t.marker.symbol for t in strip] == ["circle", "square"]


def test_by_box_stats_pooled(d):
    # the box summarizes ALL the data, not per group
    fig = X("Score", by="Grp", data=d, form="vbs", jitter_x=0)
    box = [t for t in fig.data if t.fill == "toself"
           and t.hoverinfo == "text"][0]
    _, q1, _, q3, _ = five_num(d["Score"].to_numpy())
    assert min(box.x) == pytest.approx(q1)
    assert max(box.x) == pytest.approx(q3)


def test_categorical_still_rejected(d):
    with pytest.raises(TypeError, match="Chart\\(\\)"):
        X("Grp", data=d, form="vbs")


def test_facet_panels(d):
    fig = X("Score", facet="Grp", data=d, form="vbs",
            jitter_x=0)
    # one violin and one box per level
    fills = [t for t in fig.data if t.fill == "toself"]
    assert len(fills) == 4
    boxes = [t for t in fills if t.hoverinfo == "text"]
    assert len(boxes) == 2
    # per-panel statistics: each box matches its level's hinges;
    # first level in the BOTTOM band (y centered at 0)
    for i, lvl in enumerate(["A", "B"]):
        xg = d.loc[d["Grp"] == lvl, "Score"].to_numpy()
        _, q1, _, q3, _ = five_num(xg)
        bx = [b for b in boxes
              if min(b.y) == pytest.approx(i * 3.0, abs=0.5)][0]
        assert min(bx.x) == pytest.approx(q1)
        assert max(bx.x) == pytest.approx(q3)
    # strip label rectangles and level annotations
    rects = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(rects) == 2
    labels = [a.text for a in fig.layout.annotations]
    assert labels == ["A", "B"]
    # every case appears exactly once
    n_pts = sum(len(t.x) for t in fig.data
                if t.mode == "markers")
    assert n_pts == len(d)


def test_by_with_facet(d):
    # by= within facet= panels (guard removed July 2026): points
    # per group in every band, all cases drawn once
    fig = X("Score", by="Grp", facet="Grp", data=d, form="vbs")
    n_pts = sum(len(t.x) for t in fig.data if t.mode == "markers")
    assert n_pts == len(d)
    labels = [a.text for a in fig.layout.annotations]
    assert labels == ["A", "B"]
