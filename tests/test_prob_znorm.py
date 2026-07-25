import pytest

from lessPy import prob_znorm
from lessPy.prob_znorm import ProbZnormResults, _gray_to_hex


def test_returns_results_and_figure():
    r = prob_znorm()
    assert isinstance(r, ProbZnormResults)
    assert r.mu == 0 and r.sigma == 1
    assert set(r.plots) == {"znorm"}
    fig = r.plots["znorm"]
    # curve + three nested sigma bands = 4 scatter traces
    assert len(fig.data) == 4
    # the mean segment is a shape, not a trace
    assert len(fig.layout.shapes) == 1


def test_translucent_fill_stacks():
    # all three bands share the same translucent fill, so overlap
    # darkens the center; default rgb(.10,.34,.94,.20) -> rgba
    fig = prob_znorm().plots["znorm"]
    fills = [t.fillcolor for t in fig.data if t.fill == "toself"]
    assert len(fills) == 3
    assert set(fills) == {"rgba(26,87,240,0.2)"}


def test_z_axis_annotations():
    # non-standard normal -> z-score annotations (-4..4)
    fig = prob_znorm(mu=5, sigma=2).plots["znorm"]
    zs = sorted(int(a.text) for a in fig.layout.annotations)
    assert zs == list(range(-4, 5))
    # standard normal -> z suppressed
    assert len(prob_znorm().plots["znorm"].layout.annotations) == 0


def test_gray_name_to_hex():
    assert _gray_to_hex("gray10") == "#1A1A1A"
    assert _gray_to_hex("grey50") == "#808080"
    assert _gray_to_hex("#123456") == "#123456"
    # curve/border uses the converted hex
    fig = prob_znorm().plots["znorm"]
    assert fig.data[0].line.color == "#1A1A1A"


def test_errors():
    with pytest.raises(ValueError, match="between 0 and 1"):
        prob_znorm(r=1.5)
    with pytest.raises(ValueError, match="sigma"):
        prob_znorm(sigma=0)
