import pytest

import lessPy as lp
from lessPy import Chart, X, XY, get_option, read_data, style


@pytest.fixture
def emp():
    return read_data("Employee")


def test_theme_settings_are_lessR_values():
    # each theme's settings are lessR's own (style(theme) then
    # getOption()), generated into _themes.py
    style("darkred")
    assert get_option("theme") == "darkred"
    assert get_option("bar_fill_cont") == "#8B0000E6"
    assert get_option("panel_fill") == "#FCFCFC"
    style("gray", sub_theme="black")
    assert get_option("window_fill") == "#1A1A1A"
    assert get_option("axis_text_color") == "#D9D9D9"
    style()                                   # back to the default
    assert get_option("theme") == "colors"
    assert get_option("axis_color") == "#262626"


def test_setting_alone_keeps_theme_and_sizes():
    style("darkred")
    style(axis_size=1.2)
    assert get_option("theme") == "darkred"   # only that setting
    assert get_option("axis_size") == 1.2
    style("blue")                             # a theme resets it
    assert get_option("axis_size") == 0.80


def test_get_and_set_round_trip():
    style("sienna", panel_fill="#FFFFFF")
    saved = style(get=True)
    style()
    style(set=saved)
    assert get_option("theme") == "sienna"
    assert get_option("panel_fill") == "#FFFFFF"


def test_trans_sets_both_fills():
    style(trans=0.4)
    assert get_option("trans_bar_fill") == 0.4
    assert get_option("trans_pt_fill") == 0.4


def test_invalid_settings_named():
    with pytest.raises(ValueError, match="no lessPy counterpart"):
        style(results="brief")
    with pytest.raises(ValueError, match="not a style"):
        style(bar_colour="red")
    with pytest.raises(ValueError, match="not a theme"):
        style("mauve")


def test_global_theme_reaches_every_view(emp):
    style("darkred")
    bar = Chart("Dept", data=emp, quiet=True).data[0]
    assert len(set(bar.marker.color)) == 1     # the theme's bar color
    hist = X("Salary", data=emp, quiet=True).data[0]
    assert "139,0,0" in str(hist.marker.color)    # #8B0000
    fig = XY("Years", "Salary", data=emp, quiet=True)
    assert fig.layout.plot_bgcolor.upper() == "#FCFCFC"


def test_call_theme_is_for_that_call_only(emp):
    fig = Chart("Dept", theme="darkred", data=emp, quiet=True)
    assert "139,0,0" in str(fig.data[0].marker.color[0])
    assert get_option("theme") == "colors"     # restored
    X("Salary", theme="gray", data=emp, quiet=True)
    assert get_option("theme") == "colors"


def test_black_legend_reads_light(emp):
    style("gray", sub_theme="black")
    fig = Chart("Dept", by="Gender", data=emp, quiet=True)
    assert fig.layout.legend.font.color.upper() == "#D9D9D9"


def test_black_axes_light_for_every_theme(emp):
    # on sub_theme="black" the axis lines and their labels are light
    # whatever the theme; darkred keeps axis_color dark and sets the
    # axes through axis_x_color / axis_x_text_color (lessR, Oct 2026)
    for th in ("darkred", "gray", "blue"):
        style(th, sub_theme="black")
        ax = XY("Years", "Salary", data=emp, quiet=True).layout.xaxis
        assert ax.linecolor.upper() == "#D9D9D9"
        assert ax.tickfont.color.upper() == "#D9D9D9"
    style(axis_text_color="red")
    ax = XY("Years", "Salary", data=emp, quiet=True).layout.xaxis
    assert ax.tickfont.color.upper() == "#FF0000"
