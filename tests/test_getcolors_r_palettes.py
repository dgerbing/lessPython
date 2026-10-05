# R's own palettes in getColors(): grDevices rainbow/heat/terrain and
# the colorspace "_hcl" versions, as lessR's getColors() computes them.
# Reference values from R 4.5 / lessR 4.5.7 getColors(..., quiet=TRUE).
# heat_hcl and terrain_hcl are colorspace's own palettes, identical to
# colorspace::heat_hcl(5) and colorspace::terrain_hcl(5).

import pytest

from lessPy import getColors

REF = {
    ("rainbow", 3): ["#FF0000", "#00FF00", "#0000FF"],
    ("heat", 5): ["#FF0000", "#FF5500", "#FFAA00", "#FFFF00",
                  "#FFFF80"],
    ("rainbow_hcl", 5): ["#D57388", "#A69016", "#00A666", "#00A2C0",
                         "#AF7CD1"],
    ("heat_hcl", 5): ["#D33F6A", "#E1704C", "#E99A2C", "#E8C33C",
                      "#E2E6BD"],
    ("terrain", 4): ["#00A600", "#E6E600", "#ECB176", "#F2F2F2"],
    ("terrain_hcl", 5): ["#26A63A", "#9BB306", "#E1BB4E", "#FFC59E",
                         "#F1F1F1"],
}


def _hex(cols):
    return [c[:7].upper() for c in cols]


@pytest.mark.parametrize("key", list(REF))
def test_matches_r(key):
    name, n = key
    assert _hex(getColors(name, n=n, quiet=True)) == REF[key]


@pytest.mark.parametrize("name", ["rainbow", "heat", "terrain",
                                  "rainbow_hcl", "heat_hcl",
                                  "terrain_hcl"])
@pytest.mark.parametrize("n", [1, 2, 7, 256])
def test_length(name, n):
    assert len(getColors(name, n=n, quiet=True)) == n


def test_hcl_takes_c_and_l():
    a = getColors("heat_hcl", n=5, quiet=True)
    b = getColors("heat_hcl", n=5, c=40, l=70, quiet=True)
    assert a != b


def test_treemap_terrain_matches_r():
    import lessPy as lp
    d = lp.read_data("Employee")
    f = lp.Chart("Dept", form="treemap", fill="terrain", data=d,
                 quiet=True)
    got = dict(zip(f.data[0].labels, f.data[0].marker.colors))
    assert got == {"ACCT": "#20B100", "ADMN": "#44BD00",
                   "FINC": "#00A600", "MKTG": "#44BD00",
                   "SALE": "#F2F2F2"}
