import re

import pytest

from lessPy import getColors as gc


def g(**k):
    return gc(quiet=True, **k)


def test_hcl_families_match_r():
    assert g(pal="hues", n=2) == ["#4398D0", "#B28B2A"]
    assert g(pal="hues", n=5, in_order=True) == [
        "#D57388", "#A69016", "#00A666", "#00A2C0", "#AF7CD1"]
    assert g(pal="blues", n=5) == [
        "#A2C2E2FF", "#6E9CC5FF", "#2E79A9FF", "#005992FF",
        "#00408DFF"]
    assert g(pal="grays", n=4) == [
        "#B0B0B0FF", "#858585FF", "#5D5D5DFF", "#373737FF"]
    assert g(pal="reds", end_pal=["blues"], n=6) == [
        "#7A2B40FF", "#945E68FF", "#B1989CFF", "#939FADFF",
        "#4D708FFF", "#004D7AFF"]


def test_fixed_and_manual():
    assert g(pal="Okabe-Ito")[0] == "#E69F00"
    assert len(g(pal="Okabe-Ito")) == 9
    assert len(g(pal="Tableau", n=5)) == 5
    assert g(pal="Tableau")[0] == "#4E79A7"
    assert g(pal="distinct", n=3) == ["#EEB422", "#737373",
                                      "#9ACD32"]
    # manual RGB ramp red -> blue
    assert g(pal="red", end_pal="blue", n=4) == [
        "#FF0000", "#AA0055", "#5500AA", "#0000FF"]


def test_default_and_viridis_and_errors():
    # default palette is the 12 qualitative hues
    assert g() == g(pal="hues", n=12)
    hexes = g(pal="viridis", n=5)          # plotly approximation
    assert len(hexes) == 5
    assert all(re.fullmatch(r"#[0-9A-F]{6}", h) for h in hexes)
    with pytest.raises(ValueError, match="Okabe-Ito"):
        g(pal="Okabe-Ito", n=12)
    with pytest.raises(ValueError, match="Tableau"):
        g(pal="Tableau", n=12)
