import io
import contextlib

import numpy as np
import pytest

from lessPy import corReorder, read_data

# orderings verified against lessR corReorder() on Mach4 corrs


@pytest.fixture
def R10():
    d = read_data("Mach4")
    return d[[f"m{i:02d}" for i in range(1, 11)]].corr()


def test_hclust_order_match_R(R10):
    out = corReorder(R10, order="hclust", heat_map=False)
    assert list(out.columns) == ["m04", "m03", "m09", "m10",
                                 "m06", "m07", "m01", "m08",
                                 "m02", "m05"]
    assert list(out.index) == list(out.columns)   # square
    # the heat map is attached when heat_map is on (default)
    assert "heatmap" in corReorder(R10).attrs["plots"]


def test_chain_order_match_R(R10):
    out = corReorder(R10, order="chain", heat_map=False)
    assert list(out.columns) == ["m07", "m06", "m10", "m09",
                                 "m02", "m04", "m08", "m01",
                                 "m05", "m03"]


def test_manual_and_as_is(R10):
    m = corReorder(R10, vars=["m05", "m01", "m03"],
                   heat_map=False)
    assert list(m.columns) == ["m05", "m01", "m03"]
    a = corReorder(R10, order="as_is", heat_map=False)
    assert list(a.columns) == list(R10.columns)


def test_n_clusters_grouping(R10, capsys):
    corReorder(R10, order="hclust", n_clusters=3, heat_map=False)
    out = capsys.readouterr().out
    assert "3 Cluster Solution" in out
    # R cutree: {m01,m08}=1, {m02,m05}=2, rest=3
    assert "m01: 1" in out and "m08: 1" in out
    assert "m02: 2" in out and "m05: 2" in out


def test_errors(R10):
    with pytest.raises(ValueError, match="order:"):
        corReorder(R10, order="bogus")
