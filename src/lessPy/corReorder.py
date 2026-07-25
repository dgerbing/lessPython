# corReorder.py — analog of corReorder.R.
#
# corReorder(): reorder the variables of a correlation matrix so
# that related variables are adjacent, by hierarchical clustering
# (the default), the Hunter (1973) chaining algorithm, a manual
# order, or as-is. Returns the reordered correlation matrix (a
# DataFrame); the heat map and, for hclust, the dendrogram are on
# the returned frame's .attrs["plots"]. Useful before corCFA or a
# heat map, to reveal the cluster structure.

import numpy as np
import pandas as pd

from .utils import get_option

# R hclust method -> scipy linkage method. scipy 'ward' is R's
# ward.D2 (squared updates); ward.D has no scipy equivalent, so
# it maps to ward with a note. 'weighted' is R's mcquitty.
_LINKAGE = {"complete": "complete", "single": "single",
            "average": "average", "mcquitty": "weighted",
            "centroid": "centroid", "median": "median",
            "ward.D2": "ward", "ward.D": "ward"}


def corReorder(R, order="hclust", hclust_type="complete",
               dist_type="R", n_clusters=None, vars=None,
               chain_first=0, heat_map=True, dendrogram=True,
               diagonal_new=True, main=None):
    """Reorder the variables of a correlation matrix R (a
    DataFrame or array; raw data is correlated first).
    order="hclust" (default), "chain", "manual" (vars=[...]), or
    "as_is". Returns the reordered correlation-matrix DataFrame;
    its .attrs["plots"] holds the heat map (and dendrogram).
    R analog: corReorder()"""
    if order not in ("hclust", "chain", "manual", "as_is"):
        raise ValueError('order: "hclust", "chain", "manual", '
                         '"as_is"')
    Rm = pd.DataFrame(R)
    vals = Rm.to_numpy(dtype=float)
    if (Rm.shape[0] != Rm.shape[1]
            or not np.allclose(vals, vals.T, atol=1e-8)):
        Rm = Rm.select_dtypes("number").corr()
    Rm.index = Rm.columns = [str(c) for c in Rm.columns]
    S = Rm.to_numpy(dtype=float)
    names = list(Rm.columns)
    nv = len(names)
    if vars is not None:
        order = "manual"

    Z = None
    if order == "manual":
        label = [names.index(str(v)) for v in vars]
    elif order == "as_is":
        diagonal_new = False
        label = list(range(nv))
    elif order == "chain":
        label = _chain(S, nv, int(chain_first))
    else:                                  # hclust
        label, Z = _hclust(S, hclust_type, dist_type, n_clusters,
                           names)

    out = Rm.iloc[label, label]
    plots = {}
    if heat_map:
        plots["heatmap"] = _heatmap(out, diagonal_new, main)
    if order == "hclust" and dendrogram and Z is not None:
        try:
            plots["dendrogram"] = _dendrogram(Z, names, label)
        except Exception:                  # optional; skip if it fails
            pass
    out.attrs["plots"] = plots
    out.attrs["order"] = label
    return out


def _chain(S, nv, first):
    """Hunter (1973) chaining: start with the variable of largest
    summed squared correlation (or the user's chain_first), then
    repeatedly append the unselected variable most correlated (in
    absolute value) with the last one. ~ corReorder.R chain."""
    if first == 0:
        first = int(np.argmax((S ** 2).sum(axis=1)))
    else:
        first = first - 1                  # 1-based -> 0-based
    label = [first]
    remaining = set(range(nv)) - {first}
    while remaining:
        k = label[-1]
        nxt = max(remaining, key=lambda j: abs(S[k, j]))
        label.append(nxt)
        remaining.discard(nxt)
    return label


def _hclust(S, method, dist_type, n_clusters, names):
    from scipy.cluster.hierarchy import (
        fcluster, leaves_list, linkage)
    from scipy.spatial.distance import squareform
    if method not in _LINKAGE:
        raise ValueError(f"hclust_type: {', '.join(_LINKAGE)}")
    D = S if dist_type == "dist" else 1 - S
    cond = squareform(D, checks=False)
    Z = linkage(cond, method=_LINKAGE[method])
    label = leaves_list(Z).tolist()
    if n_clusters is not None:
        cl = fcluster(Z, t=n_clusters, criterion="maxclust")
        # renumber clusters by first appearance in the original
        # variable order, as R's cutree
        seen = {}
        cl = [seen.setdefault(c, len(seen) + 1) for c in cl]
        ttl = f"{n_clusters} Cluster Solution"
        print(f"\n{ttl}\n" + "-" * len(ttl))
        for nm, c in sorted(zip(names, cl), key=lambda p: p[1]):
            print(f"{nm}: {c}")
    return label, Z


def _apply_diag(M):
    """Replace the diagonal with the average of its adjacent
    off-diagonal values, for a cleaner heat map. ~ diagonal_new."""
    nv = M.shape[0]
    D = M.copy()
    if nv >= 2:
        D[0, 0] = M[0, 1]
        for i in range(1, nv - 1):
            D[i, i] = round((M[i, i - 1] + M[i, i + 1]) / 2, 2)
        D[nv - 1, nv - 1] = M[nv - 1, nv - 2]
    return D


def _heatmap(df, diagonal_new, main):
    import plotly.graph_objects as go
    from .plotly_utils import plotly_style, to_hex
    Z = df.to_numpy(dtype=float).copy()
    if diagonal_new:
        Z = _apply_diag(Z)
    labels = list(df.columns)
    style = plotly_style()
    fig = go.Figure(go.Heatmap(
        z=Z, x=labels, y=labels, zmin=-1, zmax=1,
        colorscale="RdBu", reversescale=True,
        colorbar=dict(title="r")))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        template=None, paper_bgcolor=to_hex(style["window_fill"]),
        title=dict(text=main or "Reordered Correlations", x=0.5,
                   xanchor="center",
                   font=dict(size=round(
                       16 * get_option("main_size", 1)))))
    return fig


def _dendrogram(Z, names, label):
    from plotly.figure_factory import create_dendrogram
    fig = create_dendrogram(
        np.zeros((len(names), 1)), labels=names,
        linkagefun=lambda _x: Z)
    fig.update_layout(
        title=dict(text="Cluster Dendrogram", x=0.5,
                   xanchor="center"))
    return fig
