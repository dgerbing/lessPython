# corEFA.py — analog of corEFA.R.
#
# corEFA(): exploratory factor analysis of a correlation matrix,
# maximum-likelihood extraction (a direct port of R's factanal
# ML algorithm) with promax, varimax, or no rotation (ports of
# R's promax()/varimax()). Reports the sorted loadings (small
# ones blanked), the sum-of-squares table, and a generated
# measurement-model string assigning each item to the factor of
# its largest loading, plus any items deleted for loading below
# min_loading. Returns a corEFAResults object.
#
# The numerics reproduce R to ~4 decimals: same optimizer
# objective/gradient, same Kaiser-normalized varimax, same
# promax power=4 target. Input is a correlation matrix (a
# DataFrame or array); raw data is correlated first.

import numpy as np
import pandas as pd

from .utils import fmt


def _factanal(S, q):
    """Maximum-likelihood factor extraction: optimize the
    uniquenesses, then form the loadings from the top-q
    eigenvectors of the scaled correlation matrix. Direct port of
    R's factanal.fit.mle (FAfn/FAgr/FAout)."""
    from scipy.optimize import minimize
    p = S.shape[0]

    def eig(psi):
        sc = 1 / np.sqrt(psi)
        v, V = np.linalg.eigh(sc[:, None] * S * sc[None, :])
        return v[::-1], V[:, ::-1]                # descending

    def out(psi):
        v, V = eig(psi)
        load = V[:, :q] * np.sqrt(np.maximum(v[:q] - 1, 0))
        return np.sqrt(psi)[:, None] * load

    def fn(psi):
        v, _ = eig(psi)
        e = v[q:]
        return -(np.sum(np.log(e) - e) - q + p)

    def gr(psi):
        load = out(psi)
        g = load @ load.T + np.diag(psi) - S
        return np.diag(g) / psi ** 2

    start = (1 - 0.5 * q / p) / np.diag(np.linalg.inv(S))
    res = minimize(fn, start, jac=gr, method="L-BFGS-B",
                   bounds=[(0.005, 1)] * p)
    return out(res.x), bool(res.success)


def _sort_extract(L):
    """factanal's sortLoadings: order factors by descending SS,
    flip each column so its loadings sum to a positive value."""
    L = L[:, np.argsort(-(L ** 2).sum(0))]
    neg = L.sum(0) < 0
    L[:, neg] *= -1
    return L


def _varimax(x, normalize=True, eps=1e-5):
    """Kaiser-normalized varimax. Port of R's stats::varimax."""
    nc = x.shape[1]
    if nc < 2:
        return x
    x = x.copy()
    if normalize:
        sc = np.sqrt((x ** 2).sum(1))
        x = x / sc[:, None]
    p, TT, d = x.shape[0], np.eye(nc), 0.0
    for _ in range(1000):
        z = x @ TT
        B = x.T @ (z ** 3 - z @ np.diag((z ** 2).sum(0)) / p)
        U, s, Vt = np.linalg.svd(B)
        TT = U @ Vt
        d, dpast = s.sum(), d
        if d < dpast * (1 + eps):
            break
    z = x @ TT
    return z * sc[:, None] if normalize else z


def _promax(x, m=4):
    """Promax oblique rotation, power m=4. Port of R's
    stats::promax (varimax, then oblique target)."""
    if x.shape[1] < 2:
        return x
    xv = _varimax(x)
    Q = xv * np.abs(xv) ** (m - 1)
    U = np.linalg.lstsq(xv, Q, rcond=None)[0]
    U = U @ np.diag(np.sqrt(np.diag(np.linalg.inv(U.T @ U))))
    return xv @ U


class corEFAResults:
    """Results of corEFA(): the sorted loadings DataFrame, the
    sum-of-squares table, the measurement model string, and the
    convergence flag."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return (f"<lessPy corEFA: {self.n_factors} factors, "
                f"{self.loadings.shape[0]} items>")


def corEFA(R, n_factors, rotate="promax", min_loading=0.2,
           sort=True):
    """Exploratory factor analysis of a correlation matrix R
    (a DataFrame or array; raw data is correlated first).
    Maximum-likelihood extraction with rotate= "promax" (default),
    "varimax", or "none". Prints the loadings, the sum-of-squares
    table, and the measurement-model code; returns a
    corEFAResults. R analog: corEFA()"""
    if rotate not in ("promax", "varimax", "none"):
        raise ValueError('rotate: "promax", "varimax", "none"')
    Rm = pd.DataFrame(R)
    vals = Rm.to_numpy(dtype=float)
    if (Rm.shape[0] != Rm.shape[1]
            or not np.allclose(vals, vals.T, atol=1e-8)):
        Rm = Rm.select_dtypes("number").corr()   # raw data
    names = [str(c) for c in Rm.columns]
    S = Rm.to_numpy(dtype=float)

    Lam, converged = _factanal(S, n_factors)
    Lam = _sort_extract(Lam)
    if n_factors > 1 and rotate != "none":
        Lam = _promax(Lam) if rotate == "promax" else _varimax(Lam)

    fac = [f"Factor{i + 1}" for i in range(n_factors)]
    ld = pd.DataFrame(Lam, index=names, columns=fac)

    if sort and n_factors > 1:
        top = np.argmax(np.abs(ld.to_numpy()), axis=1)
        order = []
        for f in range(n_factors):
            items = [i for i in range(len(ld)) if top[i] == f]
            items.sort(key=lambda i: -abs(ld.iloc[i, f]))
            order += items
        ld = ld.iloc[order]

    print("\n".join(_efa_output(ld, n_factors, rotate,
                                min_loading, converged)))
    model, deleted = _cfa_model(ld, n_factors, min_loading)
    ss = _ss_table(ld, n_factors)
    return corEFAResults(
        loadings=ld, ss=ss, model=model, deleted=deleted,
        converged=converged, n_factors=n_factors)


def _ss_table(ld, n_factors):
    v = (ld ** 2).sum(axis=0)
    n = ld.shape[0]
    rows = {"SS loadings": v, "Proportion Var": v / n}
    if n_factors > 1:
        rows["Cumulative Var"] = np.cumsum(v / n)
    return pd.DataFrame(rows).T


def _efa_output(ld, n_factors, rotate, min_loading, converged):
    L = ["", "  EXPLORATORY FACTOR ANALYSIS", "",
         "Extraction: maximum likelihood"]
    if n_factors > 1:
        L.append(f"Rotation: {rotate}")
    if not converged:
        L.append(">>> Warning: extraction did not fully converge")
    L += ["", f"Loadings (except -{min_loading} to "
          f"{min_loading})", ""]
    w = max(len(s) for s in ld.index)
    L.append(" " * w + "".join(f"{c:>10}" for c in ld.columns))
    for item, row in ld.iterrows():
        cells = "".join(
            (f"{fmt(v, 3):>10}" if abs(v) >= min_loading
             else " " * 10) for v in row)
        L.append(f"{item:<{w}}{cells}")
    # sum of squares
    ss = _ss_table(ld, n_factors)
    L += ["", "Sum of Squares", ""]
    w2 = max(len(s) for s in ss.index)
    L.append(" " * w2 + "".join(f"{c:>10}" for c in ss.columns))
    for lbl, row in ss.iterrows():
        L.append(f"{lbl:<{w2}}"
                 + "".join(f"{fmt(v, 3):>10}" for v in row))
    return L


def _cfa_model(ld, n_factors, min_loading):
    """Assign each item to the factor of its largest loading (if
    above min_loading), build the F# =~ items measurement model
    string, and collect deleted items. ~ corEFA MIMM code."""
    lv = ld.to_numpy()
    items = list(ld.index)
    assign = []
    for i in range(len(items)):
        j = int(np.argmax(np.abs(lv[i])))
        assign.append(j if abs(lv[i, j]) > min_loading else -1)

    lines = []
    for f in range(n_factors):
        members = [items[i] for i in range(len(items))
                   if assign[i] == f]
        if members:
            lines.append(f"  F{f + 1} =~ " + " + ".join(members))
    model = "\n".join(lines)

    deleted = [items[i] for i in range(len(items))
               if assign[i] == -1]
    print("\n".join(
        ["", "  MEASUREMENT MODEL (for confirmatory analysis)",
         "", model]
        + (["", f"Deletion threshold: min_loading = {min_loading}",
            "Deleted items: " + " ".join(deleted)]
           if deleted else [])))
    return model, deleted
