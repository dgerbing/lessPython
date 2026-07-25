# corCFA.py — analog of corCFA.R + cor_mimm.R.
#
# corCFA(): a Multiple-Indicator Measurement Model (MIMM)
# confirmatory factor analysis of a correlation matrix, given a
# measurement model that assigns items to factors. Reports scale
# composition, reliability (Cronbach's alpha, McDonald's omega),
# the item-factor pattern loadings with an indicator diagnostic,
# the item/factor correlations, the residual correlations, and
# the lavaan specification, plus a heat map. The core is a direct
# port of lessR's deterministic .mimm() (sum-of-correlations
# composites, a communality iteration, then standardization) — no
# optimization, so it reproduces R exactly.
#
# The model is given as a lavaan-style string ("F1 =~ X1 + X2")
# or as F-keyword lists of item names (F1=["X1","X2"]).

import re

import numpy as np
import pandas as pd

from .utils import fmt, get_option


def _mimm(R, cuts, NItems, NF, Iter):
    """The MIMM computation. R is the (NItems+NF) square matrix
    with the item correlations in the top-left block and zeros in
    the factor rows/cols; cuts[j] = (start, end) item indices of
    factor j. Returns (expanded correlation matrix, alpha,
    omega). Direct port of cor_mimm.R .mimm()."""
    R = R.copy().astype(float)
    n = NItems + NF
    # item-factor and factor-factor covariances (column sums)
    for I in range(n):
        for J in range(NF):
            L = NItems + J
            s = 0.0
            for K in range(cuts[J][0], cuts[J][1] + 1):
                s += R[I, K]
            R[I, L] = s
            R[L, I] = s
    # Cronbach's alpha
    Alpha = np.zeros(NF)
    for J in range(NF):
        L = NItems + J
        N = cuts[J][1] - cuts[J][0] + 1
        if N == 1:
            Alpha[J] = 1.0
        else:
            DSum = sum(R[I, I]
                       for I in range(cuts[J][0], cuts[J][1] + 1))
            XN = float(N)
            RMean = (R[L, L] - DSum) / (XN * (XN - 1.0))
            Alpha[J] = (XN * RMean) / ((XN - 1) * RMean + 1.0)
    # communality iteration
    if Iter >= 0:
        for J in range(NF):
            if Alpha[J] > 0:
                L = NItems + J
                for _ in range(Iter):
                    OldFF = R[L, L]
                    R[L, L] = 0.0
                    for I in range(cuts[J][0], cuts[J][1] + 1):
                        OldII = R[I, I]
                        R[I, I] = R[I, L] ** 2 / OldFF
                        R[I, L] = R[I, L] + (R[I, I] - OldII)
                        R[L, L] += R[I, L]
    # standardize to correlations
    for J in range(NF):
        L = NItems + J
        FacSD = np.sqrt(R[L, L])
        for I in range(L + 1):
            R[I, L] /= FacSD
            R[L, I] = R[I, L]
        for I in range(J, NF):
            LI = NItems + I
            R[L, LI] /= FacSD
            R[LI, L] = R[L, LI]
    # McDonald's omega
    Omega = np.zeros(NF)
    for IFac in range(NF):
        FI = NItems + IFac
        SumLam = SumUnq = 0.0
        for J in range(cuts[IFac][0], cuts[IFac][1] + 1):
            Lam = R[FI, J]
            SumLam += Lam
            SumUnq += 1 - Lam ** 2
        Omega[IFac] = SumLam ** 2 / (SumLam ** 2 + SumUnq)
        if Iter == 0:
            Omega[IFac] = 1.0
    return R, Alpha, Omega


class corCFAResults:
    """Results of corCFA(): the item-factor loadings (if_cor),
    factor correlations (ff_cor), communalities, alpha, omega,
    predicted and residual correlations, and the plotly heat map
    in .plots."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return (f"<lessPy corCFA: {len(self.alpha)} factors, "
                f"{self.if_cor.shape[0]} items>")


def _parse_model(model, factors, fkw, cols):
    """Collect the factor -> item-name lists from a lavaan-style
    string, a factors dict, or F1=/F2= keyword lists."""
    fac = {}
    if model is not None:
        for line in model.splitlines():
            line = line.strip()
            if not line or "=~" not in line:
                continue
            name, rhs = line.split("=~")
            fac[name.strip()] = [t.strip()
                                 for t in rhs.split("+")]
    if factors is not None:
        fac.update({k: list(v) for k, v in factors.items()})
    for k in sorted(fkw, key=lambda s: int(s[1:])):
        fac[k] = list(fkw[k])
    if not fac:
        raise ValueError(
            "specify the measurement model: a lavaan-style "
            'string ("F1 =~ X1 + X2"), factors={...}, or '
            "F1=[...], F2=[...]")
    for name, items in fac.items():
        bad = [it for it in items if it not in cols]
        if bad:
            raise ValueError(
                f"factor {name}: items not in the correlation "
                f"matrix: {', '.join(bad)}")
    return fac


def corCFA(R, model=None, factors=None, min_cor=0.10,
           min_res=0.05, iter=50, sort=True, resid=True,
           item_cor=True, heat_map=True, **fkw):
    """Confirmatory factor analysis (MIMM) of a correlation
    matrix R (a DataFrame or array; raw data is correlated
    first). The measurement model assigns items to factors, given
    as a lavaan-style string, factors={...}, or F1=[...] lists.
    Prints reliability, the loadings and diagnostics, residuals,
    and the lavaan code; returns a corCFAResults. R analog:
    corCFA()"""
    Rm = pd.DataFrame(R)
    vals = Rm.to_numpy(dtype=float)
    if (Rm.shape[0] != Rm.shape[1]
            or not np.allclose(vals, vals.T, atol=1e-8)):
        Rm = Rm.select_dtypes("number").corr()
    Rm.index = Rm.columns = [str(c) for c in Rm.columns]
    Cmat = Rm.to_numpy(dtype=float)
    pos = {c: i for i, c in enumerate(Rm.columns)}

    fac = _parse_model(model, factors, fkw, set(Rm.columns))
    fac_names = list(fac)
    NF = len(fac_names)
    label, cuts = [], []
    for name in fac_names:
        start = len(label)
        for it in fac[name]:
            label.append(pos[it])
        cuts.append((start, len(label) - 1))
    NItems = len(label)

    def run(lab):
        sub = Cmat[np.ix_(lab, lab)]
        E = np.zeros((NItems + NF, NItems + NF))
        E[:NItems, :NItems] = sub
        return _mimm(E, cuts, NItems, NF, iter)

    Rout, alpha, omega = run(label)
    if sort:
        newlab = []
        for f in range(NF):
            n1, n2 = cuts[f]
            order = sorted(range(n1, n2 + 1),
                           key=lambda j: -Rout[NItems + f, j])
            newlab += [label[j] for j in order]
        label = newlab
        Rout, alpha, omega = run(label)

    items = [Rm.columns[i] for i in label]
    fcols = fac_names
    if_cor = pd.DataFrame(Rout[:NItems, NItems:], index=items,
                          columns=fcols)
    ff_cor = pd.DataFrame(Rout[NItems:, NItems:], index=fcols,
                          columns=fcols)
    commun = pd.Series(np.diag(Rout[:NItems, :NItems]),
                       index=items)

    lines = _cfa_text(items, fcols, cuts, alpha, omega, iter,
                      if_cor, Rout, NItems, NF)
    pred = res = None
    if resid:
        pred, res = _residuals(if_cor, ff_cor, Cmat, label,
                               items, cuts, NItems, NF, lines)
    lines += _lavaan(fcols, cuts, items)
    print("\n".join(lines))

    plots = {}
    if heat_map:
        plots["heatmap"] = _cfa_heatmap(Rout, items, fcols)
    return corCFAResults(
        if_cor=if_cor, ff_cor=ff_cor, communalities=commun,
        alpha=alpha, omega=omega, pred=pred, resid=res,
        plots=plots)


def _cfa_text(items, fcols, cuts, alpha, omega, iter, if_cor,
              Rout, NItems, NF):
    L = ["", "  FACTOR / SCALE COMPOSITION", ""]
    for f, name in enumerate(fcols):
        members = items[cuts[f][0]:cuts[f][1] + 1]
        L.append(f"{name}:  " + "  ".join(members))
    L += ["", "", "  RELIABILITY ANALYSIS", ""]
    w = max(6, max(len(s) for s in fcols))
    hdr = f"{'Scale':<{w}}{'Alpha':>9}"
    if iter > 0:
        hdr += f"{'Omega':>9}"
    L.append(hdr)
    for f, name in enumerate(fcols):
        row = f"{name:<{w}}{fmt(alpha[f], 3):>9}"
        if iter > 0:
            row += f"{fmt(omega[f], 3):>9}"
        L.append(row)
    if iter > 0:
        iw = max(10, max(len(s) for s in items) + 1)
        L += ["", "", "  SOLUTION", "", "Indicator Analysis", "",
              f"{'Factor':<8}{'Indicator':<{iw}}"
              f"{'Pattern':>9}{'Unique':>9}   Diagnostics"]
        for f, name in enumerate(fcols):
            for it_i in range(cuts[f][0], cuts[f][1] + 1):
                item = items[it_i]
                lam = Rout[NItems + f, it_i]
                unique = 1 - lam ** 2
                diag = ""
                if lam <= 0:
                    diag = "** Negative loading on own factor **"
                    unq = "   xxxx"
                elif unique <= 0:
                    diag = ("** Improper loading **"
                            if cuts[f][1] > cuts[f][0]
                            else "** Factor defined by one item **")
                    unq = fmt(unique, 3)
                else:
                    unq = fmt(unique, 3)
                    bad = [g + 1 for g in range(NF)
                           if abs(Rout[NItems + g, it_i]) > lam]
                    if bad:
                        diag = " ".join(f"F{b}" for b in bad)
                L.append(f"F{f + 1:<7}{item:<{iw}}"
                         f"{fmt(lam, 3):>9}{unq:>9}   {diag}")
    return L


def _residuals(if_cor, ff_cor, Cmat, label, items, cuts, NItems,
               NF, lines):
    lam = np.zeros((NItems, NF))
    for f in range(NF):
        for i in range(cuts[f][0], cuts[f][1] + 1):
            lam[i, f] = if_cor.iloc[i, f]
    phi = ff_cor.to_numpy()
    estR = lam @ phi @ lam.T
    np.fill_diagonal(estR, 1.0)
    Ritems = Cmat[np.ix_(label, label)]
    resR = np.round(Ritems - estR, 5)
    np.fill_diagonal(resR, 0.0)
    pred = pd.DataFrame(np.round(estR, 5), index=items,
                        columns=items)
    res = pd.DataFrame(resR, index=items, columns=items)

    ssq_tot = float((resR ** 2).sum())
    n = NItems
    rmsr = np.sqrt(ssq_tot / (n * n - n))
    avg_abs = float(np.abs(resR).sum()) / (n * n - n)
    lines += ["", "", "  RESIDUALS", "",
              f"Total sum of squares for all items: "
              f"{fmt(ssq_tot, 3)}",
              f"Root mean square residual (RMSR): {fmt(rmsr, 3)}",
              f"Average absolute residual off-diagonal: "
              f"{fmt(avg_abs, 3)}"]
    return pred, res


def _lavaan(fcols, cuts, items):
    L = ["", "", "  LAVAAN / semopy SPECIFICATION", "",
         "MeasModel = \"\"\""]
    for f, name in enumerate(fcols):
        members = items[cuts[f][0]:cuts[f][1] + 1]
        L.append(f"  {name} =~ " + " + ".join(members))
    L += ["\"\"\"",
          "# lavaan (R):  cfa(MeasModel, data=d, std.lv=True)",
          "# semopy (Python):  Model(MeasModel).fit(d)"]
    return L


def _cfa_heatmap(Rout, items, fcols):
    import plotly.graph_objects as go
    from .plotly_utils import plotly_style, to_hex
    labels = items + fcols
    style = plotly_style()
    fig = go.Figure(go.Heatmap(
        z=Rout, x=labels, y=labels, zmin=-1, zmax=1,
        colorscale="RdBu", reversescale=True,
        colorbar=dict(title="r")))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        template=None, paper_bgcolor=to_hex(style["window_fill"]),
        title=dict(text="Item / Factor Correlations", x=0.5,
                   xanchor="center",
                   font=dict(size=round(
                       16 * get_option("main_size", 1)))))
    return fig
