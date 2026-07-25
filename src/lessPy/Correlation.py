# Correlation.py — analog of Correlation.R (cr.main, cr.data.frame).
#
# Correlation(): with two variables, the correlation of a pair
# with its significance test (Pearson t-test and Fisher-z 95% CI,
# or Spearman/Kendall); with a data frame or a set of variables,
# the correlation matrix. miss= controls pairwise/listwise/
# everything deletion; show="missing" reports the pairwise
# complete-case counts. Prints as R; the two-variable form
# returns a CorrelationResults, the matrix form the correlation
# DataFrame (its .attrs["plots"] holds the heat map).

import numpy as np
import pandas as pd

from .utils import fmt, get_column, get_option


class CorrelationResults:
    """Results of a two-variable Correlation(): the coefficient,
    t-test, and (Pearson) confidence interval."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return (f"<lessPy Correlation: {self.method}, "
                f"r={self.r}, p={self.pvalue}>")


def Correlation(x=None, y=None, data=None, miss="pairwise",
                show="cor", method="pearson", brief=False,
                digits_d=None, heat_map=True, main=None):
    """Correlation of two variables (with a significance test) or
    the correlation matrix of several. x and y are column names
    (or arrays); with y omitted, x is a DataFrame or a list of
    column names (or None for all numeric columns of data).
    method: "pearson" (default), "spearman", "kendall".
    R analog: Correlation()"""
    if miss not in ("pairwise", "listwise", "everything"):
        raise ValueError('miss: "pairwise", "listwise", '
                         '"everything"')
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError('method: "pearson", "spearman", '
                         '"kendall"')
    if y is not None:
        xv = _resolve1(x, data, "x")
        yv = _resolve1(y, data, "y")
        return _two_var(xv, yv, _name(x, "x"), _name(y, "y"),
                        method, brief)
    return _matrix(_frame(x, data), miss, show, method, digits_d,
                   heat_map, main)


def _name(v, default):
    return v if isinstance(v, str) else default


def _resolve1(v, data, arg):
    if isinstance(v, str):
        if data is None:
            raise ValueError(f"{arg}='{v}' is a column name, so "
                             "data= is required")
        return get_column(data, v, arg).to_numpy(dtype=float)
    return np.asarray(v, dtype=float)


def _frame(x, data):
    if isinstance(x, pd.DataFrame):
        df = x
    elif isinstance(x, (list, tuple)):
        if data is None:
            raise ValueError("data= is required with a list of "
                             "variable names")
        df = data[list(x)]
    elif x is None:
        if data is None:
            raise ValueError("supply a DataFrame, a list of "
                             "variables with data=, or data=")
        df = data
    else:
        raise TypeError("for a correlation matrix, x is a "
                        "DataFrame or a list of variable names")
    return df


# ------------------------------------------------------------
# two variables
# ------------------------------------------------------------

def _two_var(x, y, x_name, y_name, method, brief):
    from scipy import stats as sps
    keep = ~(np.isnan(x) | np.isnan(y))
    n = int(keep.sum())
    n_del = int((~keep).sum())
    xk, yk = x[keep], y[keep]
    lb = ub = np.nan

    if method == "pearson":
        r, p = sps.pearsonr(xk, yk)
        df = n - 2
        t = r * np.sqrt(df / (1 - r ** 2))
        p = 2 * sps.t.sf(abs(t), df)
        z = np.arctanh(r)
        sig = 1 / np.sqrt(n - 3)
        zc = sps.norm.ppf(0.975)
        lb, ub = np.tanh(z - zc * sig), np.tanh(z + zc * sig)
        method_txt = "Pearson's product-moment correlation"
        sym, sym_pop = "r", "Correlation"
        cov = float(np.cov(xk, yk)[0, 1])
    elif method == "spearman":
        res = sps.spearmanr(xk, yk)
        r, p = float(res.statistic), float(res.pvalue)
        t, df = res.statistic * np.sqrt(
            (n - 2) / (1 - r ** 2)), n - 2
        method_txt = "Spearman's rank correlation rho"
        sym = sym_pop = "rho"
    else:
        res = sps.kendalltau(xk, yk)
        r, p = float(res.statistic), float(res.pvalue)
        t = res.statistic
        df = None
        method_txt = "Kendall's rank correlation tau"
        sym = sym_pop = "tau"

    L = []
    if not brief:
        L += [f"Correlation Analysis for Variables {x_name} "
              f"and {y_name}", ""]
    L += [f"\n>>> {method_txt}", "",
          "Number of paired values with neither missing, "
          f"n = {n}"]
    if not brief:
        L.append(f"Number of cases (rows of data) deleted: "
                 f"{n_del}")
    L.append("")
    if method == "pearson" and not brief:
        L += [f"Sample Covariance: s = {fmt(cov, 3)}", ""]
    if brief:
        L.append(f"Sample Correlation of {x_name} and {y_name}: "
                 f"{sym} = {fmt(r, 3)}")
    else:
        L.append(f"Sample Correlation: {sym} = {fmt(r, 3)}")
    L.append("")
    dfs = "NA" if df is None else str(df)
    L.append(f"Hypothesis Test of 0 {sym_pop}:  t = {fmt(t, 3)}, "
             f" df = {dfs},  p-value = {fmt(p, 3)}")
    if method == "pearson":
        L.append("95% Confidence Interval for Correlation:  "
                 f"{fmt(lb, 3)} to {fmt(ub, 3)}")
    print("\n".join(L))

    return CorrelationResults(
        method=method, r=round(float(r), 3),
        tvalue=round(float(t), 3),
        df=(None if df is None else int(df)),
        pvalue=round(float(p), 3),
        lb=(round(lb, 3) if np.isfinite(lb) else None),
        ub=(round(ub, 3) if np.isfinite(ub) else None), n=n)


# ------------------------------------------------------------
# correlation matrix
# ------------------------------------------------------------

def _matrix(df, miss, show, method, digits_d, heat_map, main):
    num = df.select_dtypes("number")
    dropped = [c for c in df.columns if c not in num.columns]
    if num.shape[1] < 2:
        raise ValueError(
            "a correlation matrix needs at least 2 numeric "
            "variables")
    d = digits_d if digits_d is not None else 2

    if miss == "listwise":
        crs = num.dropna().corr(method=method)
    else:
        crs = num.corr(method=method)
        if miss == "everything":
            for c in num.columns[num.isna().any()]:
                crs.loc[c, :] = np.nan
                crs.loc[:, c] = np.nan
    crs = crs.round(d)

    if dropped:
        print("The following non-numeric variables are deleted "
              "from the analysis")
        for i, c in enumerate(dropped, 1):
            print(f"{i}. {c}")
    tot_miss = int(num.isna().sum().sum())
    if tot_miss == 0:
        print("\n>>> No missing data\n")
    else:
        print(f"\nMissing data deletion: {miss}")

    if show == "missing":
        n = pd.DataFrame(
            np.zeros((num.shape[1], num.shape[1]), dtype=int),
            index=num.columns, columns=num.columns)
        for i in num.columns:
            for j in num.columns:
                n.loc[i, j] = int((~(num[i].isna()
                                     | num[j].isna())).sum())
        print("\nPairwise complete-case counts\n")
        print(n.to_string())
        crs.attrs["missing_n"] = n
        return crs

    print("\nCorrelation Matrix")
    print(crs.round(2).to_string())
    plots = {}
    if heat_map:
        plots["heatmap"] = _heatmap(crs, main)
    crs.attrs["plots"] = plots
    return crs


def _heatmap(df, main):
    import plotly.graph_objects as go

    from .plotly_utils import plotly_style, to_hex
    labels = list(df.columns)
    Z = df.to_numpy(dtype=float).copy()
    np.fill_diagonal(Z, np.nan)
    style = plotly_style()
    fig = go.Figure(go.Heatmap(
        z=Z, x=labels, y=labels, zmin=-1, zmax=1,
        colorscale="RdBu", reversescale=True,
        colorbar=dict(title="r")))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        template=None, paper_bgcolor=to_hex(style["window_fill"]),
        title=dict(text=main or "Correlations", x=0.5,
                   xanchor="center",
                   font=dict(size=round(
                       16 * get_option("main_size", 1)))))
    return fig
