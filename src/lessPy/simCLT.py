# simCLT.py — analog of simCLT.R.
#
# simCLT(): a Central Limit Theorem simulation. Draw ns samples,
# each of size n, from a chosen population (normal, uniform,
# lognormal, or a U-shaped "antinormal"), then show the sampling
# distribution of the sample mean converging toward normality.
# Prints the population and sample statistics as in R and returns
# a results object with the two plotly figures in .plots.
#
# The random draws use numpy's RNG, so they do not reproduce R's
# stream value-for-value (a simulation, like Flows). The population
# constants (mu, sigma, lognormal skew/median) and the CLT
# standardization are the exact R formulas. R's "antinormal" needs
# the external "triangle" package; here the triangular components
# come from scipy.stats.triang, so no extra dependency is added.
#
# NOTE: R's uniform branch samples runif(ns*n, 0, 4) — it ignores
# p1/p2 while still reporting mu/sigma from p1/p2. That is a latent
# bug; this port samples uniform(p1, p2) so the data match the
# reported population.

import numpy as np

from .utils import fmt

_GHOST = "#F8F8FF"
_STEEL = "#A2B5CD"          # lightsteelblue3


class SimCLTResults:
    """Numeric results of simCLT(): the population parameters, the
    data and sample-mean summaries, the observed 95% ranges, the
    per-sample means/sds, and the two plotly figures in .plots
    ("population" and "sampling")."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return (f"<lessPy simCLT: {self.dist}, "
                f"ns={self.n_samples}, n={self.n}>")


def simCLT(ns=None, n=None, p1=0, p2=1, seed=None, dist="normal",
           fill=_STEEL, n_display=0, digits_d=3, subtitle=True,
           pop=True):
    """Central Limit Theorem simulation: ns samples each of size n
    from a population (dist = "normal", "uniform", "lognormal", or
    "antinormal"). p1/p2 are the population parameters (normal:
    mean/sd; uniform: min/max; lognormal: meanlog/sdlog;
    antinormal: 0/max). Prints the statistics and returns a
    SimCLTResults with .plots. R analog: simCLT()"""
    if dist not in ("normal", "uniform", "lognormal", "antinormal"):
        raise ValueError('dist: "normal", "uniform", '
                         '"lognormal", "antinormal"')
    if ns is None:
        raise ValueError("specify the number of samples: ns")
    if n is None:
        raise ValueError("specify the size of each sample: n")

    rng = np.random.default_rng(seed)
    skew = med = None
    sigma = None

    if dist == "normal":
        mu, sigma = p1, p2
        data = rng.normal(mu, sigma, ns * n)
        fig_pop = _pop_normal(mu, sigma, fill, subtitle)

    elif dist == "uniform":
        lo, hi = p1, p2
        mu = (lo + hi) / 2
        sigma = (hi - lo) / np.sqrt(12)
        data = rng.uniform(lo, hi, ns * n)
        fig_pop = _pop_uniform(lo, hi, fill, subtitle)

    elif dist == "lognormal":
        meanlog, sdlog = p1, p2
        vlog = sdlog ** 2
        mu = np.exp(meanlog + vlog / 2)
        sigma = mu * np.sqrt(np.exp(vlog) - 1)
        skew = (np.exp(vlog) + 2) * np.sqrt(np.exp(vlog) - 1)
        med = np.exp(meanlog)
        data = rng.lognormal(meanlog, sdlog, ns * n)
        fig_pop = _pop_lognormal(meanlog, sdlog, mu, sigma, fill,
                                 subtitle)

    else:                                  # antinormal (U-shaped)
        if p1 != 0:
            raise ValueError("minimum of the anti-normal "
                             "distribution must be 0")
        xmax = p2
        mu = xmax / 2
        data = _rantinormal(rng, ns * n, xmax)
        fig_pop = _pop_antinormal(xmax, fill, subtitle)

    mx = data.mean()
    sx = data.std(ddof=1)
    by_rep = data.reshape(ns, n)
    ymean = by_rep.mean(axis=1)
    ysd = by_rep.std(axis=1, ddof=1)

    fig_samp = _samp_plot(ymean, fill, ns, n, dist, subtitle)

    q_units = np.quantile(ymean, [0.025, 0.975])
    se = (sx if dist == "antinormal" else sigma) / np.sqrt(n)
    z = (ymean - mu) / se
    q_se = np.quantile(z, [0.025, 0.975])

    _report(dist, mu, sigma, skew, med, ns, n, mx, sx, ymean,
            q_units, q_se, digits_d, n_display, by_rep)

    return SimCLTResults(
        dist=dist, mu=mu, sigma=sigma, skew=skew, median=med,
        n_samples=ns, n=n, data_mean=mx, data_sd=sx,
        mean_of_means=ymean.mean(), sd_of_means=ymean.std(ddof=1),
        range_units=q_units, range_se=q_se, ymean=ymean, ysd=ysd,
        plots={"population": fig_pop, "sampling": fig_samp})


# --- population sampling -------------------------------------------

def _rantinormal(rng, size, xmax):
    from scipy.stats import triang
    half = xmax / 2
    u = rng.random(size)
    lo = u < 0.5
    out = np.empty(size)
    # lower half: triangle a=0, b=half, mode ~0 (decreasing)
    lo_c = 0.01 / half
    out[lo] = triang.ppf(rng.random(lo.sum()), lo_c, loc=0,
                         scale=half)
    # upper half: triangle a=half, b=xmax, mode ~xmax (increasing)
    hi_c = (half - 0.01) / half
    out[~lo] = triang.ppf(rng.random((~lo).sum()), hi_c,
                          loc=half, scale=half)
    return out


# --- population figures --------------------------------------------

def _pop_fig(x, y, xlab, sub, fill):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate(([x[0]], x, [x[-1]])),
        y=np.concatenate(([0.0], y, [0.0])),
        mode="lines", fill="toself",
        fillcolor=fill, line=dict(color="black"),
        hoverinfo="skip", showlegend=False))
    fig.update_layout(
        plot_bgcolor=_GHOST, paper_bgcolor="white",
        xaxis=dict(title=xlab, showgrid=False, zeroline=False,
                   linecolor="black", mirror=True),
        yaxis=dict(showticklabels=False, showgrid=False,
                   zeroline=False, linecolor="black", mirror=True),
        margin=dict(t=30, r=20, b=50, l=30),
        title=dict(text=sub or "", x=0.5, xanchor="center",
                   font=dict(size=11)))
    return fig


def _pop_normal(mu, sigma, fill, subtitle):
    from scipy.stats import norm
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
    sub = (f"mu={fmt(mu, 3)}  sigma={fmt(sigma, 3)}"
           if subtitle else "")
    return _pop_fig(x, norm.pdf(x, mu, sigma),
                    "Normal Population", sub, fill)


def _pop_uniform(lo, hi, fill, subtitle):
    x = np.linspace(lo, hi, 500)
    y = np.full_like(x, 1.0 / (hi - lo))
    sub = f"min={lo} max={hi}" if subtitle else ""
    return _pop_fig(x, y, "Uniform Population", sub, fill)


def _pop_lognormal(meanlog, sdlog, mu, sigma, fill, subtitle):
    from scipy.stats import lognorm
    xmax = np.ceil(mu + 4 * sigma)
    x = np.linspace(1e-6, xmax, 500)
    y = lognorm.pdf(x, sdlog, scale=np.exp(meanlog))
    sub = (f"meanlog={meanlog} sdlog={sdlog}"
           if subtitle else "")
    return _pop_fig(x, y, "Lognormal Population", sub, fill)


def _pop_antinormal(xmax, fill, subtitle):
    from scipy.stats import triang
    half = xmax / 2
    x1 = np.linspace(0, half, 250)
    y1 = triang.pdf(x1, 0.01 / half, loc=0, scale=half)
    x2 = np.linspace(half, xmax, 250)
    y2 = triang.pdf(x2, (half - 0.01) / half, loc=half, scale=half)
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    sub = f"min=0 max={xmax}" if subtitle else ""
    return _pop_fig(x, y, "Anti-Normal Population", sub, fill)


# --- sampling-distribution figure ----------------------------------

def _samp_plot(ymean, fill, ns, n, dist, subtitle):
    import plotly.graph_objects as go
    from scipy.stats import norm
    m, s = ymean.mean(), ymean.std(ddof=1)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ymean, histnorm="probability density",
        marker=dict(color=fill, line=dict(color="black", width=1)),
        showlegend=False, hoverinfo="skip"))
    x = np.linspace(ymean.min(), ymean.max(), 300)
    fig.add_trace(go.Scatter(
        x=x, y=norm.pdf(x, m, s), mode="lines",
        line=dict(color="black"), showlegend=False,
        hoverinfo="skip"))
    sub = (f"{ns} samples, each of size {n} from {dist}"
           if subtitle else "")
    fig.update_layout(
        bargap=0, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Sample Mean", showgrid=False,
                   linecolor="black", mirror=True),
        yaxis=dict(showticklabels=False, showgrid=False,
                   linecolor="black", mirror=True),
        margin=dict(t=30, r=20, b=50, l=30),
        title=dict(text=sub, x=0.5, xanchor="center",
                   font=dict(size=11)))
    return fig


# --- text report ----------------------------------------------------

def _report(dist, mu, sigma, skew, med, ns, n, mx, sx, ymean,
            q_units, q_se, digits_d, n_display, by_rep):
    print()
    print(f"Population mean, mu : {mu}")
    if sigma is not None:
        print(f"Pop std dev, sigma  : {sigma}")
    if dist == "lognormal":
        print(f"Population skew:      {skew}")
        print(f"Population median:    {med}")
    print()
    print(f"Number of samples   : {ns}")
    print(f"Size of each sample : {n}")
    print()
    print(f"Mean of the data: {mx}")
    print(f"Std Dev of the data: {sx}")
    print()
    print("Analysis of Sample Means")
    print(f"   Mean: {fmt(ymean.mean(), digits_d)}")
    print(f"Std Dev: {fmt(ymean.std(ddof=1), digits_d)}")
    print()
    print("Observed 95% range of Sample Mean")
    print(f"    Original Units: {q_units[0]} {q_units[1]}")
    lead = ("   Estimated from Data,  "
            if dist == "antinormal" else "  ")
    print(f"{lead}Standard Errors: {q_se[0]} {q_se[1]}")
    if n_display > 0:
        for i in range(n_display):
            vals = " ".join(fmt(v, digits_d) for v in by_rep[i])
            print(f"\nSample {i + 1}")
            print(f" Mean: {fmt(ymean[i], digits_d)}")
            print(f" Rounded Data Values: {vals}")
    print()
