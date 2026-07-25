# simMeans.py — analog of simMeans.R.
#
# simMeans(): draw ns samples, each of size n, from a normal
# population and plot every sample mean against its sample index,
# with a horizontal centerline at the population mean mu. Shows
# how the sample means scatter about mu and how their spread (the
# empirical standard error) shrinks as n grows. Prints the per-
# sample means, SDs and raw values, then the analysis of the
# sample means, and returns a results object with the plotly
# figure in .plots.
#
# The draws use numpy's RNG (seed=), so they do not reproduce R's
# stream value-for-value (a simulation, like simCLT). R's
# interactive "pause" mode (one sample per Enter press) has no
# batch equivalent and is not ported.

import numpy as np

from .utils import fmt


class SimMeansResults:
    """Numeric results of simMeans(): the population mu/sigma, the
    per-sample means and SDs (in display order), the display order
    index, the mean of the sample means and their SD (the direct
    standard-error estimate), and the plotly figure in .plots
    ("means")."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return (f"<lessPy simMeans: ns={self.n_samples}, "
                f"n={self.n}>")


def simMeans(ns=None, n=None, mu=0, sigma=1, seed=None,
             show_title=True, show_data=True, max_data=10,
             grid="#E5E5E5", ylim_bound=None, sort=True,
             set_mu=False, digits_d=2):
    """Simulate ns samples of size n from a normal population
    (mean mu, sd sigma) and plot each sample mean vs its index,
    centered on mu. sort orders the samples by their mean;
    set_mu hides a randomly chosen mu/sigma (a guess-the-mean
    exercise); show_data lists up to max_data raw values per
    sample. Prints the tables and returns a SimMeansResults with
    .plots. R analog: simMeans()"""
    if ns is None:
        raise ValueError("specify the number of samples: ns")
    if n is None:
        raise ValueError("specify the size of each sample: n")
    if sigma < 0:
        raise ValueError("sigma cannot be negative")

    rng = np.random.default_rng(seed)
    if set_mu:
        mu = int(rng.integers(0, 101))
        sigma = int(rng.integers(1, 26))

    if not set_mu:
        print(f"\nPopulation mean, mu: {mu}")
        print(f"Pop std dev, sigma : {sigma}")
    print(f"\nNumber of samples  : {ns}")
    print(f"Size of each sample: {n}")

    data = rng.normal(mu, sigma, ns * n).reshape(ns, n)
    ymean = data.mean(axis=1)
    ysd = data.std(axis=1, ddof=1)

    if sort:
        o = np.argsort(ymean, kind="stable")
    else:
        o = np.arange(ns)
    ymean, ysd, data = ymean[o], ysd[o], data[o]

    if ylim_bound is None:
        dev = max(abs(mu - ymean.min()), abs(ymean.max() - mu))
        lo, hi = mu - dev, mu + dev
    else:
        lo, hi = mu - ylim_bound, mu + ylim_bound

    fig = _plot(ns, ymean, mu, sigma, n, lo, hi, grid,
                show_title and not set_mu)

    _report(ns, n, o, ymean, ysd, data, mu, sigma, max_data,
            show_data, digits_d)

    return SimMeansResults(
        mu=mu, sigma=sigma, n_samples=ns, n=n, order=o,
        ymean=ymean, ysd=ysd, mean_of_means=ymean.mean(),
        se=ymean.std(ddof=1), plots={"means": fig})


def _plot(ns, ymean, mu, sigma, n, lo, hi, grid, show_title):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_hline(y=mu, line=dict(color="darkslateblue", width=1.5))
    fig.add_trace(go.Scatter(
        x=list(range(1, ns + 1)), y=ymean, mode="markers",
        marker=dict(symbol="circle", size=7, color="#4398D0",
                    line=dict(color="black", width=1)),
        showlegend=False,
        hovertemplate="sample %{x}<br>mean %{y:.3f}"
                      "<extra></extra>"))
    title = (f"μ={mu}  σ={sigma}  n={n}"
             if show_title else "")
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=36, r=30, b=30, l=50),
        xaxis=dict(title="", range=[1, ns], showgrid=True,
                   gridcolor=grid, gridwidth=0.5,
                   linecolor="black", mirror=True),
        yaxis=dict(title="Sample Mean", range=[lo, hi],
                   showgrid=True, gridcolor=grid, gridwidth=0.5,
                   linecolor="black", mirror=True),
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=12)))
    return fig


def _report(ns, n, o, ymean, ysd, data, mu, sigma, max_data,
            show_data, digits_d):
    maxd = min(n, max_data)
    w = max(8, digits_d + 6)
    head = f"\n{'Sample':>6}{'Mean':>{w}}{'SD':>{w}}"
    if show_data:
        head += "   " + "".join(f"{j:>{w}}"
                                 for j in range(1, maxd + 1))
    print(head)
    for i in range(ns):
        row = (f"{int(o[i]) + 1:>6}"
               f"{fmt(ymean[i], digits_d):>{w}}"
               f"{fmt(ysd[i], digits_d):>{w}}   ")
        if show_data:
            row += "".join(f"{fmt(data[i, j], digits_d):>{w}}"
                           for j in range(maxd))
        if n > max_data:
            row += " ..."
        print(row)

    print("\nAnalysis of Sample Means")
    print(f"   Mean: {fmt(ymean.mean(), digits_d):>{w}}")
    print(f"Std Dev: {fmt(ymean.std(ddof=1), digits_d):>{w}}"
          "    Direct estimate of the standard error of the "
          "sample mean")
    print()
