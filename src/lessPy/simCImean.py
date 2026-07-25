# simCImean.py — analog of simCImean.R.
#
# simCImean(): draw ns samples of size n from a normal population
# and form a confidence interval for the mean from each. Plot the
# intervals as vertical bars against a centerline at the true mean
# mu, colored by whether they capture mu, and report the miss
# rate — a demonstration that a cl-level interval misses mu about
# (1 - cl) of the time. Prints the per-sample table and the
# structure/performance summaries, and returns a results object
# with the plotly figure in .plots.
#
# The draws use numpy's RNG (seed=), so they do not reproduce R's
# stream value-for-value (a simulation, like the other sim*).
# R's interactive "pause" mode is not ported.

import numpy as np

from .utils import fmt


class SimCImeanResults:
    """Numeric results of simCImean(): the per-sample means, SDs,
    interval bounds (lb/ub) and hit/miss flags, the miss count and
    rate, the t cutoff, and the plotly figure in .plots ("ci")."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return (f"<lessPy simCImean: ns={self.n_samples}, "
                f"n={self.n}, misses={self.n_miss}>")


def simCImean(ns=None, n=None, mu=0, sigma=1, cl=0.95, seed=None,
              show_data=False, show_title=True, miss_only=False,
              color_hit="gray", color_miss="red", grid="#E5E5E5",
              ylim_bound=None, digits_d=3):
    """Simulate ns confidence intervals for a mean: ns samples of
    size n from a normal population (mean mu, sd sigma), each
    giving a cl-level t interval. Plots the intervals about a
    centerline at mu, colored by hit/miss, and reports the miss
    rate. miss_only lists only the intervals that miss mu;
    show_data overlays the raw values. Prints the tables and
    returns a SimCImeanResults with .plots. R analog: simCImean()"""
    from scipy.stats import t as t_dist
    if ns is None:
        raise ValueError("specify the number of samples: ns")
    if n is None:
        raise ValueError("specify the size of each sample: n")
    if sigma < 0:
        raise ValueError("sigma cannot be negative")
    if not 0 < cl < 1:
        raise ValueError("cl must be between 0 and 1")

    alpha = 1 - cl
    tcut = t_dist.ppf(1 - alpha / 2, df=n - 1)

    rng = np.random.default_rng(seed)
    data = rng.normal(mu, sigma, ns * n).reshape(ns, n)
    ymean = data.mean(axis=1)
    ysd = data.std(axis=1, ddof=1)

    se = ysd / np.sqrt(n)
    E = tcut * se
    lb, ub = ymean - E, ymean + E
    hit = (lb < mu) & (ub > mu)
    n_miss = int((~hit).sum())
    miss_rate = round(n_miss / ns * 100, 2)

    if ylim_bound is None:
        if show_data:
            lo0, hi0 = data.min(), data.max()
        else:
            lo0, hi0 = lb.min(), ub.max()
        dev = max(abs(mu - lo0), abs(hi0 - mu))
        lo, hi = mu - dev, mu + dev
    else:
        lo, hi = mu - ylim_bound, mu + ylim_bound

    fig = _plot(ns, n, ymean, E, lb, ub, hit, mu, sigma, cl, data,
                lo, hi, grid, color_hit, color_miss, show_data,
                show_title)

    _report(ns, n, ymean, ysd, se, E, lb, ub, hit, mu, sigma, cl,
            n_miss, miss_rate, miss_only, digits_d)

    return SimCImeanResults(
        mu=mu, sigma=sigma, cl=cl, n_samples=ns, n=n, tcut=tcut,
        ymean=ymean, ysd=ysd, lb=lb, ub=ub, hit=hit,
        n_miss=n_miss, miss_rate=miss_rate,
        mean_of_means=ymean.mean(), sd_of_means=ymean.std(ddof=1),
        plots={"ci": fig})


def _plot(ns, n, ymean, E, lb, ub, hit, mu, sigma, cl, data, lo,
          hi, grid, color_hit, color_miss, show_data, show_title):
    import plotly.graph_objects as go
    x = np.arange(1, ns + 1)
    fig = go.Figure()
    if show_data:
        fig.add_trace(go.Scatter(
            x=np.repeat(x, n), y=data.ravel(), mode="markers",
            marker=dict(size=2.5, color="gray"), opacity=0.5,
            showlegend=False, hoverinfo="skip"))
    fig.add_hline(y=mu, line=dict(color="darkslateblue",
                                  width=1.5))
    for mask, col, nm in ((hit, color_hit, "capture μ"),
                          (~hit, color_miss, "miss μ")):
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=x[mask], y=ymean[mask], mode="markers",
            marker=dict(size=3, color=col),
            error_y=dict(type="data", array=E[mask], color=col,
                         thickness=1.5, width=3),
            name=nm, showlegend=False,
            hovertemplate="sample %{x}<br>mean %{y:.3f}"
                          "<extra></extra>"))
    title = (f"μ={mu}  σ={sigma}  cl={round(cl * 100, 2)}%  "
             f"n={n}") if show_title else ""
    fig.update_layout(
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        margin=dict(t=36, r=30, b=30, l=50),
        xaxis=dict(title="", range=[0, ns + 1], showgrid=False,
                   linecolor="black", mirror=True),
        yaxis=dict(title="", range=[lo, hi], showgrid=True,
                   gridcolor=grid, gridwidth=0.5,
                   linecolor="black", mirror=True),
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=12)))
    return fig


def _report(ns, n, ymean, ysd, se, E, lb, ub, hit, mu, sigma, cl,
            n_miss, miss_rate, miss_only, digits_d):
    w = 8
    def c(v):
        return f"{fmt(v, digits_d):>{w}}"
    print(f"\n{'Sample':>6}{'Mean':>{w}}{'StdDev':>{w}}"
          f"{'StdErr':>{w}}{'Error':>{w}}{'LB':>{w}}{'UB':>{w}}")
    for i in range(ns):
        if miss_only and hit[i]:
            continue
        row = (f"{i + 1:>6}{c(ymean[i])}{c(ysd[i])}{c(se[i])}"
               f"{c(E[i])}{c(lb[i])}{c(ub[i])}")
        if not hit[i]:
            row += "  *** MISS ***"
        print(row)

    print("\nStructure")
    print(f"Population mean, mu : {mu}")
    print(f"Pop std dev, sigma  : {sigma}")
    print(f"Number of samples   : {ns}")
    print(f"Size of each sample : {n}")
    print(f"Confidence level    : {cl}")

    print("\nPerformance")
    print(f"Number of Misses: {n_miss}")
    print(f"Percent Misses  : {miss_rate:.2f}")

    d2 = digits_d + 1
    print("\nAnalysis of Sample Means")
    print(f"   Mean: {fmt(ymean.mean(), d2):>{w}}")
    print(f"Std Dev: {fmt(ymean.std(ddof=1), d2):>{w}}")
    print()
