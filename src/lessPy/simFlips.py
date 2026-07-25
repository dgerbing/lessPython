# simFlips.py — analog of simFlips.R.
#
# simFlips(): flip a coin n times and plot the running proportion
# of heads converging toward the true probability prob — a law of
# large numbers demonstration. Optionally shows each individual
# 0/1 flip. Prints a short summary and returns a results object
# with the plotly figure in .plots.
#
# The flips use numpy's RNG (seed=), so they do not reproduce R's
# stream value-for-value (a simulation, like simCLT/simMeans).
# R's interactive "pause" mode is not ported (no batch equivalent).

import numpy as np

from .utils import fmt


class SimFlipsResults:
    """Numeric results of simFlips(): the 0/1 flips, the running
    proportion of heads, the head/tail counts, the target prob,
    the final running mean, and the plotly figure in .plots
    ("flips")."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return (f"<lessPy simFlips: n={self.n}, "
                f"heads={self.n_heads}>")


def simFlips(n=None, prob=0.5, seed=None, show_title=True,
             show_flips=True, grid="#E5E5E5"):
    """Flip a coin n times (P(heads) = prob) and plot the running
    proportion of heads against the number of flips, with a
    reference line at prob. show_flips overlays each 0/1 flip.
    Prints a summary and returns a SimFlipsResults with .plots.
    R analog: simFlips()"""
    if n is None:
        raise ValueError("specify the number of flips: n")
    if not 0 <= prob <= 1:
        raise ValueError("prob must be between 0 and 1")

    rng = np.random.default_rng(seed)
    flips = rng.binomial(1, prob, n)
    ybar = np.cumsum(flips) / np.arange(1, n + 1)
    n_heads = int(flips.sum())
    n_tails = n - n_heads

    fig = _plot(n, flips, ybar, prob, grid, show_flips,
                show_title, n_heads, n_tails)

    print(f"\nNumber of flips: {n}")
    print(f"P(heads)       : {prob}")
    print(f"Heads          : {n_heads}")
    print(f"Tails          : {n_tails}")
    print(f"Sample mean after {n} flips: {fmt(ybar[-1], 3)}")
    print()

    return SimFlipsResults(
        n=n, prob=prob, flips=flips, running_mean=ybar,
        n_heads=n_heads, n_tails=n_tails, final_mean=ybar[-1],
        plots={"flips": fig})


def _plot(n, flips, ybar, prob, grid, show_flips, show_title,
          n_heads, n_tails):
    import plotly.graph_objects as go
    fig = go.Figure()
    # reference line at the true probability
    fig.add_hline(y=prob, line=dict(color="lightsteelblue",
                                    width=2))
    if show_flips:
        fig.add_trace(go.Scatter(
            x=list(range(1, n + 1)), y=flips, mode="markers",
            marker=dict(symbol="diamond", size=6, color="darkblue",
                        line=dict(color="lightsteelblue", width=1)),
            showlegend=False,
            hovertemplate="flip %{x}: %{y}<extra></extra>"))
    # running proportion of heads
    fig.add_trace(go.Scatter(
        x=list(range(1, n + 1)), y=ybar, mode="lines",
        line=dict(color="gray", width=3), showlegend=False,
        hovertemplate="after %{x}: %{y:.3f}<extra></extra>"))

    for y, txt in ((prob, f"μ={prob}"), (1, f"{n_heads} Heads"),
                   (0, f"{n_tails} Tails")):
        fig.add_annotation(
            xref="paper", x=1.005, y=y, xanchor="left",
            text=txt, showarrow=False, font=dict(size=10))

    title = (f"Sample Mean after {n} Coin Flips: "
             f"{fmt(ybar[-1], 3)}") if show_title else ""
    fig.update_layout(
        plot_bgcolor="#F8F8FF", paper_bgcolor="white",
        margin=dict(t=36, r=70, b=40, l=50),
        xaxis=dict(title="Number of Flips", range=[1, n],
                   showgrid=False, linecolor="black", mirror=True),
        yaxis=dict(title="Estimate", range=[0, 1], showgrid=True,
                   gridcolor=grid, gridwidth=0.5,
                   linecolor="black", mirror=True),
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=12)))
    return fig
