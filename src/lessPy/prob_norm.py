# prob_norm.py — analog of prob_norm.R.
#
# prob_norm(): plot a normal density curve, shade the interval
# from lo to hi, and return the probability P(lo < Y < hi). An
# open bound (lo or hi = None) extends to the tail. Prints the
# probability and returns a results object with the plotly figure
# in .plots.

import numpy as np

from .utils import fmt

_FILL_NRM = "#E8E8E8"      # grey91
_FILL_INT = "#9FB6CD"      # slategray3


class ProbNormResults:
    """Numeric results of prob_norm(): the interval probability,
    the bounds lo/hi, mu/sigma, and the plotly figure in .plots
    ("norm")."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return f"<lessPy prob_norm: P={self.prob:.4g}>"


def prob_norm(lo=None, hi=None, mu=0, sigma=1, nrm_color="black",
              fill_nrm=_FILL_NRM, fill_int=_FILL_INT, y_axis=False,
              z=True):
    """Probability of a normal interval: shade the area under the
    N(mu, sigma) curve from lo to hi and return P(lo < Y < hi). A
    None bound extends to the tail. z adds a second axis of z
    scores (suppressed for the standard normal). Prints the
    probability and returns a ProbNormResults with .plots.
    R analog: prob_norm()"""
    from scipy.stats import norm
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"lo ({lo}) cannot be larger than hi "
                         f"({hi})")
    if sigma <= 0:
        raise ValueError("sigma must be larger than zero")

    if mu == 0 and sigma == 1:
        z = False

    lo_lbl = "..." if lo is None else str(lo)
    hi_lbl = "..." if hi is None else str(hi)
    lo_v = mu - sigma * 10 if lo is None else lo
    hi_v = mu + sigma * 10 if hi is None else hi

    prob = norm.cdf(hi_v, mu, sigma) - norm.cdf(lo_v, mu, sigma)

    fig = _plot(lo_v, hi_v, lo_lbl, hi_lbl, mu, sigma, prob,
                nrm_color, fill_nrm, fill_int, y_axis, z)

    print(f"Probability:  {prob}")

    return ProbNormResults(prob=prob, lo=lo_v, hi=hi_v, mu=mu,
                           sigma=sigma, plots={"norm": fig})


def _plot(lo, hi, lo_lbl, hi_lbl, mu, sigma, prob, nrm_color,
          fill_nrm, fill_int, y_axis, z):
    import plotly.graph_objects as go
    from scipy.stats import norm
    min_x, max_x = mu - 4 * sigma, mu + 4 * sigma
    cuts = [mu + k * sigma for k in range(-4, 5)]
    x = np.linspace(min_x, max_x, 200)
    dnrm = norm.pdf(x, mu, sigma)

    fig = go.Figure()
    # full curve, filled
    fig.add_trace(go.Scatter(
        x=np.concatenate(([min_x], x, [max_x])),
        y=np.concatenate(([0.0], dnrm, [0.0])),
        mode="lines", fill="toself", fillcolor=fill_nrm,
        line=dict(color=nrm_color), hoverinfo="skip",
        showlegend=False))
    # shaded interval
    m = (x > lo) & (x < hi)
    xs = np.concatenate(([lo], x[m], [hi]))
    ys = np.concatenate(([0.0], dnrm[m],
                         [0.0]))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", fill="toself",
        fillcolor=fill_int, line=dict(color=fill_int),
        hoverinfo="skip", showlegend=False))

    title = (f"Prob = {float(f'{prob:.4g}')} for Y from "
             f"{lo_lbl} to {hi_lbl}<br>"
             f"<sub>μ={mu}  σ={sigma}</sub>")
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=50, r=20, b=60 if z else 40, l=40),
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=13)),
        xaxis=dict(tickvals=cuts, showgrid=False,
                   linecolor="black", ticks="outside"),
        yaxis=dict(visible=y_axis, showgrid=False,
                   title="Normal Density" if y_axis else None,
                   rangemode="tozero"),
        showlegend=False)
    if z:                                  # second row: z scores
        for k, cx in zip(range(-4, 5), cuts):
            fig.add_annotation(x=cx, xref="x", y=-0.13,
                               yref="paper", text=str(k),
                               showarrow=False,
                               font=dict(size=11))
    return fig
