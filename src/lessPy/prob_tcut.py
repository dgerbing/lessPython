# prob_tcut.py — analog of prob_tcut.R.
#
# prob_tcut(): plot a t density (df degrees of freedom) with the
# central 1-alpha region shaded and the two alpha/2 tails shaded,
# overlaid with a standard normal for comparison. Both the t and
# normal two-tailed critical cutoffs are drawn as vertical lines
# and labeled. Prints and returns the upper t cutoff,
# qt(1 - alpha/2, df) — the value R reports (labeled "Probability").

import numpy as np

from .utils import fmt

_BG = "#F9F9FC"             # rgb(249,249,252)
_TAIL = "#8B475D"           # palevioletred4
_NRM = "#B3B3B3"            # gray(.7)
_T = "#141414"              # gray(.08)


class ProbTcutResults:
    """Result of prob_tcut(): the upper t cutoff, df, alpha, and the
    plotly figure in .plots ("tcut")."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return f"<lessPy prob_tcut: df={self.df} cutoff={self.cutoff:.4g}>"


def prob_tcut(df, alpha=0.05, digits_d=3, y_axis=False,
              fill="aliceblue", color_tail=_TAIL, nrm_color=_NRM,
              color_t=_T):
    """Two-tailed t cutoffs at alpha: shade the central 1-alpha
    region of the t(df) curve and its alpha/2 tails, overlay a
    standard normal, and mark both sets of critical values. Returns
    a ProbTcutResults whose cutoff is the upper t critical value
    qt(1 - alpha/2, df). R analog: prob_tcut()"""
    from scipy.stats import t as t_dist, norm
    if df < 2:
        raise ValueError("df must be 2 or larger")

    tail = alpha / 2
    t_lo = t_dist.ppf(tail, df)
    t_hi = t_dist.ppf(1 - tail, df)
    cutoff = t_hi                          # value R returns
    n_lo = norm.ppf(tail)
    n_hi = norm.ppf(1 - tail)

    fig = _plot(df, alpha, t_lo, t_hi, n_lo, n_hi, digits_d, y_axis,
                fill, color_tail, nrm_color, color_t)

    print(f"Probability:  {cutoff}")
    return ProbTcutResults(cutoff=cutoff, df=df, alpha=alpha,
                           plots={"tcut": fig})


def _cut_line(fig, x, top, color):
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=top,
                  line=dict(color=color))


def _cut_lbl(fig, x, y, value, digits_d, color):
    fig.add_annotation(x=x, y=y, text=fmt(value, digits_d),
                       showarrow=False, font=dict(size=12,
                       color=color))


def _plot(df, alpha, t_lo, t_hi, n_lo, n_hi, digits_d, y_axis,
          fill, color_tail, nrm_color, color_t):
    import plotly.graph_objects as go
    from scipy.stats import t as t_dist, norm
    xmin, xmax = -5.5, 5.5
    x = np.linspace(xmin, xmax, 400)
    yt = t_dist.pdf(x, df)

    fig = go.Figure()
    # central 1-alpha region of the t curve (aliceblue)
    fig.add_trace(go.Scatter(
        x=np.concatenate(([xmin], x, [xmax])),
        y=np.concatenate(([0.0], yt, [0.0])),
        mode="lines", fill="toself", fillcolor=fill,
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        showlegend=False))
    # alpha/2 tails (palevioletred4)
    for lo, hi in ((xmin, t_lo), (t_hi, xmax)):
        xs = np.linspace(lo, hi, 100)
        ys = t_dist.pdf(xs, df)
        fig.add_trace(go.Scatter(
            x=np.concatenate(([lo], xs, [hi])),
            y=np.concatenate(([0.0], ys, [0.0])),
            mode="lines", fill="toself", fillcolor=color_tail,
            line=dict(color=nrm_color, width=1), hoverinfo="skip",
            showlegend=False))
    # t curve on top
    fig.add_trace(go.Scatter(
        x=x, y=yt, mode="lines", line=dict(color=color_t, width=3),
        name=f"t, df={df}", hoverinfo="skip"))
    # standard normal overlay
    fig.add_trace(go.Scatter(
        x=x, y=norm.pdf(x), mode="lines",
        line=dict(color=nrm_color, width=2), name="Normal",
        hoverinfo="skip"))

    # t cutoffs: line to .19, label at .20
    for c in (t_lo, t_hi):
        _cut_line(fig, c, 0.19, color_tail)
        _cut_lbl(fig, c, 0.205, c, digits_d, color_t)
    # normal cutoffs: line to .25, label at .26
    for c in (n_lo, n_hi):
        _cut_line(fig, c, 0.25, nrm_color)
        _cut_lbl(fig, c, 0.265, c, digits_d, nrm_color)
    # central confidence label, e.g. "95%"
    fig.add_annotation(x=0, y=0.175,
                       text=f"{round(100 * (1 - alpha))}%",
                       showarrow=False,
                       font=dict(size=17, color=nrm_color))

    fig.update_layout(
        plot_bgcolor=_BG, paper_bgcolor="white",
        margin=dict(t=20, r=20, b=50, l=40),
        xaxis=dict(title="Standard Errors from Zero", range=[xmin,
                   xmax], showgrid=False, zeroline=False,
                   linecolor="black", ticks="outside"),
        yaxis=dict(visible=y_axis, range=[0, 0.42], showgrid=False,
                   zeroline=False,
                   title="Density" if y_axis else None),
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top",
                    bordercolor="black", borderwidth=0.5,
                    font=dict(size=10)))
    return fig
