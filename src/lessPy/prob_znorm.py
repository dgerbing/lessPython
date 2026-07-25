# prob_znorm.py — analog of prob_znorm.R.
#
# prob_znorm(): plot a normal density curve marked at each integer
# standard deviation, with the central +/-1, +/-2, and +/-3 sigma
# regions shaded by three nested translucent bands. Because the
# bands are translucent and stacked, the center (covered by all
# three) is darkest, then two layers, then one — the empirical-rule
# graphic. Unlike prob_norm() this returns no probability; it is a
# display, so the results object holds only the plotly figure in
# .plots ("znorm").

import numpy as np


class ProbZnormResults:
    """Result of prob_znorm(): mu/sigma and the plotly figure in
    .plots ("znorm"). No probability is computed."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return f"<lessPy prob_znorm: mu={self.mu} sigma={self.sigma}>"


def prob_znorm(mu=0, sigma=1, color_border="gray10", r=0.10,
               g=0.34, b=0.94, a=0.20, main="", y_axis=False,
               z=True):
    """Normal curve with the central 1/2/3-sigma regions shown as
    three nested translucent bands (the empirical-rule graphic).
    color_border is the curve/outline color; r/g/b/a (each in
    [0, 1]) set the translucent fill. z adds a second axis of z
    scores (suppressed for the standard normal). Returns a
    ProbZnormResults with .plots. R analog: prob_znorm()"""
    if not all(0 <= v <= 1 for v in (r, g, b, a)):
        raise ValueError("r, g, b and a must each be between 0 and "
                         "1, inclusive")
    if sigma <= 0:
        raise ValueError("sigma must be larger than zero")

    if mu == 0 and sigma == 1:
        z = False

    border = _gray_to_hex(color_border)
    fill = f"rgba({round(r * 255)},{round(g * 255)}," \
           f"{round(b * 255)},{a})"

    fig = _plot(mu, sigma, border, fill, main, y_axis, z)
    return ProbZnormResults(mu=mu, sigma=sigma, plots={"znorm": fig})


def _gray_to_hex(color):
    """plotly rejects R's grayNN/greyNN names; convert those to hex,
    pass any other color string through unchanged."""
    lc = color.lower()
    for pre in ("gray", "grey"):
        if lc.startswith(pre) and lc[len(pre):].isdigit():
            v = round(int(lc[len(pre):]) / 100 * 255)
            return f"#{v:02X}{v:02X}{v:02X}"
    return color


def _plot(mu, sigma, border, fill, main, y_axis, z):
    import plotly.graph_objects as go
    from scipy.stats import norm
    xmin, xmax = mu - 4 * sigma, mu + 4 * sigma
    cuts = [mu + k * sigma for k in range(-4, 5)]
    x = np.linspace(xmin, xmax, 200)
    y = norm.pdf(x, mu, sigma)

    fig = go.Figure()
    # normal curve
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", line=dict(color=border, width=2),
        hoverinfo="skip", showlegend=False))
    # dotted vertical segment at the mean, up to the peak
    fig.add_shape(type="line", x0=mu, x1=mu, y0=0,
                  y1=norm.pdf(mu, mu, sigma),
                  line=dict(color=border, dash="dot"))
    # nested translucent bands: +/-3, +/-2, +/-1 sigma. Drawn in
    # this order so the overlaps stack — the center is darkest.
    for k in (3, 2, 1):
        lo, hi = mu - k * sigma, mu + k * sigma
        m = (x > lo) & (x < hi)
        fig.add_trace(go.Scatter(
            x=np.concatenate(([lo], x[m], [hi])),
            y=np.concatenate(([0.0], y[m], [0.0])),
            mode="lines", fill="toself", fillcolor=fill,
            line=dict(color=border, dash="dot", width=1),
            hoverinfo="skip", showlegend=False))

    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=50 if main else 30, r=20,
                    b=60 if z else 40, l=40),
        title=dict(text=main, x=0.5, xanchor="center",
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
