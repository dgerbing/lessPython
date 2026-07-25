# plt_add.py — analog of plt.add.R
#
# The add= annotation vocabulary shared by Chart(), X(), and
# XY(): "v_line", "h_line", "line", "rect", "arrow", "point",
# and any other string draws as text at (x1, y1). Style comes
# from the add_ options (set_option): add_color (gray10),
# add_fill (#D9D9D920), add_lwd (0.5), add_lty (solid),
# add_size (1), add_trans (0) — each may be a list, recycled
# per object as in R. One object with vector coordinates
# repeats at every location; several objects consume one
# coordinate per object, in order.

import numpy as np
import plotly.graph_objects as go

from .plotly_utils import as_plotly_color, make_trans, to_hex
from .utils import get_option

_DASH = {"dashed": "dash", "dotted": "dot",
         "dotdash": "dashdot", "longdash": "longdash",
         "twodash": "longdashdot"}


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    return [v]


def plt_add(fig, add, x1=None, x2=None, y1=None, y2=None):
    """Draw the add= annotation objects onto a built figure.
    R analog: .plt.add()"""
    add = _as_list(add)
    x1v, x2v = _as_list(x1), _as_list(x2)
    y1v, y2v = _as_list(y1), _as_list(y2)
    n_obj = len(add)

    colors = _as_list(get_option("add_color", "gray10"))
    fills = _as_list(get_option("add_fill", "#D9D9D920"))
    lwds = _as_list(get_option("add_lwd", 0.5))
    ltys = _as_list(get_option("add_lty", "solid"))
    sizes = _as_list(get_option("add_size", 1.0))
    transs = _as_list(get_option("add_trans", 0.0))

    idx = {"x1": 0, "x2": 0, "y1": 0, "y2": 0}

    def locs(obj):
        """Coordinate tuples for one object: all locations when
        there is a single object, else the next one per vector."""
        need_x = obj != "h_line"
        need_y = obj != "v_line"
        two = obj in ("line", "rect", "arrow")
        if n_obj == 1:
            n_loc = max(len(x1v) if need_x else 0,
                        len(y1v) if need_y else 0, 1)
            out = []
            for k in range(n_loc):
                out.append((
                    x1v[k % len(x1v)] if need_x and x1v else None,
                    y1v[k % len(y1v)] if need_y and y1v else None,
                    x2v[k % len(x2v)] if two and x2v else None,
                    y2v[k % len(y2v)] if two and y2v else None))
            return out
        vals = []
        for name, vec, need in (("x1", x1v, need_x),
                                ("y1", y1v, need_y),
                                ("x2", x2v, two),
                                ("y2", y2v, two)):
            if need and vec:
                vals.append(vec[idx[name] % len(vec)])
                idx[name] += 1
            else:
                vals.append(None)
        return [tuple(vals)]

    for i, obj in enumerate(add):
        col = to_hex(colors[i % len(colors)])
        lwd = float(lwds[i % len(lwds)])
        lty = str(ltys[i % len(ltys)])
        cex = float(sizes[i % len(sizes)])
        trn = float(transs[i % len(transs)])
        fll = fills[i % len(fills)]
        fill_c = (make_trans(fll, 1 - trn) if trn > 0
                  else as_plotly_color(fll))
        line = dict(color=col, width=max(0.5, lwd * 2))
        if lty != "solid":
            line["dash"] = _DASH.get(lty, lty)

        for xx1, yy1, xx2, yy2 in locs(obj):
            if obj == "v_line":
                fig.add_shape(type="line", xref="x",
                              yref="paper", x0=xx1, x1=xx1,
                              y0=0, y1=1, line=line)
            elif obj == "h_line":
                fig.add_shape(type="line", xref="paper",
                              yref="y", x0=0, x1=1,
                              y0=yy1, y1=yy1, line=line)
            elif obj == "line":
                fig.add_shape(type="line", xref="x", yref="y",
                              x0=xx1, y0=yy1, x1=xx2, y1=yy2,
                              line=line)
            elif obj == "rect":
                fig.add_shape(type="rect", xref="x", yref="y",
                              x0=xx1, y0=yy1, x1=xx2, y1=yy2,
                              line=line, fillcolor=fill_c)
            elif obj == "arrow":
                fig.add_annotation(
                    x=xx2, y=yy2, ax=xx1, ay=yy1,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2,
                    arrowcolor=col,
                    arrowwidth=max(0.5, lwd * 2), text="")
            elif obj == "point":
                fig.add_trace(go.Scatter(
                    x=[xx1], y=[yy1], mode="markers",
                    marker=dict(symbol="circle",
                                size=max(3, cex * 8),
                                color=fill_c,
                                line=dict(color=col, width=1)),
                    hoverinfo="skip", showlegend=False))
            else:                      # any other string: text
                fig.add_annotation(
                    x=xx1, y=yy1, text=str(obj),
                    xref="x", yref="y", showarrow=False,
                    font=dict(color=col,
                              size=max(6, round(13 * cex))))
    return fig
