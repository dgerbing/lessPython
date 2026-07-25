# plotly_utils.py — analog of zzz_plotly.R
#
# Only the helpers needed by the render functions ported so far.
# Function names drop the leading "." of their R counterparts.

import math
import re

import numpy as np

from .utils import get_option

# R analog: .plotly_base_colors()
BASE_COLORS = [
    "#4398D0", "#B28B2A", "#5FA140", "#D57388",
    "#9A84D6", "#00A898", "#C97E5B", "#909711",
    "#00A3BA", "#D26FAF", "#00A76F", "#BD76CB",
]

# minimal named-color table; R's col2rgb() knows all R names, but
# plotly accepts CSS names directly so unknowns pass through as-is
_NAMED = {
    "black":       "#000000",
    "white":       "#FFFFFF",
    "red":         "#FF0000",
    "blue":        "#0000FF",
    "green":       "#00FF00",
    "gray":        "#BEBEBE",
    "grey":        "#BEBEBE",
    "darkred":     "#8B0000",
    "darkblue":    "#00008B",
    "steelblue":   "#4682B4",
    "transparent": "#FFFFFF00",
}


def _gray_level(name):
    m = re.fullmatch(r"gr[ae]y(\d{1,3})", name)
    if m is None:
        return None
    lvl = int(m.group(1))
    if lvl > 100:
        return None
    v = round(255 * lvl / 100)
    return f"#{v:02X}{v:02X}{v:02X}"


def to_hex(col):
    """Convert a color or list of colors to hex. R analog: .to_hex()"""
    if col is None:
        return None
    if isinstance(col, (list, tuple)):
        return [to_hex(c) for c in col]

    c1 = str(col).strip()
    if c1.lower() == "off":
        c1 = "transparent"

    # rgb()/rgba(): retain alpha if present
    m = re.fullmatch(r"rgba?\(([^)]+)\)", c1)
    if m:
        parts = [float(p) for p in m.group(1).split(",")]
        r, g, b = (int(round(max(0, min(255, p)))) for p in parts[:3])
        if len(parts) == 4:
            a = int(round(max(0, min(1, parts[3])) * 255))
            return f"#{r:02X}{g:02X}{b:02X}{a:02X}"
        return f"#{r:02X}{g:02X}{b:02X}"

    if re.fullmatch(r"#[0-9A-Fa-f]{6}", c1):
        return c1
    if re.fullmatch(r"#[0-9A-Fa-f]{3}", c1):
        r, g, b = c1[1], c1[2], c1[3]
        return f"#{r}{r}{g}{g}{b}{b}"
    if re.fullmatch(r"#[0-9A-Fa-f]{8}", c1):
        return c1

    lower = c1.lower()
    if lower in _NAMED:
        return _NAMED[lower]
    gray = _gray_level(lower)
    if gray is not None:
        return gray
    return c1        # CSS name unknown to us; plotly handles it


def as_plotly_color(col):
    """plotly.py rejects #RRGGBBAA hex; rewrite it as rgba().
    R analog: .as_plotly_rgba()"""
    if col is None:
        return None
    if isinstance(col, (list, tuple)):
        return [as_plotly_color(c) for c in col]
    h = to_hex(col)
    if isinstance(h, str) and re.fullmatch(r"#[0-9A-Fa-f]{8}", h):
        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
        a = int(h[7:9], 16) / 255
        return f"rgba({r},{g},{b},{a:.3f})"
    return h


def auto_opacity(k, mode="overlay"):
    """Fill opacity by number of overlapping series.
    R analog: .auto_opacity()"""
    if mode not in ("overlay", "fill", "lines", "stack"):
        raise ValueError("bad auto_opacity mode")
    k = max(1, int(k))
    base = 1.00 if k == 1 else 0.65 - 0.058 * k
    lo, hi = {"overlay": (0.28, 0.80), "fill": (0.20, 0.65),
              "lines": (0.40, 0.90), "stack": (1.00, 1.00)}[mode]
    return max(lo, min(hi, base))


def make_trans(col, alpha):
    """Color(s) + alpha -> rgba() string(s).
    R analog: .maketrans_plotly()"""
    if isinstance(col, (list, tuple)):
        return [make_trans(c, alpha) for c in col]
    a = max(0.0, min(1.0, float(alpha)))
    s = "" if col is None else str(col).strip().lower()
    if s in ("", "na", "transparent", "off"):
        return "rgba(0,0,0,0)"
    h = to_hex(col)
    if not str(h).startswith("#"):
        return h            # unknown CSS name; pass through opaque
    r, g, b = hex_to_rgb3(h)
    return f"rgba({r},{g},{b},{a:.3f})"


def auto_text_color(cols, bg="white", threshold=0.5):
    """Black or white text per fill, from relative luminance; rgba
    fills fall back to bg when fully transparent.
    R analog: .auto_text_color() in zzz.R"""
    def one(col):
        s = str(col)
        m = re.fullmatch(r"rgba?\(([^)]+)\)", s)
        if m:
            parts = [float(p) for p in m.group(1).split(",")]
            if len(parts) == 4 and parts[3] == 0:
                s = bg          # fully transparent: use background
                m = None
            else:
                r, g, b = parts[:3]
        if not m:
            h = to_hex(s)
            if not str(h).startswith("#"):
                h = to_hex(bg)
            r, g, b = hex_to_rgb3(h)
        lin = (_srgb_to_lin(r), _srgb_to_lin(g), _srgb_to_lin(b))
        lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
        return "white" if lum < threshold else "black"
    if isinstance(cols, (list, tuple)):
        return [one(c) for c in cols]
    return one(cols)


def build_title(x_name, by_name=None, y_name=None, stat=None,
                facet_name=None):
    """Auto-build a chart title from the variable names.
    R analog: .plotly_build_title()"""
    cap = {"mean": "Mean", "sum": "Sum", "median": "Median",
           "min": "Min", "max": "Max", "sd": "SD",
           "deviation": "Mean Deviation"}
    if y_name == "Count":
        y_name = None
    if y_name is None and stat is None:
        ttl = f"Count of {x_name}"
    elif stat is None:
        ttl = f"{y_name} by {x_name}"
    else:
        ttl = f"{cap.get(stat, stat.upper())} of {y_name} " \
              f"by {x_name}"
    if by_name:
        ttl += f" by {by_name}"
    if facet_name:
        ttl += f" across {facet_name}"
    return ttl


# where a named position puts the legend: (x, y, xanchor, yanchor).
# "right_margin" is the default, outside the panel; the rest are the
# base-R legend() keywords, placed inside the panel as they are there.
LEGEND_POSITIONS = {
    "right_margin": (1.02, 0.5, "left", "middle"),
    "topleft":      (0.02, 0.98, "left", "top"),
    "top":          (0.50, 0.98, "center", "top"),
    "topright":     (0.98, 0.98, "right", "top"),
    "left":         (0.02, 0.50, "left", "middle"),
    "center":       (0.50, 0.50, "center", "middle"),
    "right":        (0.98, 0.50, "right", "middle"),
    "bottomleft":   (0.02, 0.02, "left", "bottom"),
    "bottom":       (0.50, 0.02, "center", "bottom"),
    "bottomright":  (0.98, 0.02, "right", "bottom"),
}


def abbrev(txt, n):
    """Truncate to at most n characters. R analog: legend_abbrev"""
    if n is None or txt is None:
        return txt
    n = int(n)
    return txt if len(txt) <= n else txt[:n]


def legend_style(by_name, style_opts, title=None, position=None,
                 horiz=False, size=None, adjust=0, abbrev_n=None):
    """Legend styling for a by= grouping, shared by the histogram,
    density, and bar renderers (the VBS composite draws its own
    horizontal lattice-style key instead).

    title overrides the by-variable name, position names a slot in
    LEGEND_POSITIONS, horiz lays the keys out in a row, size is a
    character expansion factor on the text, adjust shifts the legend
    horizontally (paper units), and abbrev_n truncates the title and
    the key labels. R analog: the legend_* parameters of Chart()."""
    if position is not None and position not in LEGEND_POSITIONS:
        raise ValueError(
            f"legend_position must be one of "
            f"{', '.join(LEGEND_POSITIONS)}")

    mult = 1.0 if size is None else float(size)
    text = by_name or "" if title is None else title
    leg = dict(
        title=dict(text=abbrev(text, abbrev_n) or "",
                   font=dict(size=round(
                       15 * get_option("lab_size", 1) * mult))),
        font=dict(size=round(
            16 * get_option("axis_size", 0.9) * mult)),
        bgcolor=to_hex(style_opts["window_fill"]),
        bordercolor=to_hex(style_opts["legend_border"]),
        borderwidth=1,
    )
    # placement stays plotly's own unless asked for, so the callers
    # that only style the legend keep the layout they already had
    if horiz:
        leg["orientation"] = "h"
    # "right_margin" is plotly's own outside-right placement, so
    # leave the layout alone there unless the legend is nudged
    if (position not in (None, "right_margin")) or adjust:
        x, y, xanchor, yanchor = LEGEND_POSITIONS[
            position or "right_margin"]
        leg.update(x=x + float(adjust or 0), xanchor=xanchor,
                   y=y, yanchor=yanchor)
    return leg


def hex_to_rgb3(h):
    """R analog: .hex_to_rgb3()"""
    h = h.replace("#", "").upper()
    if len(h) in (3, 4):
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _srgb_to_lin(c):
    c = c / 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def contrast_text_for_hex(hex_col):
    """White or black text, whichever contrasts more with the fill.
    R analog: .contrast_text_for_hex()"""
    hex_col = to_hex(hex_col)
    if not str(hex_col).startswith("#"):
        return "#000000"
    r, g, b = hex_to_rgb3(hex_col)
    lin = (_srgb_to_lin(r), _srgb_to_lin(g), _srgb_to_lin(b))
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    white = 1.05 / (lum + 0.05)
    black = (lum + 0.05) / 0.05
    return "#FFFFFF" if white > black else "#000000"


def is_integer_valued(v):
    """True if every non-missing value is a whole number."""
    vals = [x for x in v if x == x]      # drop NaN
    return all(float(x).is_integer() for x in vals)


def get_tick_fmt(v, digits_d=2):
    """d3-format string for hover/tick values. R analog: .get.tick.fmt()"""
    if digits_d is None:
        digits_d = 2
    return "" if is_integer_valued(v) else f".{digits_d}f"


def axis_format(vals, digits_d=2, axis_fmt="K", prefix=""):
    """Tick labels under the lessR axis_fmt policies. "K" (the
    default): billions go scientific; multiples of 100 all past
    1000 render as thousands ("60K"); otherwise a comma
    separator past 9999. ",": comma separator. ".": European
    (period separator). "": plain. prefix prepends (e.g. "$").
    R analog: .axis.format()"""
    v = np.asarray(list(vals), dtype=float)
    if len(v) == 0:
        return []
    fmtd = get_tick_fmt(v, digits_d)
    base = [f"{u:.{digits_d}f}" if fmtd else f"{u:g}"
            for u in v]
    nz = np.abs(v)[np.abs(v) > 0]
    if axis_fmt == "K" and not fmtd:
        if np.any(np.abs(v) >= 1e9):
            base = [f"{u:g}" for u in v]
        elif (np.all(np.abs(v / 100 - np.round(v / 100))
                     < 1e-10)
              and len(nz) > 0 and np.all(nz > 1000)):
            base = [f"{u / 1000:g}K" for u in v]
        elif np.any(v > 9999):
            base = [f"{u:,.0f}" for u in v]
    elif axis_fmt == ",":
        base = [f"{u:,.0f}" for u in v]
    elif axis_fmt == ".":
        base = [f"{u:,.0f}".replace(",", ".") for u in v]
    if prefix:
        base = [prefix + b for b in base]
    return base


def grid_style():
    """R analog: .grid_style()"""
    return {
        "color": to_hex(get_option("grid_col", "gray85")),
        "width": get_option("grid_lwd", 0.5),
        "dash": get_option("grid_lty", None),
    }


def _grid_lines(vals, vertical):
    if vals is None or len(vals) == 0:
        return []
    st = grid_style()
    line = {"color": st["color"], "width": st["width"]}
    if st["dash"] is not None:
        line["dash"] = st["dash"]
    shapes = []
    for v in vals:
        if vertical:
            pos = {"xref": "x", "yref": "paper",
                   "x0": v, "x1": v, "y0": 0, "y1": 1}
        else:
            pos = {"xref": "paper", "yref": "y",
                   "x0": 0, "x1": 1, "y0": v, "y1": v}
        shapes.append({"type": "line", "layer": "below",
                       "line": dict(line), **pos})
    return shapes


def x_grid(x_vals):
    """Vertical grid lines at given x values. R analog: x_grid()"""
    return _grid_lines(x_vals, vertical=True)


def y_grid(y_vals):
    """Horizontal grid lines at given y values. R analog: y_grid()"""
    return _grid_lines(y_vals, vertical=False)


def axis_base():
    """R analog: axis_base()"""
    return {
        "zeroline": False,
        "showline": True,
        "linecolor": to_hex(get_option("axis_color", "black")),
        "linewidth": get_option("axis_lwd", 1),
        "ticks": "outside",
        "ticklen": 4,
        "automargin": True,
        "tickfont": {
            "color": to_hex(get_option("axis_color", "black")),
            "size": 16 * get_option("axis_size", 0.9),
        },
        "title": {
            "font": {
                "color": to_hex(get_option("lab_color", "black")),
                "size": 15 * get_option("lab_size", 1),
            },
            "standoff": 10,
        },
        "gridcolor": to_hex(get_option("grid_color", "gray90")),
        "gridwidth": get_option("grid_lwd", 1),
    }


def axis_num(title_txt, tickvals, ticktext):
    """Numeric axis with explicit ticks. R analog: axis_num()"""
    ax = axis_base()
    ax["title"]["text"] = title_txt
    ax.update(tickmode="array", tickvals=tickvals,
              ticktext=ticktext, showgrid=False)
    return ax


def axis_cat(title_txt):
    """Categorical axis, grid off. R analog: axis_cat()"""
    ax = axis_base()
    ax["title"]["text"] = title_txt
    ax["showgrid"] = False
    return ax


def apply_font_size(fig, mult):
    """Scale every text size in the figure by mult -- the per-call
    font_size. Covers the global font, the title, each (sub)plot's
    axis title and tick font, the legend, annotations, and trace
    text. Text without an explicit size inherits the global font,
    so scaling layout.font carries it. R analog: font_size."""
    if not mult or mult == 1:
        return fig

    def scale(obj, attr, default=None):
        if obj is None:
            return
        try:
            cur = getattr(obj, attr)
        except Exception:
            return
        try:
            if cur is not None:
                setattr(obj, attr, max(1, round(cur * mult)))
            elif default is not None:
                setattr(obj, attr, max(1, round(default * mult)))
        except Exception:
            pass

    lay = fig.layout
    scale(lay.font, "size", 12)            # inherited default
    scale(lay.title.font, "size")
    scale(lay.legend.font, "size")
    for ax in list(fig.select_xaxes()) + list(fig.select_yaxes()):
        scale(ax.title.font, "size")
        scale(ax.tickfont, "size")
    for ann in lay.annotations:
        scale(ann.font, "size")
    for tr in fig.data:
        for a in ("textfont", "insidetextfont", "outsidetextfont"):
            scale(getattr(tr, a, None), "size")
    return fig


def font_scaled(func):
    """Wrap a plot function so a font_size= keyword scales the text
    of the returned figure (font_size=1 is a no-op)."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, font_size=1, **kwargs):
        fig = func(*args, **kwargs)
        try:
            if font_size and font_size != 1 and hasattr(fig, "layout"):
                apply_font_size(fig, font_size)
        except Exception:
            pass
        return fig

    return wrapper


def sym_at(shape, i):
    """Per-group marker symbol. A scalar shape is used for every
    group (by= differentiates by color only, the default); a list/
    tuple gives one symbol per group, recycled if short."""
    if isinstance(shape, (list, tuple)):
        return shape[i % len(shape)]
    return shape


def square_layout(main=False, side=460):
    """Figure width/height/margin giving a square plot area
    (side x side px), matching R's square default device. Setting
    width = side + left + right and height = side + top + bottom
    makes the plotting box square regardless of tick-label width."""
    left, right, bottom = 78, 28, 56
    top = 60 if main else 30
    return dict(width=side + left + right,
                height=side + top + bottom,
                margin=dict(l=left, r=right, t=top, b=bottom))


def facet_fig(n_rows, n_cols=1):
    """Facet panel grid on shared axes; row n_rows is the BOTTOM
    row, where the first facet levels draw (the lattice as.table
    default, as in the faceted VBS)."""
    from plotly.subplots import make_subplots
    # plotly caps spacing at 1/(n-1); keep a 10% margin under it so
    # many-panel grids (e.g. 25 facet levels) still lay out
    v_space = max(0.055, min(0.16, 0.32 / n_rows))
    if n_rows > 1:
        v_space = min(v_space, 0.9 / (n_rows - 1))
    h_space = 0.06 if n_cols > 1 else 0.2
    if n_cols > 1:
        h_space = min(h_space, 0.9 / (n_cols - 1))
    return make_subplots(
        rows=n_rows, cols=n_cols, shared_xaxes=True,
        shared_yaxes=n_cols > 1,
        vertical_spacing=v_space, horizontal_spacing=h_space)


def facet_pos(i, n_lvl, n_col):
    """Subplot (row, col) of facet level i: the lattice fill
    order — bottom row first, left to right, then upward, so the
    bottom row is always full and carries the x ticks."""
    n_row = math.ceil(n_lvl / max(1, n_col))
    return n_row - i // n_col, i % n_col + 1


def facet_cell_label(name, level):
    """Panel-strip label  Var = "level"  (quote non-numeric levels
    only, so an integer level reads as  Year = 2020, not
    Year = "2020"). R analog: .lab.lv() in .plt.dist.facet /
    .plt.contour.facet"""
    try:
        float(level)
        q = str(level)
    except (TypeError, ValueError):
        q = f'"{level}"'
    return f"{name} = {q}"


def facet_cells2(f1_order, f2_order, f1_name, f2_name):
    """Cell order for the two-facet grid: panel rows = facet2
    levels (first level on the top row), panel columns = facet1
    levels, filled row-major — the layout of R's faceted
    dist/contour renderers and the lattice facet1 * facet2
    conditioning. Returns (labels, pos, n_row, n_col) for
    finish_facet(pos=...)."""
    labels, pos = [], []
    for i2, l2 in enumerate(f2_order):
        for i1, l1 in enumerate(f1_order):
            labels.append(facet_cell_label(f1_name, l1) + ", "
                          + facet_cell_label(f2_name, l2))
            pos.append((i2 + 1, i1 + 1))
    return labels, pos, len(f2_order), len(f1_order)


def facet_panels(facet, facet_order, facet2=None,
                 facet2_order=None, facet_name=None,
                 facet2_name=None, n_col=1):
    """Panel plan shared by the faceted renderers. One facet:
    the established bottom-up fill (facet_pos) with plain level
    strip labels. Two facets: the facet_cells2 top-down grid
    (rows = facet2, cols = facet1) with  Var = "level"  cell
    labels; n_col is then fixed by the facet1 levels. Returns
    (labels, pos, sel, n_row, n_col): sel[i] is the boolean
    row mask of panel i."""
    import numpy as np
    f1 = np.asarray(facet)
    if facet2 is None:
        n = len(facet_order)
        labels = [str(lv) for lv in facet_order]
        pos = [facet_pos(i, n, n_col) for i in range(n)]
        sel = [f1 == lv for lv in facet_order]
        return labels, pos, sel, math.ceil(n / max(1, n_col)), \
            max(1, n_col)
    labels, pos, n_row, n_c = facet_cells2(
        facet_order, facet2_order, facet_name, facet2_name)
    f2 = np.asarray(facet2)
    sel = [(f1 == l1) & (f2 == l2)
           for l2 in facet2_order for l1 in facet_order]
    return labels, pos, sel, n_row, n_c


def finish_facet(fig, levels, ax, x_lab, y_lab, gridT1,
                 style_opts=None, y_cat=None, n_col=1, pos=None,
                 height=None, width=None):
    """Facet styling shared by the faceted renderers: per-panel y
    axes on a common scale (y label on the middle panel of the
    first column), x ticks on the bottom row only, vertical grid
    and a frame per panel, and a shaded strip label above each
    panel. levels fill bottom-up, left to right (facet_pos):
    levels[0] draws bottom-left. n_col > 1 lays the panels out
    as a grid (the lattice n_col). y_cat: category labels for
    panels whose y axis is categorical (the faceted bar chart);
    ax then needs only axT1/axL1. pos: explicit (row, col) per
    level — the two-facet grid (facet_cells2), which fills
    top-down."""
    if style_opts is None:
        style_opts = plotly_style()
    n = len(levels)
    if pos is not None:
        n_row = max(r for r, _ in pos)
        n_col = max(c for _, c in pos)
    else:
        n_row = math.ceil(n / max(1, n_col))
    mid_r = (n_row + 1) // 2
    mid_c = (n_col + 1) // 2
    gs = grid_style()
    shapes = list(fig.layout.shapes or [])
    anns = list(fig.layout.annotations or [])
    s_fill = as_plotly_color(get_option("strip_fill",
                                        "#7F7F7F37"))
    s_line = to_hex(get_option("strip_color", "gray40"))
    s_text = to_hex(get_option("strip_text_color", "gray15"))
    strip_h = max(0.10, min(0.30, 0.05 * n_row))
    # strip font: shrink so the longest label fits its panel
    # (the two-facet  Var = "level"  cell labels run long)
    s_size = round(14 * get_option("axis_size", 0.9))
    panel_px = (700 - 90) / max(1, n_col)
    max_len = max(len(str(lv)) for lv in levels)
    if max_len > 0:
        s_size = max(8, min(s_size,
                            int(panel_px / (0.58 * max_len))))

    for i, lvl in enumerate(levels):
        r, c = pos[i] if pos is not None else facet_pos(i, n,
                                                        n_col)
        k_ax = (r - 1) * n_col + c
        suf = "" if k_ax == 1 else str(k_ax)
        show_ylab = y_lab if (c == 1 and r == mid_r) else ""
        if y_cat is None:
            ay = axis_num(show_ylab, ax["axT2"], ax["axL2"])
            ay.update(showgrid=True,
                      gridcolor=to_hex(style_opts["grid_col"]),
                      gridwidth=1, griddash="dot",
                      showticklabels=c == 1,
                      range=[0, float(ax["axT2"][-1]) * 1.04])
        else:
            ay = axis_cat(show_ylab)
            ay.update(categoryorder="array",
                      categoryarray=list(y_cat),
                      showticklabels=c == 1)
        fig.update_yaxes(row=r, col=c, **ay)
        if r == n_row:
            fig.update_xaxes(
                row=r, col=c,
                **axis_num(x_lab if c == mid_c else "",
                           ax["axT1"], ax["axL1"]))
        else:
            fig.update_xaxes(
                row=r, col=c, showline=True, ticks="",
                linecolor=to_hex(get_option("axis_color",
                                            "black")))
        for v in (gridT1 or []):          # vertical panel grid
            shapes.append({
                "type": "line", "layer": "below",
                "xref": f"x{suf}", "yref": f"y{suf} domain",
                "x0": v, "x1": v, "y0": 0, "y1": 1,
                "line": {"color": gs["color"],
                         "width": gs["width"]}})
        shapes.append({                   # panel frame
            "type": "rect",
            "xref": f"x{suf} domain", "yref": f"y{suf} domain",
            "x0": 0, "x1": 1, "y0": 0, "y1": 1,
            "line": {"color": to_hex("gray75"), "width": 1},
            "fillcolor": "rgba(0,0,0,0)"})
        shapes.append({                   # strip label band
            "type": "rect",
            "xref": f"x{suf} domain", "yref": f"y{suf} domain",
            "x0": 0, "x1": 1, "y0": 1.0, "y1": 1.0 + strip_h,
            "fillcolor": s_fill,
            "line": {"color": s_line, "width": 1}})
        anns.append(dict(
            xref=f"x{suf} domain", yref=f"y{suf} domain",
            x=0.5, y=1.0 + strip_h / 2, text=str(lvl),
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=s_size, color=s_text)))

    used = (set(pos) if pos is not None
            else {facet_pos(i, n, n_col) for i in range(n)})
    for r in range(1, n_row + 1):      # ragged top row: hide
        for c in range(1, n_col + 1):  # axes of empty cells
            if (r, c) not in used:
                fig.update_xaxes(row=r, col=c, visible=False)
                fig.update_yaxes(row=r, col=c, visible=False)

    layout_kw = dict(
        shapes=shapes, annotations=anns, template=None,
        plot_bgcolor=to_hex(style_opts["panel_fill"]),
        paper_bgcolor=to_hex(style_opts["window_fill"]),
        height=height if height is not None else 140 + 160 * n_row,
    )
    if width is not None:
        layout_kw["width"] = width
    fig.update_layout(**layout_kw)
    return fig


def facet_domains_grid(n_fac, n_col_max=3, gap_x=0.04, gap_y=0.11,
                       facet_size=1.0, n_col=None):
    """Equal-size grid of paper-coordinate panel domains, filled
    row-major from the TOP row down (plotly domain y runs
    bottom-up). Each domain: dict(x=(x0, x1), y=(y0, y1), row,
    col), rows/cols 1-based. facet_size < 1 shrinks each panel
    about its own center. An explicit n_col (the lattice n_col)
    overrides the n_col_max default.
    R analog: .plotly_make_domains_grid()"""
    if n_col is None:
        n_col = min(n_col_max, n_fac)
    n_col = max(1, min(int(n_col), n_fac))
    n_row = math.ceil(n_fac / n_col)

    width = max(0.01, (1 - (n_col + 1) * gap_x) / n_col)
    height = max(0.01, (1 - (n_row + 1) * gap_y) / n_row)

    domains = []
    for k in range(n_fac):
        r, c = divmod(k, n_col)
        x0 = gap_x + c * (width + gap_x)
        x1 = x0 + width
        y1 = 1 - r * (height + gap_y)
        y0 = y1 - height

        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half_w = (x1 - x0) * facet_size / 2
        half_h = (y1 - y0) * facet_size / 2
        domains.append(dict(
            x=(max(0.0, cx - half_w), min(1.0, cx + half_w)),
            y=(max(0.0, cy - half_h), min(1.0, cy + half_h)),
            row=r + 1, col=c + 1))
    return domains, n_row, n_col


def facet_layout(facet_levels, facet_name="",
                 n_col_max=3, gap_x=0.04, gap_y=0.11,
                 facet_size=1.0, domains=None, domain_adjust=None,
                 reverse_domains=False, y_base=0.054,
                 y_row_shift=-0.002, yanchor="top",
                 font_size=None, ann_adjust=None, n_col=None):
    """Panel-grid domains plus a per-panel title annotation, shared
    by the faceted hier/radar/dot/bubble renderers. Supply domains
    to skip the grid computation (panels laid out elsewhere).
    Hooks: domain_adjust(d) tweaks each domain, reverse_domains
    flips their order (treemap row fix), ann_adjust(i, ann, d)
    finalizes each annotation. An empty facet_name gives bare
    level labels. An explicit n_col (the lattice n_col) overrides
    the n_col_max grid default. R analog: .plotly_facet_layout()"""
    if domains is None:
        domains, _, _ = facet_domains_grid(
            len(facet_levels), n_col_max, gap_x, gap_y, facet_size,
            n_col=n_col)
    if domain_adjust is not None:
        domains = [domain_adjust(dict(d)) for d in domains]
    if reverse_domains:
        domains = list(reversed(domains))
    if font_size is None:
        font_size = round(14 * get_option("lab_size", 1))

    anns = []
    for i, d in enumerate(domains):
        y_title = (d["y"][1] + y_base
                   + y_row_shift * (d.get("row", 1) - 1))
        ann = dict(
            text=(f"{facet_name}: {facet_levels[i]}" if facet_name
                  else str(facet_levels[i])),
            x=(d["x"][0] + d["x"][1]) / 2, y=y_title,
            xref="paper", yref="paper",
            xanchor="center", yanchor=yanchor,
            showarrow=False, font=dict(size=font_size))
        if ann_adjust is not None:
            ann = ann_adjust(i, ann, d)
        anns.append(ann)
    return domains, anns


def plot_border(top=True, right=True, color=None, width=None):
    """Top/right frame lines of the panel. R analog: plot_border()"""
    if color is None:
        color = get_option("panel_border", "#808080")
    if width is None:
        width = get_option("panel_lwd", 1)
    if isinstance(color, str) and re.fullmatch(r"gray\d+", color):
        color = "gray75"   # make a gray a little lighter for plotly

    line = {"color": to_hex(color), "width": width}
    out = []
    if top:
        out.append({"type": "line", "xref": "paper", "yref": "paper",
                    "x0": 0, "x1": 1, "y0": 1, "y1": 1, "line": line})
    if right:
        out.append({"type": "line", "xref": "paper", "yref": "paper",
                    "x0": 1, "x1": 1, "y0": 0, "y1": 1, "line": line})
    return out


def plotly_style():
    """Current style settings for the render functions.
    R analog: .plotly_style()"""
    pb = get_option("panel_border", "#808080")
    return {
        "grid_color": get_option("grid_color", "gray85"),
        "grid_col": get_option("grid_col", "gray85"),
        "lab_color": get_option("lab_color", "black"),
        "legend_border": get_option("legend_border", pb),
        "panel_border": pb,
        "panel_fill": get_option("panel_fill", "white"),
        "window_fill": get_option("window_fill", "white"),
        "segment_color": get_option("segment_color", "gray40"),
    }
