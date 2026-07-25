# getColors.py — analog of getColors.R.
#
# getColors(): generate a color palette and return it as a list of
# hex strings, printing a small table of the colors unless quiet.
# The HCL families (hues/qualitative, sequential, divergent) are
# reproduced exactly from R's grDevices HCL and colorspace
# trajectories; the manual families interpolate in RGB; Okabe-Ito,
# Tableau and "distinct" are fixed lists. The viridis family maps
# to plotly's built-in colorscales (a documented approximation),
# and R's wesanderson palettes are not ported (external package).

import math

# --- fixed palettes -------------------------------------------------

# Okabe-Ito, in R's order minus the leading black, black appended
_OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
    "#D55E00", "#CC79A7", "#999999", "#000000",
]
_TABLEAU = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC949", "#AF7AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]
# R's "distinct" names resolved to hex (col2rgb)
_DISTINCT = [
    "#EEB422", "#737373", "#9ACD32", "#CD69C9", "#87CEEB",
    "#A9A9A9", "#F08080", "#8B795E", "#00CDCD", "#CD6600",
    "#CD2990", "#66CDAA", "#4876FF", "#8B7D7B", "#CDB5CD",
]

# hue (degrees) of each pre-defined HCL color sequence
_SEQ_HUE = {
    "reds": 0, "rusts": 30, "browns": 60, "olives": 90,
    "greens": 120, "emeralds": 150, "turquoises": 180,
    "aquas": 210, "blues": 240, "purples": 270, "violets": 300,
    "magentas": 330, "grays": 0,
}
_SEQ_NAMES = list(_SEQ_HUE)
_VIRIDIS = {"viridis": "Viridis", "cividis": "Cividis",
            "plasma": "Plasma", "spectral": "Spectral"}

# D65 white point (grDevices HCL / colorspace polarLUV)
_XN, _YN, _ZN = 95.047, 100.0, 108.883


# --- HCL -> sRGB (grDevices polarLUV, D65) --------------------------

def _gtrans(u):
    if u > 0.0031308:
        return 1.055 * u ** (1.0 / 2.4) - 0.055
    return 12.92 * u


def hcl(h, c, l, fixup=True):
    """One HCL (polarLUV) color as a hex string, matching R's
    grDevices::hcl. Returns None when out of gamut and fixup is
    False. R analog: hcl()"""
    if l <= 0:
        x = y = z = 0.0
    else:
        hrad = math.radians(h)
        U = c * math.cos(hrad)
        V = c * math.sin(hrad)
        un = 4 * _XN / (_XN + 15 * _YN + 3 * _ZN)
        vn = 9 * _YN / (_XN + 15 * _YN + 3 * _ZN)
        if l > 7.999592:
            y = _YN * ((l + 16) / 116) ** 3
        else:
            y = _YN * l / 903.3
        up = U / (13 * l) + un
        vp = V / (13 * l) + vn
        x = 9 * y * up / (4 * vp)
        z = y * (12 - 3 * up - 20 * vp) / (4 * vp)

    xr, yr, zr = x / 100, y / 100, z / 100
    r = 3.240479 * xr - 1.537150 * yr - 0.498535 * zr
    g = -0.969256 * xr + 1.875992 * yr + 0.041556 * zr
    b = 0.055648 * xr - 0.204043 * yr + 1.057311 * zr
    rgb = [_gtrans(max(0.0, v)) for v in (r, g, b)]

    if not fixup and any(v < -1e-6 or v > 1 + 1e-6 for v in rgb):
        return None
    ints = [min(255, max(0, round(v * 255))) for v in rgb]
    return "#{:02X}{:02X}{:02X}".format(*ints)


def _seq(n):
    if n == 1:
        return [0.0]
    return [1.0 - i / (n - 1) for i in range(n)]


def _sequential_hcl(n, h, c, l, power, fixup):
    c1, c2 = (c, c) if not isinstance(c, tuple) else c
    l1, l2 = (l, l) if not isinstance(l, tuple) else l
    p1, p2 = (power, power) if not isinstance(power, tuple) else power
    out = []
    for r in _seq(n):
        L = l2 - (l2 - l1) * r ** p2
        C = c2 - (c2 - c1) * r ** p1
        out.append(hcl(h, C, L, fixup) + "FF")
    return out


def _diverging_hcl(n, h1, h2, c, l, power, fixup):
    l1, l2 = (l, l) if not isinstance(l, tuple) else l
    p1, p2 = (power, power) if not isinstance(power, tuple) else power
    rvals = [1.0] if n == 1 else \
        [1.0 - 2.0 * i / (n - 1) for i in range(n)]
    out = []
    for r in rvals:
        L = l2 - (l2 - l1) * abs(r) ** p2
        C = c * abs(r) ** p1
        H = h1 if r > 0 else h2
        out.append(hcl(H, C, L, fixup) + "FF")
    return out


# --- RGB ramp (colorRampPalette default space="rgb") ----------------

def _to_rgb(col):
    from .plotly_utils import to_hex, hex_to_rgb3
    return hex_to_rgb3(to_hex(col))


def _rgb_to_hcl(r, g, b):
    """sRGB (0-255) -> HCL (polarLUV, D65): (hue_deg, chroma, lum).
    The inverse of hcl(); used to read a named color's hue and
    chroma for the value-scaled bar fills."""
    def inv(v):
        v /= 255.0
        return (((v + 0.055) / 1.055) ** 2.4 if v > 0.04045
                else v / 12.92)
    R, G, B = inv(r), inv(g), inv(b)
    X = (0.412453 * R + 0.357580 * G + 0.180423 * B) * 100
    Y = (0.212671 * R + 0.715160 * G + 0.072169 * B) * 100
    Z = (0.019334 * R + 0.119193 * G + 0.950227 * B) * 100
    if Y <= 0:
        return 0.0, 0.0, 0.0
    yr = Y / _YN
    L = (116 * yr ** (1 / 3) - 16 if yr > 0.008856 else 903.3 * yr)
    d = X + 15 * Y + 3 * Z
    if d == 0:
        return 0.0, 0.0, L
    un = 4 * _XN / (_XN + 15 * _YN + 3 * _ZN)
    vn = 9 * _YN / (_XN + 15 * _YN + 3 * _ZN)
    U = 13 * L * (4 * X / d - un)
    V = 13 * L * (9 * Y / d - vn)
    return math.degrees(math.atan2(V, U)) % 360, math.hypot(U, V), L


def _color_hc(col):
    h, c, _ = _rgb_to_hcl(*_to_rgb(col))
    return h, c


def bar_fill_colors(values, fill, fill_split, fill_scaled,
                    fill_chroma=75):
    """Per-bar fill colors for a numeric bar chart, keyed to the bar
    values. fill_split gives a two-color split at that value (below
    -> first color, above -> second). fill_scaled makes an HCL
    gradient: chroma and darkness rise with distance from the split
    (default 0), hue by side. A faithful approximation of R's
    .scale.clr, not colorspace-exact. R analog: fill_split /
    fill_scaled."""
    import numpy as np
    from .plotly_utils import to_hex
    x = np.asarray(values, dtype=float)
    fills = (None if fill is None
             else [fill] if isinstance(fill, str) else list(fill))
    split = 0.0 if fill_split is None else float(fill_split)

    if not fill_scaled:                    # discrete two-color split
        if fills is not None and len(fills) == 2:
            c1, c2 = to_hex(fills[0]), to_hex(fills[1])
        else:                              # default dark/light pair
            c1, c2 = hcl(255, 55, 30), hcl(255, 55, 70)
        return [c1 if v <= split else c2 for v in x]

    dist = np.abs(x - split)
    mx = float(dist.max()) or 1.0
    normed = dist / mx
    lum = 27 + (1 - normed) * 53           # near split light, far dark

    if fills is None or len(fills) > 2:
        h1 = h2 = 255.0
        c1 = c2 = float(fill_chroma)
    elif len(fills) == 1:
        h1, c1 = _color_hc(fills[0])
        h2, c2 = h1, c1
    else:
        h1, c1 = _color_hc(fills[0])
        h2, c2 = _color_hc(fills[1])
    g1, g2 = c1 < 5, c2 < 5                # near-gray fill colors

    out = []
    for v, nm, L in zip(x, normed, lum):
        if g1 and g2:                      # pure grayscale ramp
            hue, chroma = 0.0, 0.0
        elif g1 or g2:                     # sequential: chromatic hue
            hue = h2 if g1 else h1
            chroma = (c2 if g1 else c1) * nm
        else:                              # diverging by side
            below = v <= split
            hue = h1 if below else h2
            chroma = (c1 if below else c2) * nm
        out.append(hcl(hue, chroma, max(1.0, L)))
    return out


def _ramp(colors, n):
    pts = [_to_rgb(c) for c in colors]
    if n == 1:
        r, g, b = pts[0]
        return ["#{:02X}{:02X}{:02X}".format(r, g, b)]
    m = len(pts) - 1
    out = []
    for i in range(n):
        t = i / (n - 1) * m
        j = min(int(math.floor(t)), m - 1)
        f = t - j
        rgb = [round(pts[j][k] + f * (pts[j + 1][k] - pts[j][k]))
               for k in range(3)]
        out.append("#{:02X}{:02X}{:02X}".format(*rgb))
    return out


def _viridis(name, n):
    from plotly.colors import sample_colorscale
    xs = [0.0] if n == 1 else [i / (n - 1) for i in range(n)]
    cols = sample_colorscale(_VIRIDIS[name], xs, colortype="rgb")
    out = []
    for c in cols:
        r, g, b = (round(float(v)) for v in
                   c[c.find("(") + 1:c.find(")")].split(","))
        out.append("#{:02X}{:02X}{:02X}".format(r, g, b))
    return out


# the fixed mixed hues for a qualitative palette (n <= 24)
_MIX = [240, 60, 120, 0, 275, 180, 30, 90, 210, 330, 150, 300]


def getColors(pal=None, end_pal=None, n=None, h=0, h2=None, c=None,
              l=None, in_order=None, fixup=True, power=None,
              transparency=0, quiet=False):
    """Generate a color palette, returning a list of hex colors and
    (unless quiet) printing a table of them.

    pal: a palette name ("hues", a sequential name such as "blues",
    "viridis"/"cividis"/"plasma"/"spectral", "Okabe-Ito", "Tableau",
    "distinct") or one or more explicit colors. end_pal: an ending
    color/name for a divergent (two sequential names) or manual
    (two colors) gradient. h/h2, c, l, power: HCL parameters.
    R analog: getColors()"""
    c_miss, l_miss = c is None, l is None
    n_miss = n is None
    if n_miss:
        n = 12

    if in_order is None:
        in_order = False
    if isinstance(pal, str):
        pal = [pal]
    if isinstance(pal, list) and pal and pal[0] == "yellows":
        pal[0] = "browns"
    if isinstance(pal, str) and pal == "magma":
        pal = ["plasma"]

    if pal is None and end_pal is None and not isinstance(c, list) \
            and not isinstance(l, list):
        pal = ["hues"]

    # classify the request
    if pal is not None:
        p0 = pal[0]
        if p0 in _SEQ_NAMES:
            kind = "divergent" if (end_pal is not None and
                                   end_pal[0] in _SEQ_NAMES) \
                else "sequential"
        elif p0 == "hues":
            kind = "qualitative"
        elif p0 in _VIRIDIS:
            kind = "viridis"
        elif p0 == "Okabe-Ito":
            kind = "oi"
        elif p0 == "Tableau":
            kind = "Tableau"
        elif p0 == "distinct":
            kind = "distinct"
        else:
            kind = "manual.q" if end_pal is None else "manual.s"
    else:
        kind = "sequential" if (isinstance(c, list) or
                                isinstance(l, list)) else "qualitative"

    # pre-defined sequential/divergent hues
    if pal is not None and pal[0] in _SEQ_NAMES:
        h = _SEQ_HUE[pal[0]]
        if pal[0] == "grays":
            c, c_miss = 0, False
        if end_pal is None:
            pal = None
    if end_pal is not None and end_pal[0] in _SEQ_NAMES:
        h2 = _SEQ_HUE[end_pal[0]]
        if end_pal[0] == "grays":
            c, c_miss = 0, False
        pal = None

    ttl = ""

    if kind == "qualitative":
        if h2 is None:
            h2 = h + (360 * (n - 1) / n)
        if n <= 24:
            if not in_order:
                hues = _MIX + [x + 15 for x in _MIX]
            else:
                hues = ([h] if n == 1 else
                        [h + (h2 - h) * i / (n - 1) for i in range(n)])
        else:
            hues = [h + (h2 - h) * i / (n - 1) for i in range(n)]
        hues = [(x - 360 if x >= 360 else x + 360 if x < 0 else x)
                for x in hues]
        cc = 65 if c_miss else c
        ll = 60 if l_miss else l
        pal = [hcl(hues[i], cc, ll, fixup) for i in range(n)]

    elif kind == "sequential":
        cc = (35, 75) if c_miss else (c if isinstance(c, (tuple, list))
                                      else (c, c))
        l_dk = max(14, 35 - 3 * n)
        l_lt = min(92, 52 + 5 * n)
        ll = (l_lt, l_dk) if l_miss else (l if isinstance(l, (tuple,
              list)) else (l, l))
        pw = 1 if power is None else power
        pal = _sequential_hcl(n, h, tuple(cc), tuple(ll), pw, fixup)

    elif kind == "divergent":
        cc = 50 if c_miss else (c[0] if isinstance(c, (tuple, list))
                                else c)
        ll = (30, 80) if l_miss else (tuple(l) if isinstance(l, (tuple,
              list)) else (l, l))
        pw = 0.75 if power is None else power
        pal = _diverging_hcl(n, h, h2, cc, ll, pw, fixup)

    elif kind == "viridis":
        ttl = "Viridis Style Palette: " + pal[0]
        pal = _viridis(pal[0], n)

    elif kind == "oi":
        if n_miss:
            n = 9
        if n > 9:
            raise ValueError("only 9 Okabe-Ito colors are available")
        pal = _OKABE_ITO[:n]

    elif kind == "Tableau":
        if n_miss:
            n = 10
        if n > 10:
            raise ValueError("only 10 Tableau colors are available")
        pal = _TABLEAU[:n]

    elif kind == "distinct":
        if n > 15:
            raise ValueError("only 15 distinct colors are available")
        pal = _DISTINCT[:n]

    elif kind == "manual.s":
        anchors = list(pal) + list(
            end_pal if isinstance(end_pal, list) else [end_pal])
        pal = _ramp(anchors, n)

    else:                                  # manual.q
        from .plotly_utils import to_hex
        pal = [to_hex(c1) for c1 in pal]
        n = len(pal)

    if transparency > 0:
        from .plotly_utils import make_trans
        pal = [make_trans(col, 1 - transparency) for col in pal]

    if not quiet:
        _print_table(pal, kind)
    return pal


def _print_table(pal, kind):
    from .plotly_utils import to_hex, hex_to_rgb3
    print()
    print("  color      r    g    b")
    print("-----------------------------")
    for i, col in enumerate(pal, 1):
        r, g, b = hex_to_rgb3(to_hex(col))
        print("{:>2}  {:<9}{:>4} {:>4} {:>4}".format(
            i, col, r, g, b))
    print()
