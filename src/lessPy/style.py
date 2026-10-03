# style.py — analog of lessR's style(): the color theme, a sub_theme,
# and the individual appearance settings that every visualization
# reads, Chart(), X(), and XY() alike, until changed.
#
# The settings of each theme come from lessR itself (_themes.py,
# generated from style(theme[, sub_theme])), so a theme draws the same
# colors in both packages. Text sizes are lessPy's own, tuned for
# plotly in notebooks, and naming a theme leaves them as they were.
# R analog: style.R

import copy

from . import utils
from ._themes import THEMES

# the text sizes, which a theme does not set
_SIZE_KEYS = ("main_size", "lab_size", "lab_x_size", "lab_y_size",
              "axis_size", "axis_x_size", "axis_y_size", "sub_size",
              "add_size")

# lessR style() parameters with no lessPy counterpart: R Markdown output
#   and console settings, and parameters that lessPy takes in the call
#   of the visualization function instead
_R_ONLY = ("results", "explain", "interpret", "document", "code",
           "width", "show", "lessR.use_plotly", "notes", "suggest",
           "brief", "offset", "rotate_x", "rotate_y", "labels",
           "labels_size", "labels_digits", "labels_position")

# start-up settings that a theme does not reach: lessPy's own options,
#   captured once at import, before any style() call
_BASELINE = copy.deepcopy(utils._OPTIONS)


def _style_keys():
    keys = set(THEMES["colors"]["none"]) | set(_SIZE_KEYS)
    keys |= {"quiet", "digits_d", "n_cat", "grid_col", "legend_border"}
    return keys


def _theme_settings(theme, sub_theme):
    if theme not in THEMES:
        raise ValueError(
            f"theme='{theme}' is not a theme. Themes: "
            + ", ".join(THEMES))
    variant = "none" if sub_theme is None else sub_theme
    if variant not in THEMES[theme]:
        raise ValueError(
            f"sub_theme='{sub_theme}' is not a sub_theme. Sub-themes: "
            "default, black, wsj")
    return THEMES[theme][variant]


def style(theme=None, sub_theme=None, set=None, get=False, reset=True,
          **settings):
    """Set the appearance of every subsequent visualization.

    style("darkred") applies a color theme, resetting every other
    setting to the theme's own; sub_theme="black" draws on a black
    background, "default" on a gray panel with white grid lines, as in
    lessR. Name any setting to change only that one, as in
    style(panel_fill="gray95", axis_size=1.1), with or without a theme.
    style() alone restores the default theme. get=True returns the
    current settings as a dict, which set= restores. A visualization
    function's own theme= applies a theme for that call only.

    trans sets the transparency of both the bar and the point fills.
    Settings that lessR's style() takes for R Markdown or the console,
    or that lessPy takes in the visualization call (rotate_x, labels,
    ...), are not available here. R analog: style()"""
    keys = _style_keys()

    if set is not None:
        for k, v in set.items():
            utils._OPTIONS[k] = copy.deepcopy(v)
        return None

    # transparency names both fills at once, as in lessR
    for alias in ("trans", "transparency"):
        if alias in settings:
            t = settings.pop(alias)
            settings.setdefault("trans_bar_fill", t)
            settings.setdefault("trans_pt_fill", t)

    bad = [k for k in settings if k not in keys]
    if bad:
        r_only = [k for k in bad if k in _R_ONLY]
        other = [k for k in bad if k not in _R_ONLY]
        msg = []
        if r_only:
            msg.append(
                f"{', '.join(r_only)}: lessR style() settings with no "
                "lessPy counterpart (R Markdown and console output, or "
                "set in the call of the visualization function)")
        if other:
            msg.append(f"{', '.join(other)}: not a style() setting")
        raise ValueError(".\n".join(msg))

    nothing = (theme is None and sub_theme is None and not settings
               and not get)
    if nothing:
        theme = "colors"                 # style() restores the default

    if (theme is not None or sub_theme is not None) and reset:
        th = theme if theme is not None else utils.get_option(
            "theme", "colors")
        st = (sub_theme if sub_theme is not None
              else None if theme is not None
              else utils.get_option("sub_theme"))
        base = copy.deepcopy(_BASELINE)
        base.update(copy.deepcopy(_theme_settings(th, st)))
        utils._OPTIONS.clear()
        utils._OPTIONS.update(base)
        utils._OPTIONS["theme"] = th
        utils._OPTIONS["sub_theme"] = st

    for k, v in settings.items():
        utils._OPTIONS[k] = v

    if get:
        return {k: copy.deepcopy(utils._OPTIONS.get(k))
                for k in sorted(keys | {"theme", "sub_theme"})}
    return None


class _AppliedTheme:
    """A theme for the duration of one call, the settings restored
    after it. R analog: the theme= of Chart(), X(), and XY(), which
    call style(theme) with on.exit(style(old_theme))"""

    def __init__(self, theme):
        self.theme = theme

    def __enter__(self):
        self.saved = copy.deepcopy(utils._OPTIONS)
        if self.theme is not None and \
                self.theme != utils.get_option("theme"):
            style(self.theme)
        return self

    def __exit__(self, *exc):
        utils._OPTIONS.clear()
        utils._OPTIONS.update(self.saved)
        return False


def with_theme(func, own_param=False):
    """Give a visualization function the theme= of its lessR analog:
    the theme applies to that call only. own_param: the function takes
    theme itself (Chart), so it is passed on as well."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, theme=None, **kwargs):
        if own_param:
            kwargs["theme"] = theme
        with _AppliedTheme(theme):
            return func(*args, **kwargs)

    return wrapper
