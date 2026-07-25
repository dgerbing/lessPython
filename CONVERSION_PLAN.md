# lessPy — Python Port Conversion Plan

A Python translation of the three analytic-view functions — `Chart()`,
`X()`, `XY()` — from the lessR visual-grammar framework. The goal is to
demonstrate that the framework is conceptual, not an artifact of R.

## Scope

Port only the three analytic-view functions, not all of lessR:

- **`X()`** — one-variable analytic view (distribution of a single variable).
- **`XY()`** — two-variable analytic view (relationship between two variables).
- **`Chart()`** — the general dispatcher / composite view.

## Design decisions (settled)

1. **Plotly-only rendering.**
   - `plotly.py` is actively developed; R's plotly is in maintenance mode.
   - The base-R and lattice engines have no Python equivalent, so there is
     nothing to carry over for them.
   - One rendering backend keeps the port focused.

2. **String interface for variable names.**
   - Python has no non-standard evaluation (NSE), so the lessR
     no-quotes-on-variable-names principle does **not** carry over.
   - Variables are passed as strings referring to columns of a DataFrame.

3. **Stats layer.**
   - `pandas` `groupby` for aggregation.
   - `scipy` / `statsmodels` for statistical computation.
   - ETS (statsmodels) replaces R's `fable` for time-series smoothing.

## Differentiator vs `plotly.express`

The value over `plotly.express` is not new chart types but organization:

- **Analytic-view organization** — 3 functions instead of ~30 chart-named
  ones.
- **Auto-tuning** of plot defaults.
- **Accompanying statistics** returned alongside each view.
- **Composites** such as VBS (violin / box / scatter).

## Packaging (framed for an R-package author)

The user is a CRAN author since 2009 but new to Python packaging. Map the
familiar R concepts:

| Python | R analog |
| --- | --- |
| `pyproject.toml` | `DESCRIPTION` |
| `__init__.py` | `NAMESPACE` |
| `pytest` | `R CMD check` |
| PyPI | CRAN — but **no review**, names are first-come |

**PyPI name (decided July 2026):** distribution name is
**lessPython** (verified free on PyPI: lesspython / less-python /
lessPython all unclaimed), import name stays **lessPy**. So users:
`pip install lessPython`, then `import lessPy`. Background:
`lesspy` is TAKEN on PyPI — an abandoned 2011 LESS→CSS compiler
(PyPI names are case-insensitive, so `lessPy` is the same name).
Remaining action when ready to publish: claim lessPython on PyPI
early (upload requires a PyPI account; names are first-come).

## Sequencing

1. Settle design decisions (string interface, plotly-only) — done.
2. PyPI name checked; taken; naming deferred to publish time — done.
3. Scaffold the package (`pyproject.toml`, `src/lessPy/`, tests)
   — done July 2026. Layout mirrors lessR file names during the
   port: `bc.plotly.R` → `bc_plotly.py`, `zzz_plotly.R` →
   `plotly_utils.py`, `zzz.R` → `utils.py`.
4. Port the plotly renderers, starting with `bc_plotly()` —
   **bc_plotly done** (1-D and 2-D, stack/group, horiz, labels
   modes, hover percents, autocontrast label text). Input contract
   is pre-tabulated, faithful to R: pandas Series (1-D) or
   crosstab DataFrame (2-D, rows = by levels). One deviation:
   axis ticks/grid computed internally via `utils.pretty()` when
   `ax=None`, since there is no upstream Chart() yet.
5. `Chart()` — **bar path done** (July 2026), in `Chart.py`. Ports
   the pipeline of Chart.R, not its lines: the NSE evaluation,
   legacy-parameter migration, base-R/lattice paths, and PDF device
   code have no Python counterpart and are deliberately absent.
   Implemented: counts (1-D), `by=` crosstab (2-D, stack/beside),
   `stat_x="proportion"` (no-by), `y=` with `stat=` (mean/sum/sd/
   min/median/max via groupby, `deviation` from unweighted mean of
   group means), pre-aggregated (pivot) input detection with the
   same errors as R (raw y requires stat; summary table rejects
   stat), `filter=` (pandas query string), `sort=`, casewise NA
   deletion, aggregated-data labels default ("input", "%" if
   beside). Other `form=` values raise NotImplementedError until
   their renderers are ported. Category order: Categorical dtype
   order, else alphabetical (R factor convention).
6. `pie_plotly()` — **done** (July 2026), `pie_plotly.py` ~
   piechart.plotly.R, wired into `Chart(form="pie")` reusing the
   entire Chart data pipeline (counts / by / y+stat / sort /
   filter). 1-D single donut (hole=0.65, clamped to 0.62 for
   inside labels) and 2-D pie grid (one pie per by level, group
   name annotated in each hole). Labels default "%" for pies
   (bar default stays "input"). Auto-titles via build_title()
   (~ .plotly_build_title): main=None auto-builds, main=""
   suppresses — pie path only, matching R where bar titles are
   handled in bc.main. Deviations noted in the file header:
   theme-dependent sequential fills (.scale.clr) not ported
   (no theme system yet; defaults = BASE_COLORS hues), and R's
   post-plotly_build() border repair is unnecessary in plotly.py.
   New helpers in plotly_utils.py: make_trans()
   (~ .maketrans_plotly), auto_text_color() (~ .auto_text_color,
   zzz.R), build_title(), as_plotly_color() (~ .as_plotly_rgba;
   plotly.py rejects #RRGGBBAA everywhere R's plotly tolerates it).
7. `dot_plotly()` — **done** (July 2026), `dot_plotly.py` ~
   dot.plotly.R, wired into `Chart(form="dot")`. Two of the three
   R paths ported: single-series (lollipop: origin->value stems +
   palette dots, vertical or horiz) and paired (x=labels,
   y=[col1, col2] displayed directly, never aggregated, horizontal
   with legend, sort= orders by row mean). The faceted path is
   deliberately not ported until Chart() gains facet=.
   `_dot_origin_grid()` (~ .dot_origin_grid, Chartsub.R) lives in
   Chart.py per the single-caller convention: counts anchor at 0;
   continuous data get origin one grid step below the first tick
   (or at min value when the range hugs it). R-matching errors:
   by= rejected ("Do a bar chart"), counts of unique-x rejected
   (need y), stat rejected for paired. New Chart() params:
   pt_size, origin_x/origin_y, segments_x/segments_y. Dot border
   defaults to pt_color (#324E5C), opacity 1 - trans_pt_fill
   (0.90), stems segment_color (gray40) — added to the options
   dict. Pre-aggregated y now labels axes with the bare variable
   name (was "Value of y"), matching R.
8. `radar_plotly()` — **done** (July 2026), `radar_plotly.py` ~
   radar.plotly.R, wired into `Chart(form="radar")`. Consumes the
   standard Chart table directly: the R renderer's
   .radar_aggregate() bundle with facet/by = "(All)" collapses to
   exactly the Series/crosstab shape the pipeline already builds,
   so no separate aggregation port was needed. Closed scatterpolar
   polygons (lines+markers, toself fill), clockwise, rotation 10,
   radial range [0, 1.08*max], legend outside the paper (x=1.02)
   so plotly never shrinks the polar domain. R-matching checks in
   Chart(): x >= 3 levels, by >= 2 levels, no empty by-x cells
   (fires on real data: Cars93 has no nonUSA Large). by= defaults
   to translucent fills (trans 0.4); sort= is ignored — a radar's
   axis order stays fixed. Hover value label matches
   .radar_aggregate y.label ("Count" / "Mean Salary" / bare y
   name). Faceted radar grid deferred until Chart() gains facet=.
9. `bubble_plotly()` — **done** (July 2026), `bubble_plotly.py` ~
   bubble.plotly.R, wired into `Chart(form="bubble")`. 1-D bubble
   row (bubbles on a hidden y at 0, stems below, per-category
   hues) and 2-D bubble matrix / balloon plot (one trace per by
   level, first level top row, x-grid on, panel border rect).
   Diameter = 2*radius*dpi * v^power / max(v^power) (defaults
   radius=0.50, power=0.5, dpi=96 -> max 96 px). Labels default
   "%" like pie, drawn only on bubbles >= label_min_px (26), with
   autocontrast text. Hover uses dict customdata (%{customdata
   .xcat} etc.). Border defaults to "black" (R: color.miss).
   New Chart() params: radius, power. Small deviation: category
   axes get automargin=True (R's fixed l=25 margin clips long by
   labels). Faceted 1-D grid deferred until facet=. Remaining
   renderer: treemap/icicle (hier.plotly.R).
10. `hier_plotly()` — **done** (July 2026), `hier_plotly.py` ~
    hier.plotly.R + .hier_aggregate() + simplified
    .hier_color_resolve() (placed together per the single-caller
    convention). Covers all three hierarchical forms: sunburst,
    treemap, icicle. **Routing corrected to match R:**
    `Chart(x, by=, form="pie")` renders a SUNBURST via the hier
    path (x wedges, by rings nested inside); the pie GRID that
    by= previously produced is R's facet= treatment and now
    renders only via pie_plotly() directly until Chart() gains
    facet=. `form="sunburst"` accepted as the R alias for pie
    (plain donut without by). Unlike other renderers, hier
    aggregates from RAW columns at every nesting level (node
    table: ids/labels/parents/values, leaves valued,
    branchvalues="remainder", colors inherited from the top-level
    wedge). Labels: Chart resolves the default before dispatch,
    as in R (Chart.R:233) — "%" for counts (label+percent root;
    parents carry value 0 so % keeps inner rings meaningful),
    "input" for stat data via texttemplate on customdata.stat
    (non-additive also drops "% of parent" from hover and bumps
    digits 0 -> 2). stat="deviation" rejected (negative areas).
    Single-level icicle draws right-of-center (plotly.js
    geometry), hence R's title_x=0.71 — reproduced. Faceted hier
    grid deferred until facet=. ALL SEVEN Chart() forms are now
    ported: bar, pie/sunburst, dot, radar, bubble, treemap,
    icicle.
11. `X()` — **histogram and density done** (July 2026), `X.py` ~
    X.R (pipeline, not lines), with two new renderers:
    `hs_plotly.py` ~ hs.plotly.R and `dn_plotly.py` ~ dn.plotly.R.
    X() requires a numerical x (categorical -> TypeError pointing
    to Chart()). Histogram: R hist() binning (Sturges via
    pretty(), or bin_start/bin_width/bin_end), right-closed bins,
    grid lines at bin boundaries, dotted y-grid, hover with bin
    range / % of total / cumulative; stat="proportion" scales to
    proportions (small extension: R's plotly path passes only
    counts); by= overlays translucent groups (auto_opacity) or
    stacks (position="stack"). Density: Gaussian KDE in numpy on
    a 512-point grid extended 3*bw past the data (R density()
    geometry), bandwidth = bw_nrd0 (Silverman, R's default) *
    adjust, mean line per group, area fill toggle (area_fill),
    cumulative % in hover via trapezoid. stat="density" switches
    form to density as in R. Group fills default to BASE_COLORS
    hues when by= is present. New shared helpers: auto_opacity()
    (~.auto_opacity), bw_nrd0(), get_column() (moved to utils —
    now shared by Chart and X). NOT yet ported: form="freq_poly"
    (renderer pending) and the VBS family — VBS renders through
    lattice only in lessR, so a plotly VBS must be designed, not
    translated.
12. `XY()` — **scatter and time-series core done** (July 2026),
    `XY.py` ~ XY.R (pipeline, not lines) with renderer
    `plt_plotly.py` ~ plt.plotly.R. Points, by= groups
    (BASE_COLORS hues, legend at right), the least-squares fit
    family — lm ("ls" synonym), null, exp, quad, power, log — in
    `_plt_fit()` (~ .plt.fit); SE bands about lm fits using the
    predict(se=TRUE) formula with qt(prb, n-1) as in plt.main.R,
    fit_se default 0.95, 0 with by= (XY.R:484); data ellipse(s)
    via the Murdoch-Chow construction of ellipse::ellipse
    (scipy added as a dependency for the t and chi-square
    quantiles). A datetime x renders as a connected time series
    on plotly's native date axis (deviation: R computes its own
    date ticks). Axis ranges extend over fit lines, bands, and
    ellipses, as in R. Renderer deviations in the plt_plotly.py
    header: fits/bands/ellipses arrive as prepared coordinate
    lists rather than per-point vectors; bubble mode and the
    outlier/MD path not ported. (Everything listed as NOT
    yet ported here has since landed: items 17-28.)
    (form="contour": item 23; form="smooth": item 24.
    form="hexbin" dropped by design: plotly has no hexbin
    trace.)
    category_order() moved from Chart.py to utils.py (now
    shared by Chart and XY). (fit="loess": item 17.)
13. **Plotly VBS for X() — done** (July 2026), `vbs_plotly.py`:
    a DESIGN, not a translation (lessR renders VBS only through
    lattice). X(form=) violin/box/strip/bs/vbs map to component
    letters as X.R maps them to vbs_plot (vbs_plot itself is
    deprecated in R and deliberately not ported). Statistics
    follow R exactly: Tukey fivenum hinges (five_num(), tested
    against R), boxplot.stats whiskers, inner/outer fences at
    k/2k hinge-IQRs, outliers split firebrick4 (inner-outer) /
    firebrick2 (extreme). Auto point size ports the one-variable
    .param.VBS() formulas, including the tie-breaking jitter_x
    (R jitter() semantics, amount = factor*range/50, applied to
    the data as X.R does — but only when strip points display).
    Geometry ports the lattice panel: violin half-width
    vbs_ratio/2, box height vbs_ratio/denom with the
    adj.bx.ht/denom formula, whisker caps and median/mean/fence
    segments at box height. Design decisions in the file header:
    jitter_y = strip half-spread as a fraction of violin
    half-width (default 0.5; lattice jitter factors don't map),
    violin bw defaults to bw_nrd0 not the iterated .band.width,
    strip-only also routes fence-outliers to the outlier traces.
    Not ported: box_adj (medcouple), out_cut labels and .bx.stats
    console output (await accompanying statistics), by= and
    facet= (NEXT TASK). kde() moved to utils (shared by dn and
    vbs renderers); VBS colors added to the options: violin_fill
    #7485975A, violin_color/box_color gray15, box_fill #419BD2,
    out_fill/out_color firebrick4, out2_* firebrick2.
14. **by= for the VBS forms — done** (July 2026). As in the
    lattice con_cat panel: ONE violin and box computed over all
    the data (pooled), box fill dropped to 0.25 alpha so points
    show through, strip points per group in BASE_COLORS with
    varying symbols (.plt.shapes "vary" order: triangle-up,
    triangle-down, circle, square, diamond), outliers colored by
    their group rather than firebrick, horizontal legend at the
    top (lattice key). Auto point size and tie-jitter use the
    .param.VBS() group-branch formulas (mx.c = largest group n,
    mc.w = largest within-group tie count, reps threshold 5%).
    vbs_pt_fill does not apply with by=, as in X.R.
    facet= — **done** (July 2026): one full VBS band per level
    stacked on the shared value axis within a single hidden y
    axis (no plotly subplots needed), statistics computed per
    panel, first level in the BOTTOM band (lattice as.table
    default), shaded strip label above each band (strip_fill
    #7F7F7F37, strip_color gray40, strip_text_color gray15 added
    to the options), per-panel box hues (~ .plt.fill "hues")
    under the common slate violin, per-panel violin bandwidth,
    box height denom shrinks 0.5 per level as in plt.lattice.
    .param.VBS() group-branch sizing shared by by= and facet=,
    as in R. Guards: facet= for histogram/density and by= with
    facet= raise NotImplementedError.
    facet= for histogram and density — **done** (July 2026),
    via shared scaffolding in plotly_utils: facet_fig() (stacked
    subplots, shared x, first level in the BOTTOM panel) and
    finish_facet() (per-panel y axes on a common scale, x ticks
    on the bottom panel, per-panel grid and frame, shaded strip
    labels). Histogram (~ .bar.lattice T.type="hist"): shared
    Sturges bins from all the data, shared count scale,
    proportions per panel; the R Trellis fill rgb(150,170,195)
    is already bar_fill_cont. Density: an EXTENSION — R still
    stops with "Facets not yet working with density" — shared
    bandwidth, support, and density scale, mean line per panel.
    by= with facet= — **done** (July 2026) for every X() form.
    VBS needed only the removal of the X() guard (vbs_plotly
    already looped by-groups within each facet band); histogram
    and density gained by= in their facet paths: one series per
    group within each panel on the shared scale, overlay or
    stack (histogram), proportions per group within a panel,
    legendgroup so a legend click toggles the group in every
    panel, legend entries drawn once. A (facet, group) cell
    absent from the data draws nothing; a one-observation cell
    still errors for density. Shared legend_style() helper in
    plotly_utils replaced four identical legend blocks.
    freq_poly — **done** (July 2026), `freq_poly_plotly.py` ~
    freq_poly.plotly.R (modeled on dn_plotly as the R source is
    modeled on dn.plotly): per-group counts on SHARED breaks
    from _breaks_from_args() over all the data, vertices at bin
    midpoints, closed to zero one bin-width beyond each end,
    area fill default with area_fill="off" giving lines +
    vertex markers (zero-size at the closers, R's trick);
    stat="density" falls back to count as in X.R; digits_d
    default 3 (X.R call site). EXTENSION: facet= and by= with
    facet= work (R still errors), same panel conventions as
    histogram/density. X() now covers all forms of X.R.
    **Next tasks**: auto-tuning of plot defaults.
    (Forecasting: items 18-19; facet= for XY() and
    accompanying statistics: items 20-21; outlier/MD
    flagging: item 22.)
15. **facet= for Chart() — done** (July 2026), in Chart.py: one
    panel per facet level for every form, closing out the paths
    items 7-10 deferred. Multiple facet variables flatten to one
    " / " interaction that defines the panel grid, as in R.
    Per form: bar renders the Trellis chart via bc_facet_plotly()
    (bc_plotly.py) ~ .bar.lattice — stacked panels of horizontal
    count bars, counts of the original data only, with R's
    .bcParamValid guards (no sort/stat/y/by); radar and 1-D
    bubble build one table per level with _facet_table() on the
    GLOBAL category set so panels stay aligned (absent cells 0,
    as in xtabs); dot aggregates and sorts per panel (a panel
    shows only its own categories) and hands dot_plotly() its
    faceted path; the hier forms panel via hier_aggregate(facet=)
    -> nd_byfac, one trace per level. Treemap panels place
    right-to-left within a row — verified identical to R
    (hier.plotly.R rev_dom, annotation x 0.26/0.74 both sides).
    facet= on a plain pie renders the pie GRID: the facet becomes
    the grouping, restoring the R treatment item 10 noted.
    Guards ported: multi-y paired dot, stat="deviation", bubble
    facet + by; stat_x="proportion" for faceted radar/bubble/dot
    raises NotImplementedError. labels_size defaults to 0.85 in
    panels (Chart.R:134). Tests: facet section of test_chart.py
    (11 tests: every form, two-variable grid, values per panel,
    guards).
16. **R reference notes for VBS** (kept for the by=/facet=
    work). R references, all under lessRproject/lessR/R/:
    - X.R VBS section (~lines 1037-1300): resolves components
      from vbs_plot letters ("v" violin, "b" box, "s" strip;
      form= violin/box/strip/bs/vbs maps to letter sets), fills,
      and calls .plt.lattice (plt.lattice.R:~) at X.R:1269.
    - bx.stats.R (.bx.stats): box statistics + outlier text —
      also the basis of X()'s outlier console output later.
    - param.VBS.R: VBS style parameter definitions.
    - Key X() params to support: vbs_plot, violin_fill, box_fill,
      vbs_pt_fill, pt_size, vbs_mean, fences, k=1.5 (IQR
      multiplier), box_adj (adjusted boxplot, a=-4, b=3),
      jitter_x/jitter_y, by= (one VBS per group).
    - Plotly building blocks: go.Violin (has built-in box and
      points options, but lessR's VBS layers them explicitly —
      likely go.Violin + go.Box + jittered go.Scatter for full
      control of the lessR look: violin fill, narrow box with
      outlier styling, strip points below/overlaid per
      vbs_ratio).
    Then: freq_poly, facet= support, and auto-tuning
    (labels_position="out" for small bars, etc.).

17. **XY() fit="loess" — done** (July 2026), in XY.py: _loess()
    implements R's loess directly in numpy — local quadratic
    regression, tri-cube weights, gaussian family, evaluated
    exactly at each x (R surface="direct") — so no statsmodels
    dependency (its lowess is degree-1, robust, no SE). Fitted
    values and se.fit (from the equivalent-kernel rows, sigma^2
    via tr[(I-L)'(I-L)]) match R loess()/predict(se=TRUE) with
    statistics="exact" to ~1e-13, verified for spans 0.4, 0.75,
    and 1.2. For span > 1, R scales the max distance by
    span^(1/2), not span^(1/p) as documented — established
    empirically against R fitted values at degrees 1 and 2. SE
    bands as for lm: fit +/- qt((1+lv)/2, n-1) * se.fit
    (plt.main.R ~1349). span=0.75 is an explicit XY() parameter
    (in R it reaches .plt.main only through XY's dots).

18. **XY() classic forecasting — done** (July 2026),
    plt_forecast.py: ts_source="classic". es ports
    stats::HoltWinters directly (decompose+lm start values, the
    C_HoltWinters recursion, L-BFGS-B on SSE over only the free
    smoothing parameters; scipy sometimes lands a marginally
    LOWER SSE on the flat beta direction — forecasts then agree
    with R to ~2e-3). PIs port predict.HoltWinters (psi weights
    on var(residuals), qnorm), including its multiplicative
    indexing quirk. lm ports the stl+regression path with
    statsmodels STL (s.window="periodic" -> seasonal=10n+1,
    seasonal_deg=0; agrees with R to ~5e-3); trend-only lm is
    machine-exact vs R. Aggregation is NOT ported, so lessPy
    matches R's no-ts_unit behavior; R's ts_truncate drops an
    incomplete final period only when aggregating (ts_unit
    given) — a trap when comparing. R golden values in
    test_xy.py. Console report + forecast table print;
    overlay traces port plt.plotly.R ~343-451 (default-theme
    forecast hue rgb(.6,0,0)).

19. **XY() fable-equivalent forecasting — done** (July 2026),
    plt_forecast.py: ts_source="fable" (the default, as in R),
    per settled decision #3 (statsmodels replaces fable). es:
    ETSModel over the component grid fable searches, selected
    by AICc — on the Apple test data it picks the same model as
    fable, ETS(M,A,N); fixed components via
    ts_error/ts_trend/ts_seasons ("Ad"/"Md" damped accepted),
    fixed smoothing via fix_params. PIs as lessR builds them:
    mean +/- qnorm * sqrt(forecast_variance) — symmetric, NOT
    statsmodels' simulated asymmetric summary_frame intervals.
    Point forecasts within ~1.4% of fable, PIs within ~10% at
    h=12 (initial-state estimation differs between libraries).
    lm: TSLM analog — OLS on trend + calendar-season dummies
    (month/quarter of the date, cycle position otherwise) with
    qnorm PIs; machine-exact vs fable::TSLM on the test data,
    with the R-style coefficient table, R^2 and F reported.
    statsmodels: lazy import, now a core dependency (also used
    by Regression/Logit/ANOVA), moved from the optional
    [forecast] extra to core deps July 2026.

20. **facet= for XY() — done** (July 2026), _xy_facet in XY.py:
    one scatter panel per level on shared axes, following the
    _hs_facet conventions (facet_fig / finish_facet, first
    level bottom, strip labels, legend from first panel only).
    by= combines (grouped colors per panel); fit lines, SE
    bands, and ellipses compute per panel per group; y range
    overridden after finish_facet (scatter y spans the data,
    not [0, max]). Not for a date x or with forecasting
    (clear errors).

21. **Accompanying statistics — done** (July 2026),
    stats_out.py, printed by default with quiet= on X(), XY(),
    Chart() (default from the "quiet" option, as R
    getOption("quiet")). X(): n/missing, mean/sd, five-number
    summary, 1.5-IQR outliers (~ SummaryStats.R + bx.stats.R).
    XY(): correlation r with t test and Fisher-z 95% CI,
    overall or per by= group, plus MSE and R-squared for any
    fit line; forecast report already prints (now also gated
    by quiet). Chart(): one-way frequencies with proportions
    and Total; two-way crosstab + chi-square with by=; n and
    the stat per level with y=. DESIGN: print-only this
    increment — the functions still return the plotly Figure
    (notebooks display it automatically); a returned stats
    object would wrap the Figure and is deferred. Auto-tuning
    remains open (item 13 next tasks).

22. **Outlier/MD flagging for XY() — done** (July 2026):
    md_outliers in stats_out.py ports .plt.MD (plt.MD.R) —
    bivariate Mahalanobis distance from the centroid, MD_cut
    absolute threshold or out_cut proportion/count; distances
    match R's mahalanobis() exactly (verified on planted
    outliers). Console: the sorted MD/ID table (outliers, a
    blank line, the next 3 for comparison, "..."), printed
    with the accompanying statistics under the same quiet=.
    Display ports plt.plotly.R ~140-185: flagged points
    overdrawn with out_shape (default "circle-open" — in R the
    plotly branch maps the unrecognized default "circle" to
    circle-open), ID annotations below each point (ID= column,
    default the DataFrame index; ID_color/ID_size). With fit=
    on, a second dashed fit line excludes the outliers, as
    plt.main.R's fit.remove loop. Single scatterplot only —
    by=, facet=, and a date x raise, matching the R condition
    (plt.plotly draws outliers only in its no-groups branch).
23. **XY() form="contour" — done** (July 2026),
    `plt_contour.py` ~ plt.contour.R: filled contours of the
    2-D kernel density via go.Contour. _kde2d() ports
    MASS::kde2d() (product normal kernel, bandwidth.nrd)
    exactly — grid matches R to machine precision. The R
    axis-limit logic is ported: 12%-extended kde grid, 95%
    density-mass threshold, limits from the grid extent or the
    data/truncated-95%-ellipse union. contour_n (20) bands,
    contour_nbins (50) grid, contour_points (translucent black
    overlay), contour_legend (colorbar); setting any contour_
    parameter selects the form, as in R (XY.R ~174). Overlays:
    ellipse= draws line-only (no fill, as R) and fit= draws
    without SE bands (se_levels cleared). Fill ramp fixed at
    the default-theme blues, colorRampPalette(c("white",
    getColors("blues"))) anchors — themes not ported. plotly
    fills below the first contour level with the low end of
    the colorscale, so contours.start is the first INTERIOR
    level (lv0 + step); with start at z.min() the whole panel
    tints light blue instead of white. by=, facet=, and a date
    x raise, matching R's scatter-only conditions (XY.R
    211-225). form="hexbin" dropped: no plotly hexbin trace.
    PNG-verified against R side by side (plain and
    points+fit+ellipse variants).
24. **XY() form="smooth" — done** (July 2026),
    `plt_smooth.py` ~ the smoothScatter() branch of plt.main.R
    (~795-808): the 2-D binned kernel density as a smoothed
    go.Heatmap raster (zsmooth="best"; R's image() draws raw
    cells). _bkde2d() ports KernSmooth::bkde2D() exactly —
    linear binning (_linbin2d), separable normal kernel via
    fftconvolve, tau=3.4 truncation; grid matches R to machine
    precision. Bandwidth is smoothScatter's (q95 - q05)/25 per
    axis, grid extends data +/- 1.5*bandwidth (bkde2D default
    range.x). smooth_points (100) lowest-density points
    overplot as 2px black specks (~ pch="."), nearest-cell
    density lookup as smoothScatter; smooth_power (0.25)
    transforms z for display; smooth_bins (128) grid;
    smooth_size scales the specks. No auto-switch on smooth_
    parameters — R has none (unlike contour_). Unlike contour,
    fit= keeps its SE band and ellipse= its fill (R draws them
    through the standard plt.main sections, no form gate).
    Ramp: white -> hcl(240, 80, 16) = "#0041A5", the
    default-theme clr.den. Window: data/overlay extent + 4%,
    out-of-range ticks drop (deviation: R's window also shows
    a leading tick, e.g. 0, outside the data range). MD_cut/
    out_cut now raise for any non-scatter form (R silently
    skips drawing them on smooth). 8 tests in test_xy.py;
    PNG-verified against R (plain and fit variants).
25. **jitter for XY() — done** (July 2026), in XY.py:
    jitter_x/jitter_y port plt.main.R ~634-659. Display
    coordinates only: the fit, SE bands, ellipse, MD
    distances, and statistics all use the original data (R
    jitters, then restores before those sections). Auto-rule
    per axis: <= 14 unique values and n > 14 -> jitter =
    range/32; uniform +/- jitter via
    numpy.random.default_rng() (unseeded, as R's runif);
    explicit 0 disables. Scatter form only, single panel or
    by= (R's lattice facet path has no auto-jitter, so
    facet= ignores jitter, as R does). Console: the R "Some
    Parameter values (can be manually set)" block — size,
    jitter_y, jitter_x — appended to the accompanying
    statistics under the same quiet=. 6 tests in test_xy.py.
26. **enhance= for XY() — done** (July 2026), in XY.py:
    ports XY.R:445-450, the enhanced-scatterplot bundle. For
    components not explicitly set: ellipse 0.95, fit "lm",
    MD_cut 6, and the add="means" crosshair (_add_means():
    panel-spanning line shapes at mean x and mean y, add_color
    gray10 ~ plt.main.R annotations 1403-1424). Python has no
    missing(), so default values stand in — passing the
    default explicitly cannot opt out (deviation). MD_cut set
    only where flagging is supported (single-panel scatter
    form, non-date); with by=/facet= or a density form the
    rest of the bundle still applies, so enhance never
    triggers the flagging guards. Crosshair on scatter and
    smooth without facet (R's annotation path); contour gets
    ellipse+fit only, as .plt.contour draws. enhance with a
    date x raises. 5 tests; PNG-verified against R (ellipse,
    fit + dashed no-outliers refit, crosshair, same two
    planted outliers flagged; IDs differ by the established
    0- vs 1-based index convention).
27. **facet= for a date x — done** (July 2026), _ts_facet in
    XY.py: time-series panels on the shared facet scaffolding
    (facet_fig/finish_facet, first level bottom, strip
    labels, y label middle panel). x axis uses plotly's
    native date ticks (ax axT1=None falls through axis_num to
    auto ticks) with the native date grid enabled per panel;
    y ticks pretty() over all panels, range data +/- 4%.
    Facet order sorts with the global date sort (facet_ser
    now reordered in the is_date block). DEVIATION: panels
    connect lines+markers as the single-panel time series
    does; R's lattice cont_cont path draws points only.
    ts_unit + facet raises (XY.R:195-197); by= with a faceted
    time series raises NotImplementedError (untested territory
    in R's lattice groups path — port later if wanted);
    ts_ahead + facet already raised. 4 tests; PNG-verified
    against R (same panel order and scales).
28. **ts_ aggregation/stack/area — done** (July 2026). XY()
    is now FULLY PORTED (remaining R-only pieces are the
    deliberate exclusions: hexbin, categorical-x delegations,
    by= on a faceted ts). Three pieces:
    (a) ts_unit/ts_agg — `plt_time.py` ~ plt.time.R: infer
    the existing unit (median-gap rule; R's calendar-structure
    refinements for gappy series not ported), refuse finer
    than the data, drop the trailing incomplete period,
    aggregate sum/mean per period per by= level; dates to
    period start (months/quarters/years) or last observed
    date (weeks/days, as xts). Matches .plt.time() output
    exactly on daily->monthly and daily->weekly (golden
    check); gaps note quiet-gated (R message is not).
    Aggregation runs before forecasting, so ts_unit+ts_ahead
    now compose. (b) ts_stack — cumulative curves per by=
    level (requires identical dates), fill bands between
    successive curves (first fills to its own min, as
    plt.main.R:735), default hues = the by-line colors with
    point transparency, lines on top, pt_size defaults 0.
    Hover reports cumulative values. (c) ts_area_fill/
    ts_area_split — single-series fill toward ts_area_split
    (lattice origin semantics, clipped to the plot; R's
    plt.main fills to the axis min — same look for data
    above the split), "on" = violin_fill, explicit #RRGGBBAA
    keeps its alpha (_area_color); y range pinned so the
    fill reaches the axis edge; also per-panel in _ts_facet
    (lattice panel.xyarea parity). Guards: non-date x, by=
    without ts_stack (plt.colors.R stop), ts_stack without
    by=. area polygons render via plt_plotly's new
    area_polys param (prepared-coordinates convention).
    10 tests; stack + area PNG-verified against R.

## Directory layout (restructured July 2026)

`lessPyProject/` is itself the package project root — the former
extra `lessPy/` wrapper was removed so only the importable package
carries the name:

    lessPyProject/            ~ lessRproject/lessR (package root)
      pyproject.toml          ~ DESCRIPTION
      src/lessPy/             ~ lessR/R/  (the import package)
      tests/                  ~ R CMD check tests
      demo/

## Development environment

conda env `lessPy` (Python 3.13; plotly, pandas, pytest, kaleido).
Editable install done: `conda activate lessPy` then from
`lessPyProject/`: `pytest` to test, `python demo/demo_bc.py`
for visual output (Cars93).

## Reference

- Framework source: the three lessR analytic-view functions.
- Paper: the visual-grammar / JSS paper that frames the conceptual claim.
29. **X() histogram/density embellishments — done** (July
    2026). Histogram: cumulate= "on"/"both" (hst.main.R —
    cumsum, "Cumulative" y label; "both" overdraws the
    regular bars in reg=, default snow2 #EEE9E9, plain
    bin/value hover on cumulative bars) and counts= (labels
    above the bars, textposition outside). Density: kind=
    "normal"/"both" (normal curve at the sample mean/sd,
    line-only color_normal gray20 unless fill_normal set,
    plus the Shapiro-Wilk console report for 2 < n < 5000,
    dn.main.R); show_histogram=True default with fill_hist
    (se_fill) — the faint density-scaled histogram behind
    the curve that R always drew and lessPy had omitted,
    bins from the same bin_*/breaks policy; rug= with
    color_rug/size_rug (tick segments to -0.05*ymax; rug or
    rug styling switches form to density, X.R:136-138).
    All are single-panel features as in R: by=/facet= raise
    for cumulate/counts/kind/rug; show_histogram silently
    does not draw. 9 tests; PNG-verified against R
    (cumulate both, density default, kind both + rug).
    STILL OPEN for X(): VBS box_adj (+a, b), bw_iter,
    out_cut/ID/ID_size strip-plot outlier labeling; the
    n_row/n_col/aspect facet layout; axis-format/margin
    family; add= annotations (not ported anywhere).
30. **X() VBS refinements + facet grid — done** (July 2026).
    VBS: box_adj (+a, b) — _medcouple() in vbs_plotly.py
    matches robustbase::mc(doScale=FALSE) exactly (naive
    kernel with the signum rule for median ties; p >= 1
    covers the odd-n single-median pair), fences scaled
    exp(a*mc)/exp(b*mc) inner and outer (_adj_fences ~
    plt.lattice.R 612-627); whiskers/outliers follow.
    band_width() in utils.py = .band.width() exactly (bw_nrd0
    widened 10% until <= 1 flip or bw_iter) and is now the
    default violin bandwidth, as in R. out_cut/ID/ID_size:
    up to out_cut outliers (largest |x|, R's rule) labeled
    above the strip, rotated, ID column or DataFrame index.
    Facet grid: n_row/n_col for histogram/density/freq_poly
    facets — facet_fig(n_rows, n_cols) + facet_pos() lattice
    bottom-up fill (bottom row always full -> x ticks there),
    finish_facet(n_col=) generalized: y ticks col 1, y label
    middle-left, ragged-top empty cells hidden. VBS bands
    reject n_row/n_col (bands are the lessPy facet design);
    aspect= not ported (figure sizing). Guards: n_row/n_col
    need facet=. 12 tests.
31. **add= annotations — done** (July 2026), `plt_add.py` ~
    plt.add.R, wired into Chart(), X(), XY() with x1/y1/x2/y2.
    Full vocabulary: v_line, h_line, line, rect, arrow, point,
    any other string = text at (x1, y1). R semantics ported:
    style from the add_ options (add_color gray10, add_fill
    #D9D9D920, add_lwd .5 -> px = lwd*2, add_lty, add_size,
    add_trans), each recyclable per object; one object +
    vector coords repeats at every location, several objects
    consume one coordinate each. Scope follows where R draws
    them: Chart bar (bc.main), X histogram (hst.main), XY
    scatter/time series (plt.main) — single panel; by=/facet=/
    density/contour/smooth raise (XY contour/smooth uses R's
    own message). XY resolves add="means" (via _add_means) and
    "mean_x"/"mean_y" coordinates. Rendered as plotly shapes/
    annotations (arrow = annotation with data-coord tail;
    point = marker trace). Tests distinguish add shapes from
    grid/border shapes by layer != "below".
32. **Axis-format family — done** (July 2026), the meaningful
    subset (decided with the user; margin/label-adjust knobs
    and sub= stay unported as device concerns, aspect= too).
    axis_format() in plotly_utils ~ .axis.format: "K"
    (default, billions -> scientific, multiples of 100 all
    past 1000 -> "60K", commas past 9999), ",", ".",
    "" + prefix. NOW THE DEFAULT LABEL BUILDER in XY (single,
    _xy_facet, _ts_facet), plt_contour, hs/dn/freq_poly/vbs —
    closes the visible gap where R showed "60K" and lessPy
    "60000". axis_fmt/axis_x_pre/axis_y_pre exposed on X()
    and XY(); rotate_x/rotate_y (tickangle = -angle, all
    panels incl. facets); scale_x/scale_y (XY) and scale_x
    (X): explicit linspace ticks + range, single panel.
    Chart() keeps its own rotate_x; its value-axis labels not
    yet routed through axis_format (noted). 5 tests.
33. **Chart() finish-out — done** (July 2026), four items that
    closed the last Chart() gaps behind X()/XY():
    (a) **Value-axis labels through axis_format.** bc_plotly,
    bc_facet_plotly, and dot_plotly now format their value axis
    with axis_format() ("60K"), exposed on Chart() as axis_fmt/
    axis_x_pre/axis_y_pre and rotate_y (its own rotate_x already
    existed). The prefix follows the physical axis the values
    land on (x for horizontal bars / the Trellis, y for vertical).
    Chart's value axis was the one place still showing "60000".
    (b) **stat_x="proportion" with facet=** for radar/bubble/dot
    (was NotImplementedError). _facet_table() gained a proportion
    flag that normalizes each panel's counts to sum to 1;
    per-panel proportions, matching bc_facet_plotly's Trellis
    convention. Guards mirror the non-faceted paths: proportion
    rejects a y variable, and radar proportion + by= still
    raises. Titles/labels/digits say "Proportion" (digits_d 2).
    (c) **n_row/n_col facet layout.** facet_domains_grid()/
    facet_layout() and facet_fig()/finish_facet() gained an
    explicit n_col; Chart() resolves n_col_use from n_row/n_col
    (n_col wins, else ceil(n_levels/n_row)) and threads it to
    every faceted renderer — the Trellis bar (single-column
    default) and the radar/bubble/dot/hier grids (up-to-3-col
    default). n_row/n_col require facet=; aspect= stays unported
    (figure sizing), as in X().
    (d) **Auto-tuned label placement.** Chart()'s labels_position
    now defaults to None = plotly "auto" (each bar label sits
    inside, or moves outside when the bar is too small);
    "in"/"out" force one. Outside labels get the axis text color
    (Chart.R ~287). Stacked bars always label inside, and
    labels_position="out" with a stacked by= raises, as R does
    (Chart.R:509). pie_plotly/bubble_plotly now read None as the
    "in" default (R's %||% "in"). 15 tests in test_chart.py;
    PNG-verified (60K bars, out labels, 2x2 proportion dot/radar
    grids, Trellis grid). Chart() is now fully ported; the
    remaining open items are cross-view (theme system, a returned
    stats object).

34. **facet= orthogonality — done** (July 2026), the parity pass
    for R's July facet unification (facet1=/facet2= replaced by
    one facet= across Chart()/X()/XY()):
    (a) **Resolution** (`utils.resolve_facet`): facet= takes a
    column name, a list of two names (>2 message, first two), or
    an aligned Series/ndarray of computed values — the Python
    analog of R's facet expression, length-checked. Chart()
    keeps its " / " flatten for multiple names (as R) and gained
    the Series/array form.
    (b) **Two-facet grids** in X() and XY(): rows = facet2,
    columns = facet1, filled top-down (facet_cells2/facet_panels
    in plotly_utils; finish_facet gained pos=). Cell strips read
    `Var = "level"` pairs (quotes only on non-numeric levels, ~
    .lab.lv), with the strip font auto-shrunk to fit the panel.
    Applied to histogram, density, freq_poly, scatter, and the
    date-x time-series panels.
    (c) **VBS hybrid** (design, not translation): facet2 stacks
    one full band section per level on the single shared x axis
    under a `Var = "level"` section strip — chosen over R's
    lattice cell grid to keep the bands' full-width
    comparability (vertical sections, not side-by-side columns).
    (d) **Dist-facet alignment** (~ .plt.dist.facet): density
    facets now take the single-panel embellishments per panel
    when by= is absent (kind= normal curve from panel mean/sd,
    show_histogram backdrop on shared bins — now the default, as
    single-panel — and rug, with the y range opened below zero
    for the ticks); a too-thin cell (< 2 finite values) draws an
    empty panel and only an all-empty grid stops ("No facet cell
    has enough data"); facet x ticks now go through
    axis_format() ("60K"), also for freq_poly.
    (e) **Faceted contour/smooth** (`plt_contour_facet.py` ~
    plt.contourFacet.R): both renders on a common kde2d grid and
    bandwidth (smooth uses smoothScatter's (q95-q05)/25 sd so
    faceted and single-panel smooths agree; contour keeps
    bandwidth.nrd), shared contour levels / heatmap zlim so
    density is comparable panel-to-panel, per-panel ellipse and
    fit (callables passed from XY to avoid a circular import),
    near-square default grid as R (other facets keep the
    established single-column default). scale_x/scale_y still
    raise with facet= (R allows them here — minor deviation).
    (f) **Multi-series overlay + facet-series**: XY() x= (or y=)
    accepts a list of names — single-panel overlay
    (`_series_overlay`, per-series colors/fit/SE/ellipse,
    legend) or, with facet=, the facet-series panels
    (`_facet_series` ~ .plt.facet.series: full overlay per
    panel, per-series fit in the series color, common scale,
    single-column default). by= with a vector errors redirecting
    to facet=; both-vectors (R's scatterplot matrix) and
    multiple date series stay unported.
    (g) **XY() n_row/n_col** (aspect= stays out); faceted
    scatter prints a summary table per grouping variable
    (`stats_out.facet_summary` ~ .vbs_summary_table) after the
    correlation block; a cat/cont mix with facet= raises the R
    redirect ("use X(cont, by=cat, facet=...)").
    16 new tests (233 total); PNG-verified across all five
    faceted families.

35. **Regression() core — done** (July 2026), the first port
    beyond the three views: `Regression.py` ~ Regression.R plus
    the reg.1-4 helpers, core numeric OLS scope (user decision).
    Model is a formula STRING, "Salary ~ Years + Pre" — the
    Python analog of the R formula ("Y ~ ." expands to the other
    numeric columns; "*", ":", "^", "I(", "poly(", "log("
    raise with guidance; interactions come later). Numerics
    from statsmodels OLS (lazy import, as plt_forecast);
    OLSInfluence for rstudent/dffits/Cook's; get_prediction for
    the intervals. Sections: BACKGROUND, BASIC ANALYSIS
    (estimates + 95% CIs ~ .reg1modelBasic; Model Fit with the
    PRESS R-squared ~ .reg1fitBasic; sequential Type-I ANOVA
    with Model and response-total rows ~ .reg1anvBasic),
    RELATIONS (Tolerance/VIF via R's sterr formula ~
    .reg2Relations), RESIDUALS AND INFLUENCE (res_sort/
    n_res_rows listing ~ .reg3txtResidual), PREDICTION ERROR
    (s_pred, 95% PIs, pred_sort, the three-piece display around
    the widest/narrowest intervals ~ .reg4Pred, X1_new..X6_new
    expand-grid). brief= trims as R; quiet= does not exist, as
    R. Returns RegressionResults (DataFrames + .plots dict —
    figures NOT auto-shown; no R graphics device to open).
    Graphics: simple-reg scatter + CI/PI bands (~ .reg5Plot),
    residuals vs fitted with Cook's-cut labeling
    (~ .reg3resfitResidual), Distribution of Residuals via
    dn_plotly (~ .reg3dnResidual).
    VERIFIED AGAINST R on an identical CSV (seed-3 frame, n=98):
    every printed number matched lessR to the last digit —
    estimates/CIs, sd/ranges, R-squareds incl. PRESS, sequential
    SS, VIF, residual listing (values and order), prediction
    pieces. tests/test_regression.py hard-codes those R-verified
    values (11 tests; 244 total).
    Not ported (deferred): ANCOVA (categorical predictors
    raise, pointing here), best subsets (leaps), kfold=, mod=,
    new_scale=, the multiple-regression scatterplot matrix, Rmd
    generation, and R's empty K-FOLD heading signpost. Cosmetic
    deviations: no comma formatting (R .fmt_cm), pandas 0-based
    row labels (R's are 1-based).

36. **Logit() core — done** (July 2026), `Logit.py` ~ Logit.R +
    logit.3Residual/4Pred/5Confuse, same pattern as Regression
    (formula string, statsmodels engine — GLM Binomial, lazy
    import, fit(tol=1e-12) — lessR headings verbatim,
    LogitResults object with .plots, R cross-check on a shared
    CSV). Response: numeric 0/1 or two-level categorical; the
    second level is the reference group predicted as 1,
    re-ordered by ref_group=; a single numeric predictor
    reorders factor levels for a positive slope, as R. Sections:
    variables/cases; BASIC ANALYSIS (estimates on the logit
    scale with Wald CIs ~ confint.default, digits_d=4 default;
    odds ratios + CIs; deviances/df, AIC, iteration count);
    Collinearity via the auxiliary linear model (R's formula);
    ANALYSIS OF RESIDUALS AND INFLUENCE (P(Y=1), response
    residual, rstudent, dffits, Cook's — NOTE: R's rstudent()
    on a glm empirically returns the standardized Pearson
    residual pear/sqrt(1-hat), NOT the textbook deviance
    formula; verified by direct probe and matched);
    PREDICTION (fitted/std.err sorted by fitted with the
    around-0.5 three-piece display, X1_new..X6_new grid);
    confusion matrix per prob_cut (list OK) with x.cut for one
    predictor, Accuracy/Sensitivity/Precision. Graphics: the
    fitted-sigmoid plot with prob_cut/x.cut crosshairs and
    right-axis level labels (single numeric predictor).
    VERIFIED AGAINST R on an identical CSV (seed-5, n=98):
    estimates/ORs/deviances/AIC/VIF/influence/confusion all
    match; tests/test_logit.py hard-codes the R values (10
    tests; 254 total). Known tiny deviations: standard errors
    agree to ~1e-4 (R's IRLS stops at its 1e-8 deviance
    criterion short of full convergence); iteration count
    differs (statsmodels IRLS bookkeeping); R's base-graphics
    scatterplot matrix for a multiple logit stays unported, as
    are indicator variables for categorical predictors.

37. **Indicator variables — done** (July 2026), closing the
    categorical-predictor gap in both Regression() and Logit():
    `_expand_indicators()` in Regression.py (shared with Logit)
    builds treatment-coded 0/1 columns per non-reference level,
    named VarLevel as R's model.matrix(), first level (declared
    Categorical order, else sorted) the reference, expanded in
    formula position after casewise deletion. Each function
    prints its R announcement verbatim (Regression ">>>  X is
    not numeric. Converted to indicator variables."; Logit
    ">>> Note:" two-liner). Downstream sections work on the
    dummies automatically (estimates/CIs, ORs, collinearity,
    residual and prediction listings, X?_new post-expansion,
    Logit's positive-slope flip and sigmoid conditions
    evaluated post-expansion as R does). ANCOVA case (exactly
    one covariate + one factor): Regression prints "-- Analysis
    of Variance from Type II Sums of Squares" with TERM-level
    rows (covariate df 1, factor df k-1, each adjusted for the
    other via drop-term RSS; no Model/total rows). NOTE: this
    deliberately CORRECTS a verified bug in R's .reg1ancova —
    R's table replaces its first row (labeled with the
    covariate) with anova(lm(y ~ cont + cat))[2,], the FACTOR's
    adjusted SS under the covariate's name, leaving sequential
    dummy rows beneath; Python's Type II values were verified
    against direct R anova() drop-term probes instead. The
    R-side fix was then applied (July 17: .reg1ancova rebuilds
    the table term-level from the two entry orders), so R and
    lessPy now print identical Type II tables. R-verified on a
    seed-9 CSV (estimates/ORs/fit exact in both functions);
    6 new tests (259 total). Still open on this path: the
    ANCOVA group-lines plot (.reg5ancova) and adjusted means.

38. **Best subsets — done** (July 2026), closing another core
    Regression() gap: `_best_subsets()` scores every non-empty
    predictor subset by adjusted R-squared (default) or Mallows'
    Cp, keeps the nbest=10 best of each size (the leaps()
    default), and sorts — R-matching output. New Regression()
    params subsets= (None/True default on with >1 predictor;
    an int n caps the printed lines, as R's subsets=n) and
    best_sub= ("adjr2"/"Cp"). Display ports .reg2Relations: the
    0/1 indicator columns, the R2adj/Cp and X's columns, the
    "up to 10 subsets of each number of predictors" note when
    n_pred > 5, the 50-line cap with ">>> Only first m of n
    rows printed", and the every-30-rows header repeat.
    Operates on the EXPANDED indicator columns and gates on the
    pre-expansion predictor count (n_pred_orig > 1), both as R.
    ENGINE DEVIATION (documented): an exhaustive all-subsets
    search (itertools.combinations, capped at 15 predictors =
    2^15 fits) replaces leaps' branch-and-bound — identical
    results, and the footer reads "[exhaustive search of all
    predictor subsets]" rather than crediting leaps. Result
    object gains .subsets (a DataFrame, None when off).
    VERIFIED AGAINST R on the seed-3 (adjr2 and Cp) and seed-9
    (indicator columns) CSVs — subset membership, criteria, and
    order all match leaps to display precision; 4 new tests
    (263 total). Regression() core is now complete except
    moderation (mod=), k-fold, new_scale=, the scatterplot
    matrix, and Rmd.
