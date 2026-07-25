# lessPy Function Reference

The public API is flat: every function below is reached from the
top-level package.

```python
import lessPy as lp
lp.Chart("Type", data=d)
lp.Regression("Salary ~ Years + Pre", data=d)
```

or import names individually:

```python
from lessPy import Chart, Regression, simMeans
```

Variables are passed as **strings** naming columns of a pandas
`DataFrame` — Python has no equivalent of R's non-standard
evaluation, so `Chart("Type", data=d)`, not `Chart(Type)`.

The categories below are for navigation only; they are not
separate import paths.

---

## Graphics output

Every plot function (`Chart`, `X`, `XY`, `Regression`, the `sim*`
and `prob_*` functions, …) returns a standard **plotly** figure.
How it appears in a notebook is set by plotly's *renderer* — pick
it once, right after importing, and every figure follows it.

```python
import lessPy as lp
import plotly.io as pio
```

**Interactive widgets** (hover, zoom, pan):

- `pio.renderers.default = "notebook"` — self-contained and works
  **offline** (embeds plotly.js, larger file); best for the
  classic Jupyter Notebook.
- `"notebook_connected"` — same, but loads plotly.js from a CDN:
  smaller file, **requires a connection**.
- `"plotly_mimetype"` — the native renderer for **JupyterLab**
  and **VS Code**; it is the automatic default there, so usually
  nothing need be set.
- `"colab"` — use in **Google Colab**.

**Static images** (PNG — no interactivity, smallest and most
robust for large notebooks):

- `pio.renderers.default = "png"` — requires the `kaleido`
  package (`pip install kaleido`, or `conda install -c
  conda-forge python-kaleido`).

With a renderer set, let the figure be a cell's last line, or
call `.show()`:

```python
d = lp.read_data("Employee")
fig = lp.XY("Years", "Salary", data=d)
fig.show()
```

Override per figure without changing the session default with
`fig.show(renderer="png")` (or `"notebook"`). Save to a file with
`pio.write_image(fig, "salary.png", scale=2)` (static, needs
kaleido) or `fig.write_html("salary.html")` (interactive,
standalone, no kaleido).

| Environment | Interactive | Static PNG |
|---|---|---|
| Classic Jupyter Notebook | `"notebook"` | `"png"` |
| JupyterLab / VS Code | (default) | `"png"` |
| Google Colab | `"colab"` | `"png"` |

---

## Plots

Analytic-view visualizations, rendered with plotly. Each returns
its plotly figure (and prints supporting statistics). `Chart`,
`X`, and `XY` accept `font_size=` (a multiplier, e.g. `1.3`) to
scale all of the figure's text. `by=` distinguishes groups by
color; pass a `pt_shape=` vector to vary the plot symbol as well.

- **`Chart(x, y=, by=, facet=, data=, form='bar', stat=, ...)`**
  — General view of a categorical variable, optionally crossed
  with a second (`by=`) or summarizing a numeric variable
  (`y=` with `stat=`). As in R, the second positional argument is
  `y`, the numeric variable being aggregated, so a second
  categorical variable is named: `by='Gender'`. Forms: bar,
  pie/sunburst, dot, radar,
  bubble, treemap, icicle. Numeric bars can be colored by value:
  `fill_split=v` (two colors split at `v`) or `fill_scaled=True`
  (an HCL gradient by distance from the split). `theme=` (green,
  slatered, sienna, blue, …) fills with a sequential palette in the
  theme's hue. With `by=`, `beside=True` groups the bars side by
  side and `stack100=True` rescales each bar to sum to 1.0 for a
  100% stacked chart of the composition within every `x` category
  (`labels="input"` then shows the underlying counts). Bar layout:
  `gap=` sets the spacing between bars in units of bar width — one
  value, or `(within, between)` with `beside=` — `scale_y=(min,
  max, n_intervals)` fixes the value axis, and `break_x=` breaks
  category labels at their spaces onto separate lines (on by
  default for unrotated vertical bars; a `~` in a label is a
  non-breaking space). Value labels: `labels=` chooses `"%"`,
  `"input"`, `"prop"`, or `"off"`, and `labels_decimals=` sets
  their decimal places for bar, pie, and bubble. The legend of a
  two-variable bar chart is controlled by `legend_title=`,
  `legend_labels=` (rename the `by` levels), `legend_abbrev=n`
  (shorten title and labels), `legend_size=` (a text expansion
  factor), `legend_horiz=True`, `legend_position=` (`"right_margin"`,
  the default, or `"topleft"`, `"top"`, `"topright"`, `"left"`,
  `"center"`, `"right"`, `"bottomleft"`, `"bottom"`,
  `"bottomright"`), and `legend_adjust=` to nudge it sideways.
- **`X(x, by=, facet=, data=, form='histogram', ...)`** —
  Distribution of one numeric variable: histogram, density, or
  violin/box/scatter, optionally grouped (`by=`).
- **`XY(x, y=, by=, facet=, data=, form='scatter', fit=, ts_unit=, ...)`**
  — Relationship between two continuous variables: scatter (with
  fit lines via `fit=`, `fit_new=` to predict the fitted y at new
  x values, ellipses, and outlier ID) or time series (with
  aggregation and forecasting). `x=".Index"` makes a run chart
  (the row number, 1..n); `show_runs=True` adds the runs test
  (connected points, a median center line, the run analysis).
  `x` or `y="row_names"` plots against the data-frame row names as
  a Cleveland dot-plot axis. A categorical variable paired with
  `stat=` (e.g. `stat='mean'`) gives a Cleveland dot plot of the
  aggregate, ordered by `sort=`.
- **`Flows(value, stage1, stage2, stage3=None, data=, ...)`** —
  Sankey flow diagram of `value` across two or three stages.

## Models

Inferential analyses. Each prints the full analysis and returns a
results object with numeric fields and any figures in `.plots`.

- **`Regression(my_formula, data=, ...)`** — Least-squares
  regression from a formula string (`"Y ~ X1 + X2"`): estimates,
  fit, ANOVA, collinearity, residuals and influence, prediction
  intervals, k-fold validation, moderation, and graphics.
- **`Logit(my_formula, data=, ...)`** — Logistic regression:
  estimates and odds ratios, fit, collinearity, residuals and
  influence, classification with confusion matrices, and the
  fitted-sigmoid plot.
- **`ANOVA(my_formula, data=, ...)`** — Analysis of variance:
  one-way (`Y ~ X`), two-way between groups (`Y ~ X1 * X2`), or
  randomized blocks (`Y ~ X + Block`).
- **`ttest(x=, y=, data=, paired=, mu=, ...)`** — t-test of a
  mean (one group vs `mu`) or a mean difference (two independent
  groups or paired); also from summary statistics.
- **`Prop_test(variable=, success=, by=, data=, ...)`** — Test of
  one or more proportions, goodness-of-fit, or cross-tab
  independence, from data columns or summary counts.

## Correlation structure / factor analysis

Compute correlations, then analyze the correlation matrix `R` (a
`DataFrame` or array; raw data is correlated first).

- **`Correlation(x=, y=, data=, method='pearson', ...)`** —
  Correlation of two variables (with a significance test) or the
  correlation matrix of several.
- **`corEFA(R, n_factors, rotate='promax', ...)`** — Exploratory
  factor analysis (ML), with rotation and sorted loadings.
- **`corCFA(R, model=, factors=, ...)`** — Confirmatory factor
  analysis (MIMM), with residuals and item correlations.
- **`corScree(R, main=)`** — Scree analysis: eigenvalues and
  their successive differences, printed and plotted.
- **`corReorder(R, order='hclust', ...)`** — Reorder the
  variables of `R` (hierarchical clustering, manual, or chained).
- **`corProp(R, ...)`** — Item proportionality coefficients: the
  profile similarity of each item pair across the other items.
- **`corReflect(R, vars, ...)`** — Reflect (negate) the
  correlations of the named variables within `R`.
- **`corRead(file=, var_names=, sep=)`** — Read a square
  correlation matrix from a whitespace-delimited text file.
- **`corPrint(R, min_value=0)`** — Print `R` in the compact lessR
  style, blanking coefficients with `|r| < min_value`.

## Simulations

Teaching simulations. Each draws from a population with a numpy
RNG (`seed=` for reproducibility), prints a summary, and returns
a results object with the plotly figure(s) in `.plots`.

- **`simCLT(ns, n, p1=0, p2=1, dist='normal', ...)`** — Central
  Limit Theorem: sampling distribution of the mean from a normal,
  uniform, lognormal, or U-shaped "antinormal" population.
- **`simMeans(ns, n, mu=0, sigma=1, ...)`** — Plot each sample
  mean against its index, centered on `mu`.
- **`simFlips(n, prob=0.5, ...)`** — Law of large numbers: the
  running proportion of heads over `n` coin flips.
- **`simCImean(ns, n, mu=0, sigma=1, cl=0.95, ...)`** — Coverage
  of confidence intervals: `ns` intervals about `mu`, colored by
  whether they capture it, with the miss rate.

## Probability

Normal- and t-curve probability graphics. Each returns a results
object with the plotly figure in `.plots`.

- **`prob_norm(lo=, hi=, mu=0, sigma=1, ...)`** — Normal-curve
  interval probability: shade `N(mu, sigma)` from `lo` to `hi` and
  return `P(lo < Y < hi)`. An open bound (`None`) extends to the
  tail.
- **`prob_znorm(mu=0, sigma=1, ...)`** — Empirical-rule graphic:
  the central ±1, ±2, and ±3 sigma regions of a normal curve as
  three nested translucent bands (a display, no probability
  returned).
- **`prob_tcut(df, alpha=0.05, ...)`** — Two-tailed t cutoffs at
  `alpha`: shade the central `1 - alpha` region and the two tails
  of the `t(df)` curve, overlay a standard normal, and return the
  upper t critical value.

## Data

Load the bundled example datasets.

- **`read_data(name)`** — Load a bundled example dataset as a
  `DataFrame`.
- **`datasets()`** — Names of the bundled example datasets.

## Utility

Operate on a `DataFrame` — reshape, aggregate, sort, rename, or
inspect. Each returns a new `DataFrame` (or, for `details`, a
results object) rather than mutating the input.

- **`details(data, ...)`** — Diagnostic report on a data frame:
  dimensions, per-variable type / missing / unique counts with
  first and last values, ID-column and numeric-category notes, and
  a missing-data analysis; returns the summary as a `DataFrame`.
- **`VariableLabels(data, labels=, units=)`** — Get or set the
  variable labels and units of a data frame (stored in
  `data.attrs`, read by `details`), from a dict or a CSV/Excel
  file.
- **`pivot(data, compute, variable=, by=, ...)`** — Aggregate a
  numeric variable over groups, or tabulate frequencies.
- **`reshape_long(data, transform=, ...)`** — Reshape wide data
  to long.
- **`reshape_wide(data, widen, response, ID, ...)`** — Reshape
  long data to wide.
- **`order_by(data, by, direction=, ...)`** — Sort rows by one or
  more columns (each ascending or descending), or by the row names
  or at random, returning the sorted copy.
- **`rename(data, old, new)`** — Rename columns, returning a
  renamed copy.

## Color

- **`getColors(pal=, end_pal=, n=, ...)`** — Generate a color
  palette (HCL qualitative/sequential/divergent, viridis family,
  or fixed lists) as a list of hex colors.
- **`showColors(color=, n_col=6)`** — Display the named colors as
  a labeled grid of swatches.

## Dates

- **`date_infer(x, quiet=True)`** — Infer the format of a column
  of date strings and return a datetime `Series`.
- **`format_date_labels(dates, ts_unit)`** — Format dates as axis
  labels for a time unit (years, quarters, months, weeks, days).

## Options

- **`get_option(name, default=None)`** — Read a lessPy style /
  behavior option.
- **`set_option(name, value)`** — Set a lessPy option.

## Lower-level plot renderers

These back `Chart`, `X`, and `XY`; most users call the analytic
functions above rather than these directly. Each returns a plotly
figure.

- **`bc_plotly`** — bar chart · **`pie_plotly`** — pie / sunburst
  · **`hier_plotly`** — sunburst / treemap / icicle
- **`bubble_plotly`** — bubble plot · **`radar_plotly`** — radar
  · **`dot_plotly`** — dot / Cleveland plot
- **`hs_plotly`** — histogram · **`dn_plotly`** — density ·
  **`freq_poly_plotly`** — frequency polygon
- **`vbs_plotly`** — violin / box / scatter composite ·
  **`plt_plotly`** — scatterplot / time-series renderer
