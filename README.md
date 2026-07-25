# lessPython

[![PyPI](https://img.shields.io/pypi/v/lessPython.svg)](https://pypi.org/project/lessPython/)
[![Python versions](https://img.shields.io/pypi/pyversions/lessPython.svg)](https://pypi.org/project/lessPython/)
[![License](https://img.shields.io/pypi/l/lessPython.svg)](https://pypi.org/project/lessPython/)

A Python port of [lessR](https://cran.r-project.org/package=lessR):
analytic-view visualizations, statistical models, correlation and
factor analysis, teaching simulations, and data utilities — the
graphics rendered with plotly.

> **New to Python?** See the step-by-step
> **[Getting Started guide](docs/getting-started.md)** — Google Colab
> or Miniconda, from zero to your first chart. Written for students.

Install as `lessPython`, import as `lessPy`:

```
pip install lessPython
```

### With Miniconda / conda

lessPython is published on PyPI, not conda-forge, so inside a conda
environment install it with `pip` (conda and pip coexist here — every
dependency ships as a standard wheel):

```
conda create -n lesspy python=3.12
conda activate lesspy
pip install lessPython
```

Reuse the environment in later sessions with just `conda activate
lesspy`. To confirm the install:

```
python -c "import lessPy as lp; print(lp.read_data('Employee').shape)"
```

```python
import lessPy as lp

d = lp.read_data("Employee")
lp.Chart("JobSat", data=d)                       # bar chart
lp.Chart("JobSat", by="Gender", data=d, form="pie")
lp.Regression("Salary ~ Years + Pre", data=d)
lp.simCLT(ns=10000, n=30, dist="uniform")
```

Variables are passed as **strings** naming columns of a pandas
`DataFrame` — Python has no equivalent of R's non-standard
evaluation, so `Chart("JobSat", data=d)`, not `Chart(JobSat)`.

## The API

The public API is **flat**: every function is reached from the
top-level package, either as `lp.<name>` or via
`from lessPy import <name>`. There are no sub-packages to import.

See **[docs/reference.md](docs/reference.md)** for the full list
of functions, grouped by category for navigation:

- **Plots** — `Chart`, `X`, `XY`, `Flows`
- **Models** — `Regression`, `Logit`, `ANOVA`, `ttest`,
  `Correlation`, `Prop_test`
- **Correlation / factor analysis** — `corEFA`, `corCFA`,
  `corScree`, `corReorder`, `corProp`, `corReflect`, `corRead`,
  `corPrint`
- **Simulations** — `simCLT`, `simMeans`, `simFlips`, `simCImean`
- **Data** — `read_data`, `datasets`, `reshape_long`,
  `reshape_wide`, `pivot`, `rename`
- **Color** — `getColors`, `showColors`
- **Dates** — `date_infer`, `format_date_labels`

The inferential functions print their analysis as in lessR and
also return a results object whose numeric fields and plotly
figures (in `.plots`) are available for programmatic use.

## Development install

```
conda activate lessPy
pip install -e ".[dev]"
pytest
python demo/demo_chart.py
```
