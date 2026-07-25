# Getting Started with lessPython

A step-by-step guide to installing lessPython and making your first
chart. Written for students and anyone new to Python — no prior setup
assumed.

You install the package as **`lessPython`** but import it as
**`lessPy`**:

```python
pip install lessPython      # the name on the package index
import lessPy as lp          # the name you use in code
```

## Two ways to use lessPython

- **[Option 1 — Google Colab](#option-1--google-colab):** nothing to
  install on your computer — it runs notebooks in your web browser.
- **[Option 2 — Miniconda](#option-2--miniconda-your-own-computer):**
  installs on your own computer; works offline and keeps your work
  local.

Both give you the same lessPython. Pick one.

---

## Option 1 — Google Colab

### 1. Open a new notebook

Go to **https://colab.research.google.com** and choose
**File → New notebook**.

### 2. Install and import lessPython

In the first cell, put both lines below and press **Shift+Enter**.
Together they give you access to lessPy: the first downloads and
installs the package (the leading `!` runs it as an install command),
the second makes it available in your code as `lp`.

```python
!pip install lessPython      # install the package
import lessPy as lp          # make it available as lp
```

It takes a few seconds. If Colab shows a **"Restart runtime"** button
when it finishes, click it once and then re-run this cell.

### 3. Read the data and look at it

**Always look at your data before you analyze it.** In the next cell,
read the dataset, then check its size and first rows:

```python
d = lp.read_data("Employee")   # a bundled example dataset
print(d.shape)                 # (rows, columns)
d.head()                       # the first few rows
```

You should see `(37, 8)` — 37 rows and 8 columns — and a table of the
first rows. Confirming the shape and eyeballing the values catches a
bad file or a wrong column before it derails your analysis.

### 4. Make your first chart

Now that the data checks out, in the next cell:

```python
lp.Chart("Dept", data=d)       # bar chart of counts by department
```

A bar chart appears below the cell. (If it doesn't, assign and show
it: `fig = lp.Chart("Dept", data=d)` then `fig.show()`.)

> **Note:** you must re-run that first cell each time you open a
> *fresh* Colab session — Colab resets its environment between
> sessions. Your notebook and its code are saved to Google Drive; the
> installed package is not.

Skip ahead to [Your first analysis](#your-first-analysis) for more to
try.

---

## Option 2 — Miniconda (your own computer)

This installs lessPython locally so it works offline. You do steps 1–3
**once**; after that, everyday use is two commands.

### 1. Install Miniconda

Miniconda gives you Python plus `conda`, a tool for managing isolated
environments. Download and run the installer for your system:

- **https://www.anaconda.com/download/success** → "Miniconda Installers"
- macOS: the `.pkg` installer. Windows: the `.exe` installer.

Accept the defaults. When it finishes, **open a new terminal window**
(macOS: Terminal; Windows: "Anaconda Prompt"). Check it worked:

```
conda --version
```

### 2. Create an environment and install the packages

An *environment* is a self-contained sandbox so this course's packages
don't collide with anything else on your computer. Create one named
`lesspy`, then install lessPython **and** Jupyter Notebook into it —
keeping them together is what makes charts appear inline:

```
conda create -n lesspy python=3.12
conda activate lesspy
pip install lessPython notebook
```

Answer `y` when prompted. That's the whole setup.

### 3. Start Jupyter, read the data, and check it

Every time you want to work, open a terminal and run:

```
conda activate lesspy
jupyter notebook
```

Jupyter opens in your browser. Create a new notebook
(**New → Python 3**). **Always look at your data before you analyze
it** — in the first cell, import lessPython, read the dataset, and
check its size and first rows:

```python
import lessPy as lp

d = lp.read_data("Employee")   # a bundled example dataset
print(d.shape)                 # (rows, columns)
d.head()                       # the first few rows
```

Press **Shift+Enter**. You should see `(37, 8)` — 37 rows and 8
columns — and a table of the first rows.

### 4. Make your first chart

In the next cell:

```python
lp.Chart("Dept", data=d)       # bar chart of counts by department
```

Press **Shift+Enter** — a bar chart appears below the cell.

> The `conda activate lesspy` step matters **every** session — it's
> what puts lessPython on the path. If `import lessPy` fails with
> "No module named lessPy", you forgot to activate the environment.

---

## Your first analysis

These work the same in both Colab and Miniconda. A few more to try:

```python
lp.Chart("JobSat", data=d)                        # another bar chart
lp.Chart("Dept", by="Gender", data=d)             # grouped by gender
lp.Regression("Salary ~ Years + Pre", data=d)     # a regression
```

**Variable names are passed as strings** naming columns of the data
frame — `lp.Chart("Dept", data=d)`, not `lp.Chart(Dept)`. Unlike R's
lessR, Python has no way to use a bare column name.

### Using your own data

`read_data` loads the bundled example datasets by name. For your own
file, read it with pandas and pass the data frame the same way:

```python
import pandas as pd
d = pd.read_csv("my_data.csv")
lp.Chart("MyColumn", data=d)
```

Use the pandas function that matches your file type — read and write
come as a matched pair for each format:

| File type | Read | Write |
|-----------|------|-------|
| comma- or tab-separated text (`.csv`, `.tsv`, `.txt`) | `pd.read_csv` | `d.to_csv` |
| Excel workbook (`.xlsx`, `.xls`) | `pd.read_excel` | `d.to_excel` |
| Parquet columnar file (`.parquet`) | `pd.read_parquet` | `d.to_parquet` |
| Feather / Arrow file (`.feather`) | `pd.read_feather` | `d.to_feather` |

Note the asymmetry: you **read** with a pandas function that takes the
path — `pd.read_csv("my_data.csv")` — but **write** with a method on
the data frame itself — `d.to_csv("my_data.csv", index=False)`.

pandas also reads data saved from other statistical systems — for
example `pd.read_spss` (SPSS `.sav`), `pd.read_stata` (Stata `.dta`),
and `pd.read_sas` (SAS). Of these it can also *write* Stata
(`d.to_stata`); SAS and SPSS are read-only in pandas. Some formats need
an extra package installed (e.g. `openpyxl` for Excel, `pyarrow` for
Parquet/Feather, `pyreadstat` for SPSS); pandas tells you which if it's
missing.

(In Colab, upload your file first with the folder icon in the left
sidebar, or mount Google Drive.)

---

## Troubleshooting

**A chart cell runs but no chart appears (blank output).**

- *Colab:* assign and show it — `fig = lp.Chart("Dept", data=d)`
  then `fig.show()`.
- *Miniconda:* this happens if Jupyter was started from a *different*
  environment than lessPython. Make this the first cell, then restart
  the kernel (**Kernel → Restart**) and run again:

  ```python
  import plotly.io as pio
  pio.renderers.default = "iframe"
  ```

**`import lessPy` fails with "No module named lessPy".**
- *Colab:* re-run the `!pip install lessPython` cell (a new session
  starts empty).
- *Miniconda:* you didn't activate the environment — run
  `conda activate lesspy` first, then start Jupyter from that same
  terminal.

**`conda: command not found` (Miniconda).**
Close and reopen your terminal after installing Miniconda. On Windows,
use the "Anaconda Prompt" rather than the default command prompt.

**Verify a Miniconda install from the terminal (no notebook needed):**

```
conda activate lesspy
python -c "import lessPy as lp; print('lessPython', lp.read_data('Employee').shape)"
```

Expected output: `lessPython (37, 8)`.

---

## Quick reference

**Google Colab** — in a notebook cell:

```python
!pip install lessPython       # run once per session
import lessPy as lp
```

**Miniconda** — in a terminal:

| Task | Command |
|------|---------|
| Set up (once) | `conda create -n lesspy python=3.12` then `conda activate lesspy` then `pip install lessPython notebook` |
| Work (every session) | `conda activate lesspy` then `jupyter notebook` |
| In a notebook | `import lessPy as lp` |
| Update lessPython later | `conda activate lesspy` then `pip install -U lessPython` |
