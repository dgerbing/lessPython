# datasets.py — the bundled example datasets, the Python analog
# of lessR's data/*.rda files loaded via Read(name,
# format="lessR"). The .rda frames were exported to CSV (no R
# factors exist in them, so CSV round-trips faithfully); read_data
# loads one into a pandas DataFrame.

from importlib import resources

import pandas as pd

# datasets whose first CSV column is a meaningful row label (the
# R row names) rather than data: used as the DataFrame index
_INDEXED = {"Employee", "Cars93", "WeightLoss", "Employee_lbl",
            "Mach4_lbl"}
# datasets with a date column to parse
_DATE_COL = {"StockPrice": "Month"}


def _data_dir():
    return resources.files(__package__).joinpath("data")


def datasets():
    """Sorted names of the bundled example datasets, each usable
    with read_data(). R analog: the lessR data/*.rda files."""
    return sorted(p.name[:-4] for p in _data_dir().iterdir()
                  if p.name.endswith(".csv"))


def read_data(name):
    """Load a bundled example dataset as a pandas DataFrame, the
    analog of lessR's Read("<name>", format="lessR"). The
    _lbl datasets (Employee_lbl, Mach4_lbl) are the variable-label
    tables. See datasets() for the available names."""
    res = _data_dir().joinpath(f"{name}.csv")
    if not res.is_file():
        raise ValueError(
            f"no bundled dataset '{name}'. Available: "
            f"{', '.join(datasets())}")
    idx = 0 if name in _INDEXED else None
    parse = [_DATE_COL[name]] if name in _DATE_COL else False
    with resources.as_file(res) as path:
        df = pd.read_csv(path, index_col=idx, parse_dates=parse)
    if idx == 0:
        df.index.name = None
    return df
