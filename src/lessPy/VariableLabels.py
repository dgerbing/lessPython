# VariableLabels.py — analog of VariableLabels.R.
#
# VariableLabels(): get or set the variable labels (and optional
# units) of a data frame. lessPy stores this metadata in the pandas
# attrs of the frame -- data.attrs["variable_labels"] and
# data.attrs["variable_units"], each a {column: text} mapping --
# which is where details() reads it.
#
# R's VariableLabels relies on global objects (the label table `l`,
# the data frame `d`) and non-standard evaluation of a bare variable
# name; those modes do not translate. This port keeps the name and
# purpose but takes the frame explicitly:
#   VariableLabels(data)                 -> return / show the labels
#   VariableLabels(data, {"Salary": ...}) -> set from a dict
#   VariableLabels(data, "labels.csv")   -> set from a file
# Setting updates data.attrs in place and also returns the resulting
# labels as a DataFrame (the single return shape for both modes).

import pandas as pd

from .utils import get_option


def VariableLabels(data, labels=None, units=None, quiet=None):
    """Get or set a data frame's variable labels. With only data,
    return (and print) the current labels as a DataFrame. With
    labels as a {column: label} dict or a CSV/Excel file path
    (column, label, optional unit per row, no header), attach them
    to data.attrs and return the resulting labels. units is an
    optional {column: unit} dict when setting from a dict.
    R analog: VariableLabels()"""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if quiet is None:
        quiet = get_option("quiet", False)

    if labels is None:
        return _get(data, quiet)
    return _set(data, labels, units, quiet)


def _get(data, quiet):
    lab = data.attrs.get("variable_labels")
    if not lab:
        if not quiet:
            print("\nNo variable labels present\n")
        return _frame({}, {})
    frame = _frame(lab, data.attrs.get("variable_units", {}))
    if not quiet:
        _show(frame)
    return frame


def _set(data, labels, units, quiet):
    if isinstance(labels, str):
        labels, file_units = _read_file(labels)
        if file_units:
            units = file_units
    if not isinstance(labels, dict):
        raise TypeError(
            "labels must be a {column: label} dict, a CSV/Excel "
            "file path, or None to display the current labels")

    unknown = [k for k in labels if k not in data.columns]
    if unknown:
        raise ValueError(
            f"not column(s) of data: {unknown}. Columns: "
            f"{', '.join(map(str, data.columns))}")

    lab = dict(data.attrs.get("variable_labels", {}))
    lab.update({str(k): str(v) for k, v in labels.items()})
    data.attrs["variable_labels"] = lab
    if units:
        unt = dict(data.attrs.get("variable_units", {}))
        unt.update({str(k): str(v) for k, v in units.items()})
        data.attrs["variable_units"] = unt

    frame = _frame(lab, data.attrs.get("variable_units", {}))
    if not quiet:
        _show(frame)
    return frame


def _frame(labels, units):
    """Build the labels DataFrame: index of column names, a 'label'
    column, and a 'unit' column when any units are present."""
    names = list(labels.keys())
    data = {"label": [labels[n] for n in names]}
    if units:
        data["unit"] = [units.get(n, "") for n in names]
    return pd.DataFrame(data, index=pd.Index(names, name="variable"))


def _show(frame):
    print()
    for name, row in frame.iterrows():
        unit = f"  ({row['unit']})" if "unit" in frame.columns \
            and row["unit"] else ""
        print(f"{name}: {row['label']}{unit}")
    print()


def _read_file(path):
    """Read labels from a headerless file: column name, label, and
    an optional unit per row. Returns (labels, units) dicts."""
    if path.endswith(".xlsx"):
        raw = pd.read_excel(path, header=None, index_col=0)
    else:
        raw = pd.read_csv(path, header=None, index_col=0)
    names = [str(k) for k in raw.index]
    labels = {n: str(raw.iloc[i, 0]) for i, n in enumerate(names)}
    units = {}
    if raw.shape[1] >= 2:
        for i, n in enumerate(names):
            u = raw.iloc[i, 1]
            units[n] = "" if pd.isna(u) else str(u)
        if not any(units.values()):
            units = {}
    return labels, units
