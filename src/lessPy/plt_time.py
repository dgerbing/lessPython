# plt_time.py — analog of plt.time.R
#
# ts_unit/ts_agg for XY(): aggregate a date-x time series to a
# coarser time unit, sum or mean per period. Ports .plt.time():
# infer the existing unit from the date spacing, refuse a
# requested unit finer than the data, drop the trailing
# incomplete period, aggregate per period — per by= level when
# present — with the date placed at the period start for
# months/quarters/years (weeks and days keep the last observed
# date of the period, as xts::period.apply indexes), and note
# gaps in the final series. pandas periods replace the xts/zoo
# machinery; the unit inference is the median-gap rule (the
# calendar-structure refinements of .plt.time() for series with
# missing periods are not ported).

import numpy as np
import pandas as pd

from .utils import category_order

_UNIT_ORDER = ("days", "weeks", "months", "quarters", "years")
_PERIOD_FREQ = {"weeks": "W-SUN", "months": "M",
                "quarters": "Q", "years": "Y"}


def _infer_unit(dates):
    """Existing time unit from the median day gap, "unknown"
    when irregular. R analog: the tu_exist logic of .plt.time()"""
    d = np.sort(pd.unique(pd.Series(dates).dropna()))
    if len(d) <= 1:
        return "days"
    gaps = np.diff(d).astype("timedelta64[D]").astype(int)
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return "days"
    med = float(np.median(gaps))
    if med <= 1.5:
        return "days"
    if 6 <= med <= 8:                  # weekly, unless biweekly
        return ("unknown" if (gaps == 14).mean() >= 0.40
                else "weeks")
    if 28 <= med <= 31:
        return "months"
    if 90 <= med <= 92:
        return "quarters"
    if 365 <= med <= 366:
        return "years"
    return "unknown"


def _agg_one(dser, yser, ts_unit, aggfun):
    """Aggregate one series: truncate the trailing incomplete
    period, then one value per period. R analogs: ts_truncate()
    and the xts::period.apply block of .plt.time()"""
    if ts_unit == "days":              # collapse duplicate dates
        per = dser
    else:
        per = dser.dt.to_period(_PERIOD_FREQ[ts_unit])
        # trailing partial period: last date short of period end
        last_per = per.iloc[int(np.argmax(dser.to_numpy()))]
        if dser.max() < last_per.end_time.normalize():
            keep = (per != last_per).to_numpy()
            dser, yser, per = dser[keep], yser[keep], per[keep]
    if len(dser) == 0:
        return (np.array([], dtype="datetime64[ns]"),
                np.array([]))
    y_out = yser.groupby(per.to_numpy()).agg(
        lambda s: aggfun(s.to_numpy()))  # NaN propagates, as R
    if ts_unit in ("months", "quarters", "years"):
        x_out = y_out.index.to_timestamp(how="start")
    else:                              # last observed date
        x_out = dser.groupby(per.to_numpy()).max().to_numpy()
    return np.asarray(x_out), y_out.to_numpy()


def plt_time(x_ser, y_ser, by_ser, ts_unit, ts_agg, quiet=True):
    """Aggregate the time series to ts_unit by ts_agg. Returns
    (x_ser, y_ser, by_ser, ts_unit). R analog: .plt.time()"""
    hold7 = ts_unit == "days7"
    if hold7:
        ts_unit = "days"
    if ts_unit not in _UNIT_ORDER:
        raise ValueError(
            "ts_unit must be one of days, days7, weeks, months, "
            "quarters, years")

    unit_exist = _infer_unit(
        x_ser.to_numpy() if by_ser is None
        else x_ser[by_ser == category_order(by_ser)[0]]
        .to_numpy())
    if (unit_exist != "unknown"
            and _UNIT_ORDER.index(ts_unit)
            < _UNIT_ORDER.index(unit_exist)):
        raise ValueError(
            f"Resolution of data is {unit_exist}; requested "
            f"{ts_unit} data is not available")

    aggfun = np.sum if ts_agg == "sum" else np.mean
    if by_ser is None:
        xo, yo = _agg_one(x_ser, y_ser, ts_unit, aggfun)
        x_out = pd.Series(xo, name=x_ser.name)
        y_out = pd.Series(yo, name=y_ser.name)
        by_out = None
    else:
        levels = category_order(by_ser)
        xs, ys, bs = [], [], []
        for lvl in levels:
            m = (by_ser == lvl).to_numpy()
            xl, yl = _agg_one(x_ser[m], y_ser[m], ts_unit,
                              aggfun)
            xs.append(xl)
            ys.append(yl)
            bs.extend([lvl] * len(xl))
        x_out = pd.Series(np.concatenate(xs), name=x_ser.name)
        y_out = pd.Series(np.concatenate(ys), name=y_ser.name)
        by_out = pd.Series(
            pd.Categorical(bs, categories=levels),
            name=by_ser.name)

    # gaps in the aggregated series (first by level, as R)
    d = np.sort(x_out[by_out == levels[0]].to_numpy()
                if by_out is not None else x_out.to_numpy())
    gaps = False
    if len(d) > 1:
        dd = pd.DatetimeIndex(d)
        if ts_unit == "days":
            gaps = (np.diff(d).astype("timedelta64[D]")
                    .astype(int) > 1).any()
        elif ts_unit == "weeks":
            gaps = (np.diff(d).astype("timedelta64[D]")
                    .astype(int) > 7).any()
        elif ts_unit == "months":
            gaps = (np.diff(dd.year * 12 + dd.month) > 1).any()
        elif ts_unit == "quarters":
            gaps = (np.diff(dd.year * 4 + dd.quarter) > 1).any()
        elif ts_unit == "years":
            gaps = (np.diff(dd.year) > 1).any()
    if gaps and not quiet:
        print("There are gaps in the dates, so that there are "
              "not regular\nintervals between all the dates.")

    return (x_out, y_out, by_out,
            "days7" if hold7 else ts_unit)
