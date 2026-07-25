# plt_forecast.py — analog of plt.forecast.R.
#
# ts_source="fable" (the default, as in R) maps fable's models
# to statsmodels — the plan's settled decision #3 (ETS replaces
# fable). ts_method="es" fits ETSModel over the component grid
# fable::ETS searches (error A/M; trend N/A/Ad; seasons N/A/M;
# Md also accepted when specified), selecting by AICc as fable
# does, or fixing components the caller gives via
# ts_error/ts_trend/ts_seasons. ts_method="lm" is the TSLM
# analog: OLS on trend and calendar-season dummies, with the
# lessR fable-branch prediction intervals (qnorm quantiles on
# the forecast sigma). Results track R closely but not exactly:
# initial-state estimation differs between statsmodels and
# fable.
#
# ts_source="classic":
# ts_method="es" ports stats::HoltWinters directly — the same
# initialization (decompose of the first two periods, linear fit
# for level/trend start), the same one-step filter recursion, and
# the same L-BFGS-B minimization of SSE over only the free
# smoothing parameters (start 0.3/0.1/0.1, bounds [0, 1]) — so
# results track R closely. Prediction intervals port
# predict.HoltWinters: qnorm quantiles on var(residuals) scaled
# by the psi weights.
#
# ts_method="lm" ports the stl + regression path. The stl
# periodic decomposition comes from statsmodels STL (a port of
# the same Fortran algorithm as R's stl), with R's
# s.window="periodic" mapped to seasonal=10n+1, seasonal_deg=0.
# statsmodels is imported lazily, only on this path — the
# Python analog of an R Suggests dependency.

import numpy as np
import pandas as pd
from scipy import optimize as spo
from scipy import stats as sps

from .utils import fmt

_UNITS = ("days", "days7", "weeks", "months", "quarters", "years")
_FREQ = {"days": 365, "days7": 7, "weeks": 52, "months": 12,
         "quarters": 4, "years": 1}
_OFFSET = {"days": pd.DateOffset(days=1),
           "days7": pd.DateOffset(days=1),
           "weeks": pd.DateOffset(weeks=1),
           "months": pd.DateOffset(months=1),
           "quarters": pd.DateOffset(months=3),
           "years": pd.DateOffset(years=1)}


def _ts_unit(dates, ts_unit):
    """Time unit and seasonal frequency from the date spacing.
    R analog: .tsMake() interval logic; frequencies match
    tsMake.R (days 365, days7 7, weeks 52, months 12,
    quarters 4, years 1)."""
    if ts_unit is not None and ts_unit not in _UNITS:
        raise ValueError(f"ts_unit must be one of {_UNITS}")
    dd = np.diff(dates).astype("timedelta64[D]").astype(int)
    if len(dd) == 0 or (dd <= 0).any():
        raise ValueError("Cannot forecast without a consistent "
                         "time unit: need increasing dates")
    med = float(np.median(dd))
    if med <= 1.5:
        unit = "days"
    elif 6 <= med <= 8:
        unit = "weeks"
    elif 27 <= med <= 32:
        unit = "months"
    elif 88 <= med <= 93:
        unit = "quarters"
    elif med >= 350:
        unit = "years"
    else:
        raise ValueError("Cannot forecast without a consistent "
                         f"time unit: median spacing {med} days")
    if ts_unit is None:
        ts_unit = unit
    elif ts_unit == "days7" and unit == "days":
        pass                           # weekly cycle in daily data
    elif ts_unit != unit:
        raise ValueError(f'dates are spaced as "{unit}" but '
                         f'ts_unit="{ts_unit}"')
    return ts_unit, _FREQ[ts_unit]


def _decompose(x, f, mult):
    """Classical decomposition for the Holt-Winters start values.
    R analog: stats::decompose() — centered moving-average trend,
    seasonal figure as centered cycle-position means."""
    n = len(x)
    if f % 2 == 0:
        filt = np.r_[0.5, np.ones(f - 1), 0.5] / f
    else:
        filt = np.ones(f) / f
    half = len(filt) // 2
    trend = np.full(n, np.nan)
    conv = np.convolve(x, filt[::-1], mode="valid")
    trend[half:half + len(conv)] = conv
    season = x / trend if mult else x - trend
    figure = np.empty(f)
    for i in range(f):
        figure[i] = np.nanmean(season[i::f])
    figure = figure / figure.mean() if mult \
        else figure - figure.mean()
    return trend, figure


def _hw_filter(x, f, alpha, beta, gamma, do_trend, do_seasons,
               mult, l0, b0, s0, start):
    """One-step Holt-Winters filter from observation `start`
    (0-based). R analog: C_HoltWinters (src/library/stats)."""
    n = len(x)
    a, b = l0, b0 if do_trend else 0.0
    s = list(s0) if do_seasons else []
    xhat = np.empty(n - start)
    sse = 0.0
    for j, i in enumerate(range(start, n)):
        pred = a + b
        if do_seasons:
            pred = pred * s[j] if mult else pred + s[j]
        xhat[j] = pred
        res = x[i] - pred
        sse += res * res
        if do_seasons:
            desea = x[i] / s[j] if mult else x[i] - s[j]
            anew = alpha * desea + (1 - alpha) * (a + b)
        else:
            anew = alpha * x[i] + (1 - alpha) * (a + b)
        if do_trend:
            b = beta * (anew - a) + (1 - beta) * b
        if do_seasons:
            s.append(gamma * ((x[i] / anew) if mult
                              else (x[i] - anew))
                     + (1 - gamma) * s[j])
        a = anew
    coefs = {"a": a}
    if do_trend:
        coefs["b"] = b
    if do_seasons:
        coefs["s"] = np.array(s[-f:])
    return xhat, sse, coefs


def _holt_winters(x, f, do_trend, do_seasons, mult,
                  alpha, beta, gamma):
    """stats::HoltWinters port: R start values, then minimize
    SSE over only the free smoothing parameters, as R optim
    (L-BFGS-B, start 0.3/0.1/0.1, bounds [0, 1])."""
    if do_seasons:
        if len(x) < 2 * f:
            raise ValueError("need at least 2 periods to compute "
                             "seasonal start values")
        start = f
        trend, figure = _decompose(x[:2 * f], f, mult)
        dat = trend[np.isfinite(trend)]
        # R .lm.fit on (1..m): l.start intercept, b.start slope
        b0, l0 = np.polyfit(np.arange(1, len(dat) + 1), dat, 1)
        s0 = figure
    else:
        if do_trend:
            l0, b0, s0, start = x[1], x[1] - x[0], None, 2
        else:
            l0, b0, s0, start = x[0], None, None, 1

    free = []                          # optimized parameters
    if alpha is None:
        free.append(("alpha", 0.3))
    if do_trend and beta is None:
        free.append(("beta", 0.1))
    if do_seasons and gamma is None:
        free.append(("gamma", 0.1))

    def run(vals):
        p = {"alpha": alpha, "beta": beta, "gamma": gamma}
        p.update(dict(zip([nm for nm, _ in free], vals)))
        return _hw_filter(x, f, p["alpha"],
                          p["beta"] if do_trend else 0.0,
                          p["gamma"] if do_seasons else 0.0,
                          do_trend, do_seasons, mult,
                          l0, b0 if do_trend else 0.0, s0, start)

    if free:
        if len(free) == 1:
            sol = spo.minimize_scalar(
                lambda v: run([v])[1], bounds=(0, 1),
                method="bounded")
            best = [sol.x]
        else:
            sol = spo.minimize(
                lambda v: run(v)[1],
                x0=[st for _, st in free], method="L-BFGS-B",
                bounds=[(0, 1)] * len(free))
            best = list(sol.x)
    else:
        best = []
    params = {"alpha": alpha, "beta": beta, "gamma": gamma}
    params.update(dict(zip([nm for nm, _ in free], best)))
    if not do_trend:
        params["beta"] = False
    if not do_seasons:
        params["gamma"] = False
    xhat, sse, coefs = run(best)
    return params, coefs, xhat, sse, start


def _hw_predict(x, xhat, start, f, params, coefs, mult,
                do_trend, do_seasons, h, level):
    """Forecasts and prediction intervals.
    R analog: predict.HoltWinters — psi weights on
    var(residuals), qnorm quantiles."""
    steps = np.arange(1, h + 1)
    y_hat = np.full(h, coefs["a"])
    if do_trend:
        y_hat = y_hat + steps * coefs["b"]
    if do_seasons:
        s_rep = np.resize(coefs["s"], h)
        y_hat = y_hat * s_rep if mult else y_hat + s_rep

    resid = x[start:] - xhat
    s2 = float(np.var(resid, ddof=1))
    alpha = params["alpha"]
    beta_v = params["beta"] if do_trend else 0.0
    gamma_v = params["gamma"] if do_seasons else 0.0

    def psi(j):
        return (alpha * (1 + j * beta_v)
                + (j % f == 0) * gamma_v * (1 - alpha))

    var_h = np.empty(h)
    if not mult:
        for hh in steps:
            js = np.arange(1, hh)
            var_h[hh - 1] = s2 * (1 + np.sum(psi(js) ** 2))
    else:
        # R indexes (a, b, s1..sf); (rel-j) mod f == 0 falls on b,
        # an R quirk preserved for parity
        cvec = np.r_[coefs["a"],
                     coefs["b"] if do_trend else np.nan,
                     coefs["s"]]
        for hh in steps:
            rel = 1 + (hh - 1) % f
            tot = 0.0
            for j in range(hh):
                tot += (psi(j) * cvec[1 + rel]
                        / cvec[1 + (rel - j) % f]) ** 2
            var_h[hh - 1] = s2 * tot
    zq = sps.norm.ppf((1 + level) / 2)
    half = zq * np.sqrt(var_h)
    return y_hat, y_hat - half, y_hat + half


def _stl_periodic(y, f):
    """Seasonal component, R stl(y, s.window="periodic"):
    statsmodels STL with the periodic mapping s.window = 10n+1,
    s.degree = 0. Lazy import — the Python analog of Suggests."""
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        raise ImportError(
            'ts_method="lm" with seasonality needs statsmodels: '
            "pip install statsmodels") from None
    res = STL(y, period=f, seasonal=10 * len(y) + 1,
              seasonal_deg=0, robust=False).fit()
    return np.asarray(res.seasonal), np.asarray(res.trend)


_ETS_ERR = {"A": "add", "M": "mul"}
_ETS_TREND = {"N": (None, False), "A": ("add", False),
              "M": ("mul", False), "Ad": ("add", True),
              "Md": ("mul", True)}
_ETS_SEAS = {"N": None, "A": "add", "M": "mul"}


def _ets_forecast(yv, f, ts_error, ts_trend, ts_seasons,
                  alpha, beta, gamma, h, level, digits_d):
    """fable::ETS analog on statsmodels ETSModel: fixed
    components when specified, otherwise AICc selection over the
    grid fable searches. R analog: plt.forecast.R fable es."""
    try:
        from statsmodels.tsa.exponential_smoothing.ets import (
            ETSModel)
    except ImportError:
        raise ImportError(
            'ts_source="fable" needs statsmodels: '
            "pip install statsmodels") from None

    errs = [ts_error] if ts_error else ["A", "M"]
    trds = [ts_trend] if ts_trend else ["N", "A", "Ad"]
    seas_ok = f > 1 and len(yv) >= 2 * f
    seas = ([ts_seasons] if ts_seasons
            else (["N", "A", "M"] if seas_ok else ["N"]))
    pos = bool((yv > 0).all())

    y_ser = pd.Series(np.asarray(yv, float))
    best = None
    for e in errs:
        for t in trds:
            for s in seas:
                if (e == "M" or s == "M") and not pos:
                    continue
                trend, damped = _ETS_TREND[t]
                try:
                    m = ETSModel(
                        y_ser, error=_ETS_ERR[e], trend=trend,
                        damped_trend=damped,
                        seasonal=_ETS_SEAS[s],
                        seasonal_periods=(f if _ETS_SEAS[s]
                                          else None))
                    fixes = {}
                    if alpha is not None:
                        fixes["smoothing_level"] = alpha
                    if beta is not None and trend:
                        fixes["smoothing_trend"] = beta
                    if gamma is not None and _ETS_SEAS[s]:
                        fixes["smoothing_seasonal"] = gamma
                    if fixes:
                        with m.fix_params(fixes):
                            res = m.fit(disp=False)
                    else:
                        res = m.fit(disp=False)
                except Exception:
                    continue
                if np.isfinite(res.aicc) and (
                        best is None or res.aicc < best[0]):
                    best = (res.aicc, e, t, s, res)
    if best is None:
        raise ValueError(
            "Model fitting failed. Simplify the model such as "
            'ts_trend="N" and ts_seasons="N", or use more data.')
    _, e, t, s, res = best

    n = len(yv)
    y_fit = res.fittedvalues.to_numpy()
    pred = res.get_prediction(start=n, end=n + h - 1)
    y_hat = np.asarray(pred.predicted_mean)
    # PI as lessR builds it from the fable distribution:
    # mean +/- qnorm((1+PI)/2) * sigma (plt.forecast.R ~330)
    sigma = np.sqrt(np.asarray(pred.forecast_variance))
    zq = sps.norm.ppf((1 + level) / 2)
    y_lwr = y_hat - zq * sigma
    y_upr = y_hat + zq * sigma

    mse = float(np.mean((yv - y_fit) ** 2))
    nd = max(digits_d - 1, 3)
    report = ["[ETS from statsmodels; standard reference: "
              "https://otexts.com/fpp3/]", "",
              "Estimated model\n---------------",
              f"ETS({e},{t},{s})", "",
              "Smoothing Parameters"]
    tx = " alpha: " + fmt(res.smoothing_level, nd)
    if _ETS_TREND[t][0]:
        tx += "  beta: " + fmt(res.smoothing_trend, nd)
        if _ETS_TREND[t][1]:
            tx += "  phi: " + fmt(res.damping_trend, nd)
    if _ETS_SEAS[s]:
        tx += "  gamma: " + fmt(res.smoothing_seasonal, nd)
    report += [tx, "",
               "AIC: " + fmt(res.aic, 2)
               + "  AICc: " + fmt(res.aicc, 2)
               + "  BIC: " + fmt(res.bic, 2), "",
               "Mean squared error of fit to data: "
               + fmt(mse, 3),
               "Root mean squared error (RMSE) fit: "
               + fmt(np.sqrt(mse), 3), ""]
    return y_fit, y_hat, y_lwr, y_upr, report


def _season_pos(dates, f, ts_unit):
    """Calendar season of each date for the TSLM dummies:
    month or quarter number, else position in the cycle."""
    dd = pd.DatetimeIndex(dates)
    if ts_unit == "months":
        return dd.month.to_numpy()
    if ts_unit == "quarters":
        return dd.quarter.to_numpy()
    return (np.arange(len(dd)) % f) + 1


def _tslm_forecast(x_dates, yv, f, ts_unit, ts_trend,
                   ts_seasons, x_hat, level, digits_d):
    """fable::TSLM analog: OLS on trend and calendar-season
    dummies; prediction intervals as in the lessR fable branch
    (qnorm quantiles on the forecast sigma).
    R analog: plt.forecast.R fable lm."""
    n = len(yv)
    h = len(x_hat)
    do_trend = ts_trend == "A"
    do_seasons = ts_seasons == "A"

    cols, names = [np.ones(n)], ["(Intercept)"]
    fcols = [np.ones(h)]
    if do_trend:
        cols.append(np.arange(1, n + 1, dtype=float))
        fcols.append(np.arange(n + 1, n + h + 1, dtype=float))
        names.append("trend")
    if do_seasons:
        p_now = _season_pos(x_dates, f, ts_unit)
        if ts_unit in ("months", "quarters"):
            p_new = _season_pos(x_hat, f, ts_unit)
        else:
            p_new = ((n + np.arange(h)) % f) + 1
        unit_lbl = ts_unit[:-1] if ts_unit.endswith("s") \
            else ts_unit
        for k in range(2, f + 1):
            cols.append((p_now == k).astype(float))
            fcols.append((p_new == k).astype(float))
            names.append(f"{unit_lbl}{k}")
    X = np.column_stack(cols)
    Xf = np.column_stack(fcols)

    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    y_fit = X @ coef
    resid = yv - y_fit
    p = X.shape[1]
    sse = float(resid @ resid)
    sigma2 = sse / (n - p)
    xtx_inv = np.linalg.pinv(X.T @ X)

    y_hat = Xf @ coef
    se_pred = np.sqrt(sigma2
                      * (1 + np.einsum("ij,jk,ik->i",
                                       Xf, xtx_inv, Xf)))
    zq = sps.norm.ppf((1 + level) / 2)
    y_lwr = y_hat - zq * se_pred
    y_upr = y_hat + zq * se_pred

    # coefficient table, as the lessR fable lm report
    se_b = np.sqrt(np.diag(xtx_inv) * sigma2)
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = coef / se_b
    pvals = 2 * sps.t.sf(np.abs(tvals), n - p)
    report = ["Estimated model\n---------------",
              "Model: TSLM",
              "", "Coefficients:"]
    w = max(len(nm) for nm in names)
    report.append(f"{'':{w}}  {'Estimate':>12} {'Std.Err':>12}"
                  f" {'t-value':>9} {'p-value':>8}")
    for i, nm in enumerate(names):
        report.append(
            f"{nm:{w}}  {fmt(coef[i], digits_d):>12}"
            f" {fmt(se_b[i], digits_d):>12}"
            f" {fmt(tvals[i], 3):>9}"
            f" {fmt(pvals[i], 4):>8}")
    tss = float(((yv - yv.mean()) ** 2).sum())
    if p > 1 and tss > 0:
        r2 = 1 - sse / tss
        adj = 1 - (1 - r2) * (n - 1) / (n - p)
        # a (numerically) perfect fit gives r2 == 1 exactly, so
        # 1 - r2 underflows to 0: F is infinite, as R reports
        if r2 >= 1:
            fstat, fp = np.inf, 0.0
        else:
            fstat = (r2 / (p - 1)) / ((1 - r2) / (n - p))
            fp = sps.f.sf(fstat, p - 1, n - p)
        report += ["",
                   f"Residual standard error: "
                   f"{fmt(np.sqrt(sigma2), 3)} on {n - p} "
                   "degrees of freedom",
                   f"Multiple R-squared: {fmt(r2, 4)},"
                   f"\tAdjusted R-squared: {fmt(adj, 4)}",
                   f"F-statistic: {fmt(fstat, 2)} on {p - 1} "
                   f"and {n - p} DF, p-value: {fmt(fp, 5)}"]
    else:
        report += ["", "[consider adding terms: ts_trend and "
                   "ts_seasons]"]
    mse = sse / n
    report += ["",
               "Mean squared error of fit to data: "
               + fmt(mse, 3),
               "Root mean squared error (RMSE) fit: "
               + fmt(np.sqrt(mse), 3), ""]
    return y_fit, y_hat, y_lwr, y_upr, report


def plt_forecast(x_dates, yv, x_name, y_name,
                 ts_unit=None, ts_ahead=0, ts_method="es",
                 ts_source="fable", ts_error=None,
                 ts_trend=None, ts_seasons=None, ts_alpha=None,
                 ts_beta=None, ts_gamma=None, ts_PI=0.95,
                 digits_d=None):
    """Classic-path time series forecast: fitted values, forecasts
    with prediction intervals, and the console report lines.
    R analog: .plt.forecast(), ts_source="classic" branch."""
    if digits_d is None:
        digits_d = 2
    ts_unit, f = _ts_unit(x_dates, ts_unit)

    # seasonality eligibility (plt.forecast.R lines 36-44)
    if ts_seasons is not None and ts_seasons != "N":
        if ts_unit == "years":
            print("Seasonal effects are not possible with "
                  "annual data.")
            ts_seasons = "N"
        elif len(yv) < 2 * f:
            print("\nUsually need two years worth of data to "
                  "estimate seasonality.\n")
            ts_seasons = "N"

    # forecast dates, needed by the TSLM dummies
    last = pd.Timestamp(x_dates[-1])
    off = _OFFSET[ts_unit]
    x_hat = pd.DatetimeIndex([last + k * off
                              for k in range(1, ts_ahead + 1)])

    report = []
    coef_nm = None                     # classic report tail flag

    # fable-equivalent: statsmodels ETS / TSLM analog
    if ts_source == "fable":
        if ts_method == "es":
            if ts_error is not None and ts_error not in _ETS_ERR:
                raise ValueError('ts_error: "A" or "M"')
            if ts_trend is not None and \
                    ts_trend not in _ETS_TREND:
                raise ValueError(
                    'ts_trend: "N", "A", "M", "Ad", or "Md"')
            if ts_seasons is not None and \
                    ts_seasons not in _ETS_SEAS:
                raise ValueError('ts_seasons: "N", "A", or "M"')
            y_fit, y_hat, y_lwr, y_upr, rep = _ets_forecast(
                yv, f, ts_error, ts_trend, ts_seasons,
                ts_alpha, ts_beta, ts_gamma, ts_ahead, ts_PI,
                digits_d)
        elif ts_method == "lm":
            if ts_trend is not None and \
                    ts_trend not in ("A", "N"):
                raise ValueError('Enter either "A" for additive '
                                 'trend or "N" for no trend.')
            if ts_seasons is not None and \
                    ts_seasons not in ("A", "N"):
                raise ValueError(
                    'Enter either "A" for additive seasons or '
                    '"N" for no seasons.')
            y_fit, y_hat, y_lwr, y_upr, rep = _tslm_forecast(
                x_dates, yv, f, ts_unit, ts_trend, ts_seasons,
                x_hat, ts_PI, digits_d)
        else:
            raise ValueError('ts_method must be "es" or "lm"')
        report += rep
        x_fit = x_dates

    elif ts_source != "classic":
        raise ValueError('ts_source: "fable" or "classic"')

    # es: Holt-Winters (plt.forecast.R classic es branch)
    elif ts_method == "es":
        if ts_trend is not None and ts_trend not in ("A", "N"):
            raise ValueError(
                'HoltWinters() only supports "A" for additive '
                'trend or "N" for no trend.')
        if ts_seasons is not None and \
                ts_seasons not in ("A", "M", "N"):
            raise ValueError(
                'HoltWinters() only supports "A" for additive, '
                '"M" for multiplicative, or "N" for no '
                "seasonality.")
        do_trend = ts_trend == "A"
        do_seasons = ts_seasons in ("A", "M")
        mult = ts_seasons == "M"

        params, coefs, xhat, sse, start = _holt_winters(
            yv, f, do_trend, do_seasons, mult,
            ts_alpha, ts_beta if do_trend else None,
            ts_gamma if do_seasons else None)
        x_fit = x_dates[start:]
        y_fit = xhat
        y_hat, y_lwr, y_upr = _hw_predict(
            yv, xhat, start, f, params, coefs, mult,
            do_trend, do_seasons, ts_ahead, ts_PI)

        n_param = 1 + do_trend + do_seasons
        mse = sse / (len(x_fit) - n_param)

        nd = max(digits_d - 1, 3)
        tx = " alpha: " + fmt(params["alpha"], nd)
        if do_trend:
            tx += "  beta: " + fmt(params["beta"], nd)
        if do_seasons:
            tx += "  gamma: " + fmt(params["gamma"], nd)
        report += ["Smoothing Parameters", tx, ""]
        coef_v = np.r_[coefs["a"],
                       coefs["b"] if do_trend else [],
                       coefs["s"] if do_seasons else []]
        coef_nm = (["b0"] + (["b1"] if do_trend else [])
                   + ([f"s{i+1}" for i in range(f)]
                      if do_seasons else []))

    # lm: regression on (usually) deseasonalized data
    elif ts_method == "lm":
        if ts_trend is not None and ts_trend not in ("A", "N"):
            raise ValueError('Enter either "A" for additive '
                             'trend or "N" for no trend.')
        if ts_seasons is not None and ts_seasons not in ("A", "N"):
            raise ValueError('Enter either "A" for additive '
                             'seasons or "N" for no seasons.')
        do_trend = ts_trend == "A"
        do_seasons = ts_seasons == "A"
        n = len(yv)

        if do_seasons:
            seas, stl_trend = _stl_periodic(yv, f)
            # R: trend+remainder, or remainder + mean(y)
            y_trend = yv - seas if do_trend \
                else (yv - seas - stl_trend) + yv.mean()
        else:
            seas = np.zeros(n)
            y_trend = yv if do_trend \
                else np.full(n, yv.mean())

        x_seq = np.arange(1, n + 1)
        b1, b0 = np.polyfit(x_seq, y_trend, 1)
        y_fit = (b0 + b1 * x_seq) + seas
        x_fit = x_dates

        sse = float(np.sum((yv - y_fit) ** 2))
        n_param = 2 + (f if do_seasons else 0)
        mse = sse / (n - n_param)

        # seasonal indices of the forecast (plt.forecast.R
        # lines 512-529): continue the cycle past the data
        new_seq = np.arange(n + 1, n + ts_ahead + 1)
        if do_seasons:
            start_ind = (n % f)        # 0-based next position
            new_ind = (start_ind + np.arange(ts_ahead)) % f
            new_seas = seas[:f][new_ind]
        else:
            new_seas = np.zeros(ts_ahead)
        y_hat = (b0 + b1 * new_seq) + new_seas

        x_mean = x_seq.mean()
        sxx = np.sum((x_seq - x_mean) ** 2)
        se_fore = np.sqrt(mse * (1 + 1 / n
                                 + (new_seq - x_mean) ** 2 / sxx))
        tq = sps.t.ppf((1 + ts_PI) / 2, n - n_param)
        y_lwr = y_hat - tq * se_fore
        y_upr = y_hat + tq * se_fore

        coef_v = np.r_[b0, b1, seas[:f] if do_seasons else []]
        coef_nm = ["b0", "b1"] + ([f"s{i+1}" for i in range(f)]
                                  if do_seasons else [])
    else:
        raise ValueError('ts_method must be "es" or "lm"')

    if coef_nm is not None:            # classic report tail
        report += ["Mean squared error of fit to data: "
                   + fmt(mse, digits_d + 1), ""]
        ttl = "Coefficients for Linear Trend"
        if ts_seasons not in (None, "N"):
            ttl += " and Seasonality"
        report.append(ttl)
        line = ""
        for nm, v in zip(coef_nm, coef_v):
            if nm == "s1":
                report.append(" " + line)
                line = ""
            line += f"{nm}: {fmt(v, digits_d + 1)}  "
        report.append(" " + line)

    # forecast output data frame
    dfmt = {"months": "%b %Y", "years": "%Y"}.get(ts_unit)
    if ts_unit == "quarters":
        dates_out = [f"{d.year} Q{d.quarter}" for d in x_hat]
    elif dfmt:
        dates_out = x_hat.strftime(dfmt)
    else:
        dates_out = x_hat.strftime("%Y-%m-%d")
    forecast = pd.DataFrame({
        x_name: dates_out,
        "predicted": y_hat,
        "lower": y_lwr,
        "upper": y_upr,
        "width": y_upr - y_lwr})

    return {"x_fit": x_fit, "y_fit": y_fit,
            "x_hat": x_hat, "y_hat": y_hat,
            "y_lwr": y_lwr, "y_upr": y_upr,
            "forecast": forecast, "report": report}
