"""Multivariate VARMA forecasting (numpy port of drvarma's forecast.c).

`forecast_w` computes point forecasts of the stationary modelled series from a
given origin; `forecast_levels` integrates them back to original units via
transform.integrate_forecast.
"""

import numpy as np
from . import transform


def forecast_w(phi, theta, mu, w, a, L, b=0):
    """Point forecasts of the differenced/stationary series (port of forecast_mean).

    Parameters
    ----------
    phi : (p, m, m), theta : (q, m, m), mu : (m,)
    w : (nobs, m) modelled stationary series used in estimation
    a : (nobs, m) residuals from estimation
    L : horizon;  b : origin offset (forecast from observation nobs-b)

    Returns
    -------
    f : (L, m) forecasts of the modelled series.
    """
    w = np.asarray(w, float); a = np.asarray(a, float)
    mu = np.asarray(mu, float)
    n, m = w.shape
    p = len(phi); q = len(theta)
    f = np.zeros((L, m))
    for l in range(1, L + 1):
        ar = np.zeros(m)
        for i in range(1, p + 1):
            src = (f[l - i - 1] - mu) if l > i else (w[n - b - i + l - 1] - mu)
            ar += phi[i - 1] @ src
        ma = np.zeros(m)
        for j in range(1, q + 1):
            if l <= j:
                ma += theta[j - 1] @ a[n - b - j + l - 1]
        f[l - 1] = mu + ar - ma
    return f


def recursive_forecast(series, estwin, H, lam, d, D, scale, p, q,
                       include_mean=False, diag_ar=False, diag_ma=False,
                       diag_cov=False, deseason=None):
    """Fixed-parameter recursive forecasts from multiple origins (port of -estwin).

    Estimate params once on the first `estwin` raw observations, then forecast H
    steps from every origin e in [estwin_eff, nobs_eff] with those FIXED params,
    integrating anchored at the origin and re-seasonalising.

    Currently supports q=0 (VAR), the documented use of -estwin; for q>0 the
    residuals over the full data at fixed params would be required.

    Returns
    -------
    (rows, result) where rows is a list of (origin_raw, series_idx, horizon, level).
    """
    if q != 0:
        raise NotImplementedError("recursive_forecast supports q=0 (VAR) only")
    from ._engine import estimate_w
    from . import transform as _t
    from .deseason import deseasonalize_raw

    levels = np.asarray(series.data, float)
    m = series.m
    freq = series.freq
    sub = series.start[1]
    off = d + D * freq

    dummies = None
    if deseason:
        _, dummies, _ = deseasonalize_raw(levels[:estwin], s=freq,
                                          start_sub=sub, mode=deseason)
        levels = levels.copy()
        periods = (np.arange(len(levels)) + sub - 1) % freq
        for j in range(m):
            levels[:, j] = levels[:, j] - dummies[j][periods]

    w_full, bc_full = _t.transform(levels, lam=lam, d=d, D=D, s=freq, scale=scale)
    nobs_eff = w_full.shape[0]
    estwin_eff = estwin - off
    if estwin_eff < p + 1 or estwin_eff > nobs_eff:
        raise ValueError(f"invalid estwin {estwin} (effective {estwin_eff}, nobs {nobs_eff})")

    result = estimate_w(w_full[:estwin_eff], p=p, q=q, include_mean=include_mean,
                        diag_ar=diag_ar, diag_ma=diag_ma, diag_cov=diag_cov)
    a_full = np.zeros_like(w_full)          # q=0: residuals unused for the mean

    rows = []
    for e in range(estwin_eff, nobs_eff + 1):
        b = nobs_eff - e
        rraw = e + off
        wf = forecast_w(result["phi"], result["theta"], result["mu"],
                        w_full, a_full, H, b=b)
        lvl = _t.integrate_forecast(bc_full[:rraw], wf, lam=lam, scale=scale,
                                    d=d, D=D, s=freq)
        for l in range(1, H + 1):
            for j in range(m):
                v = lvl[l - 1, j]
                if deseason and dummies is not None:
                    period = (rraw + l + sub - 2) % freq
                    v += dummies[j][period]
                rows.append((rraw, j, l, float(v)))
    return rows, result


def forecast_levels(result, w, bc, lam, scale, d, D, s, L, b=0):
    """Forecast L steps and integrate back to original units.

    `result` is the dict from `_engine.estimate_w` (contains phi/theta/mu and
    residuals).  Returns (levels (L, m), wf (L, m)).
    """
    wf = forecast_w(result["phi"], result["theta"], result["mu"],
                    w, result["residuals"], L, b=b)
    levels = transform.integrate_forecast(bc, wf, lam=lam, scale=scale,
                                          d=d, D=D, s=s)
    return levels, wf
