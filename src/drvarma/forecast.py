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
