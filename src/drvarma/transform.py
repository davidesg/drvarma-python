"""Box-Cox transform and regular/seasonal differencing.

Mirrors drvarma's `transform.c`: the modelled series is

    w = (1-B)^d (1-B^s)^D [ scale * BoxCox_lambda(level) ]

with the default `scale = 100` matching the C engine (rescaling improves the
conditioning of the convergence criteria; it does not change the optimum).
"""

import numpy as np

DEFAULT_SCALE = 100.0


def boxcox_fwd(x, lam):
    """Box-Cox forward transform. lam=0 -> log, lam=1 -> identity."""
    x = np.asarray(x, dtype=float)
    if lam == 0.0:
        return np.log(x)
    return (np.power(x, lam) - 1.0) / lam


def boxcox_inv(y, lam):
    """Inverse Box-Cox transform."""
    y = np.asarray(y, dtype=float)
    if lam == 0.0:
        return np.exp(y)
    return np.power(lam * y + 1.0, 1.0 / lam)


def difference(x, d=0, D=0, s=1):
    """Apply (1-B)^d (1-B^s)^D column-wise to a (nobs, m) array.

    Returns the differenced array, shorter by d + D*s rows.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    for _ in range(d):
        x = x[1:] - x[:-1]
    for _ in range(D):
        x = x[s:] - x[:-s]
    return x


def transform(level, lam=0.0, d=1, D=0, s=12, scale=DEFAULT_SCALE):
    """Full forward transform: scale * BoxCox(level), then differencing.

    Parameters
    ----------
    level : (nobs, m) raw level data
    lam, d, D, s : Box-Cox lambda, regular/seasonal diff orders, seasonal lag
    scale : multiplier applied after Box-Cox (default 100)

    Returns
    -------
    (w, bc) : differenced modelled series (neff, m) and the Box-Cox-scaled
              level series (nobs, m) kept for integrating forecasts back.
    """
    level = np.atleast_2d(np.asarray(level, dtype=float))
    if level.ndim == 1:
        level = level.reshape(-1, 1)
    bc = scale * boxcox_fwd(level, lam)
    w = difference(bc, d=d, D=D, s=s)
    return w, bc


def integrate_forecast(bc, wf, lam=0.0, scale=DEFAULT_SCALE, d=1, D=0, s=12):
    """Invert differencing + Box-Cox on a forecast path (per series).

    Port of transform.c:integrate_forecast.

    Parameters
    ----------
    bc : (nobs_raw, m) Box-Cox*scale level history (the `bc` from transform()).
    wf : (L, m) forecast of the fully-differenced modelled series.

    Returns
    -------
    level_out : (L, m) forecasts in original units.
    """
    bc = np.atleast_2d(np.asarray(bc, dtype=float))
    wf = np.atleast_2d(np.asarray(wf, dtype=float))
    nobs_raw, m = bc.shape
    L = wf.shape[0]
    nstage = d + D
    out = np.zeros((L, m))
    for i in range(m):
        chain = [bc[:, i].copy()]
        for k in range(1, nstage + 1):
            lag = 1 if k <= d else s
            prev = chain[k - 1]
            chain.append(prev[lag:] - prev[:-lag])
        cf = wf[:, i].copy()
        for k in range(nstage, 0, -1):
            lag = 1 if k <= d else s
            prev_series = chain[k - 1]
            n_prev = len(prev_series)
            pf = np.zeros(L)
            for l in range(1, L + 1):
                if l - lag <= 0:
                    prev = prev_series[n_prev + (l - lag) - 1]
                else:
                    prev = pf[l - lag - 1]
                pf[l - 1] = cf[l - 1] + prev
            cf = pf
        out[:, i] = boxcox_inv(cf / scale, lam)
    return out
