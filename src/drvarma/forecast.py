"""Multivariate VARMA forecasting (numpy port of drvarma's forecast.c).

`forecast_w` computes point forecasts of the stationary modelled series from a
given origin; `forecast_levels` integrates them back to original units via
transform.integrate_forecast.
"""

import math

import numpy as np
from . import transform


def forecast_w(phi, theta, mu, w, a, L, b=0, const=None):
    """Point forecasts of the differenced/stationary series (port of forecast_mean).

    Parameters
    ----------
    phi : (p, m, m), theta : (q, m, m), mu : (m,)
    w : (nobs, m) modelled stationary series used in estimation
    a : (nobs, m) residuals from estimation
    L : horizon;  b : origin offset (forecast from observation nobs-b)
    const : (m,) or None, a constant added to every step, for a model
        written as Phi(B)(w_t - mu) = const + Theta(B) a_t. A model with unit
        roots in Phi(B) needs it: its drift is not of the form Phi(1) mu
        (forecast_given). None, the default, adds nothing.

    The recursion is the same with or without unit roots in Phi(B): nothing
    here assumes stationarity.

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
        if const is not None:
            f[l - 1] += const
    return f


def series_psi(psi, c, rn, L):
    """The psi weights of series c's LEVEL: row c of the system's psi weights,
    psi (L+1, m, m), convolved with 1 / r(B), r its univariate non-stationary
    operator (fue's coefficients, r(B) = 1 - rn[0] B - rn[1] B^2 ...; empty
    when the system's w is already the level). -> P (m, L+1), P[:, t] the
    weights of the m shocks at lead t."""
    uu = np.zeros(L + 1); uu[0] = 1.0
    for t in range(1, L + 1):
        uu[t] = sum(rn[k - 1] * uu[t - k] for k in range(1, min(len(rn), t) + 1))
    return np.array([[sum(uu[k] * psi[t - k][c, j] for k in range(t + 1))
                      for t in range(L + 1)] for j in range(psi.shape[1])])


def lead_sd(P, sigma, l, d=0):
    """The s.d. of the l-step error of a series whose shock weights are P
    (series_psi): of its level (d = 0), of its d-period change (d = 1, the
    period; d = s, the year)."""
    v = 0.0
    for t in range(l):
        col = P[:, t] - (P[:, t - d] if d and t >= d else 0.0)
        v += col @ sigma @ col
    return math.sqrt(max(v, 0.0))


def forecast_given(phi, theta, sigma, w, a, L, mu=None, const=None, det=None,
                   lam=None, refactor=None, z=1.96):
    """L-step forecasts of a VARMA given its parameters: no estimation, and no
    stationarity asked of Phi(B).

        Phi(B) (w_t - mu) = const + Theta(B) a_t,    a_t ~ N(0, sigma)

    This is the forecasting entry point for a model that drvarma did not fit,
    such as a VEC written in levels (drvec's ``to_levels_varma``): Phi(B) may
    carry unit roots (Phi(1) singular), and a drift that is not of the form
    Phi(1) mu goes in ``const``. It is the ladder's forecast (Ladder.forecast)
    with the univariate operator empty: the same recursion (forecast_w), the
    same psi weights (irf.psi_weights) and the same band (series_psi,
    lead_sd), and the level through fue's inverse Box-Cox.

    Parameters
    ----------
    phi : (p, m, m); theta : (q, m, m); sigma : (m, m), the covariance of a_t.
    w : (n, m), the series up to the origin (the last row), in the model's
        units: for a ``.pre`` model, refactor * BoxCox(y) - det.
    a : (n, m), the residuals on the same rows (the MA part reads the last q).
    mu, const : (m,) or None.
    det : (m, L) or None, each series' deterministic component at the L
        forecast dates, added back before the Box-Cox is inverted.
    lam, refactor : (m,) or None; None leaves the levels in the model's units.
    z : the band's multiplier, +/- z s.d. in the model's units, transformed
        at both ends (the ladder's 1.96 by default).

    Returns
    -------
    One dict per series: ``w`` (the forecast in the model's units, det not
    added), ``sd`` (its s.d.), ``level``, ``low``, ``high`` (in the series'
    own units), and ``psi`` (the system's psi weights, (L+1, m, m)).
    """
    from .irf import psi_weights
    phi = np.asarray(phi, float); theta = np.asarray(theta, float)
    w = np.asarray(w, float); a = np.asarray(a, float)
    sigma = np.asarray(sigma, float)
    n, m = w.shape
    if a.shape != w.shape:
        raise ValueError(f"w is {w.shape} and a is {a.shape}: one residual per row")
    if theta.shape[0] > n:
        raise ValueError(f"q = {theta.shape[0]} needs that many residuals, there are {n}")
    if phi.shape[0] > n:
        raise ValueError(f"p = {phi.shape[0]} needs that many rows of w, there are {n}")
    mu = np.zeros(m) if mu is None else np.asarray(mu, float)
    f = forecast_w(phi, theta, mu, w, a, L,
                   const=None if const is None else np.asarray(const, float))
    psi = psi_weights(phi, theta, L)
    if lam is not None:
        from fue.forecast import _inv_boxcox

        def inv(i, v):
            return _inv_boxcox(float(v), float(lam[i]), float(refactor[i]))
    else:
        def inv(i, v):
            return float(v)
    out = []
    for c in range(m):
        P = series_psi(psi, c, [], L)
        sd = np.array([lead_sd(P, sigma, l) for l in range(1, L + 1)])
        ys = f[:, c] + (np.asarray(det[c], float) if det is not None else 0.0)
        out.append({"w": f[:, c].copy(), "sd": sd,
                    "level": np.array([inv(c, v) for v in ys]),
                    "low": np.array([inv(c, v - z * e) for v, e in zip(ys, sd)]),
                    "high": np.array([inv(c, v + z * e) for v, e in zip(ys, sd)]),
                    "psi": psi})
    return out


def _forecast_variance(psis, sigma):
    """var[l] = sum_{j=0}^{l-1} psis[j] sigma psis[j]^T (l=1..L)."""
    L = psis.shape[0] - 1
    m = sigma.shape[0]
    var = np.zeros((L + 1, m, m))
    for l in range(1, L + 1):
        acc = np.zeros((m, m))
        for j in range(0, l):
            acc += psis[j] @ sigma @ psis[j].T
        var[l] = acc
    return var


def forecast_level_variances(phi, theta, sigma, L, d, D, s):
    """Level / monthly / annual forecast-error variances (port of forecast.c).

    Returns (v_level, v_mon, v_ann), each (L+1, m, m), on the scale*Box-Cox scale.
    """
    from .irf import psi_weights
    sigma = np.asarray(sigma, float)
    m = sigma.shape[0]
    psi = psi_weights(phi, theta, L)
    deg = d + D * s
    delta = np.zeros(deg + 1)
    delta[0] = 1.0
    for _ in range(d):
        for k in range(deg, 0, -1):
            delta[k] -= delta[k - 1]
    for _ in range(D):
        for k in range(deg, s - 1, -1):
            delta[k] -= delta[k - s]
    psis = np.zeros((L + 1, m, m))
    for l in range(L + 1):
        val = psi[l].copy()
        for k in range(1, min(l, deg) + 1):
            val -= delta[k] * psis[l - k]
        psis[l] = val
    v_level = _forecast_variance(psis, sigma)
    op = np.array([psis[l] - (psis[l - 1] if l >= 1 else 0.0) for l in range(L + 1)])
    v_mon = _forecast_variance(op, sigma)
    op = np.array([psis[l] - (psis[l - s] if l >= s else 0.0) for l in range(L + 1)])
    v_ann = _forecast_variance(op, sigma)
    return v_level, v_mon, v_ann


def recursive_forecast(series, estwin, H, lam, d, D, scale, p, q,
                       include_mean=False, diag_ar=False, diag_ma=False,
                       diag_cov=False, deseason=None):
    """Fixed-parameter recursive forecasts from multiple origins (port of -estwin).

    Estimate params once on the first `estwin` raw observations, then forecast H
    steps from every origin e in [estwin_eff, nobs_eff] with those FIXED params,
    integrating anchored at the origin and re-seasonalising.

    Supports general VARMA(p, q).  For q>0 the MA term uses the estimation-window
    residuals ``varma1.a`` (zeros beyond the window), matching the C binary's
    ``forecast_mean`` (which indexes ``a[e]`` at each origin e and finds 0 past
    the estimation window).

    Returns
    -------
    (rows, result) where rows is a list of (origin_raw, series_idx, horizon, level).
    """
    from ._engine import estimate_w
    from . import transform as _t
    from .deseason import deseasonalize_raw, reseasonalize

    levels = np.asarray(series.data, float)
    m = series.m
    freq = series.freq
    sub = series.start[1]
    off = d + D * freq

    dummies = None
    if deseason:
        # BUG-0003: estimated and removed on the Box-Cox scale (log first)
        g = _t.boxcox_fwd(levels, lam)
        _, dummies, _ = deseasonalize_raw(g[:estwin], s=freq,
                                          start_sub=sub, mode=deseason)
        periods = (np.arange(len(levels)) + sub - 1) % freq
        for j in range(m):
            g[:, j] = g[:, j] - dummies[j][periods]
        levels = _t.boxcox_inv(g, lam)

    w_full, bc_full = _t.transform(levels, lam=lam, d=d, D=D, s=freq, scale=scale)
    nobs_eff = w_full.shape[0]
    estwin_eff = estwin - off
    if estwin_eff < p + 1 or estwin_eff > nobs_eff:
        raise ValueError(f"invalid estwin {estwin} (effective {estwin_eff}, nobs {nobs_eff})")

    result = estimate_w(w_full[:estwin_eff], p=p, q=q, include_mean=include_mean,
                        diag_ar=diag_ar, diag_ma=diag_ma, diag_cov=diag_cov)
    # MA term uses the estimation-window residuals (varma1.a), zero beyond the
    # window — exactly as the C.  Harmless for q=0 (forecast_w ignores `a`).
    a_full = np.zeros_like(w_full)
    a_full[:estwin_eff] = result["residuals"]

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
                    v = float(reseasonalize(v, dummies[j][period], lam))
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
