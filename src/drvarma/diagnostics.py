"""Multivariate residual diagnostics (numpy ports of diagnose.c).

- hosking_q : multivariate portmanteau test (Hosking).
- jarque_bera_mv : multivariate normality (sum of univariate Jarque-Bera).
"""

import numpy as np

try:
    from scipy.stats import chi2
    def _chisq_sf(x, df):
        return float(chi2.sf(x, df))
except Exception:  # pragma: no cover - scipy expected, but degrade gracefully
    import math
    def _chisq_sf(x, df):
        # regularised upper incomplete gamma via series/continued fraction
        a = df / 2.0
        xx = x / 2.0
        if xx <= 0:
            return 1.0
        if xx < a + 1.0:
            term = 1.0 / a; s = term; n = a
            for _ in range(500):
                n += 1.0; term *= xx / n; s += term
                if abs(term) < abs(s) * 1e-15:
                    break
            return 1.0 - s * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
        b = xx + 1.0 - a; c = 1e300; d = 1.0 / b; h = d
        for i in range(1, 500):
            an = -i * (i - a); b += 2.0
            d = an * d + b
            if abs(d) < 1e-300: d = 1e-300
            c = b + an / c
            if abs(c) < 1e-300: c = 1e-300
            d = 1.0 / d; de = d * c; h *= de
            if abs(de - 1.0) < 1e-15:
                break
        return h * math.exp(-xx + a * math.log(xx) - math.lgamma(a))


def hosking_q(res, s):
    """Hosking multivariate portmanteau Q on residuals (nobs x m), s lags.

    Returns (Q, df, pvalue);  df = m^2 * s.
    """
    res = np.atleast_2d(np.asarray(res, float))
    n, m = res.shape
    x = res - res.mean(axis=0)
    C = [(x[:n - r].T @ x[r:]) / n for r in range(s + 1)]
    C0inv = np.linalg.inv(C[0])
    Q = 0.0
    for r in range(1, s + 1):
        Cr = C[r]
        Q += np.trace(Cr.T @ C0inv @ Cr @ C0inv)
    Q *= n
    df = m * m * s
    return float(Q), df, _chisq_sf(Q, df)


def ccf(w1, w2, lags):
    """Two-sided sample cross-correlation function between two series.

    Returns ``rho`` of length ``2*lags+1`` for lags ``-lags..+lags``, with
    ``rho[lags+k]`` the correlation at lag k.  At lag k>0 it pairs ``w1_t`` with
    ``w2_{t-k}`` (w2 lagging); at lag k<0, ``w2_t`` with ``w1_{t-|k|}``.  Cross-
    covariances follow drvus ``ccf.c`` (divided by N); standardised by the
    contemporaneous standard deviations.
    """
    w1 = np.asarray(w1, float).ravel()
    w2 = np.asarray(w2, float).ravel()
    n = w1.shape[0]
    x1 = w1 - w1.mean()
    x2 = w2 - w2.mean()
    c11 = float((x1 * x1).mean())
    c22 = float((x2 * x2).mean())
    den = np.sqrt(c11 * c22)

    def c12(k):                          # (1/N) sum_t x1_t x2_{t-k}, k>=0
        return float((x1[k:] * x2[:n - k]).sum() / n)

    def c21(k):                          # (1/N) sum_t x2_t x1_{t-k}, k>=0
        return float((x2[k:] * x1[:n - k]).sum() / n)

    rho = np.zeros(2 * lags + 1)
    for k in range(lags + 1):
        rho[lags + k] = c12(k) / den     # lag +k: w2 lags w1
        rho[lags - k] = c21(k) / den     # lag -k: w1 lags w2
    return rho


def qccf(w1, w2, lags):
    """Hosking bivariate portmanteau Q for the pair (w1, w2) up to `lags`.

    Port of drvus ``ccf.c``/``qccf.c``; equivalent to `hosking_q` on the
    stacked 2-variate series.  Returns (Q, df, pvalue).
    """
    res = np.column_stack([np.asarray(w1, float).ravel(),
                           np.asarray(w2, float).ravel()])
    return hosking_q(res, lags)


def jarque_bera_mv(res):
    """Multivariate Jarque-Bera (sum of univariate JB) on residuals (nobs x m).

    Returns (JB, df, pvalue);  df = 2*m.
    """
    res = np.atleast_2d(np.asarray(res, float))
    n, m = res.shape
    x = res - res.mean(axis=0)
    sd = np.sqrt((x ** 2).mean(axis=0))
    sd = np.where(sd > 1e-12, sd, np.inf)
    skew = (x ** 3).mean(axis=0) / sd ** 3
    kurt = (x ** 4).mean(axis=0) / sd ** 4 - 3.0
    JB = float(np.sum(n * (skew ** 2 / 6.0 + kurt ** 2 / 24.0)))
    df = 2 * m
    return JB, df, _chisq_sf(JB, df)
