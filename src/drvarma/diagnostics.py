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


def series_stats(x):
    """Per-series descriptive statistics (port of diagnose.c Mean/Stdev/Skew/Kurt).

    Uses the population (÷n) standard deviation, as the C engine does.  Returns a
    dict: n, mean, variance, std, std_error (=std/√n), skew, kurt (excess),
    max_idx/max_val, min_idx/min_val (indices are 0-based).
    """
    x = np.asarray(x, float).ravel()
    n = x.shape[0]
    mean = float(x.mean())
    xc = x - mean
    var = float((xc ** 2).mean())
    sd = float(np.sqrt(var))
    if sd < 1e-20:
        skew = kurt = 0.0
    else:
        skew = float(((xc / sd) ** 3).mean())
        kurt = float(((xc / sd) ** 4).mean() - 3.0)
    imax = int(np.argmax(x))
    imin = int(np.argmin(x))
    return {"n": n, "mean": mean, "variance": var, "std": sd,
            "std_error": sd / np.sqrt(n), "skew": skew, "kurt": kurt,
            "max_idx": imax, "max_val": float(x[imax]),
            "min_idx": imin, "min_val": float(x[imin])}


def acf(x, lags):
    """Autocorrelations corr[1..lags] (port of diagnose.c Acf; biased ÷n var)."""
    x = np.asarray(x, float).ravel()
    n = x.shape[0]
    xc = x - x.mean()
    var = float((xc ** 2).mean())
    out = np.zeros(lags)
    if var < 1e-300:
        return out
    for j in range(1, lags + 1):
        out[j - 1] = float((xc[:n - j] * xc[j:]).sum()) / (n * var)
    return out


def pacf(acf_vals):
    """Partial autocorrelations from the ACF via Durbin-Levinson (port of Pacf)."""
    corr = np.asarray(acf_vals, float).ravel()
    lags = corr.shape[0]
    M = np.zeros((lags + 1, lags + 1))           # 1-indexed phi_{i,j}
    c = np.zeros(lags + 1)
    c[1:lags + 1] = corr
    pc = np.zeros(lags)
    if lags == 0:
        return pc
    M[1][1] = c[1]
    pc[0] = M[1][1]
    for i in range(2, lags + 1):
        s1 = sum(M[i - 1][j] * c[i - j] for j in range(1, i))
        s2 = sum(M[i - 1][j] * c[j] for j in range(1, i))
        M[i][i] = (c[i] - s1) / (1.0 - s2) if abs(1.0 - s2) > 1e-300 else 0.0
        for j in range(1, i):
            M[i][j] = M[i - 1][j] - M[i][i] * M[i - 1][i - j]
        pc[i - 1] = M[i][i]
    return pc


def ljung_box(acf_vals, nobs, npar=0):
    """Ljung-Box Q on the ACF (port of diagnose.c ChiTest).

    Q = n(n+2) Σ_{i=1}^{lags} r_i² / (n-i);  returns (Q, df, pvalue), df=lags-npar.
    """
    corr = np.asarray(acf_vals, float).ravel()
    lags = corr.shape[0]
    Q = float(nobs * (nobs + 2) *
              np.sum([corr[i] ** 2 / (nobs - (i + 1)) for i in range(lags)]))
    df = max(lags - npar, 1)
    return Q, df, _chisq_sf(Q, df)


def _acf_lags(nobs, freq):
    """Lag count for the residual ACF (port of AnalyzeOneSeries' rule)."""
    if nobs < 3 * (freq + 1):
        lags = nobs - freq // 2
    else:
        lags = 3 * (freq + 2)
    return max(1, min(lags, nobs - 2))


def residual_diagnostics(res, freq, lags=None, npar=0):
    """Per-series residual diagnostics for a residual matrix (nobs x m).

    Returns a list (one dict per series) of {stats, acf, pacf, ljung_box, lags},
    mirroring diagnose.c's AnalyzeOneSeries computations.  `npar` adjusts the
    Ljung-Box degrees of freedom.
    """
    res = np.atleast_2d(np.asarray(res, float))
    nobs, m = res.shape
    L = _acf_lags(nobs, freq) if lags is None else lags
    out = []
    for j in range(m):
        col = res[:, j]
        a = acf(col, L)
        p = pacf(a)
        Q, qdf, qp = ljung_box(a, nobs, npar=npar)
        out.append({"stats": series_stats(col), "acf": a, "pacf": p,
                    "lags": L, "ljung_box": {"Q": Q, "df": qdf, "p": qp}})
    return out


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
