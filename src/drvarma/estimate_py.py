"""Pure-Python VARMA estimator (C-free fallback for the CFFI engine).

`estimate_w_py` mirrors the C `_engine.estimate_w` faithfully: it uses the **same
parameterisation, initialisation, objective and optimiser** as the C engine, so
the engine-free result matches the C to floating-point tolerance — including the
``sigma2``/``Q`` split and the ``cov[]`` standard errors.

How it mirrors the C (`drvarma_api.c` + `drvmlest.c` + `qnewtopt.c`):

* **Parameterisation** (`shootx` layout): the flat vector packs ``mu``, then each
  AR lag's coefficients (``m`` diagonal or ``m*m`` full, row-major), each MA lag's,
  then the **raw** innovation-covariance lower triangle ``qq`` (``cov[i,j]``).
  ``phi[0]=theta[0]=I``, so the C's "normalisation" is the identity here.
* **Initialisation** (`init_varma`): sample means; OLS VAR(p) coefficients; and
  ``qq`` started at the **correlation matrix** of the OLS residuals.  Because the
  concentrated objective is scale-invariant in ``qq`` (``f1 -> f1/c``,
  ``f2 -> c^m f2``), this starting scale is what pins the reported ``sigma2``/``Q``
  split, exactly as in the C.
* **Objective** (`objcfunc`): ``(f1/f1_0)^m * (f2/f2_0)`` from Mauricio's AS 311
  ``elf`` with ``sigma2=1`` — i.e. the exact concentrated Gaussian likelihood.
* **Optimiser** (`raxopt`, `_qnewt`): the factored BFGS quasi-Newton.  Its final
  factored Hessian ``b`` gives ``cov = 2*f*b^{-1}/n`` (`drvmlest.c:est`).

The exact log-likelihood is Mauricio's AS 311 (`_as311`, faithful port of
`elfvarma.c`).  Parameters line up with `report._param_labels` (mu, phi, theta,
cov lower-tri).

License: GPL-2.0-or-later
"""

import numpy as np

from . import _as311
from . import _qnewt
from .elfvarma_py import elf_varma, var_residuals, varma_residuals, _to_1based_cube

_LOG2PI = 1.837877066                     # drvmlest.c constant (matches AS 311)


# -- parameter packing ------------------------------------------------------ #

def _lag_mask(m, k, diag):
    """Boolean (k, m, m) of free entries (diagonal only if `diag`)."""
    mask = np.ones((k, m, m), bool)
    if diag:
        for i in range(k):
            mask[i] = np.eye(m, dtype=bool)
    return mask


def _cov_indices(m, diag_cov):
    """List of (i, j) lower-triangle covariance entries, in C (`shootx`) order."""
    if diag_cov:
        return [(i, i) for i in range(m)]
    return [(i, j) for i in range(m) for j in range(i + 1)]


def _pack(mu, phi, theta, qq, include_mean, phi_mask, theta_mask, cov_idx):
    """Pack (mu, phi, theta, qq lower-tri) into a flat vector in C label order."""
    parts = []
    if include_mean:
        parts.append(np.asarray(mu, float))
    if phi.size:
        parts.append(phi[phi_mask])
    if theta.size:
        parts.append(theta[theta_mask])
    parts.append(np.array([qq[i, j] for (i, j) in cov_idx]))
    return np.concatenate(parts) if parts else np.zeros(0)


def _unpack(vec, m, p, q, include_mean, phi_mask, theta_mask, cov_idx):
    """Unpack a flat vector (mu, phi, theta, qq lower-tri) -> model arrays."""
    k = 0
    mu = np.zeros(m)
    if include_mean:
        mu = vec[k:k + m]; k += m
    phi = np.zeros((p, m, m))
    if p:
        nphi = int(phi_mask.sum())
        phi[phi_mask] = vec[k:k + nphi]; k += nphi
    theta = np.zeros((q, m, m))
    if q:
        nth = int(theta_mask.sum())
        theta[theta_mask] = vec[k:k + nth]; k += nth
    qq = np.zeros((m, m))
    for (i, j) in cov_idx:
        qq[i, j] = qq[j, i] = vec[k]; k += 1
    return mu, phi, theta, qq


# -- AS 311 objective pieces ------------------------------------------------ #

def _elf_f1f2(w, mu, phi, theta, qq, xitol):
    """Return (f1, f2, ifault) from AS 311 `elf` with sigma2=1 (concentrated)."""
    n, m = w.shape
    p = phi.shape[0]
    q = theta.shape[0]
    Mu = np.zeros(m + 1); Mu[1:] = mu
    Phi = _to_1based_cube(phi, p, m) if p else np.zeros((1, m + 1, m + 1))
    Theta = _to_1based_cube(theta, q, m) if q else np.zeros((1, m + 1, m + 1))
    Qq = np.zeros((m + 1, m + 1)); Qq[1:, 1:] = qq
    W = np.zeros((n + 1, m + 1)); W[1:, 1:] = w
    _logelf, f1, f2, _a, ifault = _as311.elf(
        m, n, p, q, Mu, Phi, Theta, Qq, W, 1.0, xitol, False)
    return float(f1), float(f2), int(ifault)


# -- initialisation (port of init_varma, drvarma_api.c) --------------------- #

def _init_varma(w, p, q, include_mean, diag_ar, diag_ma, diag_cov,
                phi_mask, theta_mask, cov_idx):
    """Starting vector: sample means, OLS VAR(p), qq = residual correlation matrix.

    Faithful port of `init_varma`: theta starts at 0, and the covariance block is
    the *correlation* matrix of the OLS residuals (unit diagonal), which fixes the
    qq scale (hence the sigma2/Q split) like the C.
    """
    n, m = w.shape
    mu = w.mean(axis=0) if include_mean else np.zeros(m)
    xc = w - (mu if include_mean else 0.0)

    phi = np.zeros((p, m, m))
    if p > 0:
        T = n - p
        Z = np.zeros((T, m * p))
        Y = np.zeros((T, m))
        for t in range(p, n):
            r = t - p
            Y[r] = xc[t]
            for k in range(1, p + 1):
                Z[r, (k - 1) * m:k * m] = xc[t - k]
        coef, *_ = np.linalg.lstsq(Z, Y, rcond=None)      # (m*p, m), coef[:,eq]
        for k in range(p):
            phi[k] = coef[k * m:(k + 1) * m, :].T
        if diag_ar:
            for k in range(p):
                phi[k] = np.diag(np.diag(phi[k]))
        # residuals over the full sample (C uses the t>k guard)
        resid = np.zeros((n, m))
        for t in range(n):
            pred = np.zeros(m)
            for k in range(1, p + 1):
                if t >= k:
                    pred += phi[k - 1] @ xc[t - k]
            resid[t] = xc[t] - pred
        rc = resid.T @ resid / n
    else:
        rc = (xc.T @ xc) / n

    # correlation matrix (regularised diagonal = 1), as in init_varma
    sd = np.sqrt(np.clip(np.diag(rc), 1e-24, None))
    corr = rc / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)
    if diag_cov:
        qq = np.eye(m)                                    # init variance = 1
    else:
        qq = corr

    return _pack(mu, phi, np.zeros((q, m, m)), qq, include_mean,
                 phi_mask, theta_mask, cov_idx)


# -- Hannan-Rissanen two-step initialisation (port of drvarma_api.c) -------- #

def _ols(Z, y):
    """Solve the normal equations (Z'Z) b = Z'y, mirroring the C ludcp/lusol."""
    ZtZ = Z.T @ Z
    return np.linalg.solve(ZtZ, Z.T @ y)


def _hannan_rissanen_diag(w, p, q, include_mean):
    """Diagonal (per-series) HR two-step estimate; port of `hannan_rissanen_diag`.

    Returns (phi_d, theta_d, var_d): phi_d (p, m) and theta_d (q, m) diagonal
    coefficients per series, var_d (m,) residual variances scaled to average 1.
    For q==0 each series is a plain OLS AR(p).
    """
    n, m = w.shape
    mean = w.mean(axis=0) if include_mean else np.zeros(m)
    datac = w - mean
    maxpq = max(p, q)

    phi_d = np.zeros((p, m))
    theta_d = np.zeros((q, m))

    for i in range(m):
        xi = datac[:, i]
        if q == 0:
            if n - p <= 0:
                continue
            Z = np.column_stack([xi[p - k:n - k] for k in range(1, p + 1)])  # cols t-1..t-p
            yv = xi[p:]
            b = _ols(Z, yv)
            phi_d[:, i] = b
            continue

        # q > 0: Hannan-Rissanen
        L = int(np.floor(np.sqrt(n)))
        if L < p + q:
            L = p + q
        if L + maxpq >= n:
            L = n - maxpq - 1
        if L < 1:
            L = 1

        T_ar = n - L
        if T_ar <= L:
            continue
        # Step 1: AR(L) OLS -> residuals e_hat
        Z_ar = np.column_stack([xi[L - k:n - k] for k in range(1, L + 1)])     # t-1..t-L
        y_ar = xi[L:]
        ar_coef = _ols(Z_ar, y_ar)
        e_hat = np.zeros(n)
        pred = Z_ar @ ar_coef                                                 # t = L..n-1
        e_hat[L:] = xi[L:] - pred

        # Step 2: regress y_t on [y_{t-1..t-p}, e_{t-1..t-q}]
        start = max(maxpq + 1, L + 2)                                          # 1-indexed t
        if start > n:
            continue
        s0 = start - 1                                                         # 0-indexed first row
        rows = np.arange(s0, n)
        Xcols = [xi[rows - k] for k in range(1, p + 1)]                        # AR lags
        Xcols += [e_hat[rows - k] for k in range(1, q + 1)]                    # MA lags
        X = np.column_stack(Xcols)
        b = _ols(X, xi[rows])
        phi_d[:, i] = b[:p]
        theta_d[:, i] = -b[p:p + q]

    # ARMA residual variances (t = maxpq+1 .. n), scaled to average 1
    var_d = np.zeros(m)
    for i in range(m):
        xi = datac[:, i]
        resid = np.zeros(n)
        for t in range(maxpq, n):
            pr = 0.0
            for k in range(1, p + 1):
                pr += phi_d[k - 1, i] * xi[t - k]
            for k in range(1, q + 1):
                pr += theta_d[k - 1, i] * resid[t - k]
            resid[t] = xi[t] - pr
        seg = resid[maxpq:]
        var_d[i] = float(seg @ seg) / len(seg) if len(seg) else 1.0
    sf = var_d.mean()
    if sf < 1e-12:
        sf = 1.0
    var_d = var_d / sf
    return phi_d, theta_d, var_d


def _apply_twostep(x0, w, m, p, q, include_mean, diag_ar, diag_ma, diag_cov,
                   phi_mask, theta_mask, cov_idx):
    """Merge the diagonal HR estimate into the full start vector (`combine_vectors`)."""
    phi_d, theta_d, var_d = _hannan_rissanen_diag(w, p, q, include_mean)
    mu, phi, theta, qq = _unpack(x0, m, p, q, include_mean,
                                 phi_mask, theta_mask, cov_idx)
    for k in range(p):
        for i in range(m):
            phi[k, i, i] = phi_d[k, i]
    for k in range(q):
        for i in range(m):
            theta[k, i, i] = theta_d[k, i]
    for i in range(m):
        qq[i, i] = var_d[i]
    return _pack(mu, phi, theta, qq, include_mean, phi_mask, theta_mask, cov_idx)


# -- estimator -------------------------------------------------------------- #

def estimate_w_py(w, p, q, include_mean=False, diag_ar=False, diag_ma=False,
                  diag_cov=False, method=1, twostep=False, maxits=500,
                  grtol=1e-7, sptol=1e-7, **_ignored):
    """Estimate a VARMA(p,q) on the stationary series `w` by exact ML (pure Python).

    Returns the same dict shape as the C `_engine.estimate_w`, with the C's
    ``sigma2``/``Q`` split (``params`` carries the raw ``qq`` lower triangle,
    ``sigma`` = ``sigma2 * qq``) and ``cov``/``std_errors`` from the factored
    BFGS Hessian.
    """
    w = np.ascontiguousarray(np.atleast_2d(np.asarray(w, float)))
    n, m = w.shape
    phi_mask = _lag_mask(m, p, diag_ar)
    theta_mask = _lag_mask(m, q, diag_ma)
    cov_idx = _cov_indices(m, diag_cov)
    xitol = -1e-3 if method == 2 else 1e-3

    x0 = _init_varma(w, p, q, include_mean, diag_ar, diag_ma, diag_cov,
                     phi_mask, theta_mask, cov_idx)
    # Hannan-Rissanen two-step: seed the diagonal AR/MA/cov from per-series HR
    # (only for q>0 and a not-fully-diagonal model, exactly as the C).
    if twostep and q > 0 and not (diag_ar and diag_ma and diag_cov):
        x0 = _apply_twostep(x0, w, m, p, q, include_mean, diag_ar, diag_ma,
                            diag_cov, phi_mask, theta_mask, cov_idx)
    npar = x0.size

    def model_of(vec):
        return _unpack(vec, m, p, q, include_mean, phi_mask, theta_mask, cov_idx)

    f1_0, f2_0, if0 = _elf_f1f2(w, *model_of(x0), xitol)
    ifault = if0

    def objcfunc(vec):
        mu, phi, theta, qq = model_of(np.asarray(vec))
        f1, f2, ifa = _elf_f1f2(w, mu, phi, theta, qq, xitol)
        if ifa or not np.isfinite(f1) or f1 <= 0.0 or f2 <= 0.0:
            return 1.0
        return (f1 / f1_0) ** m * (f2 / f2_0)

    if npar and not if0:
        # raxopt works on a 1-indexed vector (leading unused slot).
        xk = np.zeros(npar + 1)
        xk[1:] = x0

        def func1(xk1):
            return objcfunc(xk1[1:npar + 1])

        fk, bfac, nit, termcode = _qnewt.raxopt(
            func1, npar, xk, maxits, grtol, sptol)
        x_hat = xk[1:npar + 1].copy()
        ifault = 0 if termcode in (1, 2) else 6
    else:
        fk, bfac = 1.0, None
        x_hat = x0
        nit, termcode = 0, 0

    mu, phi, theta, qq = model_of(x_hat)
    f1, f2, lf = _elf_f1f2(w, mu, phi, theta, qq, xitol)
    if lf:
        ifault = ifault or lf

    # Concentrated log-likelihood and variance (drvmlest.c:est, [4]).
    sigma2 = f1 / (n * m)
    logelf = (-0.5 * m * n * (_LOG2PI - np.log(m) - np.log(n) + 1.0)
              - 0.5 * n * (m * np.log(f1) + np.log(f2)))
    sigma = sigma2 * qq                                   # r->sigma = sigma2 * qq

    # AS 311 exact residuals at the final estimate (scale-invariant in qq).
    _ll, _ifr, resid = elf_varma(w, mu, phi, theta, qq, xitol=xitol,
                                 compute_residuals=True)
    if resid is None:
        resid = var_residuals(w, mu, phi) if q == 0 else varma_residuals(w, mu, phi, theta)

    params = _pack(mu, phi, theta, qq, include_mean, phi_mask, theta_mask, cov_idx)
    cov, std = _covariance(bfac, fk, n, npar)

    return {
        "ifault": int(ifault),
        "npar": int(npar),
        "m": m, "p": p, "q": q,
        "sigma2": float(sigma2),
        "logelf": float(logelf),
        "params": params,
        "std_errors": std,
        "cov": cov,
        "residuals": resid,
        "mu": mu,
        "phi": phi,
        "theta": theta,
        "sigma": sigma,
        "nit": int(nit),            # BFGS iterations (pure-Python only)
        "termcode": int(termcode),  # raxopt termination code (1/2 = converged)
    }


def _covariance(bfac, fk, n, npar):
    """Parameter covariance / std errors from the factored BFGS Hessian.

    ``cov = 2*fk*b^{-1}/n`` solved column by column with `cholsol` (the factored
    Hessian ``b`` is the Cholesky factor of the BFGS approximation), mirroring
    `drvmlest.c:est` [3].
    """
    if not npar or bfac is None:
        return np.zeros((npar, npar)), np.zeros(npar)
    cov = np.zeros((npar, npar))
    for i in range(1, npar + 1):
        e = np.zeros(npar + 1)
        e[i] = 1.0
        _qnewt.cholsol(bfac, npar, e)
        for j in range(1, npar + 1):
            cov[j - 1, i - 1] = 2.0 * fk * e[j] / n
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return cov, std
