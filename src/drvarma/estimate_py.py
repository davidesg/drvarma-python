"""Pure-Python VARMA estimator (C-free fallback for the CFFI engine).

`estimate_w_py` mirrors the result dict of the C `_engine.estimate_w` so the rest
of the package (model, forecast, irf, diagnostics, report) works unchanged when
the compiled engine is absent.  The exact log-likelihood is Mauricio's AS 311
(`elfvarma_py.elf_varma`, a faithful port of `elfvarma.c`); q=0 uses the faster
vectorised `elf_var`.  Parameters are packed in the same order the C uses
(mu, phi, theta, then the covariance lower triangle) so they line up with
`report._param_labels`.
"""

import numpy as np
from scipy.optimize import minimize

from .elfvarma_py import elf_var, elf_varma, var_residuals, varma_residuals


# -- parameter packing ------------------------------------------------------ #

def _lag_mask(m, k, diag):
    """Boolean (k, m, m) of free entries (diagonal only if `diag`)."""
    mask = np.ones((k, m, m), bool)
    if diag:
        for i in range(k):
            mask[i] = np.eye(m, dtype=bool)
    return mask


def _cov_indices(m, diag_cov):
    """List of (i, j) lower-triangle covariance entries, in C order."""
    if diag_cov:
        return [(i, i) for i in range(m)]
    return [(i, j) for i in range(m) for j in range(i + 1)]


def _pack_params(mu, phi, theta, sigma, include_mean, phi_mask, theta_mask, cov_idx):
    """Pack (mu, phi, theta, cov-lower-tri) into a flat vector in C label order."""
    parts = []
    if include_mean:
        parts.append(np.asarray(mu, float))
    if phi.size:
        parts.append(phi[phi_mask])
    if theta.size:
        parts.append(theta[theta_mask])
    parts.append(np.array([sigma[i, j] for (i, j) in cov_idx]))
    return np.concatenate(parts) if parts else np.zeros(0)


# -- log-likelihood dispatch ------------------------------------------------ #

def _loglik(w, mu, phi, theta, sigma):
    """Exact log-likelihood (ll, ifault): AS 311 for q>0, fast VAR for q=0."""
    if theta.shape[0] == 0:
        return elf_var(w, mu, phi, sigma)
    ll, ifault, _ = elf_varma(w, mu, phi, theta, sigma)
    return ll, ifault


# -- optimiser (un)packing -------------------------------------------------- #

def _unpack_opt(theta_vec, m, p, q, include_mean, phi_mask, theta_mask,
                n_chol, diag_cov):
    """Unpack an optimiser vector (mu, phi, theta, chol(Sigma)) -> model."""
    k = 0
    mu = np.zeros(m)
    if include_mean:
        mu = theta_vec[k:k + m]; k += m
    phi = np.zeros((p, m, m))
    if p:
        nphi = int(phi_mask.sum())
        phi[phi_mask] = theta_vec[k:k + nphi]; k += nphi
    th = np.zeros((q, m, m))
    if q:
        nth = int(theta_mask.sum())
        th[theta_mask] = theta_vec[k:k + nth]; k += nth
    Lc = np.zeros((m, m))
    if diag_cov:
        for i in range(m):
            Lc[i, i] = theta_vec[k]; k += 1
    else:
        idx = np.tril_indices(m)
        Lc[idx] = theta_vec[k:k + n_chol]; k += n_chol
    sigma = Lc @ Lc.T
    return mu, phi, th, sigma


def _initial(w, p, q, include_mean, diag_ar, diag_cov):
    """Starting values: OLS VAR(p) for (mu, phi, Sigma); theta = 0."""
    n, m = w.shape
    mu = w.mean(axis=0) if include_mean else np.zeros(m)
    x = w - mu
    theta = np.zeros((q, m, m))
    if p == 0:
        sigma = np.cov(x.T, bias=True).reshape(m, m)
        return mu, np.zeros((0, m, m)), theta, np.atleast_2d(sigma)
    X = np.column_stack([x[p - 1 - i:n - 1 - i] for i in range(p)])
    Y = x[p:]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    phi = np.zeros((p, m, m))
    for i in range(p):
        phi[i] = beta[i * m:(i + 1) * m].T
    if diag_ar:
        for i in range(p):
            phi[i] = np.diag(np.diag(phi[i]))
    resid = Y - X @ beta
    sigma = (resid.T @ resid) / (n - p)
    if diag_cov:
        sigma = np.diag(np.diag(sigma))
    return mu, phi, theta, np.atleast_2d(sigma)


def estimate_w_py(w, p, q, include_mean=False, diag_ar=False, diag_ma=False,
                  diag_cov=False, method=1, twostep=False, **_ignored):
    """Estimate a VARMA(p,q) on the stationary series `w` by exact ML (pure Python).

    Returns the same dict shape as the C `_engine.estimate_w`.  Uses Mauricio's
    AS 311 likelihood (q>0) or the fast VAR likelihood (q=0).
    """
    w = np.ascontiguousarray(np.atleast_2d(np.asarray(w, float)))
    n, m = w.shape
    phi_mask = _lag_mask(m, p, diag_ar)
    theta_mask = _lag_mask(m, q, diag_ma)
    cov_idx = _cov_indices(m, diag_cov)

    mu0, phi0, theta0, sigma0 = _initial(w, p, q, include_mean, diag_ar, diag_cov)

    try:
        L0 = np.linalg.cholesky(sigma0)
    except np.linalg.LinAlgError:
        L0 = np.diag(np.sqrt(np.clip(np.diag(sigma0), 1e-8, None)))
    if diag_cov:
        chol0 = np.diag(L0); n_chol = m
    else:
        chol0 = L0[np.tril_indices(m)]; n_chol = len(chol0)

    parts = []
    if include_mean:
        parts.append(mu0)
    if p:
        parts.append(phi0[phi_mask])
    if q:
        parts.append(theta0[theta_mask])
    parts.append(chol0)
    x0 = np.concatenate(parts) if parts else np.zeros(0)

    def nll(vec):
        mu, phi, th, sigma = _unpack_opt(vec, m, p, q, include_mean, phi_mask,
                                         theta_mask, n_chol, diag_cov)
        ll, ifault = _loglik(w, mu, phi, th, sigma)
        if ifault or not np.isfinite(ll):
            return 1e10
        return -ll

    if x0.size:
        res = minimize(nll, x0, method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
        x_hat = res.x
        ifault = 0 if res.success else 6
    else:
        x_hat = x0
        ifault = 0

    mu, phi, th, sigma = _unpack_opt(x_hat, m, p, q, include_mean, phi_mask,
                                     theta_mask, n_chol, diag_cov)
    if q == 0:
        logelf, lf = elf_var(w, mu, phi, sigma)
        resid = var_residuals(w, mu, phi)
    else:
        logelf, lf, resid = elf_varma(w, mu, phi, th, sigma,
                                      compute_residuals=True)
        if resid is None:
            resid = varma_residuals(w, mu, phi, th)
    if lf:
        ifault = ifault or lf

    params = _pack_params(mu, phi, th, sigma, include_mean, phi_mask,
                          theta_mask, cov_idx)
    cov, std = _information(w, mu, phi, th, sigma, include_mean, phi_mask,
                           theta_mask, cov_idx)

    return {
        "ifault": int(ifault),
        "npar": int(params.size),
        "m": m, "p": p, "q": q,
        "sigma2": 1.0,                      # fallback split: Sigma = 1.0 * Q
        "logelf": float(logelf),
        "params": params,
        "std_errors": std,
        "cov": cov,
        "residuals": resid,
        "mu": mu,
        "phi": phi,
        "theta": th,
        "sigma": sigma,
    }


def _information(w, mu, phi, theta, sigma, include_mean, phi_mask, theta_mask,
                cov_idx):
    """Covariance/std errors via the numerical Hessian of the neg log-likelihood.

    Parameterised in C label order (mu, phi, theta, cov lower-tri) so the
    returned cov/std line up with `params`.  Best-effort: NaNs if singular.
    """
    m = w.shape[1]
    p = phi.shape[0]
    q = theta.shape[0]

    def nll_packed(vec):
        k = 0
        mu_ = np.zeros(m)
        if include_mean:
            mu_ = vec[k:k + m]; k += m
        phi_ = np.zeros((p, m, m))
        if p:
            nphi = int(phi_mask.sum())
            phi_[phi_mask] = vec[k:k + nphi]; k += nphi
        th_ = np.zeros((q, m, m))
        if q:
            nth = int(theta_mask.sum())
            th_[theta_mask] = vec[k:k + nth]; k += nth
        sig_ = np.zeros((m, m))
        for (i, j) in cov_idx:
            sig_[i, j] = sig_[j, i] = vec[k]; k += 1
        ll, ifault = _loglik(w, mu_, phi_, th_, sig_)
        if ifault or not np.isfinite(ll):
            return np.inf
        return -ll

    x0 = _pack_params(mu, phi, theta, sigma, include_mean, phi_mask, theta_mask,
                      cov_idx)
    npar = x0.size
    if npar == 0:
        return np.zeros((0, 0)), np.zeros(0)
    h = 1e-4 * (np.abs(x0) + 1e-3)
    H = np.zeros((npar, npar))
    for i in range(npar):
        for j in range(i, npar):
            xpp = x0.copy(); xpp[i] += h[i]; xpp[j] += h[j]
            xpm = x0.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
            xmp = x0.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
            xmm = x0.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
            val = (nll_packed(xpp) - nll_packed(xpm)
                   - nll_packed(xmp) + nll_packed(xmm)) / (4 * h[i] * h[j])
            H[i, j] = H[j, i] = val
    try:
        cov = np.linalg.inv(H)
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        cov = np.full((npar, npar), np.nan)
        std = np.full(npar, np.nan)
    return cov, std
