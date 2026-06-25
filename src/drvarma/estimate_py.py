"""Pure-Python VAR estimator (C-free fallback for the CFFI engine).

`estimate_w_py` mirrors the result dict of the C `_engine.estimate_w` so the rest
of the package (model, forecast, irf, diagnostics, report) works unchanged when
the compiled engine is absent.  Currently VAR only (q=0); for VARMA build the C
engine.  Parameters are packed in the same order the C uses (mu, phi by lag/row,
then the covariance lower triangle) so they line up with `report._param_labels`.
"""

import numpy as np
from scipy.optimize import minimize

from .elfvarma_py import elf_var, var_residuals


# -- parameter packing ------------------------------------------------------ #

def _phi_mask(m, p, diag_ar):
    """Boolean (p, m, m) of free AR entries (diagonal only if diag_ar)."""
    mask = np.ones((p, m, m), bool)
    if diag_ar:
        for k in range(p):
            mask[k] = np.eye(m, dtype=bool)
    return mask


def _cov_indices(m, diag_cov):
    """List of (i, j) lower-triangle covariance entries, in C order."""
    if diag_cov:
        return [(i, i) for i in range(m)]
    return [(i, j) for i in range(m) for j in range(i + 1)]


def _pack_params(mu, phi, sigma, include_mean, phi_mask, cov_idx):
    """Pack (mu, phi, cov-lower-tri) into a flat vector in C label order."""
    parts = []
    if include_mean:
        parts.append(np.asarray(mu, float))
    if phi.size:
        parts.append(phi[phi_mask])
    parts.append(np.array([sigma[i, j] for (i, j) in cov_idx]))
    return np.concatenate(parts) if parts else np.zeros(0)


# -- objective (negative exact log-likelihood) ------------------------------ #

def _unpack_opt(theta, m, p, include_mean, phi_mask, n_chol, diag_cov):
    """Unpack an optimiser vector (mu, phi-free, chol(Sigma)-free) -> model."""
    k = 0
    mu = np.zeros(m)
    if include_mean:
        mu = theta[k:k + m]; k += m
    phi = np.zeros((p, m, m))
    if p:
        nphi = int(phi_mask.sum())
        phi[phi_mask] = theta[k:k + nphi]; k += nphi
    Lc = np.zeros((m, m))
    if diag_cov:
        for i in range(m):
            Lc[i, i] = theta[k]; k += 1
    else:
        idx = np.tril_indices(m)
        Lc[idx] = theta[k:k + n_chol]; k += n_chol
    sigma = Lc @ Lc.T
    return mu, phi, sigma


def _initial_ols(w, p, include_mean, diag_ar, diag_cov):
    """OLS VAR starting values (mu, phi, Sigma)."""
    n, m = w.shape
    mu = w.mean(axis=0) if include_mean else np.zeros(m)
    x = w - mu
    if p == 0:
        a = x
        return mu, np.zeros((0, m, m)), np.cov(a.T, bias=True).reshape(m, m)
    X = np.column_stack([x[p - 1 - i:n - 1 - i] for i in range(p)])  # (n-p, m*p)
    Y = x[p:]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)                    # (m*p, m)
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
    return mu, phi, np.atleast_2d(sigma)


def estimate_w_py(w, p, q, include_mean=False, diag_ar=False, diag_ma=False,
                  diag_cov=False, method=1, twostep=False, **_ignored):
    """Estimate a VAR(p) on the stationary series `w` by exact ML (pure Python).

    Returns the same dict shape as the C `_engine.estimate_w`.  Raises
    NotImplementedError for q>0 (build the C engine for VARMA).
    """
    if q != 0:
        raise NotImplementedError(
            "pure-Python estimator supports q=0 (VAR) only; build the C engine "
            "(python setup.py build_ext --inplace) for VARMA(p>0, q>0).")
    w = np.ascontiguousarray(np.atleast_2d(np.asarray(w, float)))
    n, m = w.shape
    phi_mask = _phi_mask(m, p, diag_ar)
    cov_idx = _cov_indices(m, diag_cov)

    mu0, phi0, sigma0 = _initial_ols(w, p, include_mean, diag_ar, diag_cov)

    # optimiser packing: mu, phi-free, chol(Sigma)-free (Cholesky keeps Sigma PD)
    try:
        L0 = np.linalg.cholesky(sigma0)
    except np.linalg.LinAlgError:
        L0 = np.diag(np.sqrt(np.clip(np.diag(sigma0), 1e-8, None)))
    if diag_cov:
        chol0 = np.diag(L0)
        n_chol = m
    else:
        chol0 = L0[np.tril_indices(m)]
        n_chol = len(chol0)
    parts = []
    if include_mean:
        parts.append(mu0)
    if p:
        parts.append(phi0[phi_mask])
    parts.append(chol0)
    theta0 = np.concatenate(parts) if parts else np.zeros(0)

    def nll(theta):
        mu, phi, sigma = _unpack_opt(theta, m, p, include_mean, phi_mask,
                                     n_chol, diag_cov)
        ll, ifault = elf_var(w, mu, phi, sigma)
        if ifault or not np.isfinite(ll):
            return 1e10
        return -ll

    if theta0.size:
        res = minimize(nll, theta0, method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
        theta_hat = res.x
        ifault = 0 if res.success else 6
    else:
        theta_hat = theta0
        ifault = 0

    mu, phi, sigma = _unpack_opt(theta_hat, m, p, include_mean, phi_mask,
                                 n_chol, diag_cov)
    logelf, lf = elf_var(w, mu, phi, sigma)
    if lf:
        ifault = ifault or 3
    resid = var_residuals(w, mu, phi)

    # params in C label order + standard errors from the observed information
    params = _pack_params(mu, phi, sigma, include_mean, phi_mask, cov_idx)
    cov, std = _information(w, mu, phi, sigma, include_mean, phi_mask, cov_idx)

    return {
        "ifault": int(ifault),
        "npar": int(params.size),
        "m": m, "p": p, "q": 0,
        "sigma2": 1.0,                      # fallback split: Sigma = 1.0 * Q
        "logelf": float(logelf),
        "params": params,
        "std_errors": std,
        "cov": cov,
        "residuals": resid,
        "mu": mu,
        "phi": phi,
        "theta": np.zeros((0, m, m)),
        "sigma": sigma,
    }


def _information(w, mu, phi, sigma, include_mean, phi_mask, cov_idx):
    """Covariance/std errors via the numerical Hessian of the neg log-likelihood.

    Parameterised directly in C label order (mu, phi-free, cov lower-tri), so the
    returned cov/std line up with `params`.  Best-effort: returns NaNs if the
    Hessian is singular.
    """
    m = w.shape[1]
    p = phi.shape[0]

    def nll_packed(vec):
        k = 0
        mu_ = np.zeros(m)
        if include_mean:
            mu_ = vec[k:k + m]; k += m
        phi_ = np.zeros((p, m, m))
        if p:
            nphi = int(phi_mask.sum())
            phi_[phi_mask] = vec[k:k + nphi]; k += nphi
        sig_ = np.zeros((m, m))
        for (i, j) in cov_idx:
            sig_[i, j] = sig_[j, i] = vec[k]; k += 1
        ll, ifault = elf_var(w, mu_, phi_, sig_)
        if ifault or not np.isfinite(ll):
            return np.inf
        return -ll

    x0 = _pack_params(mu, phi, sigma, include_mean, phi_mask, cov_idx)
    npar = x0.size
    if npar == 0:
        return np.zeros((0, 0)), np.zeros(0)
    # scaled central-difference step per parameter
    h = 1e-4 * (np.abs(x0) + 1e-3)
    H = np.zeros((npar, npar))
    f0 = nll_packed(x0)
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
    _ = f0
    return cov, std
