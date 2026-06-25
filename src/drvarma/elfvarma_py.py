"""Pure-Python exact Gaussian VARMA log-likelihood (reference / C-free fallback).

This is the multivariate counterpart of fue's m=1 ``elfvarma.py``.  The C engine
(``csrc/internal/elfvarma.c`` + ``multshea.c``, AS 311) remains the reference and
the fast path; this module lets the package run without the compiled engine.

Currently implemented: the **exact VAR(p)** (q=0) log-likelihood for general m,
using the stationary distribution of the first p observations (companion-form
Lyapunov equation) plus the conditional density of the rest.  This reproduces
the C ``logelf`` to ~1e-7.  The MA case (q>0) is not yet implemented here.

References
----------
Mauricio (1997) Applied Statistics 46, 157-171.   [AS 311]
Mauricio (2002) J. Time Series Analysis 23, 473-486.

License: GPL-2.0-or-later
"""

import numpy as np

_LOG2PI = float(np.log(2.0 * np.pi))


def companion(phi):
    """Companion matrix A (mp x mp) of a VAR(p) with coefficient stack `phi`.

    State convention: s_t = [x_t; x_{t-1}; ...; x_{t-p+1}], so the top block-row
    is [Phi_1 ... Phi_p] and the sub-diagonal is the identity.
    """
    phi = np.asarray(phi, float)
    p, m, _ = phi.shape
    A = np.zeros((m * p, m * p))
    for i in range(p):
        A[:m, i * m:(i + 1) * m] = phi[i]
    if p > 1:
        A[m:, :m * (p - 1)] = np.eye(m * (p - 1))
    return A


def stationary_cov(phi, sigma):
    """Stationary covariance Gamma (mp x mp) of the VAR(p) companion state.

    Solves the discrete Lyapunov equation Gamma = A Gamma A' + Q_c, where Q_c has
    the innovation covariance `sigma` in its top-left m x m block (Bartels-Stewart
    via SciPy).  Returns None if the equation cannot be solved.
    """
    from scipy.linalg import solve_discrete_lyapunov
    phi = np.asarray(phi, float)
    sigma = np.asarray(sigma, float)
    p, m, _ = phi.shape
    A = companion(phi)
    mp = m * p
    Qc = np.zeros((mp, mp))
    Qc[:m, :m] = sigma
    try:
        G = solve_discrete_lyapunov(A, Qc)
    except (np.linalg.LinAlgError, ValueError):
        return None
    return 0.5 * (G + G.T)


def var_residuals(w, mu, phi):
    """Conditional residuals a_t = x_t - sum Phi_i x_{t-i} for t = p..n-1.

    `w` is (n, m); returns an (n, m) array with the first p rows left as the
    centered series x_t = w_t - mu (no residual defined there).
    """
    w = np.asarray(w, float)
    phi = np.asarray(phi, float)
    n, m = w.shape
    p = phi.shape[0]
    x = w - mu
    a = x.copy()
    if p:
        acc = x[p:].copy()
        for i in range(1, p + 1):                  # a_t = x_t - sum_i Phi_i x_{t-i}
            acc -= x[p - i:n - i] @ phi[i - 1].T
        a[p:] = acc
    return a


def elf_var(w, mu, phi, sigma):
    """Exact Gaussian VAR(p) log-likelihood for general m (q=0).

    Parameters
    ----------
    w : (n, m) modelled stationary series (already differenced / centered basis)
    mu : (m,) process mean (subtracted from w)
    phi : (p, m, m) AR coefficient matrices
    sigma : (m, m) innovation covariance

    Returns
    -------
    (loglik, ifault) : the exact log-likelihood and a fault code
        0 = OK, 3 = non-stationary / not positive definite.
    """
    w = np.asarray(w, float)
    phi = np.asarray(phi, float)
    sigma = np.asarray(sigma, float)
    n, m = w.shape
    p = phi.shape[0]
    x = w - np.asarray(mu, float)

    # conditional log-density of x_{p..n-1}
    try:
        sign_s, logdet_s = np.linalg.slogdet(sigma)
        if sign_s <= 0:
            return -np.inf, 3
        sinv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        return -np.inf, 3
    a = x[p:].copy()
    for i in range(1, p + 1):                      # a_t = x_t - sum_i Phi_i x_{t-i}
        a -= x[p - i:n - i] @ phi[i - 1].T
    qf = float(np.sum((a @ sinv) * a))
    ll_cond = -0.5 * ((n - p) * (m * _LOG2PI + logdet_s) + qf)

    if p == 0:
        return float(ll_cond), 0

    # exact marginal of the first p observations via the stationary covariance
    G = stationary_cov(phi, sigma)
    if G is None:
        return -np.inf, 3
    sign_g, logdet_g = np.linalg.slogdet(G)
    if sign_g <= 0:
        return -np.inf, 3
    s = np.concatenate([x[p - 1 - k] for k in range(p)])   # [x_p; x_{p-1}; ...; x_1]
    try:
        qf_init = float(s @ np.linalg.solve(G, s))
    except np.linalg.LinAlgError:
        return -np.inf, 3
    ll_init = -0.5 * (p * m * _LOG2PI + logdet_g + qf_init)
    return float(ll_init + ll_cond), 0
