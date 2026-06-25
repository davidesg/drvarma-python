"""Synthetic VARMA data generation, for validating the migration.

Uses drvarma's parameterisation:

    (w_t - mu) = sum_i Phi_i (w_{t-i} - mu) + a_t - sum_j Theta_j a_{t-j},
    a_t ~ N(0, Sigma)

so simulated series can be fed back to the estimator and the parameters
recovered (same MA sign convention as forecast.c / the C engine).
"""

import numpy as np
from .series import MultiSeries


def simulate_varma(phi=None, theta=None, sigma=None, n=200, mu=None,
                   burnin=200, seed=None, freq=12, start=(2000, 1), names=None):
    """Simulate a stationary VARMA(p, q) process.

    Parameters
    ----------
    phi : list of (m, m) arrays  (AR matrices Phi_1..Phi_p), or None
    theta : list of (m, m) arrays (MA matrices Theta_1..Theta_q), or None
    sigma : (m, m) innovation covariance (default identity)
    n : number of observations to return
    mu : (m,) mean vector (default zeros)
    burnin : warm-up samples discarded
    seed : RNG seed

    Returns
    -------
    MultiSeries of shape (n, m).
    """
    rng = np.random.default_rng(seed)
    phi = [np.asarray(P, float) for P in (phi or [])]
    theta = [np.asarray(T, float) for T in (theta or [])]
    p, q = len(phi), len(theta)

    # infer m
    m = None
    for M in phi + theta:
        m = M.shape[0]; break
    if m is None:
        m = (np.asarray(sigma).shape[0] if sigma is not None
             else (len(mu) if mu is not None else 1))
    sigma = np.eye(m) if sigma is None else np.asarray(sigma, float)
    mu = np.zeros(m) if mu is None else np.asarray(mu, float)

    L = np.linalg.cholesky(sigma)
    T = n + burnin
    a = (rng.standard_normal((T, m)) @ L.T)        # innovations ~ N(0, Sigma)
    w = np.zeros((T, m))
    for t in range(T):
        v = a[t].copy()
        for i in range(1, p + 1):
            if t - i >= 0:
                v += phi[i - 1] @ (w[t - i] - mu)
        for j in range(1, q + 1):
            if t - j >= 0:
                v -= theta[j - 1] @ a[t - j]
        w[t] = mu + v

    data = w[burnin:]
    return MultiSeries(data, freq=freq, start=start, names=names)


# --------------------------------------------------------------------------- #
#  Stationarity / invertibility helpers and a registry of ground-truth cases  #
# --------------------------------------------------------------------------- #

def _companion_eigmax(mats):
    """Largest companion-matrix eigenvalue modulus of a coefficient stack.

    `mats` is a list/array of (m, m) matrices (Phi_1..Phi_p or Theta_1..Theta_q).
    Returns 0.0 for an empty stack.
    """
    mats = [np.asarray(M, float) for M in mats]
    if not mats:
        return 0.0
    p = len(mats)
    m = mats[0].shape[0]
    comp = np.zeros((m * p, m * p))
    for i in range(p):
        comp[:m, i * m:(i + 1) * m] = mats[i]
    if p > 1:
        comp[m:, :m * (p - 1)] = np.eye(m * (p - 1))
    return float(np.max(np.abs(np.linalg.eigvals(comp))))


def is_stationary(phi, tol=1.0):
    """True if the AR operator is stationary (all companion |eig| < tol)."""
    return _companion_eigmax(phi) < tol


def is_invertible(theta, tol=1.0):
    """True if the MA operator is invertible (all companion |eig| < tol)."""
    return _companion_eigmax(theta) < tol


def varma_cases():
    """Registry of seeded VARMA ground-truth cases for recovery/reliability tests.

    Each entry is a dict with keys: name, phi (list of (m,m)), theta (list of
    (m,m)), sigma (m,m), mu (m,), well_identified (bool — VARs and simple VARMAs
    where the MLE recovers the truth at large n), and notes.  All cases are
    verified stationary and invertible.
    """
    cases = [
        dict(name="var1_m2",
             phi=[[[0.5, 0.1], [-0.2, 0.4]]], theta=[],
             sigma=[[1.0, 0.3], [0.3, 0.8]], mu=[0.1, -0.2],
             well_identified=True, notes="simple bivariate VAR(1)"),
        dict(name="var2_m3",
             phi=[[[0.4, 0.0, 0.1], [0.0, 0.3, 0.0], [0.1, 0.0, 0.35]],
                  [[-0.2, 0.0, 0.0], [0.0, -0.15, 0.0], [0.0, 0.0, -0.1]]],
             theta=[],
             sigma=[[1.0, 0.2, 0.1], [0.2, 0.9, 0.25], [0.1, 0.25, 1.1]],
             mu=[0.0, 0.0, 0.0],
             well_identified=True, notes="VAR(2), m=3, full Sigma"),
        dict(name="varma11_m2",
             phi=[[[0.5, 0.1], [0.0, 0.4]]], theta=[[[0.3, 0.0], [0.1, 0.2]]],
             sigma=[[1.0, 0.2], [0.2, 0.7]], mu=[0.0, 0.0],
             well_identified=False, notes="bivariate VARMA(1,1), weakly id."),
        dict(name="fullsigma_m3",
             phi=[[[0.3, 0.05, 0.0], [0.0, 0.25, 0.05], [0.05, 0.0, 0.3]]],
             theta=[],
             sigma=[[1.2, 0.5, 0.4], [0.5, 1.0, 0.45], [0.4, 0.45, 1.3]],
             mu=[0.0, 0.0, 0.0],
             well_identified=True, notes="VAR(1) with strongly-correlated Sigma"),
        dict(name="near_unit_root_m2",
             phi=[[[0.92, 0.0], [0.0, 0.88]]], theta=[],
             sigma=[[1.0, 0.2], [0.2, 1.0]], mu=[0.0, 0.0],
             well_identified=True, notes="near-unit-root diagonal VAR(1)"),
        dict(name="diag_var1_m2",
             phi=[[[0.6, 0.0], [0.0, 0.3]]], theta=[],
             sigma=[[1.0, 0.0], [0.0, 0.5]], mu=[0.0, 0.0],
             well_identified=True, notes="diagonal VAR(1), diagonal Sigma"),
    ]
    for c in cases:                                  # sanity-check the registry
        assert is_stationary(c["phi"]), c["name"]
        assert is_invertible(c["theta"]), c["name"]
    return cases
