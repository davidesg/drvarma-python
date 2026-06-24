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
