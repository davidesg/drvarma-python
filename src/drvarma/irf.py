"""Impulse-response (orthogonalised, Cholesky) and forecast-error variance
decomposition (numpy ports of diagnose.c).
"""

import numpy as np


def psi_weights(phi, theta, horizon):
    """MA(inf) weights Psi[0..horizon]; Psi[0]=I, Psi[h]=sum phi_i Psi[h-i] - theta_h."""
    phi = np.asarray(phi, float); theta = np.asarray(theta, float)
    p = phi.shape[0]; q = theta.shape[0]
    m = phi.shape[1] if p else theta.shape[1]
    Psi = np.zeros((horizon + 1, m, m))
    Psi[0] = np.eye(m)
    for h in range(1, horizon + 1):
        for i in range(1, min(h, p) + 1):
            Psi[h] += phi[i - 1] @ Psi[h - i]
        if h <= q:
            Psi[h] -= theta[h - 1]
    return Psi


def oirf(phi, theta, sigma, horizon):
    """Orthogonalised IRF: OIRF[h] = Psi[h] @ chol(sigma) (lower). Shape (H+1, m, m)."""
    sigma = np.asarray(sigma, float)
    Psi = psi_weights(phi, theta, horizon)
    L = np.linalg.cholesky(sigma)
    return np.array([Psi[h] @ L for h in range(horizon + 1)])


def fevd(phi, theta, sigma, horizon):
    """Forecast-error variance decomposition (percentages).

    Returns array (horizon, m, m): out[H-1, i, j] = % of the H-step forecast-error
    variance of variable i explained by orthogonal shock j.
    """
    O = oirf(phi, theta, sigma, horizon)
    m = O.shape[1]
    cum = np.cumsum(O ** 2, axis=0)          # cum[h, i, j]
    out = np.zeros((horizon, m, m))
    for H in range(1, horizon + 1):
        c = cum[H - 1]
        tot = c.sum(axis=1, keepdims=True)
        tot[tot == 0] = 1.0
        out[H - 1] = 100.0 * c / tot
    return out
