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


# --------------------------------------------------------------------------- #
#  Monte-Carlo bands for OIRF / FEVD                                          #
# --------------------------------------------------------------------------- #

def _bands_from_draws(draws, alpha):
    """Percentile band and point estimate spread from a stack of draws."""
    lo = np.nanpercentile(draws, 100.0 * alpha / 2.0, axis=0)
    hi = np.nanpercentile(draws, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return lo, hi


def irf_fevd_bands(result, horizon, ndraws=800, alpha=0.05, seed=0,
                   include_mean=False, diag_ar=False, diag_ma=False,
                   diag_cov=False):
    """Monte-Carlo confidence bands for the OIRF and the FEVD.

    Without bands there is no way to tell a pass-through share of 5 % from one
    of 26 %: the point estimate alone cannot say whether the difference is
    signal. The impulse response is a smooth but strongly non-linear function of
    the parameters (powers of Phi, and a Cholesky of Sigma), so the honest cheap
    route is to propagate the parameter uncertainty rather than linearise it —
    draw theta ~ N(theta_hat, cov), recompute, and take percentiles. This is the
    standard construction in the VAR literature (Lütkepohl §3.7).

    Draws that leave the stationary/invertible region, or that give a Sigma that
    is not positive definite, are DISCARDED rather than clipped: they are not
    admissible models, and folding them in would widen the band with points the
    likelihood never considered. The count is returned so the caller can see how
    much of the draw was thrown away — a large fraction is itself a diagnostic
    that the fit sits near the boundary.

    Returns a dict with `oirf_lo/hi`, `fevd_lo/hi` (same shapes as `oirf`/`fevd`
    at that horizon), `ndraws_used` and `ndraws_rejected`.
    """
    from .estimate_py import _unpack, _lag_mask, _cov_indices

    par = np.asarray(result["params"], float).ravel()
    cov = result.get("cov")
    if cov is None:
        raise ValueError("the fit carries no parameter covariance; cannot band")
    cov = np.asarray(cov, float)
    m, p, q = int(result["m"]), int(result["p"]), int(result["q"])
    phi_mask = _lag_mask(m, p, diag_ar)
    theta_mask = _lag_mask(m, q, diag_ma)
    cov_idx = _cov_indices(m, diag_cov)
    sigma2 = float(result["sigma2"])

    # Symmetrise and nudge onto the PSD cone before factoring: `cov` comes from a
    # finite-difference Hessian and can be a hair indefinite in the flat
    # directions, which is precisely where the draws matter.
    cov = 0.5 * (cov + cov.T)
    ev, V = np.linalg.eigh(cov)
    ev = np.clip(ev, 0.0, None)
    L = V @ np.diag(np.sqrt(ev))

    rng = np.random.default_rng(seed)
    oirf_draws, fevd_draws = [], []
    rejected = 0
    for _ in range(int(ndraws)):
        vec = par + L @ rng.normal(size=par.shape[0])
        try:
            mu_d, phi_d, theta_d, qq_d = _unpack(vec, m, p, q, include_mean,
                                                 phi_mask, theta_mask, cov_idx)
            sig_d = sigma2 * np.asarray(qq_d, float)
            np.linalg.cholesky(sig_d)                 # PD or reject
            if p and not _stable(phi_d):
                rejected += 1
                continue
            if q and not _stable(-np.asarray(theta_d, float)):
                rejected += 1
                continue
            o = oirf(phi_d, theta_d, sig_d, horizon)
            f = fevd(phi_d, theta_d, sig_d, horizon)
        except Exception:                             # noqa: BLE001
            rejected += 1
            continue
        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(f))):
            rejected += 1
            continue
        oirf_draws.append(o)
        fevd_draws.append(f[-1])

    if len(oirf_draws) < 20:
        raise ValueError(
            f"only {len(oirf_draws)} of {ndraws} draws were admissible; the fit "
            "is too close to the stationarity/invertibility boundary for a "
            "meaningful band")

    o_lo, o_hi = _bands_from_draws(np.stack(oirf_draws), alpha)
    f_lo, f_hi = _bands_from_draws(np.stack(fevd_draws), alpha)
    return {"oirf_lo": o_lo, "oirf_hi": o_hi, "fevd_lo": f_lo, "fevd_hi": f_hi,
            "ndraws_used": len(oirf_draws), "ndraws_rejected": rejected,
            "alpha": alpha}


def _stable(phi):
    """True if the VAR companion matrix of `phi` has all roots inside the circle."""
    phi = np.asarray(phi, float)
    if phi.ndim != 3 or phi.shape[0] == 0:
        return True
    P, m = phi.shape[0], phi.shape[1]
    C = np.zeros((P * m, P * m))
    C[:m] = np.concatenate([phi[k] for k in range(P)], axis=1)
    if P > 1:
        C[m:, :-m] = np.eye((P - 1) * m)
    return bool(np.max(np.abs(np.linalg.eigvals(C))) < 1.0)
