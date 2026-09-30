"""Multivariate identification statistics of Jenkins and Alavi (1981).

G. M. Jenkins and A. S. Alavi, "Some aspects of modelling and forecasting
multivariate time series", JTSA 2, 1-47 (section numbers in brackets). The
school's base for the multivariate stochastic model; sima reads these in its
identification node (sima-python `docs/DESIGN-jenkins-alavi.md`, phase 1).

For a stationary vector series x_t (n x m) — the w_t of the univariate models
(their method 1, no prewhitening) or the residuals of the diagonal system
(method 2, prewhitened):

* `corr_matrices`: the sample correlation matrices R_k, r_ij(k) =
  corr(x_i,t , x_j,t-k) [§3.3], with standard errors: Bartlett's (3.13) for
  series that are not prewhitened, 1/sqrt(n) for prewhitened ones. A cut-off
  after q suggests an MA(q).
* `partial_corr_matrices`: the partial correlation matrices S_k, the last
  matrix of the AR(k) fitted by the multivariate Yule-Walker equations (3.11)
  to the STANDARDISED series, standard error 1/sqrt(n) [§3.3]. A cut-off after
  p suggests an AR(p).
* `q_partial_corr_matrices`: the q-conditioned partial correlation matrices
  S_k(q) of Alavi (1973), the last matrix of (3.14)-(3.15), which uses the
  covariances from lag q + 1 on. A cut-off after p suggests an ARMA(p, q).
  S_k(0) = S_k.
* `determinants`: |R_k|, |S_k|, |S_k(q)| — scalar guides with the same cut-off
  properties, for many series ("the curse of higher dimensionality").
* `symbols`: the + - . table (beyond +-2 standard errors).
* `haugh`: Haugh's (1976) independence test of two prewhitened series, by side.
* `two_sided`: one pair of a stack as a two-sided function over -K..K, the
  layout of drvus' CCF, for R_k and S_k alike (sima's figures).

Facts only: which order or structure to entertain is the analyst's reading.
Convention throughout: the AR model is x_t = SUM_l Phi_l x_{t-l} + a_t, so the
(i, j) element of Phi_l (and of S_k) is the effect of x_j at lag l on x_i.
"""
from __future__ import annotations

import numpy as np


def _cov(x, k):
    """C(k) = (1/n) SUM_t (x_t - xbar)(x_{t-k} - xbar)'; C(-k) = C(k)'."""
    x = np.asarray(x, float)
    n = x.shape[0]
    xc = x - x.mean(0)
    if k < 0:
        return _cov(x, -k).T
    return xc[k:].T @ xc[:n - k] / n


def _standardise(x):
    x = np.asarray(x, float)
    xc = x - x.mean(0)
    sd = np.sqrt((xc ** 2).mean(0))
    return xc / sd


def corr_matrices(x, K, prewhitened=False):
    """R_1..R_K (K x m x m) and their standard errors (K x m x m).

    prewhitened=False: Bartlett's standard error (3.13) under the null of no
    correlation beyond lag k-1 — for an autocorrelation, 1 + 2 SUM_{h<k} r_ii(h)^2;
    for a cross correlation of two series unrelated, 1 + 2 SUM_h r_ii(h) r_jj(h)
    over h < k — divided by n. prewhitened=True: 1/sqrt(n) for every element,
    which is the point of prewhitening [§3.4].
    """
    z = _standardise(x)
    n, m = z.shape
    R = np.array([_cov(z, k) for k in range(1, K + 1)])
    if prewhitened:
        return R, np.full(R.shape, 1.0 / np.sqrt(n))
    auto = np.array([np.diag(R[h]) for h in range(K)])          # K x m, r_ii(h)
    se = np.empty_like(R)
    for k in range(1, K + 1):
        a = auto[:k - 1]                                          # r_ii(h), h < k
        v = 1.0 + 2.0 * (a.T @ a)                                 # m x m: SUM r_ii r_jj
        se[k - 1] = np.sqrt(v / n)
    return R, se


def _last_ar_matrix(z, k, q=0):
    """The last matrix of the AR(k) from the covariances C(q+1..q+k):
    SUM_l Phi_l C(h - l) = C(h), h = q+1..q+k (3.11 for q = 0; 3.14-3.15)."""
    m = z.shape[1]
    C = {h: _cov(z, h) for h in range(-(k + q), k + q + 1)}
    # unknown Phi = [Phi_1 ... Phi_k] (m x mk); equations Phi M = [C(q+1) ... C(q+k)]
    M = np.zeros((m * k, m * k))
    for l in range(1, k + 1):
        for j, h in enumerate(range(q + 1, q + k + 1)):
            M[(l - 1) * m:l * m, j * m:(j + 1) * m] = C[h - l]
    rhs = np.hstack([C[h] for h in range(q + 1, q + k + 1)])
    Phi = np.linalg.solve(M.T, rhs.T).T
    return Phi[:, (k - 1) * m:]


def yule_walker(x, p):
    """Phi_1..Phi_p (p x m x m) of the AR(p) by the multivariate Yule-Walker
    equations (3.11), in the units of x (not standardised): the preliminary
    AR estimates, "very good approximations to the maximum likelihood
    estimates" [§3.2]."""
    x = np.asarray(x, float)
    m = x.shape[1]
    C = {h: _cov(x, h) for h in range(-p, p + 1)}
    M = np.zeros((m * p, m * p))
    for l in range(1, p + 1):
        for j, h in enumerate(range(1, p + 1)):
            M[(l - 1) * m:l * m, j * m:(j + 1) * m] = C[h - l]
    rhs = np.hstack([C[h] for h in range(1, p + 1)])
    Phi = np.linalg.solve(M.T, rhs.T).T
    return np.array([Phi[:, l * m:(l + 1) * m] for l in range(p)])


def residual_ma_preliminary(a, q):
    """U_1..U_q (q x m x m) of the MA residual model a_t = alpha_t - SUM U_k
    alpha_{t-k}, from the cross covariances of the prewhitened residuals:
    cov(a_i,t, a_j,t-k) ~ -U_k[i, j] Sigma_jj, so U_k[i, j] ~ -c_ij(k)/c_jj(0)
    (off-diagonal; the diagonal is left at zero, the univariate models' part).
    Jenkins and Alavi's rule theta_ij,k = -r_ji(k) [§3.4] is this with equal
    variances; valid while the cross terms are small."""
    a = np.asarray(a, float)
    m = a.shape[1]
    c0 = np.diag(_cov(a, 0))
    U = np.zeros((q, m, m))
    for k in range(1, q + 1):
        Ck = _cov(a, k)
        for i in range(m):
            for j in range(m):
                if i != j:
                    U[k - 1, i, j] = -Ck[i, j] / c0[j]
    return U


def partial_corr_matrices(x, K):
    """S_1..S_K (K x m x m) by the multivariate Yule-Walker equations (3.11),
    on the standardised series; standard error 1/sqrt(n)."""
    z = _standardise(x)
    return np.array([_last_ar_matrix(z, k) for k in range(1, K + 1)]), 1.0 / np.sqrt(z.shape[0])


def q_partial_corr_matrices(x, K, q):
    """S_1(q)..S_K(q) (K x m x m) of Alavi (1973), (3.14)-(3.15). S_k(0) = S_k.

    Jenkins and Alavi warn that their sampling behaviour is unstable: large
    values can appear at higher k [§4.1, series 3]. Read the cut-off, not the
    size, and with the 1/sqrt(n) guide only as a guide."""
    z = _standardise(x)
    return np.array([_last_ar_matrix(z, k, q) for k in range(1, K + 1)]), 1.0 / np.sqrt(z.shape[0])


def two_sided(mats, i, j, lag0=0.0):
    """The pair (i, j) of a K x m x m stack as ONE two-sided function over lags
    -K..K, the drvus CCF layout: k > 0 is mats[k-1, i, j] (series j leads, on
    series i), k < 0 is mats[|k|-1, j, i] (series i leads, on series j), and
    lag 0 is `lag0` (the contemporaneous correlation for R_k; S_k has none).
    For R_k this is exactly drvus' `ccf(x_i, x_j)`; for S_k it is the same
    reading of the partial matrices. Returns an array of 2K + 1."""
    M = np.asarray(mats, float)
    K = M.shape[0]
    out = np.empty(2 * K + 1)
    out[K] = lag0
    out[K + 1:] = M[:, i, j]
    out[:K] = M[::-1, j, i]
    return out


def haugh(a1, a2, K):
    """Haugh's (1976) test that two prewhitened series are independent, from
    their cross correlations only: S* = n^2 SUM_{|k|<=K} r(k)^2 / (n - |k|),
    chi-squared with 2K + 1 degrees of freedom — the portmanteau of method 2
    (each series prewhitened by its own model; Hosking's Q adds the
    autocorrelations, which is the checking of a fitted model, not this).
    Also split by side, as Haugh and Box (1977) read direction: k > 0 (the
    second series leads, K d.f.) and k < 0 (the first leads, K d.f.).
    Returns {"all": (S, df, p), "k>0": (...), "k<0": (...)}."""
    from scipy.stats import chi2
    from .diagnostics import ccf
    a1 = np.asarray(a1, float).ravel()
    a2 = np.asarray(a2, float).ravel()
    n = a1.shape[0]
    r = ccf(a1, a2, K)
    k = np.arange(-K, K + 1)
    term = n * n * r ** 2 / (n - np.abs(k))

    def one(mask):
        S = float(term[mask].sum())
        df = int(mask.sum())
        return S, df, float(chi2.sf(S, df))
    return {"all": one(np.ones_like(k, bool)), "k>0": one(k > 0), "k<0": one(k < 0)}


def determinants(mats):
    """|M_k| for each matrix of a K x m x m stack."""
    return np.array([float(np.linalg.det(M)) for M in np.asarray(mats, float)])


def symbols(mats, se, width=2.0):
    """The + - . table: beyond +-width standard errors. `se` is a scalar or an
    array of the same shape. Returns a K x m x m array of '+', '-', '.'."""
    mats = np.asarray(mats, float)
    se = np.broadcast_to(np.asarray(se, float), mats.shape)
    out = np.full(mats.shape, ".", dtype="<U1")
    out[mats > width * se] = "+"
    out[mats < -width * se] = "-"
    return out
