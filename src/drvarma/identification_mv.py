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

And from the Wisconsin line (sima-python `docs/STUDY-tiao-box.md`):

* `stepwise_ar`: Tiao and Box's (1981, §4.1) stepwise autoregression — the
  last coefficient matrix of each AR(l) by least squares with its t-ratios,
  the likelihood-ratio statistic M(l) (4.3), chi-squared with m^2 d.f., and
  the diagonal of the residual covariance matrix. The test beside S_k's guide.
* `canonical`: Box and Tiao's (1977) canonical analysis of a VAR(p) —
  components ordered from least to most predictable; near-white ones are
  static relations among the series, near-non-stationary ones their common
  growth.

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


# --------------------------------------------------------------------------- #
#  Tiao and Box (1981): the stepwise autoregression                           #
# --------------------------------------------------------------------------- #

def _ls_var(x, l, start, const=True):
    """Least-squares AR(l) on x[start:], with a constant: (B, resid, XtX^-1).
    B is (1 + l m) x m, rows [const, Phi_1', ..., Phi_l']."""
    n = x.shape[0]
    Y = x[start:]
    cols = [np.ones(n - start)] if const else []
    cols += [x[start - j:n - j] for j in range(1, l + 1)]
    X = np.column_stack(cols) if cols else np.zeros((n - start, 0))
    XtXi = np.linalg.pinv(X.T @ X) if X.shape[1] else np.zeros((0, 0))
    B = XtXi @ X.T @ Y if X.shape[1] else np.zeros((0, x.shape[1]))
    return B, Y - X @ B, XtXi


def stepwise_ar(x, L):
    """Tiao and Box's (1981) stepwise autoregression [§4.1], l = 1..L.

    Every AR(l) is fitted by multivariate least squares, with a constant, on
    the COMMON sample t = L+1..n, so that the determinants compare; then

    * P[l-1] = Phi_l of the AR(l), the partial autoregression matrix, and
      T[l-1] its t-ratios (their indicator symbols use +-2);
    * M[l-1] = -(N - 1/2 - l m) ln(|S(l)| / |S(l-1)|) (4.3), N = n - L - 1
      the effective number of observations with a constant, asymptotically
      chi-squared with m^2 d.f. under Phi_l = 0; pvalue its tail;
    * Sigma[l-1] = S(l)/(n - L), the residual covariance matrix, sigma[l-1]
      its diagonal and det_sigma its determinant — their Table 14 to the
      printed digit on the gas furnace (the t-ratios use the residual degrees
      of freedom instead, as least squares does);
    * aic[l-1] = N ln|S(l)/N| + 2 l m^2, as information only.

    This reproduces their gas furnace M(l) (Table 12(b)) to the printed digit
    for l = 1..8; see the test. Phi_all[l-1] and T_all[l-1] hold every
    coefficient matrix of the AR(l) and its t-ratios (their Table 14, the
    successive fits). Returns a dict of arrays, and n_eff = n - L
    observations used."""
    from scipy.stats import chi2
    x = np.asarray(x, float)
    n, m = x.shape
    L = int(L)
    T = n - L
    N = n - L - 1
    _B, e0, _ = _ls_var(x, 0, L)
    S_prev = e0.T @ e0
    out = {k: [] for k in ("P", "T", "M", "pvalue", "sigma", "det_sigma", "aic")}
    out["Sigma"], out["Phi_all"], out["T_all"] = [], [], []
    for l in range(1, L + 1):
        B, e, XtXi = _ls_var(x, l, L)
        S = e.T @ e
        dof = T - (1 + l * m)
        Sig = S / T
        r0 = 1 + (l - 1) * m
        Phi = B[r0:r0 + m].T                       # (i, j): x_j at lag l on x_i
        se = np.sqrt(np.outer(np.diag(S / dof), np.diag(XtXi)[r0:r0 + m]))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(se > 0, Phi / se, 0.0)
        M = -(N - 0.5 - l * m) * np.log(np.linalg.det(S) / np.linalg.det(S_prev))
        out["P"].append(Phi)
        out["T"].append(t)
        out["M"].append(M)
        out["pvalue"].append(float(chi2.sf(M, m * m)))
        out["Sigma"].append(Sig)
        sea = np.sqrt(np.outer(np.diag(S / dof), np.diag(XtXi)[1:]))
        allP = B[1:].T                                         # m x l m
        out["Phi_all"].append(np.array([allP[:, j * m:(j + 1) * m] for j in range(l)]))
        out["T_all"].append(np.array([(allP / sea)[:, j * m:(j + 1) * m] for j in range(l)]))
        out["sigma"].append(np.diag(Sig))
        out["det_sigma"].append(float(np.linalg.det(Sig)))
        out["aic"].append(float(N * np.log(np.linalg.det(S / N)) + 2 * l * m * m))
        S_prev = S
    res = {k: (v if k in ("Phi_all", "T_all") else np.array(v)) for k, v in out.items()}
    res["df"] = m * m
    res["n_eff"] = T
    return res


def stepwise_order(sw, alpha=0.05):
    """Tiao and Box's reading of the M(l) column: the order beyond which no
    M(l) is significant at `alpha` — "for l > 2 ... the M(l) statistic fails
    to show significant improvement" [§5.1]; on the gas furnace AR(6), with
    M(5) not significant on the way. Returns (p, gaps): p the last significant
    l (0 if none) and the non-significant l below it, which the analyst reads
    (a gap is not a reason to stop: the gas furnace's delay of 3 shows at
    l = 4, and its input's AR(3) at l = 3, Table 14)."""
    sig = [l + 1 for l, pv in enumerate(sw["pvalue"]) if pv < alpha]
    p = max(sig) if sig else 0
    return p, [l for l in range(1, p) if l not in sig]


# --------------------------------------------------------------------------- #
#  Box and Tiao (1977): the canonical analysis                                #
# --------------------------------------------------------------------------- #

def canonical_from_moments(G0, Gpred):
    """The canonical analysis from the variance of the series G0 = Gamma_0(z)
    and that of its predictable part Gpred = Gamma_0(z_hat) (2.1)-(2.5):
    the eigenvalues lam of G0^-1 Gpred, ascending (least predictable first),
    and the rows of M, scaled so that M G0 M' = I (3.7): every component has
    unit variance and lam_j is its predictable share (2.2). Returns (lam, M)."""
    from scipy.linalg import eigh
    G0 = np.asarray(G0, float)
    Gp = np.asarray(Gpred, float)
    Gp = (Gp + Gp.T) / 2
    lam, V = eigh(Gp, (G0 + G0.T) / 2)            # V' G0 V = I, ascending
    return lam, V.T


def canonical(x, p):
    """Box and Tiao's (1977) canonical analysis of x (n x m) under a VAR(p)
    fitted by least squares with a constant.

    Returns a dict:
    * lam: the predictabilities, ascending, in [0, 1]; near 0, a (nearly)
      white component — a static relation among the series (2.9); near 1, a
      nearly non-stationary one — their common growth (§3.2, for p = 1 if and
      only if roots of Phi approach the unit circle);
    * M: the rows m_j' (component y_jt = m_j' (x_t - mean)), unit-variance
      scaling M G0 M' = I;
    * weights: each row normalised to its largest absolute element 1, for
      reading the combination;
    * components: the series y_t = M (x_t - mean), n_eff x m;
    * phi_bar and shares, for p = 1 only: the transformed AR matrix
      M Phi M^-1, whose j-th row sums in squares to lam_j (3.8), and the
      proportional contributions (Table 4.3): shares[j, i] = phi_bar[j, i]^2
      from component i's past, and shares[j, m] = 1 - lam_j from its shock.

    A reading, not a test: lam near 1 also arises from a nearly singular
    innovation covariance (§5.1), and lam near 0 from exact identities in the
    data (§5.2). Johansen's reduced-rank tests are the formal successors."""
    x = np.asarray(x, float)
    n, m = x.shape
    p = int(p)
    B, e, _ = _ls_var(x, p, p)
    Y = x[p:]
    fit = Y - e
    mu = Y.mean(0)
    T = Y.shape[0]
    G0 = (Y - mu).T @ (Y - mu) / T
    Gp = (fit - fit.mean(0)).T @ (fit - fit.mean(0)) / T
    lam, M = canonical_from_moments(G0, Gp)
    lam = np.clip(lam, 0.0, 1.0)
    W = M / np.abs(M).max(axis=1, keepdims=True)
    out = {"lam": lam, "M": M, "weights": W, "components": (Y - mu) @ M.T,
           "p": p, "n_eff": T}
    if p == 1:
        Phi = B[1:1 + m].T
        pb = M @ Phi @ np.linalg.inv(M)
        sh = np.zeros((m, m + 1))
        sh[:, :m] = pb ** 2
        sh[:, m] = 1.0 - lam
        out["phi_bar"] = pb
        out["shares"] = sh
    return out
