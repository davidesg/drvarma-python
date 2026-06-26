"""Conditional volatility of VARMA residuals (port of ``volatility.c``).

Two methods, matching the C engine byte-for-byte:

* **Exponential** (``-volexp``, ``.volexp``): ``H_t = Σ_{k=0}^{w-1} φ(1-φ)^k
  ε_{t-k}ε_{t-k}'`` with weights renormalised to sum 1.  ``φ`` is estimated as the
  proportion of residual Mahalanobis distances (w.r.t. the unconditional
  ``Σ = σ²·Q``) exceeding the empirical ``100·(1-α)`` percentile.
* **Moving window** (``-volmov``, ``.volmov``): the unbiased sample covariance of
  the residuals over the trailing ``window`` observations, for ``t = window..n``.

Both writers emit ``t var1..varm cov12 cov13 .. cov(m-1)m`` with C ``%g``
formatting.

Authors of the C original: A. Garcia-Hiernaux, M.T. Gonzalez-Perez, D.E. Guerrero.
License: GPL-2.0-or-later
"""

import numpy as np

DEFAULT_ALPHA = 0.05
DEFAULT_WINDOW = 20


def _g(x):
    """C printf ``%g`` of a float."""
    return "%g" % x


def mahalanobis(res, sigma):
    """Mahalanobis distances ``d_t = ε_t' Σ^{-1} ε_t`` (res is (n, m))."""
    res = np.asarray(res, float)
    sinv = np.linalg.inv(np.asarray(sigma, float))
    return np.einsum("ti,ij,tj->t", res, sinv, res)


def estimate_phi(d, alpha):
    """Return (phi, threshold): φ = #{d_t > thr}/n, thr the (1-α) percentile.

    Mirrors ``estimate_phi``: ``idx = (int)((1-α)·n)`` clamped to ``[1, n]`` (1-based),
    ``thr = sorted(d)[idx]``, ``φ = count(d > thr)/n``.
    """
    d = np.asarray(d, float)
    n = d.size
    ds = np.sort(d)
    idx = int((1.0 - alpha) * n)          # truncation, as C (int) cast
    if idx < 1:
        idx = 1
    if idx > n:
        idx = n
    threshold = ds[idx - 1]               # 1-based d_sorted[idx]
    count = int(np.count_nonzero(d > threshold))
    return count / n, threshold


def exponential_volatility(res, sigma, alpha=DEFAULT_ALPHA, window=DEFAULT_WINDOW):
    """Exponential-weight conditional covariances.

    Returns (phi, threshold, H) where H is an (n, m, m) array (H[t-1] = H_t).
    """
    res = np.asarray(res, float)
    n, m = res.shape
    d = mahalanobis(res, sigma)
    phi, threshold = estimate_phi(d, alpha)

    w = np.array([phi * (1.0 - phi) ** k for k in range(window)])
    sumw = w.sum()
    if not (sumw > 0.0):                  # phi == 0 (or NaN): equal weights
        w = np.full(window, 1.0 / window)
    else:
        w = w / sumw

    H = np.zeros((n, m, m))
    for t in range(1, n + 1):
        maxlag = min(t - 1, window - 1)
        acc = np.zeros((m, m))
        for k in range(maxlag + 1):
            e = res[t - 1 - k]            # res[t-k] (1-based) -> res[t-1-k] (0-based)
            acc += w[k] * np.outer(e, e)
        H[t - 1] = acc
    return phi, threshold, H


def moving_window_volatility(res, window=DEFAULT_WINDOW):
    """Unbiased sample covariance over the trailing `window` residuals.

    Returns (t_index, H) where t_index is the 1-based times (window..n) and
    H[k] is the covariance at t_index[k].
    """
    res = np.asarray(res, float)
    n, m = res.shape
    if window > n:
        raise ValueError("moving window (%d) > number of observations (%d)"
                         % (window, n))
    if window < 2:
        raise ValueError("moving window must be at least 2")
    times = list(range(window, n + 1))
    H = np.zeros((len(times), m, m))
    for r, t in enumerate(times):
        block = res[t - window:t]         # res[t-window+1 .. t] (1-based)
        mean = block.mean(axis=0)
        c = block - mean
        H[r] = (c.T @ c) / (window - 1)
    return times, H


# --------------------------------------------------------------------------- #
#  file writers (byte-exact vs the C .volexp / .volmov)                       #
# --------------------------------------------------------------------------- #

def _header(m):
    cols = ["t"] + ["var%d" % i for i in range(1, m + 1)]
    cols += ["cov%d%d" % (i, j) for i in range(1, m + 1)
             for j in range(i + 1, m + 1)]
    return " ".join(cols) + "\n"


def _row(t, Hmat, m):
    parts = [str(t)]
    parts += [_g(Hmat[i, i]) for i in range(m)]
    parts += [_g(Hmat[i, j]) for i in range(m) for j in range(i + 1, m)]
    return " ".join(parts) + "\n"


def volexp_text(res, sigma, alpha=DEFAULT_ALPHA, window=DEFAULT_WINDOW):
    """Full ``.volexp`` text (and the estimated phi/threshold)."""
    res = np.asarray(res, float)
    n, m = res.shape
    phi, threshold, H = exponential_volatility(res, sigma, alpha, window)
    out = [_header(m)]
    for t in range(1, n + 1):
        out.append(_row(t, H[t - 1], m))
    return "".join(out), phi, threshold


def volmov_text(res, window=DEFAULT_WINDOW):
    """Full ``.volmov`` text."""
    res = np.asarray(res, float)
    m = res.shape[1]
    times, H = moving_window_volatility(res, window)
    out = [_header(m)]
    for r, t in enumerate(times):
        out.append(_row(t, H[r], m))
    return "".join(out)


def volexp_info_section(alpha, threshold, phi):
    """The ``.out`` info line the C appends for ``-volexp`` (byte-exact)."""
    return ("\nExponential volatility (rational inattention):\n"
            "  α = %.3f, threshold = %s, φ = %.4f\n"
            % (alpha, _g(threshold), phi))


def write_volexp(path, res, sigma, alpha=DEFAULT_ALPHA, window=DEFAULT_WINDOW):
    text, phi, threshold = volexp_text(res, sigma, alpha, window)
    with open(path, "w") as f:
        f.write(text)
    return phi, threshold


def write_volmov(path, res, window=DEFAULT_WINDOW):
    with open(path, "w") as f:
        f.write(volmov_text(res, window))
