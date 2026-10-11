"""Echelon-form VARMA systems with known Kronecker indices, for stage 0 of
docs/ECHELON_SEEDED_DESIGN.md: do the univariate orders predict the indices?

Lutkepohl and Poskitt (1996, section 2), the reversed echelon form

    A(L) y_t = M(L) a_t,    A(z) = A_0 + A_1 z + ...,   M_0 = A_0,

with A_0 lower triangular, unit diagonal, and the index table (their 2.3)

    n_rc = min(n_r + 1, n_c)  for c < r,     min(n_r, n_c)  for c >= r.

Row r: a_rr has its free coefficients at lags 1..n_r; a_rc (r != c) at lags
n_r - n_rc + 1 .. n_r (lag 0 only when n_rc = n_r + 1); m_rc at lags
1..n_r (lag 0 is A_0's).

Three things are computed from a system, all exactly (no sampling):
  - the Kronecker indices of its impulse response (a check that the draw
    has the indices it was built with);
  - the standard form, Phi_k = -A_0^{-1} A_k and Theta_k = -A_0^{-1} M_k,
    in drvarma.datasets.simulate_varma's sign convention
    (w_t = sum Phi w_{t-k} + a_t - sum Theta a_{t-k});
  - the orders of each component's MARGINAL ARMA, by Cramer's rule:
    det A(z) y_i = sum_j [adj A(z) M(z)]_ij a_j, common roots cancelled.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import polynomial as P

# -- the index table and the free positions --------------------------------- #

def index_table(n):
    m = len(n)
    return [[(min(n[r] + 1, n[c]) if c < r else min(n[r], n[c])) for c in range(m)]
            for r in range(m)]


def free_positions(n):
    """The free coefficients: lists of (k, r, c) for A (k >= 0) and M (k >= 1)."""
    m, nt = len(n), index_table(n)
    a, mm = [], []
    for r in range(m):
        for c in range(m):
            if r == c:
                a += [(k, r, r) for k in range(1, n[r] + 1)]
            else:
                a += [(k, r, c) for k in range(n[r] - nt[r][c] + 1, n[r] + 1)]
            mm += [(k, r, c) for k in range(1, n[r] + 1)]
    return a, mm


def build(n, a_vals, m_vals):
    """A and M as (deg+1, m, m) arrays from the free values."""
    m, p = len(n), max(n)
    A = np.zeros((p + 1, m, m))
    M = np.zeros((p + 1, m, m))
    A[0] = np.eye(m)
    pa, pm = free_positions(n)
    for (k, r, c), v in zip(pa, a_vals):
        A[k, r, c] = v
    M[0] = A[0]
    for (k, r, c), v in zip(pm, m_vals):
        M[k, r, c] = v
    return A, M


def standard(A, M):
    """Phi_1..p, Theta_1..p in simulate_varma's convention."""
    inv = np.linalg.inv(A[0])
    return ([-inv @ A[k] for k in range(1, len(A))],
            [-inv @ M[k] for k in range(1, len(M))])

# -- polynomial matrices ----------------------------------------------------- #

def _poly(Amat, r, c):
    return np.trim_zeros(Amat[:, r, c], "b") if np.any(Amat[:, r, c]) else np.zeros(1)


def _det(polys):
    """Determinant of a square list-of-lists of polynomials (m <= 3)."""
    m = len(polys)
    if m == 1:
        return polys[0][0]
    out = np.zeros(1)
    for c in range(m):
        minor = [[polys[i][j] for j in range(m) if j != c] for i in range(1, m)]
        term = P.polymul(polys[0][c], _det(minor))
        out = P.polyadd(out, term) if c % 2 == 0 else P.polysub(out, term)
    return out


def _adj(polys):
    m = len(polys)
    if m == 1:
        return [[np.ones(1)]]
    adj = [[None] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            minor = [[polys[r][c] for c in range(m) if c != j] for r in range(m) if r != i]
            d = _det(minor)
            adj[j][i] = d if (i + j) % 2 == 0 else -d
    return adj


def _deg(p, tol=1e-10):
    p = np.atleast_1d(p)
    nz = np.nonzero(np.abs(p) > tol * max(1.0, np.max(np.abs(p))))[0]
    return int(nz[-1]) if nz.size else -1


def det_poly(A):
    m = A.shape[1]
    return _det([[_poly(A, r, c) for c in range(m)] for r in range(m)])


def marginal_orders(A, M, tol=1e-7):
    """(p*, q*) of each component's marginal ARMA, exactly: det A(z) and the
    row of adj(A) M, with their common roots cancelled."""
    m = A.shape[1]
    Ap = [[_poly(A, r, c) for c in range(m)] for r in range(m)]
    Mp = [[_poly(M, r, c) for c in range(m)] for r in range(m)]
    d = _det(Ap)
    adj = _adj(Ap)
    out = []
    for i in range(m):
        C = []
        for j in range(m):
            s = np.zeros(1)
            for k in range(m):
                s = P.polyadd(s, P.polymul(adj[i][k], Mp[k][j]))
            C.append(s)
        dd = _deg(d)
        roots = P.polyroots(d[:dd + 1]) if dd > 0 else np.array([])
        cancel = 0
        for z in roots:
            scale = max(np.max(np.abs(c)) for c in C) or 1.0
            if max(abs(P.polyval(z, c)) for c in C) < tol * scale * (1 + abs(z)) ** 6:
                cancel += 1
        out.append((dd - cancel, max(_deg(c) for c in C) - cancel))
    return out

# -- Kronecker indices of the impulse response ------------------------------ #

def psi_weights(phi, theta, K):
    m = (phi or theta)[0].shape[0]
    psi = [np.eye(m)]
    for k in range(1, K + 1):
        s = -theta[k - 1] if k <= len(theta) else np.zeros((m, m))
        for i in range(1, min(k, len(phi)) + 1):
            s = s + phi[i - 1] @ psi[k - i]
        psi.append(s)
    return psi


def kronecker_indices(phi, theta, K=None, tol=1e-8):
    """From the block Hankel [Psi_{i+j-1}]: the first linearly independent
    rows, taken in order (block row by block row); n_r counts variable r's."""
    m = (phi or theta)[0].shape[0]
    p = max(len(phi), len(theta))
    K = K or (m * p + 2)
    psi = psi_weights(phi, theta, 2 * K)
    H = np.block([[psi[i + j + 1] for j in range(K)] for i in range(K)])
    sel, n = [], [0] * m
    base_rank = 0
    for i in range(K):
        for r in range(m):
            cand = sel + [i * m + r]
            rk = np.linalg.matrix_rank(H[cand], tol=tol * np.linalg.norm(H))
            if rk > base_rank:
                sel, base_rank = cand, rk
                n[r] += 1
    return n

# -- drawing a system --------------------------------------------------------- #

def inverse_root_max(A):
    """Largest modulus of the inverse roots of det A(z) (0 if constant)."""
    d = det_poly(A)
    dd = _deg(d)
    if dd <= 0:
        return 0.0
    return float(np.max(1.0 / np.abs(P.polyroots(d[:dd + 1]))))


def draw(n, seed, lo=0.2, hi=0.7, cross=1.0, rmax=0.85, tries=20000):
    """A system with Kronecker indices n: free values uniform in
    +-[lo, hi] (the off-diagonal ones, A_0's included, times `cross`), AR and
    MA inverse roots at most rmax, and the indices of its impulse response
    equal to n (left coprime, row reduced)."""
    rng = np.random.default_rng(seed)
    pa, pm = free_positions(n)
    sa = np.array([1.0 if r == c else cross for (_, r, c) in pa])
    sm = np.array([1.0 if r == c else cross for (_, r, c) in pm])
    for _ in range(tries):
        av = sa * rng.uniform(lo, hi, len(pa)) * rng.choice([-1, 1], len(pa))
        mv = sm * rng.uniform(lo, hi, len(pm)) * rng.choice([-1, 1], len(pm))
        A, M = build(n, av, mv)
        if inverse_root_max(A) > rmax or inverse_root_max(M) > rmax:
            continue
        phi, theta = standard(A, M)
        if kronecker_indices(phi, theta) != list(n):
            continue
        return A, M
    raise RuntimeError(f"no system with indices {n} in {tries} tries")
