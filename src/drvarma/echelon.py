"""The echelon form's index search and its least-squares estimates:
Lutkepohl and Poskitt (1996, section 3), stages I and II.

The model is the reversed echelon form of ``Ladder(echelon=n)``,

    A(B) w_t = M(B) a_t,   A(z) = A_0 + A_1 z + ...,   M_0 = A_0,

with A_0 lower triangular, unit diagonal, and LP's index table (their 2.3)

    n_rc = min(n_r + 1, n_c)  for c < r,     min(n_r, n_c)  for c >= r.

Row r: a_rr at lags 1..n_r; a_rc (r != c) at lags n_r - n_rc + 1 .. n_r
(lag 0, i.e. A_0, only when n_rc = n_r + 1); m_rc at lags 1..n_r.

- **Stage I**: a long VAR(h) by least squares, h by AIC; its residuals are
  the proxies of the innovations.
- **Stage II**: for each candidate n, least squares row by row on

      w_t = (I - A_0)(w_t - e_t) - SUM A_k w_t-k + SUM M_k e_t-k + a_t,

  and the criterion log|Sigma(n)| + c_T dim(n) / T, c_T = 2 log log T (HQ)
  or log T (SC). Each candidate then gets one round of Hannan and
  Kavalieris' (1984) correction: its residuals recomputed from its own
  stage II model, and stage II again.
- **The search**: every n in {0..n_max}^m while there are at most 256;
  beyond that Hannan and Kavalieris' two steps, about m n_max fits: all
  indices equal, 0..n_max; then, from the best, each index in turn lowered
  while the criterion improves.

A and M are in LP's convention (the Ladder's coefficients are -A, -M).
The stage II estimates are the start of stage III, exact ML
(``Ladder(echelon=n, start="echelon")``). docs/ECHELON_PLAN.md, stage 3b.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

CRITERIA = ("HQ", "SC")


# -- the structure ------------------------------------------------------------ #

def index_table(n):
    m = len(n)
    return [[(min(n[r] + 1, n[c]) if c < r else min(n[r], n[c])) for c in range(m)]
            for r in range(m)]


def free_positions(n):
    """The free coefficients: lists of (k, r, c) for A (k >= 0; k = 0 is
    A_0's) and for M (k >= 1)."""
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


def dim(n):
    a, mm = free_positions(n)
    return len(a) + len(mm)


def standard(A, M):
    """Phi_k = -A_0^-1 A_k, Theta_k = -A_0^-1 M_k: the standard form in
    drvarma's convention (w_t = SUM Phi w_t-k + a_t - SUM Theta a_t-k)."""
    inv = np.linalg.inv(A[0])
    return ([-inv @ A[k] for k in range(1, len(A))],
            [-inv @ M[k] for k in range(1, len(M))])


def max_inverse_root(C):
    """The largest inverse root of I - SUM C_k z^k (C: k x m x m), by the
    companion matrix; 0 for no lags."""
    C = [np.asarray(c, float) for c in C]
    if not C:
        return 0.0
    m, p = C[0].shape[0], len(C)
    K = np.zeros((m * p, m * p))
    K[:m] = np.hstack(C)
    K[m:, :-m] = np.eye(m * (p - 1))
    return float(np.abs(np.linalg.eigvals(K)).max())


def pull_inside(A, M, rho=0.95):
    """A, M with A_k scaled by lambda_A^k and M_k by lambda_M^k (k >= 1),
    lambda = rho / (the largest inverse root) where that root is >= rho:
    every inverse root of the AR (or the MA) is scaled by exactly lambda.
    Least squares can leave the MA outside the invertibility region, which
    elf does not accept as a start. Returns (A, M, lambda_A, lambda_M)."""
    phi, theta = standard(A, M)
    lam = []
    for C in (phi, theta):
        r = max_inverse_root(C)
        lam.append(rho / r if r >= rho else 1.0)
    A, M = A.copy(), M.copy()
    for k in range(1, len(A)):
        A[k] *= lam[0] ** k
        M[k] *= lam[1] ** k
    return A, M, lam[0], lam[1]


# -- stage I ---------------------------------------------------------------- #

def _var_ls(w, h, t0):
    """VAR(h) by least squares on t = t0..T-1: (residuals, Sigma)."""
    T = w.shape[0]
    y = w[t0:]
    if h == 0:
        e = y - y.mean(0)
    else:
        X = np.hstack([np.ones((T - t0, 1))] + [w[t0 - k:T - k] for k in range(1, h + 1)])
        B, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ B
    return e, e.T @ e / e.shape[0]


def long_var(w, h_max=None):
    """Stage I: (h, residuals e_h..e_T-1, AIC table). h by AIC on the common
    sample t >= h_max; the residuals are those of the chosen VAR(h) on its
    own sample t >= h."""
    w = np.asarray(w, float)
    T, m = w.shape
    if h_max is None:
        h_max = int(math.log(T) ** 1.5)
    h_max = max(1, min(int(h_max), (T - 1) // (m + 1) - 1))
    aic = []
    for h in range(1, h_max + 1):
        _e, S = _var_ls(w, h, h_max)
        aic.append(math.log(np.linalg.det(S)) + 2.0 * h * m * m / (T - h_max))
    h = 1 + int(np.argmin(aic))
    e, _S = _var_ls(w, h, h)
    return h, e, aic


# -- stage II --------------------------------------------------------------- #

@dataclass
class StageII:
    """The least-squares fit of one echelon form."""
    n: tuple
    A: np.ndarray            # (p + 1, m, m), A[0] = A_0
    M: np.ndarray            # (p + 1, m, m), M[0] = A_0
    sigma: np.ndarray        # the residual covariance
    logdet: float
    dim: int                 # the free coefficients
    T: int                   # the observations of the regression

    def criterion(self, name="HQ"):
        cT = 2.0 * math.log(math.log(self.T)) if name == "HQ" else math.log(self.T)
        return self.logdet + cT * self.dim / self.T


def stage2(w, e, h, n, t0=None, exclude=()):
    """Stage II for indices n: row-by-row least squares on the demeaned w
    and the stage I residuals e (``long_var``'s: e[t - h] is the residual of
    time t). ``t0`` is the first time used (default h + max n: pass the same
    for every candidate to compare criteria). ``exclude``: positions
    ("A", k, r, c) or ("M", k, r, c) held at zero (links, zeros)."""
    w = np.asarray(w, float)
    w = w - w.mean(0)
    T, m = w.shape
    n = tuple(int(k) for k in n)
    p = max(n) if n else 0
    t0 = h + p if t0 is None else int(t0)
    if t0 < h + p:
        raise ValueError(f"t0 {t0} < h + max n = {h + p}")
    ex = set(exclude)
    pa, pm = free_positions(n)
    ts = np.arange(t0, T)
    E = np.full((T, m), np.nan)
    E[h:] = e
    A = np.zeros((p + 1, m, m)); A[0] = np.eye(m)
    M = np.zeros((p + 1, m, m))
    U = np.empty((ts.size, m))
    d = 0
    for r in range(m):
        cols, keys = [], []
        for (k, rr, c) in pa:
            if rr != r or ("A", k, r, c) in ex:
                continue
            cols.append(w[ts, c] - E[ts, c] if k == 0 else w[ts - k, c])
            keys.append(("A", k, c))
        for (k, rr, c) in pm:
            if rr != r or ("M", k, r, c) in ex:
                continue
            cols.append(E[ts - k, c])
            keys.append(("M", k, c))
        y = w[ts, r]
        d += len(cols)
        if cols:
            X = np.column_stack(cols)
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            U[:, r] = y - X @ b
        else:
            b = []
            U[:, r] = y
        for (kind, k, c), v in zip(keys, b):
            if kind == "A":                  # w_t = -A_k w_t-k, (I - A_0)(w - e)
                A[k, r, c] = -v
            else:
                M[k, r, c] = v
    M[0] = A[0]
    S = U.T @ U / ts.size
    sign, ld = np.linalg.slogdet(S)
    return StageII(n, A, M, S, ld if sign > 0 else -np.inf, d, int(ts.size))


def residuals(w, A, M):
    """The innovations of the echelon model on the demeaned w, by the
    recursion A_0 e_t = A_0 w_t + SUM A_k w_t-k - SUM M_k e_t-k, with zero
    pre-sample values (T x m)."""
    w = np.asarray(w, float)
    w = w - w.mean(0)
    T, m = w.shape
    p = len(A) - 1
    inv = np.linalg.inv(A[0])
    Ak = [inv @ A[k] for k in range(1, p + 1)]
    Mk = [inv @ M[k] for k in range(1, p + 1)]
    e = np.zeros_like(w)
    for t in range(T):
        s = w[t].copy()
        for k in range(1, min(p, t) + 1):
            s += Ak[k - 1] @ w[t - k] - Mk[k - 1] @ e[t - k]
        e[t] = s
    return e


def stage2_refined(w, e, h, n, t0=None, exclude=(), iterate=1):
    """Stage II, then ``iterate`` rounds of Hannan and Kavalieris' (1984)
    correction: the residuals recomputed from the model just fitted (its
    roots pulled inside first, ``pull_inside``), and stage II again with
    them in place of the long VAR's. ``iterate=0`` is plain stage II."""
    f = stage2(w, e, h, n, t0=t0, exclude=exclude)
    for _ in range(int(iterate)):
        A, M, _la, _lm = pull_inside(f.A, f.M)
        f = stage2(w, residuals(w, A, M)[h:], h, n, t0=t0, exclude=exclude)
    return f


# -- the search ------------------------------------------------------------- #

@dataclass
class Search:
    """The chosen indices, every fit the search made, and stage I."""
    indices: tuple
    criterion: str
    fits: dict = field(default_factory=dict)      # n -> StageII
    h: int = 0
    aic: list = field(default_factory=list)
    steps: list = field(default_factory=list)     # (step, n) in the order tried

    @property
    def best(self):
        return self.fits[self.indices]

    def table(self):
        """Rows (n, dim, logdet, HQ, SC), in the order tried."""
        return [(n, f.dim, f.logdet, f.criterion("HQ"), f.criterion("SC"))
                for n, f in self.fits.items()]

    def text(self, names=None):
        """The criterion table, as sima prints it."""
        m = len(self.indices)
        names = names or [f"y{i + 1}" for i in range(m)]
        out = [f"Echelon index search (Lutkepohl-Poskitt 1996, stages I-II): "
               f"long VAR({self.h}) by AIC, then least squares per candidate",
               f"Variables, in order: {', '.join(names)}  "
               f"(the order is part of the echelon form)", "",
               f"{'indices':>12} {'dim':>5} {'log|S|':>10} {'HQ':>10} {'SC':>10}"]
        for n, d, ld, hq, sc in self.table():
            mark = "  <-" if n == self.indices else ""
            out.append(f"{str(n):>12} {d:5d} {ld:10.4f} {hq:10.4f} {sc:10.4f}{mark}")
        out += ["", f"Chosen by {self.criterion}: {self.indices}"]
        return "\n".join(out)


FULL_MAX = 256          # candidates up to which the search enumerates them all


def search(w, n_max=None, criterion="SC", h_max=None, full=None, iterate=1):
    """Choose the Kronecker indices of the stationary vector w (T x m).

    n_max defaults to 3 (LP's examples stop there). ``full`` enumerates
    every n in {0..n_max}^m ((n_max + 1)^m fits) instead of Hannan and
    Kavalieris' two steps; None (default) enumerates while that is at most
    FULL_MAX. ``iterate``: the rounds of the residual correction per
    candidate (``stage2_refined``).

    The defaults -- SC, one round, full enumeration -- are what
    docs/STUDY-echelon-search.md measured best: the only setting whose hit
    rate grows with the sample. HQ overfits."""
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}")
    w = np.asarray(w, float)
    T, m = w.shape
    n_max = 3 if n_max is None else int(n_max)
    h, e, aic = long_var(w, h_max)
    t0 = h + n_max
    if T - t0 < 2 * dim((n_max,) * m) // m + 10:
        raise ValueError(f"{T} observations are too few for n_max {n_max} with a "
                         f"VAR({h}) first stage")
    if full is None:
        full = (n_max + 1) ** m <= FULL_MAX
    out = Search((), criterion, h=h, aic=aic)

    def fit(n, step):
        n = tuple(n)
        if n not in out.fits:
            out.fits[n] = stage2_refined(w, e, h, n, t0=t0, iterate=iterate)
            out.steps.append((step, n))
        return out.fits[n].criterion(criterion)

    if full:
        for n in itertools.product(range(n_max + 1), repeat=m):
            fit(n, "full")
        out.indices = min(out.fits, key=lambda n: out.fits[n].criterion(criterion))
        return out
    best = min(range(n_max + 1), key=lambda k: fit((k,) * m, "a"))
    cur = [best] * m
    c_cur = fit(cur, "a")
    for i in range(m):
        while cur[i] > 0:
            trial = cur.copy(); trial[i] -= 1
            c = fit(trial, "b")
            if c < c_cur:
                cur, c_cur = trial, c
            else:
                break
    out.indices = tuple(cur)
    return out
