"""Bridge from Python to the cffi-compiled drvarma C estimator.

`estimate_w(...)` estimates a VARMA(p,q) on an already-transformed stationary
series `w` (shape nobs x m) via the C engine and returns a result dict.  When the
C extension is not built it falls back to the pure-Python estimator
(`estimate_py`, the full VARMA(p,q)) — mirroring fue's `_engine.py` — and says so
(BUG-0005).
"""
import os
import numpy as np


# -- Which engine computes (BUG-0005) ---------------------------------------- #
#
# Where no binary wheel exists pip installs the py3-none-any one, and every
# entry point here fell back to the pure-Python port in silence: same numbers,
# 20-250x slower, and the user did not know why. Now the first fallback in a
# process warns with the import error, and `engine_backend()` says which
# engine is in use. DRVARMA_NO_ENGINE is the
# user asking for the port: no warning then.

_C_ERROR = None           # the ImportError text when it does not load
_WARNED = False


def _load_c():
    """(ffi, lib) of the C extension, or None.

    Not cached: after the first time the import is a sys.modules lookup, and
    not caching keeps honest the tests that block the module to force the port.
    """
    global _C_ERROR
    try:
        from drvarma._drvarma_engine import ffi, lib
    except ImportError as e:
        _C_ERROR = f"{type(e).__name__}: {e}"
        return None
    _C_ERROR = None
    return ffi, lib


def engine_backend() -> str:
    """"c" when the compiled engine loads, "python" when drvarma falls back to
    the pure-Python port (or DRVARMA_NO_ENGINE asks for it)."""
    if os.environ.get("DRVARMA_NO_ENGINE"):
        return "python"
    return "c" if _load_c() is not None else "python"


def engine_load_error():
    """Why the C engine did not load (the ImportError), or None."""
    _load_c()
    return _C_ERROR


def _warn_python_fallback(what="", stacklevel=3):
    """Once per process; never when DRVARMA_NO_ENGINE asked for the port."""
    global _WARNED
    if _WARNED or os.environ.get("DRVARMA_NO_ENGINE"):
        return
    _WARNED = True
    import warnings
    warnings.warn(
        f"drvarma: the C engine did not load ({_C_ERROR}); "
        + (what or "computing with the pure-Python port") +
        ". Same results, 20-250x slower. Install a binary wheel (pip install "
        "drvarma) or build it (pip install 'drvarma[c-engine]'); "
        "DRVARMA_NO_ENGINE=1 asks for the port and silences this. "
        "drvarma.engine_backend() tells the engine in use (BUG-0005).",
        RuntimeWarning, stacklevel=stacklevel)


def estimate_w(w, p, q, include_mean=False,
               diag_ar=False, diag_ma=False, diag_cov=False,
               method=1, twostep=False, maxits=0, grtol=0.0, sptol=0.0):
    """Estimate VARMA(p,q) on the stationary series w (nobs x m).

    Uses the compiled C engine when available; otherwise falls back to the
    pure-Python exact-ML estimator (`estimate_py.estimate_w_py`), a faithful
    port of the C engine: same logelf and estimates to ~1e-10, std errors to
    ~1e-4 (the finite-difference Hessian).
    """
    # Runtime opt-out: DRVARMA_NO_ENGINE forces the pure-Python estimator (e.g. to
    # dodge a C-engine issue) without rebuilding.  Otherwise use the C engine if
    # built, falling back to pure-Python when it is not.
    c = None
    if not os.environ.get("DRVARMA_NO_ENGINE"):
        c = _load_c()
        if c is None:
            _warn_python_fallback()
    if c is None:
        from .estimate_py import estimate_w_py
        return estimate_w_py(w, p, q, include_mean=include_mean,
                             diag_ar=diag_ar, diag_ma=diag_ma, diag_cov=diag_cov,
                             method=method, twostep=twostep)
    ffi, lib = c

    w = np.ascontiguousarray(np.atleast_2d(np.asarray(w, dtype=np.float64)))
    nobs, m = w.shape

    spec = ffi.new("DrvarmaModelSpec *")
    lib.drvarma_defaults(spec)
    _wbuf = ffi.from_buffer("double[]", w.reshape(-1))
    spec.m = m
    spec.nobs = nobs
    spec.w = _wbuf
    spec.p = p
    spec.q = q
    spec.include_mean = 1 if include_mean else 0
    spec.diag_ar = 1 if diag_ar else 0
    spec.diag_ma = 1 if diag_ma else 0
    spec.diag_cov = 1 if diag_cov else 0
    spec.method = method
    spec.twostep = 1 if twostep else 0
    if maxits:
        spec.maxits = maxits
    if grtol:
        spec.grtol = grtol
    if sptol:
        spec.sptol = sptol

    res = lib.drvarma_estimate(spec)
    try:
        npar = res.npar
        rp, rq, rm = res.p, res.q, res.m
        out = {
            "ifault": res.ifault,
            "npar": npar,
            "m": rm,
            "p": rp,
            "q": rq,
            "sigma2": res.sigma2,
            "logelf": res.logelf,
            # Same keys as the pure-Python estimator, so the convergence
            # diagnosis works identically whichever engine ran. The C used to
            # omit them — `report()` announces the termcode to `outputv`, which
            # the binding points at /dev/null and `quiet_mode` silences anyway —
            # so `Model.termcode` was None on the DEFAULT path and every warning
            # built on it was inert.
            "termcode": res.termcode,
            "nit": res.nit,
            "params": np.frombuffer(ffi.buffer(res.params, npar * 8), float).copy()
                      if npar else np.zeros(0),
            "std_errors": np.frombuffer(ffi.buffer(res.std_errors, npar * 8), float).copy()
                          if npar else np.zeros(0),
            "cov": (np.frombuffer(ffi.buffer(res.cov_matrix, npar * npar * 8), float)
                    .reshape(npar, npar).copy() if npar else np.zeros((0, 0))),
            "residuals": np.frombuffer(ffi.buffer(res.residuals, nobs * m * 8), float)
                         .reshape(nobs, m).copy(),
            "mu": np.frombuffer(ffi.buffer(res.mu, rm * 8), float).copy(),
            "phi": (np.frombuffer(ffi.buffer(res.phi, rp * rm * rm * 8), float)
                    .reshape(rp, rm, rm).copy() if rp else np.zeros((0, rm, rm))),
            "theta": (np.frombuffer(ffi.buffer(res.theta, rq * rm * rm * 8), float)
                      .reshape(rq, rm, rm).copy() if rq else np.zeros((0, rm, rm))),
            "sigma": np.frombuffer(ffi.buffer(res.sigma, rm * rm * 8), float)
                     .reshape(rm, rm).copy(),
        }
    finally:
        lib.drvarma_result_free(res)
    _ = _wbuf  # keep buffer alive until here
    return out


# -- marma: Shea's exact likelihood at a GIVEN structure ------------------- #

def marma_c(m, n, p, q, mu, phi, theta, qq, w, sigma2=1.0):
    """Compiled Shea (1989, AS 242): the independent benchmark beside `elf`.

    Same 0-based arrays as `elf_c`. Returns `(logelf, f1, f2, ifault)`, with
    f1, f2 on elf's scale (the concentrated objective is (f1)^m * f2). Always
    exact; the MA frontier is elf's (ifault 4 where elf would refuse). No
    residuals: Shea's are the one-step innovations, and forecasts need elf's.
    The C engines' `-lik shea` (atsw-gui lib/lik) runs the same code.

    Falls back to the pure-Python port, `_as242.marma`, when the extension is
    not built (same returns; 1e-14 relative on the logL).
    """
    import numpy as np
    c = _load_c()
    if c is None:                                          # pragma: no cover
        _warn_python_fallback()
        from ._as242 import marma
        return marma(m, n, p, q, mu, phi, theta, qq, w, sigma2)
    ffi, lib = c

    mu = np.ascontiguousarray(mu, dtype=np.float64)
    qq = np.ascontiguousarray(qq, dtype=np.float64)
    w = np.ascontiguousarray(w, dtype=np.float64)
    phi = np.ascontiguousarray(phi, dtype=np.float64) if p else np.zeros((0, m, m))
    theta = np.ascontiguousarray(theta, dtype=np.float64) if q else np.zeros((0, m, m))
    f1 = ffi.new("double *"); f2 = ffi.new("double *"); lg = ffi.new("double *")
    ifault = lib.drvarma_marma(
        m, n, p, q,
        ffi.from_buffer("double[]", mu.ravel()),
        ffi.from_buffer("double[]", phi.ravel()) if p else ffi.NULL,
        ffi.from_buffer("double[]", theta.ravel()) if q else ffi.NULL,
        ffi.from_buffer("double[]", qq.ravel()),
        ffi.from_buffer("double[]", w.ravel()),
        float(sigma2), f1, f2, lg)
    return float(lg[0]), float(f1[0]), float(f2[0]), int(ifault)


# -- elf: the exact likelihood at a GIVEN structure ------------------------- #

def elf_c(m, n, p, q, mu, phi, theta, qq, w, sigma2=1.0, xitol=-1e-3, atf=False):
    """Compiled `elf`, evaluated at a structure the caller built.

    Same contract as `_as311.elf` but with **0-based, flat** arrays, which is
    what a caller naturally has: `mu` (m,), `phi` (p, m, m), `theta` (q, m, m),
    `qq` (m, m), `w` (n, m). Returns `(logelf, f1, f2, a, ifault)`, with `a`
    (n, m) filled only when `atf=True`.

    Why this exists: `estimate_w` fits a FREE VARMA(p, q); a restricted model —
    a transfer function, a network, anything whose structure comes from a cast —
    needs the likelihood *scored* at a given Phi/Theta/Sigma, and that could not
    be asked for through the estimate entry point. The pure-Python `_as311.elf`
    could, but it is ~250x slower, which is the difference between validating a
    six-series system and being able to work with it.

    Falls back to `_as311.elf` when the extension is not built.
    """
    import numpy as np

    mu = np.ascontiguousarray(mu, dtype=np.float64)
    qq = np.ascontiguousarray(qq, dtype=np.float64)
    w = np.ascontiguousarray(w, dtype=np.float64)
    phi = (np.ascontiguousarray(phi, dtype=np.float64) if p
           else np.zeros((0, m, m)))
    theta = (np.ascontiguousarray(theta, dtype=np.float64) if q
             else np.zeros((0, m, m)))

    c = _load_c()
    if c is None:                                          # pragma: no cover
        _warn_python_fallback()
        from ._as311 import elf as _elf_py
        Mu = np.zeros(m + 1); Mu[1:] = mu
        Phi = np.zeros((max(p, 1) + 1, m + 1, m + 1))
        for k in range(p):
            Phi[k + 1, 1:, 1:] = phi[k]
        Theta = np.zeros((max(q, 1) + 1, m + 1, m + 1))
        for k in range(q):
            Theta[k + 1, 1:, 1:] = theta[k]
        Qq = np.zeros((m + 1, m + 1)); Qq[1:, 1:] = qq
        W = np.zeros((n + 1, m + 1)); W[1:, 1:] = w
        return _elf_py(m, n, p, q, Mu, Phi, Theta, Qq, W, sigma2, xitol, atf)
    ffi, lib = c

    a_out = np.zeros(n * m, dtype=np.float64) if atf else None
    f1 = ffi.new("double *")
    f2 = ffi.new("double *")
    lg = ffi.new("double *")

    ifault = lib.drvarma_elf(
        m, n, p, q,
        ffi.from_buffer("double[]", mu.ravel()),
        ffi.from_buffer("double[]", phi.ravel()) if p else ffi.NULL,
        ffi.from_buffer("double[]", theta.ravel()) if q else ffi.NULL,
        ffi.from_buffer("double[]", qq.ravel()),
        ffi.from_buffer("double[]", w.ravel()),
        float(sigma2), float(xitol), 1 if atf else 0,
        ffi.from_buffer("double[]", a_out) if atf else ffi.NULL,
        f1, f2, lg)

    # 1-based (n+1, m+1) on the way out, to match `_as311.elf`
    a = np.zeros((n + 1, m + 1))
    if atf:
        a[1:, 1:] = a_out.reshape(n, m)
    return float(lg[0]), float(f1[0]), float(f2[0]), a, int(ifault)


# -- GSL's eigensolvers, from the compiled engine --------------------------- #
#
# drvec's Johansen start (canonical_b2) solves A v = l B v with GSL's
# gsl_eigen_gensymmv + gsl_eigen_gensymmv_sort, and its reports print the
# eigenvalues of a general matrix in gsl_eigen_nonsymm's order. Its Python port
# must call the same routines to be bit-identical with the C (scipy's `eigh` is
# a few ulps off, and on Lutkepohl's E1 that moves an optimum by 0.47 in logL).
# The engine already links GSL on every wheel platform, so it exports them and
# nobody has to look for a system libgsl.
#
# There is NO pure-Python fallback here, on purpose: what these functions
# promise is GSL's bits, and scipy's `eigh` would answer a different question
# in silence. Without the engine (no extension, or DRVARMA_NO_ENGINE) they
# raise EngineUnavailableError; `gsl_available()` asks first, and the caller
# chooses its own fallback.

class EngineUnavailableError(RuntimeError):
    """The compiled engine is not loaded (or DRVARMA_NO_ENGINE asks for the
    port), so GSL's routines are not available."""


GSL_EIGEN_SORT_VAL_ASC, GSL_EIGEN_SORT_VAL_DESC = 0, 1
GSL_EIGEN_SORT_ABS_ASC, GSL_EIGEN_SORT_ABS_DESC = 2, 3
_SORTS = {"val_asc": 0, "val_desc": 1, "abs_asc": 2, "abs_desc": 3, None: -1}


def gsl_available() -> bool:
    """True when `gensymmv` and `eigen_nonsymm` can run: the compiled engine
    loads, carries them, and DRVARMA_NO_ENGINE is not set."""
    if os.environ.get("DRVARMA_NO_ENGINE"):
        return False
    c = _load_c()
    return c is not None and hasattr(c[1], "drvarma_gensymmv")


def _gsl_lib():
    if os.environ.get("DRVARMA_NO_ENGINE"):
        raise EngineUnavailableError(
            "drvarma: DRVARMA_NO_ENGINE is set; GSL's eigensolvers come only "
            "from the compiled engine")
    c = _load_c()
    if c is None:
        raise EngineUnavailableError(
            f"drvarma: the compiled engine did not load ({_C_ERROR}); GSL's "
            "eigensolvers come only from it. drvarma.engine_backend() says "
            "which engine is in use.")
    if not hasattr(c[1], "drvarma_gensymmv"):
        raise EngineUnavailableError(
            "drvarma: the compiled engine predates GSL's eigensolvers "
            "(drvarma < 0.3.0); rebuild it")
    return c


def _square(X, name):
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != X.shape[1] or X.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix, got shape {X.shape}")
    return X


def gensymmv(A, B, sort="val_desc"):
    """GSL's generalized symmetric-definite eigensolver: A v = l B v.

    `gsl_eigen_gensymmv` on copies of A (symmetric) and B (symmetric positive
    definite), then `gsl_eigen_gensymmv_sort` with `sort` ("val_desc" — the
    default, GSL_EIGEN_SORT_VAL_DESC, what drvec's Johansen start uses —,
    "val_asc", "abs_asc", "abs_desc", or None for GSL's own order). GSL reads
    the lower triangle of A and B.

    Returns `(eval, evec)`: eval (n,), evec (n, n) with the eigenvectors as
    COLUMNS, of unit Euclidean norm as GSL leaves them (scipy's `eigh`
    B-normalises them instead). Bit for bit what a C program
    calling GSL gets, with the GSL this engine was linked against.

    Raises `numpy.linalg.LinAlgError` when GSL fails (B not positive
    definite), and `EngineUnavailableError` without the compiled engine.
    """
    if sort not in _SORTS:
        raise ValueError(f"sort must be one of {sorted(k for k in _SORTS if k)} or None")
    A, B = _square(A, "A"), _square(B, "B")
    n = A.shape[0]
    if B.shape != A.shape:
        raise ValueError(f"A {A.shape} and B {B.shape} differ in shape")
    ffi, lib = _gsl_lib()
    ev = np.empty(n, dtype=np.float64)
    V = np.empty((n, n), dtype=np.float64)
    st = lib.drvarma_gensymmv(n, ffi.from_buffer("double[]", A.ravel()),
                              ffi.from_buffer("double[]", B.ravel()),
                              ffi.from_buffer("double[]", ev),
                              ffi.from_buffer("double[]", V.reshape(-1)),
                              _SORTS[sort])
    if st != 0:
        raise np.linalg.LinAlgError(
            f"gsl_eigen_gensymmv failed (GSL status {st}): B is not positive definite"
            if st == 1 else f"gsl_eigen_gensymmv failed (GSL status {st})")
    return ev, V


def eigen_nonsymm(A, balance=True):
    """GSL's eigenvalues of a real general matrix, in GSL's order.

    `gsl_eigen_nonsymm` on a copy of A, without the Schur form, balanced first
    when `balance` (as the suite's C `eigenqr` does). Returns `(wr, wi)`, two
    (n,) arrays, the eigenvalues being wr[k] + i wi[k]. The order is GSL's,
    which is not numpy's: a report that prints them in that order needs this.

    Raises `numpy.linalg.LinAlgError` when the QR iteration does not converge
    (GSL_EMAXITER), and `EngineUnavailableError` without the compiled engine.
    """
    A = _square(A, "A")
    n = A.shape[0]
    ffi, lib = _gsl_lib()
    wr = np.empty(n, dtype=np.float64)
    wi = np.empty(n, dtype=np.float64)
    st = lib.drvarma_eigen_nonsymm(n, ffi.from_buffer("double[]", A.ravel()),
                                   1 if balance else 0,
                                   ffi.from_buffer("double[]", wr),
                                   ffi.from_buffer("double[]", wi))
    if st != 0:
        raise np.linalg.LinAlgError(f"gsl_eigen_nonsymm failed (GSL status {st})")
    return wr, wi
