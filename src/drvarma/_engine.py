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
