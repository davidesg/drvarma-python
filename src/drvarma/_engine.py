"""Bridge from Python to the cffi-compiled drvarma C estimator.

`estimate_w(...)` estimates a VARMA(p,q) on an already-transformed stationary
series `w` (shape nobs x m) via the C engine and returns a result dict.  When the
C extension is not built it falls back to the pure-Python estimator
(`estimate_py`, exact VAR only) — mirroring fue's `_engine.py`.
"""
import os
import numpy as np


def estimate_w(w, p, q, include_mean=False,
               diag_ar=False, diag_ma=False, diag_cov=False,
               method=1, twostep=False, maxits=0, grtol=0.0, sptol=0.0):
    """Estimate VARMA(p,q) on the stationary series w (nobs x m).

    Uses the compiled C engine when available; otherwise falls back to the
    pure-Python exact-ML estimator (`estimate_py.estimate_w_py`, q=0 only).
    """
    # Runtime opt-out: DRVARMA_NO_ENGINE forces the pure-Python estimator (e.g. to
    # dodge a C-engine issue) without rebuilding.  Otherwise use the C engine if
    # built, falling back to pure-Python when it is not.
    ffi = lib = None
    if not os.environ.get("DRVARMA_NO_ENGINE"):
        try:
            from drvarma._drvarma_engine import ffi, lib
        except ImportError:
            ffi = lib = None
    if lib is None:
        from .estimate_py import estimate_w_py
        return estimate_w_py(w, p, q, include_mean=include_mean,
                             diag_ar=diag_ar, diag_ma=diag_ma, diag_cov=diag_cov,
                             method=method, twostep=twostep)

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
