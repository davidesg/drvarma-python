"""Standard errors from Mauricio's ``fdhess``: the curvature AT the optimum.

The covariance of the estimates is ``2 F H^-1 / n`` (drvmlest.c:est [3]), with
``F`` the objective at the optimum and ``H`` its second-derivative matrix. Which
``H`` is the whole question:

* **BFGS** (the default since the C of the 1990s): the Hessian ``raxopt``
  ACCUMULATED along the search. It steers the search; it is not the curvature at
  the optimum. It depends on the path (two starts, two sets of standard errors),
  it degrades in the flattest directions -- the ones with the largest standard
  errors -- and it is never built if the search starts at the optimum. Measured
  in fue (engines/fue/ERRORES_ESTANDAR.md: SE(mu) 0.073 against the exact GLS
  0.0285) and in drvarma (the 2026-09-27 review: t = -99 for a coefficient with
  t ~ -1.1 when two series have very different scales).
* **fdhess** (Mauricio's code, Dennis & Schnabel A5.6.2): finite differences at
  the optimum, forward, step ``eps^(1/3) max(|x|, 1)``. It is in the published
  code and was left COMMENTED OUT in every version of fue and drvarma; drtran
  enabled it. This module is the one place drvarma-python computes it, with
  ``_qnewt.fdhess`` -- the faithful port, not fue's ``_fdhess`` whose step was
  mistranslated (fue BUG-0015).

Two things make ``fdhess`` safe to use:

1. **The flat direction.** The likelihood concentrates ``sigma2``, so it is
   exactly invariant to ``Q -> cQ`` (Mauricio 1995, eq. 2.1): along that
   direction the Hessian is singular. The ladder removes it by construction
   (``Q11 = 1``). The ``.inp`` path estimates the raw lower triangle of Q, so
   here the direction is removed by holding ``qq[1,1]`` at its optimum while
   the Hessian is taken. Its standard error is then not reported (NaN): alone,
   it is not identified.
2. **The positive-definiteness guard.** The C's ``choldcp`` is a MODIFIED
   Cholesky: it patches small pivots and only fails on clearly negative ones,
   so a Hessian that is not positive definite still yields numbers. Here the
   check is a plain Cholesky. If it fails, the BFGS standard errors are kept
   and ``method`` says so: never a silent fallback.
"""
from __future__ import annotations

import numpy as np

from . import _qnewt

MACHEPS = float(np.finfo(float).eps)


def fd_covariance(objective, x_hat, n, fixed=None):
    """Covariance of the estimates from ``fdhess`` at ``x_hat``.

    Parameters
    ----------
    objective : callable, vector -> float
        The objective the optimiser minimised (any positive scaling of it: the
        constant cancels in ``2 F H^-1``).
    x_hat : the optimum.
    n : the number of observations in the likelihood.
    fixed : index held at its optimum while the Hessian is taken (the flat
        direction), or None.

    Returns
    -------
    (cov, std, info): ``cov`` and ``std`` are None if the Hessian is not
    positive definite. ``info`` = {"positive_definite", "min_eig", "cond"}.
    """
    x_hat = np.asarray(x_hat, float)
    idx = [i for i in range(x_hat.size) if i != fixed]
    k = len(idx)
    if k == 0:
        return np.zeros((0, 0)), np.zeros(0), {"positive_definite": True,
                                              "min_eig": float("nan"), "cond": float("nan")}

    boundary = []

    def f_red(z1):
        v = x_hat.copy()
        v[idx] = z1[1:k + 1]
        f = float(objective(v))
        # A neighbour the objective rejects (non-invertible MA, a unit root,
        # Q not positive definite) comes back as a sentinel. Then the optimum
        # is ON the boundary of the admissible region and no unrestricted
        # Hessian exists there (drvec BUG-34/BUG-49): no fdhess SEs, said.
        if not np.isfinite(f) or f >= 1.0e299:
            boundary.append(1)
        return f

    z = np.zeros(k + 1)
    z[1:] = x_hat[idx]
    F = f_red(z)
    H1 = np.zeros((k + 1, k + 1))
    with np.errstate(over="ignore", invalid="ignore"):   # a sentinel neighbour
        _qnewt.fdhess(f_red, k, z, F, MACHEPS, H1)
    H = H1[1:, 1:]
    H = 0.5 * (H + H.T)
    if boundary or not np.all(np.isfinite(H)):
        return None, None, {"positive_definite": False, "boundary": True,
                            "min_eig": float("nan"), "cond": float("nan")}
    eig = np.linalg.eigvalsh(H)
    info = {"min_eig": float(eig[0]),
            "cond": float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf")}
    try:
        np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        info["positive_definite"] = False
        return None, None, info
    info["positive_definite"] = True
    cov_red = 2.0 * F * np.linalg.inv(H) / n
    cov = np.full((x_hat.size, x_hat.size), np.nan)
    cov[np.ix_(idx, idx)] = cov_red
    std = np.full(x_hat.size, np.nan)
    std[idx] = np.sqrt(np.diag(cov_red))
    return cov, std, info


def legacy_objective(w, p, q, include_mean, diag_ar, diag_ma, diag_cov, method=1):
    """The objective of the ``.inp`` path (estimate_py/drvmlest), as a function of
    the packed vector, scored with the compiled ``elf`` when it is there."""
    from .estimate_py import _lag_mask, _cov_indices, _unpack
    from ._engine import elf_c
    w = np.ascontiguousarray(np.atleast_2d(np.asarray(w, float)))
    n, m = w.shape
    phi_mask = _lag_mask(m, p, diag_ar)
    theta_mask = _lag_mask(m, q, diag_ma)
    cov_idx = _cov_indices(m, diag_cov)
    xitol = -1e-3 if method == 2 else 1e-3

    def f1f2(vec):
        mu, phi, theta, qq = _unpack(np.asarray(vec, float), m, p, q, include_mean,
                                     phi_mask, theta_mask, cov_idx)
        _ll, f1, f2, _a, ifa = elf_c(m, n, p, q, mu, phi, theta, qq, w, 1.0, xitol, False)
        return f1, f2, ifa

    ref = {}

    def objective(vec):
        """(f1/f1_0)^m (f2/f2_0), as drvmlest.c:objcfunc, normalised at the
        first point it is asked (the optimum): the constant cancels anyway."""
        f1, f2, ifa = f1f2(vec)
        if ifa or not np.isfinite(f1) or f1 <= 0.0 or f2 <= 0.0:
            return 1.0e300
        if not ref:
            ref["f1"], ref["f2"] = f1, f2
        return (f1 / ref["f1"]) ** m * (f2 / ref["f2"])

    ncov = len(cov_idx)
    return objective, n, ncov


def legacy_fd_std_errors(result, w, p, q, include_mean=False, diag_ar=False,
                         diag_ma=False, diag_cov=False, method=1):
    """``fdhess`` standard errors for a fit of the ``.inp`` path, whichever engine
    produced it (the C through CFFI or the pure-Python port): same packing.

    Returns (cov, std, method, info). ``qq[1,1]`` is held fixed (the flat
    direction) and its standard error is NaN. If the Hessian is not positive
    definite the BFGS ones in ``result`` are returned and ``method`` says so.
    """
    objective, n, ncov = legacy_objective(w, p, q, include_mean, diag_ar, diag_ma,
                                          diag_cov, method)
    x_hat = np.asarray(result["params"], float)
    objective(x_hat)                              # fixes the normalisation there
    fixed = x_hat.size - ncov                     # qq[1,1]: the first of Q
    cov, std, info = fd_covariance(objective, x_hat, n, fixed=fixed)
    if cov is None:
        why = ("the optimum is on the boundary of the admissible region"
               if info.get("boundary") else "the Hessian is not positive definite")
        return (result.get("cov"), result.get("std_errors"), f"bfgs (fdhess: {why})", info)
    return cov, std, "fdhess", info
