"""Faithful Python port of Mauricio's factored BFGS quasi-Newton optimiser.

Direct transcription of ``csrc/internal/qnewtopt.c`` (``raxopt``, ``bfgsfac``,
``qrupdate``, ``jacrot``, ``cdgrad``, ``lnsrch``, ``umstop0``, ``umstop``) plus
the Cholesky solve primitives from ``nlatools.c`` (``cholfor``/``cholbak``/
``cholsol``).  Source: Dennis & Schnabel (1983).  Copyright (C) J.A. Mauricio.

This is *not* a generic optimiser: it reproduces the C estimator's exact
quasi-Newton trajectory and, crucially, the factored Hessian approximation ``b``
that the C uses for the parameter covariance (``cov = 2*f*b^{-1}/n`` in
``drvmlest.c:est``).  Keeping the same trajectory is what makes the pure-Python
estimate match the C engine's sigma2/Q split and ``cov[]`` standard errors.

The C is 1-indexed (Numerical-Recipes style); the arrays here are likewise
1-indexed (a leading unused slot, addressed from 1) so the index arithmetic is
an exact transcription.

License: GPL-2.0-or-later
"""

import numpy as np

MACHEPS = float(np.finfo(float).eps)        # cmacheps(): 2^-52


def _rmax(a, b):
    return a if a > b else b


def _rmin(a, b):
    return a if a < b else b


# --------------------------------------------------------------------------- #
#  Cholesky solve (nlatools.c): matl holds the lower factor L (H = L L').      #
# --------------------------------------------------------------------------- #

def cholfor(matl, n, rhsol):
    """Solve L y = rhsol in place (L lower-triangular, 1-indexed)."""
    rhsol[1] /= matl[1][1]
    for i in range(2, n + 1):
        tmp = 0.0
        for j in range(1, i):
            tmp += matl[i][j] * rhsol[j]
        rhsol[i] = (rhsol[i] - tmp) / matl[i][i]


def cholbak(matl, n, rhsol):
    """Solve L' x = rhsol in place."""
    rhsol[n] /= matl[n][n]
    for i in range(n - 1, 0, -1):
        tmp = 0.0
        for j in range(i + 1, n + 1):
            tmp += matl[j][i] * rhsol[j]
        rhsol[i] = (rhsol[i] - tmp) / matl[i][i]


def cholsol(matl, n, rhsol):
    """Solve (L L') x = rhsol in place."""
    cholfor(matl, n, rhsol)
    cholbak(matl, n, rhsol)


# --------------------------------------------------------------------------- #
#  cdgrad : central-difference gradient                                       #
# --------------------------------------------------------------------------- #

def cdgrad(func, n, x, eta, g):
    third = pow(eta, 1.0 / 3.0)
    for i in range(1, n + 1):
        if x[i] == abs(x[i]):
            stepi = third * _rmax(x[i], 1.0)
        else:
            stepi = third * _rmin(x[i], -1.0)
        tempi = x[i]
        x[i] = x[i] + stepi
        stepi = x[i] - tempi             # reduce finite-precision error
        fpls = func(x)
        x[i] = tempi - stepi
        fmns = func(x)
        g[i] = (fpls - fmns) / (2.0 * stepi)
        x[i] = tempi


# --------------------------------------------------------------------------- #
#  fdhess : finite-difference Hessian                                         #
# --------------------------------------------------------------------------- #

def fdhess(func, n, x, f, eta, H):
    """Second-derivative matrix by finite differences at `x`. Port of
    `qnewtopt.c:fdhess` (Dennis & Schnabel, Algorithm A5.6.2).

    Why this exists and is not the optimiser's matrix: `raxopt` leaves in `b`
    the Hessian ACCUMULATED by BFGS along the search path. That is what steers
    the search, but it is not the curvature at the optimum -- it depends on the
    path taken (two different starts give different standard errors) and it
    degrades precisely in the flattest directions, which are the ones with the
    largest standard errors. It is also never built at all when the search
    starts AT the optimum and stops immediately, which is the normal situation
    when the seeds come from a previous rung of the ladder.

    `f` is `func(x)`, already evaluated. `H` is 1-indexed (n+1, n+1) and is
    written in place; `x` is 1-indexed and is restored on return.

    Cost: (n^2 + 3n)/2 evaluations of `func`.
    """
    third = pow(eta, 1.0 / 3.0)
    step = np.zeros(n + 1)
    fneigh = np.zeros(n + 1)

    for i in range(1, n + 1):
        if x[i] == abs(x[i]):
            step[i] = third * _rmax(x[i], 1.0)
        else:
            step[i] = third * _rmin(x[i], -1.0)
        tempi = x[i]
        x[i] = x[i] + step[i]
        step[i] = x[i] - tempi           # reduce finite-precision error
        fneigh[i] = func(x)
        x[i] = tempi

    for i in range(1, n + 1):
        tempi = x[i]
        x[i] = x[i] + 2.0 * step[i]
        fii = func(x)
        H[i][i] = (f + fii - 2.0 * fneigh[i]) / (step[i] * step[i])
        x[i] = tempi + step[i]

        for j in range(i + 1, n + 1):
            tempj = x[j]
            x[j] = x[j] + step[j]
            fij = func(x)
            H[i][j] = (f - fneigh[i] + fij - fneigh[j]) / (step[i] * step[j])
            H[j][i] = H[i][j]
            x[j] = tempj

        x[i] = tempi


# --------------------------------------------------------------------------- #
#  jacrot / qrupdate / bfgsfac : factored BFGS update                         #
# --------------------------------------------------------------------------- #

def jacrot(n, i, a, b, M):
    if a == 0.0:
        c = 0.0
        s = 1.0 if b >= 0.0 else -1.0
    else:
        den = np.sqrt(a * a + b * b)
        c = a / den
        s = b / den
    for j in range(i, n + 1):
        y = M[i][j]
        w = M[i + 1][j]
        M[i][j] = c * y - s * w
        M[i + 1][j] = s * y + c * w


def qrupdate(n, u, v, M):
    k = 0
    for kk in range(n, 0, -1):
        if u[kk]:
            k = kk
            break
    if k < 1:
        k = 1
    for i in range(k - 1, 0, -1):
        jacrot(n, i, u[i], -u[i + 1], M)
        if u[i] == 0.0:
            u[i] = abs(u[i + 1])
        else:
            u[i] = np.sqrt(u[i] * u[i] + u[i + 1] * u[i + 1])
    for j in range(1, n + 1):
        M[1][j] += u[1] * v[j]
    for i in range(1, k):
        jacrot(n, i, M[i][i], -M[i + 1][i], M)


def bfgsfac(n, xk, xkp1, gk, gkp1, eta, B):
    s = np.zeros(n + 1)
    y = np.zeros(n + 1)
    u = np.zeros(n + 1)
    t = np.zeros(n + 1)

    for i in range(1, n + 1):
        s[i] = xkp1[i] - xk[i]
        y[i] = gkp1[i] - gk[i]

    tmp1 = tmp2 = tmp3 = 0.0
    for i in range(1, n + 1):
        tmp1 += y[i] * s[i]
        tmp2 += s[i] * s[i]
        tmp3 += y[i] * y[i]

    if tmp1 > np.sqrt(MACHEPS * tmp2 * tmp3):
        tmp2 = 0.0
        for i in range(1, n + 1):
            t[i] = 0.0
            for j in range(i, n + 1):
                t[i] += B[j][i] * s[j]
            tmp2 += t[i] * t[i]

        alpha = np.sqrt(tmp1 / tmp2)
        tol = np.sqrt(eta)
        skpupd = 1

        for i in range(1, n + 1):
            tmp3 = 0.0
            for j in range(1, i + 1):
                tmp3 += B[i][j] * t[j]
            if abs(y[i] - tmp3) >= tol * _rmax(abs(gk[i]), abs(gkp1[i])):
                skpupd = 0
            u[i] = y[i] - alpha * tmp3

        if skpupd == 0:
            tmp3 = np.sqrt(tmp1 * tmp2)
            for i in range(1, n + 1):
                t[i] /= tmp3
            for i in range(2, n + 1):
                for j in range(1, i):
                    B[j][i] = B[i][j]
                    B[i][j] = 0.0
            qrupdate(n, t, u, B)
            for i in range(2, n + 1):
                for j in range(1, i):
                    B[i][j] = B[j][i]
                    B[j][i] = 0.0


# --------------------------------------------------------------------------- #
#  stopping criteria                                                          #
# --------------------------------------------------------------------------- #

def umstop0(n, x, f, g, gradtol, maxits):
    if maxits == 0:
        return 4, 0
    consecmax = 0
    max1 = abs(g[1]) * (abs(x[1]) + 1.0) / (abs(f) + 1.0)
    for i in range(2, n + 1):
        tmp = abs(g[i]) * (abs(x[i]) + 1.0) / (abs(f) + 1.0)
        if tmp > max1:
            max1 = tmp
    return (1 if max1 <= gradtol else 0), consecmax


def umstop(n, xk, xkp1, fkp1, gkp1, retcode, gradtol, steptol, k, maxits,
           maxcmax, maxtaken, consecmax):
    max1 = abs(gkp1[1]) * (abs(xkp1[1]) + 1.0) / (abs(fkp1) + 1.0)
    for i in range(2, n + 1):
        tmp = abs(gkp1[i]) * (abs(xkp1[i]) + 1.0) / (abs(fkp1) + 1.0)
        if tmp > max1:
            max1 = tmp
    max2 = abs(xkp1[1] - xk[1]) / (abs(xkp1[1]) + 1.0)
    for i in range(2, n + 1):
        tmp = abs(xkp1[i] - xk[i]) / (abs(xkp1[i]) + 1.0)
        if tmp > max2:
            max2 = tmp

    if max1 <= gradtol:
        return 1, consecmax
    elif retcode == 1:
        return 3, consecmax
    elif max2 <= steptol:
        return 2, consecmax
    elif k >= maxits:
        return 4, consecmax
    elif maxtaken:
        consecmax += 1
        if consecmax == maxcmax:
            return 5, consecmax
        return 0, consecmax
    else:
        return 0, 0


# --------------------------------------------------------------------------- #
#  lnsrch : Dennis-Schnabel line search                                       #
# --------------------------------------------------------------------------- #

def lnsrch(n, xk, fk, gk, dk, xkp1, maxstep, steptol, func):
    """Returns (lambda, fkp1, retcode, maxtaken).  xkp1 written in place."""
    maxtaken = 0
    retcode = 2
    alpha = 1.0e-4

    newtlen = 0.0
    for i in range(1, n + 1):
        newtlen += dk[i] * dk[i]
    newtlen = np.sqrt(newtlen)
    if newtlen > maxstep:
        tmp = maxstep / newtlen
        for i in range(1, n + 1):
            dk[i] *= tmp
        newtlen = maxstep

    initslp = 0.0
    for i in range(1, n + 1):
        initslp += gk[i] * dk[i]

    rellen = 0.0
    for i in range(1, n + 1):
        tmp = abs(dk[i]) / _rmax(abs(xk[i]), 1.0)
        if tmp > rellen:
            rellen = tmp
    minlam = steptol / rellen
    lam = 1.0

    fkp1 = fk
    prelam = 0.0
    pfkp1 = 0.0
    while True:
        for i in range(1, n + 1):
            xkp1[i] = xk[i] + lam * dk[i]
        fkp1 = func(xkp1)

        if fkp1 <= fk + alpha * lam * initslp:
            retcode = 0
            if (lam == 1.0) and (newtlen > 0.99 * maxstep):
                maxtaken = 1
        elif lam < minlam:
            retcode = 1
            for i in range(1, n + 1):
                xkp1[i] = xk[i]
        else:
            if lam == 1.0:
                tlambda = -initslp / (2.0 * (fkp1 - fk - initslp))
            else:
                t1 = fkp1 - fk - lam * initslp
                t2 = pfkp1 - fk - prelam * initslp
                t3 = 1.0 / (lam - prelam)
                a = (t1 / (lam * lam) - t2 / (prelam * prelam)) * t3
                b = (t2 * lam / (prelam * prelam)
                     - t1 * prelam / (lam * lam)) * t3
                if a == 0.0:
                    tlambda = -initslp / (2.0 * b)
                else:
                    disc = b * b - 3.0 * a * initslp
                    if disc < 0.0:
                        raise RuntimeError("ROUNDOFF PROBLEM IN LINE SEARCH")
                    tlambda = (-b + np.sqrt(disc)) / (3.0 * a)
                if tlambda > 0.5 * lam:
                    tlambda = 0.5 * lam
            prelam = lam
            pfkp1 = fkp1
            if tlambda <= 0.1 * lam:
                lam = 0.1 * lam
            else:
                lam = tlambda

        if retcode != 2:
            break
    return lam, fkp1, retcode, maxtaken


# --------------------------------------------------------------------------- #
#  raxopt : the driver                                                        #
# --------------------------------------------------------------------------- #

def raxopt(func, n, xk, maxits, gradtol, steptol):
    """Factored BFGS minimisation of `func` from `xk` (1-indexed, modified in place).

    Returns (fk, b, k, termcode): final objective, the factored Hessian b
    (1-indexed (n+1,n+1) lower-triangular Cholesky factor of the BFGS Hessian
    approximation), iteration count, and the termination code.
    """
    xkp1 = np.zeros(n + 1)
    gk = np.zeros(n + 1)
    gkp1 = np.zeros(n + 1)
    dk = np.zeros(n + 1)
    b = np.zeros((n + 1, n + 1))

    maxstep = 0.0
    for i in range(1, n + 1):
        maxstep += xk[i] * xk[i]
    maxstep = np.sqrt(maxstep)
    maxstep = _rmax(maxstep, 1.0)
    maxstep = 1.0e3 * maxstep
    eta = MACHEPS
    maxcmax = 5

    k = 0
    fk = 1.0
    cdgrad(func, n, xk, eta, gk)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            b[i][j] = 0.0
        b[i][i] = 1.0

    termcode, consecmax = umstop0(n, xk, fk, gk, gradtol, maxits)

    while termcode == 0:
        for i in range(1, n + 1):
            dk[i] = -gk[i]
        cholsol(b, n, dk)

        lam, fkp1, retcode, maxtaken = lnsrch(
            n, xk, fk, gk, dk, xkp1, maxstep, steptol, func)

        cdgrad(func, n, xkp1, eta, gkp1)

        termcode, consecmax = umstop(
            n, xk, xkp1, fkp1, gkp1, retcode, gradtol, steptol,
            k + 1, maxits, maxcmax, maxtaken, consecmax)

        bfgsfac(n, xk, xkp1, gk, gkp1, eta, b)

        k += 1
        for i in range(1, n + 1):
            xk[i] = xkp1[i]
            gk[i] = gkp1[i]
        fk = fkp1

    return fk, b, k, termcode
