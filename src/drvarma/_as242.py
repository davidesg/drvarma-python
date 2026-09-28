"""Pure-Python port of Shea (1989), ALGORITHM AS 242: the exact likelihood of
a vector ARMA(p, q), the independent benchmark beside elf (AS 311).

A line-by-line port of atsw-gui lib/lik/multshea.c (J.A. Mauricio's C of
Shea's Fortran), kept 1-indexed so each block can be checked against the C.
The entry point `marma` has the same arguments and returns as the compiled
`_engine.marma_c`: always exact (xtol < 0), elf's MA frontier first (ifault 4),
and (logelf, f1, f2, ifault) with f1, f2 on elf's scale. The compiled engine
is ~100x faster; this port is for when it is not built, and as a second
reading of the C.
"""

import math

import numpy as np

from ._as311 import _to_1based_cube, chekma

_LOG2PI = 1.837877066


# --------------------------------------------------------------------------- #
#  chol, bksb : multshea.c's own Cholesky and triangular solves (1-indexed)    #
# --------------------------------------------------------------------------- #

def _chol(a, k, l):
    """Lower Cholesky factor of a into l (upper triangle zeroed). 1 if not PD."""
    for j in range(1, k + 1):
        s = a[j, j] - np.dot(l[j, 1:j], l[j, 1:j])
        if s <= 0.0:
            return 1
        l[j, j] = math.sqrt(s)
        for i in range(j + 1, k + 1):
            l[i, j] = (a[i, j] - np.dot(l[i, 1:j], l[j, 1:j])) / l[j, j]
            l[j, i] = 0.0
    return 0


def _bksb(l, k, m, upper, b):
    """Solve l x = b in place for m columns; l upper or lower triangular."""
    for j2 in range(1, m + 1):
        rows = range(k, 0, -1) if upper else range(1, k + 1)
        for i in rows:
            if upper:
                s = b[i, j2] - np.dot(l[i, i + 1:k + 1], b[i + 1:k + 1, j2])
            else:
                s = b[i, j2] - np.dot(l[i, 1:i], b[1:i, j2])
            if l[i, i] == 0.0:
                return 1
            b[i, j2] = s / l[i, i]
    return 0


# --------------------------------------------------------------------------- #
#  ludcp, lusol : nlatools.c's LU (LINPACK dgefa/dgesl), used by covars        #
# --------------------------------------------------------------------------- #

def _ludcp(a, n):
    ip = np.zeros(n + 1, dtype=int)
    ip[n] = 1
    for k in range(1, n + 1):
        if k != n:
            m = k + int(np.argmax(np.abs(a[k:n + 1, k])))
            # the C keeps the FIRST maximum, as argmax does
            ip[k] = m
            if m != k:
                ip[n] = -ip[n]
            tmp = a[m, k]
            a[m, k] = a[k, k]
            a[k, k] = tmp
            if tmp != 0.0:
                a[k + 1:n + 1, k] /= -tmp
                for j in range(k + 1, n + 1):
                    tmp = a[m, j]
                    a[m, j] = a[k, j]
                    a[k, j] = tmp
                    if tmp != 0.0:
                        a[k + 1:n + 1, j] += a[k + 1:n + 1, k] * tmp
        if a[k, k] == 0.0:
            ip[n] = 0
    return ip


def _lusol(a, b, n, ip):
    if n != 1:
        for k in range(1, n):
            m = ip[k]
            tmp = b[m]
            b[m] = b[k]
            b[k] = tmp
            b[k + 1:n + 1] += a[k + 1:n + 1, k] * tmp
        for k1 in range(1, n):
            k = n - k1 + 1
            b[k] /= a[k, k]
            b[1:k] += a[1:k, k] * (-b[k])
    b[1] /= a[1, 1]


# --------------------------------------------------------------------------- #
#  covars                                                                      #
# --------------------------------------------------------------------------- #

def _covars(k, p, q, r, phi, theta, qq, gamwa, gamma):
    """Autocovariances of W(t) into gamma (flat, 1-indexed) and the cross
    covariances W/E into gamwa. 1 if the AR part has a unit root."""
    if q > 0:
        gamwa[0, 1:, 1:] = qq[1:, 1:]
    for m in range(1, q + 1):
        for i in range(1, k + 1):
            for j in range(1, k + 1):
                s = -np.dot(theta[m, i, 1:], qq[1:, j])
                for k2 in range(1, p + 1):
                    if m >= k2:
                        s += np.dot(phi[k2, i, 1:], gamwa[m - k2, 1:, j])
                gamwa[m, i, j] = s

    kw = k * k * (p + 1)
    mat = np.zeros((kw + 1, kw + 1))
    gamma[1:kw + 1] = 0.0
    for m in range(0, p + 1):
        for i in range(1, k + 1):
            for j in range(1, k + 1):
                l = m * k * k + (i - 1) * k + j
                if m == 0:
                    gamma[l] = qq[i, j]
                if 0 < m <= q:
                    for i2 in range(1, k + 1):
                        gamma[l] -= qq[i, i2] * theta[m, j, i2]
                for l4 in range(m + 1, q + 1):
                    for i2 in range(1, k + 1):
                        gamma[l] -= gamwa[l4 - m, i, i2] * theta[l4, j, i2]
                mat[l, l] = 1.0
                for i2 in range(1, p + 1):
                    for k2 in range(1, k + 1):
                        if m >= i2:
                            l4 = (m - i2) * k * k + (i - 1) * k + k2
                        else:
                            l4 = (i2 - m) * k * k + (k2 - 1) * k + i
                        mat[l, l4] -= phi[i2, j, k2]

    if p > 0:
        ip = _ludcp(mat, kw)
        if ip[kw] == 0:
            return 1
        _lusol(mat, gamma, kw, ip)

    for m in range(p + 1, r + 1):
        for i in range(1, k + 1):
            for j in range(1, k + 1):
                s = 0.0
                for l4 in range(1, p + 1):
                    for i2 in range(1, k + 1):
                        s += gamma[(m - l4) * k * k + (i - 1) * k + i2] * phi[l4, j, i2]
                if m <= q:
                    for i2 in range(1, k + 1):
                        s -= qq[i, i2] * theta[m, j, i2]
                    for i2 in range(m + 1, q + 1):
                        for l4 in range(1, k + 1):
                            s -= gamwa[i2 - m, i, l4] * theta[i2, j, l4]
                gamma[m * k * k + (i - 1) * k + j] = s
    return 0


# --------------------------------------------------------------------------- #
#  marma : the likelihood (multshea.c:marma, xtol < 0, chkma through chekma)   #
# --------------------------------------------------------------------------- #

def _marma(k, n, p, q, mu, phi, theta, qq, w, sigma2, xtol):
    """1-indexed core. Returns (r1, r2, rlogl, ifault)."""
    r = max(p, q)
    kr = k * r
    annot = p > q

    gamma = np.zeros((r + 1, k + 1, k + 1))
    mat = np.zeros((r + 1, k + 1, k + 1))
    temp = np.zeros((r + 1, k + 1, k + 1))
    tempk = np.zeros((r + 1, k + 1, k + 1))
    a = np.zeros((k + 1, k + 1))
    f = np.zeros((k + 1, k + 1))
    invf = np.zeros((k + 1, k + 1))
    mt = np.zeros((k + 1, k + 1))
    templ = np.zeros((k + 1, k + 1))
    tempm = np.zeros((k + 1, k + 1))
    wa = np.zeros((k + 1, k + 1))
    b = np.zeros(k + 1)
    z = np.zeros(k * k * (r + 1) + 1)
    v = np.zeros((k + 1, n + 1))

    for i in range(2, k + 1):
        for j in range(1, i):
            qq[j, i] = qq[i, j]

    if _chol(qq, k, a):
        return 0.0, 0.0, 0.0, 1
    tsig = float(np.prod(np.diag(a)[1:])) ** 2

    if _covars(k, p, q, r, phi, theta, qq, gamma, z):
        return 0.0, 0.0, 0.0, 2

    # the r (k x k) components of P(1/0)h, as tempk
    for k1 in range(1, r + 1):
        for i in range(1, k + 1):
            for j in range(1, k + 1):
                s = 0.0
                for k2 in range(1, k + 1):
                    for m in range(k1, p + 1):
                        s += phi[m, i, k2] * z[(m - k1 + 1) * k * k + (k2 - 1) * k + j]
                for m in range(k1, q + 1):
                    for k2 in range(1, k + 1):
                        s -= theta[m, i, k2] * gamma[m - k1 + 1, j, k2]
                tempk[k1, i, j] = s

    # A(1/0), V(1), F(1), ssq, detp
    z[1:kr + 1] = 0.0
    for i in range(1, k + 1):
        for j in range(1, k + 1):
            f[i, j] = tempk[1, i, j] + qq[i, j]
        v[i, 1] = w[i, 1] - mu[i]
        b[i] = v[i, 1]

    if _chol(f, k, a):
        return 0.0, 0.0, 0.0, 3
    detp = float(np.prod(np.diag(a)[1:])) ** 2

    mt[:, :] = 0.0
    for i in range(1, k + 1):
        mt[i, i] = 1.0
    _bksb(a, k, k, False, mt)
    for j in range(1, k + 1):
        for i in range(1, j + 1):
            invf[i, j] = a[j, i]
    _bksb(invf, k, k, True, mt)
    for i in range(1, k + 1):
        b[i] = (b[i] - np.dot(a[i, 1:i], b[1:i])) / a[i, i]

    ssq = float(np.dot(b[1:], b[1:]))
    detp = math.log(detp)

    # L(1) and K(1)
    mat[1:, :, :] = 0.0
    gamma[1:, :, :] = 0.0
    for l in range(1, r + 1):
        for i in range(1, k + 1):
            for j in range(1, k + 1):
                s = 0.0
                if l <= p:
                    s += np.dot(phi[l, i, 1:], tempk[1, 1:, j])
                if l < r:
                    s += tempk[l + 1, i, j]
                for k2 in range(1, k + 1):
                    sm = phi[l, i, k2] if l <= p else 0.0
                    if l <= q:
                        sm -= theta[l, i, k2]
                    s += sm * qq[k2, j]
                mat[l, i, j] = s

    if annot:
        for l in range(1, r + 1):
            for i in range(1, k + 1):
                for j in range(1, k + 1):
                    s = 0.0
                    if l <= q:
                        s += np.dot(theta[l, i, 1:], tempk[1, 1:, j])
                    if l < r:
                        s += tempk[l + 1, i, j]
                    gamma[l, i, j] = s
    else:
        gamma[1:, :, :] = mat[1:, :, :]

    # the recursions
    delta = False
    pearl = False
    j7 = n
    sig = 0.0

    for t in range(2, n + 1):
        if annot and t > p - q:
            pearl = True
        if not delta:
            for j in range(1, k + 1):
                for i in range(1, k + 1):
                    tempk[1, i, j] = a[j, i]
                    invf[i, j] = 0.0
                invf[j, j] = 1.0
            _bksb(tempk[1], k, k, True, invf)
            for l in range(1, r + 1):
                tempk[l, 1:, 1:] = gamma[l, 1:, 1:] @ invf[1:, 1:]

        temp[1:, :, :] = 0.0

        if not annot:
            if t != 2:
                for l in range(1, r + 1):
                    for i in range(1, k + 1):
                        s = 0.0
                        if l <= p:
                            s += np.dot(phi[l, i, 1:], z[1:k + 1])
                        if l < r:
                            s += z[l * k + i]
                        temp[l, 1, i] = s
        else:                                                   # r1:
            if t > 2:
                for l in range(1, r + 1):
                    for i in range(1, k + 1):
                        s = 0.0
                        if l <= q:
                            s += np.dot(theta[l, i, 1:], z[1:k + 1])
                        if l < r:
                            s += z[l * k + i]
                        temp[l, 1, i] = s
            for l in range(1, r + 1):
                for i in range(1, k + 1):
                    s = 0.0
                    for k2 in range(1, k + 1):
                        sm = phi[l, i, k2] if l <= p else 0.0
                        if l <= q:
                            sm -= theta[l, i, k2]
                        s += sm * (w[k2, t - 1] - mu[k2])
                    temp[l, 1, i] += s

        # r2:
        for l in range(1, r + 1):
            for l8 in range(1, k + 1):
                s = temp[l, 1, l8]
                if (not delta) or (not annot):
                    s += np.dot(tempk[l, l8, 1:], b[1:])
                z[(l - 1) * k + l8] = s

        if not delta:
            templ[1:, 1:] = mat[1, 1:, 1:]
            coef = phi if not annot else theta
            lim = p if not annot else q
            for l in range(1, r + 1):
                for i in range(1, k + 1):
                    for j in range(1, k + 1):
                        s = 0.0
                        if l <= lim:
                            s += np.dot(coef[l, i, 1:], mat[1, 1:, j])
                        if l < r:
                            s += mat[l + 1, i, j]
                        temp[l, i, j] = s

            _chol(mt, k, invf)

            for i in range(1, k + 1):
                for j in range(1, k + 1):
                    wa[i, j] = np.dot(templ[i, j:k + 1], invf[j:k + 1, j])
            for i in range(1, k + 1):
                for j in range(1, k + 1):
                    tempm[i, j] = np.dot(invf[i, 1:i + 1], wa[j, 1:i + 1])
            for l in range(1, r + 1):
                for i in range(1, k + 1):
                    for j in range(1, k + 1):
                        if pearl and l >= q + 1:
                            gamma[l, i, j] = 0.0
                        else:
                            gamma[l, i, j] += -np.dot(temp[l, i, 1:], tempm[1:, j])
            for i in range(1, k + 1):
                for j in range(i, k + 1):
                    s = f[i, j] - np.dot(wa[i, 1:], wa[j, 1:])
                    f[i, j] = s
                    f[j, i] = s

            s = 0.0
            for i in range(1, k + 1):
                d = abs(f[i, i] - qq[i, i])
                if qq[i, i] > 0.0:
                    d /= qq[i, i]
                s = s if s > d else d
            if s < xtol:
                delta = True
            if delta:
                j7 = t
                for i in range(1, k + 1):
                    for j in range(1, k + 1):
                        for l in range(1, r + 1):
                            s = phi[l, i, j] if l <= p else 0.0
                            if l <= q:
                                s -= theta[l, i, j]
                            tempk[l, i, j] = s
                for l in range(1, r + 1):
                    for i in range(1, k + 1):
                        for j in range(1, k + 1):
                            wa[i, j] = np.dot(tempk[l, i, j:k + 1], a[j:k + 1, j])
                    tempk[l, 1:, 1:] = wa[1:, 1:]

            # r3:
            wa[1:, 1:] = a[1:, 1:]
            if _chol(f, k, a):
                return 0.0, 0.0, 0.0, 3
            sig = float(np.prod(np.diag(a)[1:])) ** 2

            if not delta:
                tempm[1:, 1:] = tempm[1:, 1:].T.copy()
                _bksb(a, k, k, False, tempm)
                for i in range(1, k + 1):
                    for j in range(i, k + 1):
                        s = mt[i, j] + np.dot(tempm[1:, i], tempm[1:, j])
                        mt[i, j] = s
                        mt[j, i] = s
                _bksb(wa, k, k, False, templ)
                for l in range(1, r + 1):
                    for i in range(1, k + 1):
                        for j in range(1, k + 1):
                            if pearl and l >= q + 1:
                                s = 0.0
                            else:
                                s = np.dot(tempk[l, i, 1:], templ[1:, j])
                            mat[l, i, j] = temp[l, i, j] - s

        # r4:
        for i in range(1, k + 1):
            v[i, t] = w[i, t] - z[i] - mu[i]
            b[i] = v[i, t]
        for i in range(1, k + 1):
            b[i] = (b[i] - np.dot(a[i, 1:i], b[1:i])) / a[i, i]
        ssq += float(np.dot(b[1:], b[1:]))
        if (not delta) or t <= j7:
            detp += math.log(sig)

    r1 = ssq
    # in logarithms (the C since 2026-09-28): the plain product underflows
    r2 = math.exp((detp + (n - j7) * math.log(tsig)) / n)
    rlogl = -0.5 * (n * k * (_LOG2PI + math.log(sigma2)) + (n - j7) * math.log(tsig)
                    + detp + ssq / sigma2)
    return r1, r2, rlogl, 0


def marma(m, n, p, q, mu, phi, theta, qq, w, sigma2=1.0):
    """Shea's exact likelihood, pure Python: the same call and returns as
    `_engine.marma_c` (0-based arrays; w is (n, m); (logelf, f1, f2, ifault))."""
    if p == 0 and q == 0:
        return 0.0, 0.0, 0.0, 6
    Phi = _to_1based_cube(np.asarray(phi, float).reshape(p, m, m), p, m)
    Theta = _to_1based_cube(np.asarray(theta, float).reshape(q, m, m), q, m)
    if q > 0 and chekma(m, q, Theta):
        return 0.0, 0.0, 0.0, 4
    Mu = np.zeros(m + 1)
    Mu[1:] = np.asarray(mu, float)
    Qq = np.zeros((m + 1, m + 1))
    Qq[1:, 1:] = np.asarray(qq, float)
    Wt = np.zeros((m + 1, n + 1))
    Wt[1:, 1:] = np.asarray(w, float).T
    r1, r2, rl, ifa = _marma(m, n, p, q, Mu, Phi, Theta, Qq, Wt, sigma2, -1.0)
    return rl, r1, r2, ifa
