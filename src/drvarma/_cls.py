"""The least squares of the ladder's starts, in the C's own arithmetic.

The C (escalera.c ``start_ols``, ``start_ols_v``) forms the normal equations
with plain loops, each sum accumulated in sequence, and solves them with its
LU (nlatools.c ``ludcp``/``lusol``). numpy gives the same numbers to rounding,
with other summation orders and LAPACK's pivoting, and a seed that differs in
the sixteenth digit is another path wherever the likelihood has no interior
maximum (BUG-0030). So the seeds are computed here as the C does: the same
operations in the same order, on Python floats (IEEE doubles, as the C's).

Everything is 0-based; the loops are the C's, line for line.
"""
import math


def mean(col):
    """SUM / n, the sum accumulated in sequence."""
    s = 0.0
    for v in col:
        s += v
    return s / len(col)


def ludcp(a, n):
    """nlatools.c ludcp: LU in place with partial pivoting. Returns ip."""
    ip = [0] * n
    ip[n - 1] = 1
    for k in range(n):
        if k != n - 1:
            m = k
            for i in range(k + 1, n):
                if abs(a[i][k]) > abs(a[m][k]):
                    m = i
            ip[k] = m
            if m != k:
                ip[n - 1] = -ip[n - 1]
            tmp = a[m][k]
            a[m][k] = a[k][k]
            a[k][k] = tmp
            if tmp != 0.0:
                for i in range(k + 1, n):
                    a[i][k] /= -tmp
                for j in range(k + 1, n):
                    tmp = a[m][j]
                    a[m][j] = a[k][j]
                    a[k][j] = tmp
                    if tmp != 0.0:
                        for i in range(k + 1, n):
                            a[i][j] += a[i][k] * tmp
        if a[k][k] == 0.0:
            ip[n - 1] = 0
    return ip


def lusol(a, b, n, ip):
    """nlatools.c lusol: solves in place with ludcp's factors."""
    if n != 1:
        for k in range(n - 1):
            m = ip[k]
            tmp = b[m]
            b[m] = b[k]
            b[k] = tmp
            for i in range(k + 1, n):
                b[i] += a[i][k] * tmp
        for k1 in range(1, n):
            km1 = n - k1                 # the C's km1 (1-based count)
            k = km1                      # 0-based index of the C's k = km1 + 1
            b[k] /= a[k][k]
            tmp = -b[k]
            for i in range(km1):
                b[i] += a[i][k] * tmp
    b[0] /= a[0][0]


def gram(Z, T, n):
    """Z'Z, each entry a sum over the rows in sequence."""
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            s = 0.0
            for r in range(T):
                s += Z[r][i] * Z[r][k]
            out[i][k] = s
    return out


def cross(Z, y, T, n):
    """Z'y, each entry a sum over the rows in sequence."""
    out = [0.0] * n
    for i in range(n):
        s = 0.0
        for r in range(T):
            s += Z[r][i] * y[r]
        out[i] = s
    return out


def corr_to_cov(s_ab, s_aa, s_bb, qa, qb, route_v):
    """The covariance seed from the residuals' correlation. The two routes
    of the C write the same product in two orders, and they are kept."""
    if route_v:
        return s_ab / math.sqrt(s_aa * s_bb) * math.sqrt(qa * qb)
    rr = s_ab / math.sqrt(s_aa * s_bb)
    return rr * math.sqrt(qa * qb)
