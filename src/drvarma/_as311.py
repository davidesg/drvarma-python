"""Faithful Python port of Mauricio's AS 311 exact VARMA log-likelihood.

Direct translation of ``csrc/internal/elfvarma.c`` (functions ``elf``, ``cgamma``,
``cxi``, ``cres``, ``chekma``) by J.A. Mauricio — *not* a Kalman/state-space
reimplementation.  Mauricio (1995) JASA 90, 282-291; (1997) Applied Statistics
46, 157-171 [AS 311].

The C is 1-indexed (Numerical-Recipes style); to keep the index arithmetic an
exact transcription, the arrays here are also 1-indexed — allocated with a
leading unused slot and addressed from 1.  Linear-algebra primitives (Cholesky,
forward substitution, quadratic forms) use NumPy, matching the C ``choldcp`` /
``cholfor`` / ``cholbak`` which compute the same factorisations.

License: GPL-2.0-or-later
"""

import numpy as np

_LOG2PI = 1.837877066


# --------------------------------------------------------------------------- #
#  1-indexed helpers                                                          #
# --------------------------------------------------------------------------- #

def _m1(r, c):
    return np.zeros((r + 1, c + 1))


def _to_1based_cube(arr, p, m):
    """(p, m, m) 0-indexed -> (p+1, m+1, m+1) 1-indexed cube."""
    out = np.zeros((p + 1, m + 1, m + 1))
    for k in range(p):
        out[k + 1, 1:, 1:] = arr[k]
    return out


def _chol_lower(A_1, n):
    """Cholesky lower factor of a 1-indexed symmetric PD matrix A_1 (n x n).

    Returns (L_1, detfac, ifault): L_1 is 1-indexed lower-triangular with
    A = L L'; detfac = det(A); ifault=1 if not positive definite.
    """
    A = A_1[1:n + 1, 1:n + 1]
    A = 0.5 * (A + A.T)
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return None, 0.0, 1
    L1 = np.zeros((n + 1, n + 1))
    L1[1:n + 1, 1:n + 1] = L
    detfac = float(np.prod(np.diag(L)) ** 2)
    return L1, detfac, 0


# --------------------------------------------------------------------------- #
#  chekma : MA invertibility via companion eigenvalues                        #
# --------------------------------------------------------------------------- #

def chekma(m, q, Theta):
    """Return ifault=1 if the MA operator is (near) non-invertible.

    Theta is 1-indexed (q+1, m+1, m+1); mirrors elfvarma.c:chekma.
    """
    if q == 0:
        return 0
    n = m * q
    A = np.zeros((n, n))
    for k in range(1, q + 1):
        for i in range(1, m + 1):
            for j in range(1, m + 1):
                A[i - 1, j - 1 + (k - 1) * m] = Theta[k][i][j]
    for k in range(1, q):
        for j in range(1, m + 1):
            A[j - 1 + k * m, j - 1 + (k - 1) * m] = 1.0
    wmod = np.abs(np.linalg.eigvals(A))
    return 1 if np.any(wmod >= 1.00005) else 0


# --------------------------------------------------------------------------- #
#  cgamma : autocovariances Gamma(k) and cross-covariances Gamma_wa(k)        #
# --------------------------------------------------------------------------- #

def cgamma(m, p, q, Phi, Theta, Qq):
    """Port of elfvarma.c:cgamma.

    Returns (gamma, gamwa, ifault):
      gamma : 1-indexed packed vector, length big = m(m+1)/2 + m^2 (p-1)
      gamwa : dict {k: (m+1,m+1) 1-indexed} for k = -q+1 .. 0
      ifault: 1 if the Yule-Walker system is singular (AR unit root)
    """
    gamwa = {k: _m1(m, m) for k in range(-q + 1, 1)}

    # [1]: cross-covariance matrices
    if q > 0:
        for i in range(1, m + 1):
            for j in range(1, m + 1):
                gamwa[0][i][j] = Qq[i][j]
    for k in range(1, q):                       # k = 1 .. q-1
        for i in range(1, m + 1):
            for j in range(1, m + 1):
                s = 0.0
                for h in range(1, m + 1):
                    s -= Theta[k][i][h] * Qq[h][j]
                for l in range(1, k + 1):
                    if l <= p:
                        for h in range(1, m + 1):
                            s += Phi[l][i][h] * gamwa[l - k][h][j]
                gamwa[-k][i][j] = s

    big = m * (m + 1) // 2 + m * m * (p - 1)
    if p == 0:
        return np.zeros(0), gamwa, 0

    # [2]: w(0)
    wzero = _m1(m, m)
    mzero = _m1(m, m)
    for i in range(1, p + 1):
        for j in range(i, q + 1):
            for ii in range(1, m + 1):
                for jj in range(1, m + 1):
                    s = 0.0
                    for k in range(1, m + 1):
                        s += Phi[i][ii][k] * gamwa[i - j][k][jj]
                    mzero[ii][jj] = s
            for ii in range(1, m + 1):
                for jj in range(1, m + 1):
                    s = 0.0
                    for k in range(1, m + 1):
                        s += mzero[ii][k] * Theta[j][jj][k]
                    wzero[ii][jj] += s
    for i in range(1, m + 1):
        for j in range(i, m + 1):
            wzero[i][j] = Qq[i][j] - wzero[i][j] - wzero[j][i]
    for j in range(1, q + 1):
        for ii in range(1, m + 1):
            for jj in range(1, m + 1):
                s = 0.0
                for k in range(1, m + 1):
                    s += Theta[j][ii][k] * Qq[k][jj]
                mzero[ii][jj] = s
        for ii in range(1, m + 1):
            for jj in range(ii, m + 1):
                s = 0.0
                for k in range(1, m + 1):
                    s += mzero[ii][k] * Theta[j][jj][k]
                wzero[ii][jj] += s

    # [3]: linear system
    mat = _m1(big, big)
    rhs = np.zeros(big + 1)

    # [3.1] first m(m+1)/2 rows
    for j in range(1, m + 1):
        for i in range(1, j + 1):
            row = j * (j - 1) // 2 + i
            for l in range(1, m + 1):
                for k in range(1, l + 1):
                    col = l * (l - 1) // 2 + k
                    s = 0.0
                    if k == l:
                        for r in range(1, p + 1):
                            s -= Phi[r][i][k] * Phi[r][j][l]
                    else:
                        for r in range(1, p + 1):
                            s -= (Phi[r][i][k] * Phi[r][j][l]
                                  + Phi[r][i][l] * Phi[r][j][k])
                    mat[row][col] = s
            for sv in range(1, p):
                for l in range(1, m + 1):
                    for k in range(1, m + 1):
                        col = m * (m + 1) // 2 + m * m * (sv - 1) + m * (l - 1) + k
                        s = 0.0
                        for r in range(1, p - sv + 1):
                            s -= (Phi[r + sv][i][k] * Phi[r][j][l]
                                  + Phi[r + sv][j][k] * Phi[r][i][l])
                        mat[row][col] = s
            mat[row][row] += 1.0
            rhs[row] = wzero[i][j]

    # [3.2] remaining m^2 (p-1) rows
    for sv in range(1, p):
        for i in range(1, m + 1):
            for j in range(1, m + 1):
                row = m * (m + 1) // 2 + m * m * (sv - 1) + m * (i - 1) + j
                for l in range(1, m + 1):
                    if l <= j:
                        col = j * (j - 1) // 2 + l
                    else:
                        col = l * (l - 1) // 2 + j
                    mat[row][col] = -Phi[sv][i][l]
                for r in range(1, p):
                    for l in range(1, m + 1):
                        col = m * (m + 1) // 2 + m * m * (r - 1) + m * (j - 1) + l
                        if r + sv <= p:
                            mat[row][col] = -Phi[r + sv][i][l]
                        if sv > r:
                            col2 = m * (m + 1) // 2 + m * m * (r - 1) + m * (l - 1) + j
                            mat[row][col2] -= Phi[sv - r][i][l]
                mat[row][row] += 1.0
                val = 0.0
                for h in range(sv, q + 1):
                    for k in range(1, m + 1):
                        val -= gamwa[sv - h][j][k] * Theta[h][i][k]
                rhs[row] = val

    # [4]: solve
    try:
        sol = np.linalg.solve(mat[1:big + 1, 1:big + 1], rhs[1:big + 1])
    except np.linalg.LinAlgError:
        return np.zeros(big + 1), gamwa, 1
    gamma = np.zeros(big + 1)
    gamma[1:big + 1] = sol
    return gamma, gamwa, 0


# --------------------------------------------------------------------------- #
#  cxi : Green's-function matrix sequence Xi_k, premultiplied by q1inv         #
# --------------------------------------------------------------------------- #

def cxi(m, n, q, Theta, Q1inv, xitol):
    """Port of elfvarma.c:cxi.  Returns (nlim, rxi) with rxi[k] (k=0..n-1)
    a (m+1,m+1) 1-indexed matrix; rxi premultiplied by the lower-triangular q1inv.
    """
    rxi = np.zeros((n, m + 1, m + 1))
    for i in range(1, m + 1):
        rxi[0][i][i] = 1.0

    r = 0
    delta = False
    while True:
        r += 1
        for jx in range(1, q + 1):
            if r >= jx:
                for ii in range(1, m + 1):
                    for jj in range(1, m + 1):
                        s1 = 0.0
                        for h in range(1, m + 1):
                            s1 += Theta[jx][ii][h] * rxi[r - jx][h][jj]
                        rxi[r][ii][jj] += s1
        s2 = float(np.sum(np.abs(rxi[r])))
        if s2 < xitol:
            nq = 1
            delta = True
            while (nq <= q) and (r < n - 1) and delta:
                nq += 1
                r += 1
                for jx in range(1, q + 1):
                    if r >= jx:
                        for ii in range(1, m + 1):
                            for jj in range(1, m + 1):
                                s1 = 0.0
                                for h in range(1, m + 1):
                                    s1 += Theta[jx][ii][h] * rxi[r - jx][h][jj]
                                rxi[r][ii][jj] += s1
                s2 = float(np.sum(np.abs(rxi[r])))
                if s2 > xitol:
                    delta = False
            if delta:
                r -= nq
        if delta or r >= n - 1:
            break
    nlim = r

    # [2]: premultiply each rxi[k] by q1inv (lower triangular)
    for k in range(0, nlim + 1):
        mtmp = _m1(m, m)
        for i in range(1, m + 1):
            for jx in range(1, m + 1):
                s1 = 0.0
                for h in range(1, i + 1):
                    s1 += Q1inv[i][h] * rxi[k][h][jx]
                mtmp[i][jx] = s1
        rxi[k][1:, 1:] = mtmp[1:, 1:]
    return nlim, rxi


# --------------------------------------------------------------------------- #
#  cres : exact residuals                                                      #
# --------------------------------------------------------------------------- #

def cres(m, n, g, nlim, rxi, Q1, M, L, lam, res):
    """Port of elfvarma.c:cres.  Overwrites res (1-indexed (n+1, m+1))."""
    mg = m * g
    # [1] solve L' c = lambda  (L lower 1-indexed) -> overwrite lam[1..mg]
    Lmat = L[1:mg + 1, 1:mg + 1]
    from scipy.linalg import solve_triangular
    c = solve_triangular(Lmat, lam[1:mg + 1], lower=True, trans='T')
    lam[1:mg + 1] = c
    # [2] d = M c  (M lower 1-indexed)
    for i in range(mg, 0, -1):
        s = 0.0
        for j in range(1, i + 1):
            s += M[i][j] * lam[j]
        lam[i] = s
    # [3] residual correction
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            if (i - j <= nlim) and (j <= g):
                for jj in range(1, m + 1):
                    s = 0.0
                    for h in range(1, m + 1):
                        s += rxi[i - j][jj][h] * lam[h + (j - 1) * m]
                    res[i][jj] -= s
    for j in range(1, n + 1):
        for i in range(m, 0, -1):
            s = 0.0
            for h in range(1, i + 1):
                s += Q1[i][h] * res[j][h]
            res[j][i] = s


# --------------------------------------------------------------------------- #
#  elf : the exact log-likelihood                                             #
# --------------------------------------------------------------------------- #

def elf(m, n, p, q, Mu, Phi, Theta, Qq, W, sigma2, xitol, atf):
    """Port of elfvarma.c:elf.

    All matrix arguments are 1-indexed (Mu (m+1,), Phi/Theta cubes, Qq (m+1,m+1),
    W (n+1, m+1)).  Returns (logelf, f1, f2, a, ifault) where a is the 1-indexed
    residual matrix (n+1, m+1) (filled only if atf=True).
    """
    g = max(p, q)
    a = np.zeros((n + 1, m + 1))

    # [0]/[1]: q1 = chol(Qq), q1inv, detq
    Q1full = _m1(m, m)
    for i in range(1, m + 1):
        for j in range(i, m + 1):
            Q1full[i][j] = Qq[i][j]
            Q1full[j][i] = Qq[i][j]
    Q1, detq, ifault = _chol_lower(Q1full, m)
    if ifault:
        return 0.0, 0.0, 0.0, a, 1
    from scipy.linalg import solve_triangular
    Lq = Q1[1:m + 1, 1:m + 1]
    Q1inv_full = solve_triangular(Lq, np.eye(m), lower=True)   # inv of lower chol
    Q1inv = _m1(m, m)
    Q1inv[1:m + 1, 1:m + 1] = Q1inv_full

    # [2]: autocovariances
    gamma = np.zeros(0)
    gamwa = {}
    if p > 0:
        gamma, gamwa, ifg = cgamma(m, p, q, Phi, Theta, Qq)
        if ifg:
            return 0.0, 0.0, 0.0, a, 2

    half = m * (m + 1) // 2

    def gval(k, ii, kk):
        """gamma packing access used in [3.1] (see elfvarma.c)."""
        if k > 0:
            jl = half + m * m * (k - 1) + m * (kk - 1) + ii
        elif k < 0:
            jl = half - m * m * (k + 1) + m * (ii - 1) + kk
        else:
            if kk >= ii:
                jl = kk * (kk - 1) // 2 + ii
            else:
                jl = ii * (ii - 1) // 2 + kk
        return gamma[jl]

    # [3]: M = chol(v1 omega v1')
    mg = m * g
    mtmp0 = _m1(mg, mg)

    # [3.1]: omega * v1'  -> mtmp1  ((p+q)m x g m)
    mtmp1 = _m1(m * (p + q), mg)
    for i in range(1, p + 1):
        for j in range(1, g + 1):
            for k in range(j - i, p - i + 1):
                for ii in range(1, m + 1):
                    for jj in range(1, m + 1):
                        s1 = 0.0
                        for kk in range(1, m + 1):
                            s1 += gval(k, ii, kk) * Phi[p - k - i + j][jj][kk]
                        mtmp1[ii + (i - 1) * m][jj + (j - 1) * m] += s1
            for k in range(j - i, q - i + 1):
                if p + k <= q:
                    for ii in range(1, m + 1):
                        for jj in range(1, m + 1):
                            s1 = 0.0
                            for kk in range(1, m + 1):
                                s1 += gamwa[-q + p + k][ii][kk] * Theta[q - k - i + j][jj][kk]
                            mtmp1[ii + (i - 1) * m][jj + (j - 1) * m] -= s1
    for i in range(p + 1, p + q + 1):
        for j in range(1, g + 1):
            for k in range(p + j - i, p + p - i + 1):
                if p - k <= q:
                    for ii in range(1, m + 1):
                        for jj in range(1, m + 1):
                            s1 = 0.0
                            for kk in range(1, m + 1):
                                s1 += gamwa[-q + p - k][kk][ii] * Phi[p + p - k - i + j][jj][kk]
                            mtmp1[ii + (i - 1) * m][jj + (j - 1) * m] += s1
            if p - i + j <= 0:
                for ii in range(1, m + 1):
                    for jj in range(1, m + 1):
                        s1 = 0.0
                        for kk in range(1, m + 1):
                            s1 += Qq[ii][kk] * Theta[q + p - i + j][jj][kk]
                        mtmp1[ii + (i - 1) * m][jj + (j - 1) * m] -= s1

    # [3.2]: v1 omega v1' -> mtmp0 (upper triangle)
    for i in range(1, g + 1):
        for j in range(i, g + 1):
            for k in range(0, p - i + 1):
                for ii in range(1, m + 1):
                    jl = ii if i == j else 1
                    for jj in range(jl, m + 1):
                        s1 = 0.0
                        for kk in range(1, m + 1):
                            s1 += Phi[p - k][ii][kk] * mtmp1[kk + (k + i - 1) * m][jj + (j - 1) * m]
                        mtmp0[ii + (i - 1) * m][jj + (j - 1) * m] += s1
            for k in range(0, q - i + 1):
                for ii in range(1, m + 1):
                    jl = ii if i == j else 1
                    for jj in range(jl, m + 1):
                        s1 = 0.0
                        for kk in range(1, m + 1):
                            s1 += Theta[q - k][ii][kk] * mtmp1[kk + (k + p + i - 1) * m][jj + (j - 1) * m]
                        mtmp0[ii + (i - 1) * m][jj + (j - 1) * m] -= s1

    # symmetrise mtmp0 (it was filled on/above the block-diagonal upper triangle)
    Vfull = mtmp0[1:mg + 1, 1:mg + 1].copy()
    Vfull = np.triu(Vfull) + np.triu(Vfull, 1).T
    mtmp0[1:mg + 1, 1:mg + 1] = Vfull

    # [3.3]: M = chol(mtmp0)
    M, _detM, ifault = _chol_lower(mtmp0, mg)
    if ifault:
        return 0.0, 0.0, 0.0, a, 3

    # [4.1]: MA invertibility
    if q > 0 and chekma(m, q, Theta):
        return 0.0, 0.0, 0.0, a, 4

    # [4.2]: rxi
    nlim, rxi = cxi(m, n, q, Theta, Q1inv, xitol)

    # [5]: eta -> a  (conditional residuals, then premultiply by q1inv).
    # Vectorised equivalent of the C double loop: AR part by shifted matmuls,
    # MA part by the (inherently sequential) feedback recursion.
    x0 = W[1:n + 1, 1:m + 1] - Mu[1:m + 1]            # (n, m), 0-indexed
    phi0 = Phi[1:p + 1, 1:m + 1, 1:m + 1] if p else np.zeros((0, m, m))
    theta0 = Theta[1:q + 1, 1:m + 1, 1:m + 1] if q else np.zeros((0, m, m))
    b = x0.copy()
    for jx in range(1, p + 1):
        b[jx:] -= x0[:n - jx] @ phi0[jx - 1].T       # - Phi_j x_{t-j}
    a0 = b
    if q:
        a0 = b.copy()
        for t in range(n):
            acc = b[t]
            for jx in range(1, q + 1):
                if t - jx >= 0:
                    acc = acc + theta0[jx - 1] @ a0[t - jx]
            a0[t] = acc
    a0 = a0 @ Q1inv[1:m + 1, 1:m + 1].T              # [5.2] premultiply by q1inv
    a[1:n + 1, 1:m + 1] = a0

    # [6]: M' h -> vechh.  h_j = sum_i Xi_i' a_{i+j}  (i = 0..min(nlim, n-j)).
    vechh = np.zeros(mg + 1)
    R = rxi[:nlim + 1, 1:m + 1, 1:m + 1]             # (nlim+1, m, m)
    a0 = a[1:n + 1, 1:m + 1]                          # (n, m), 0-indexed rows
    for j in range(1, g + 1):
        kmax = min(nlim, n - j)
        if kmax >= 0:
            # sum_i Xi_i^T a_{i+j} ; a_{i+j} (1-indexed) = a0[i+j-1]
            Hj = np.einsum('iab,ia->b', R[:kmax + 1], a0[j - 1:j + kmax])
            vechh[1 + (j - 1) * m:j * m + 1] = Hj
    # premultiply by M' : vechh = M^T vechh
    Mmat = M[1:mg + 1, 1:mg + 1]
    vechh[1:mg + 1] = Mmat.T @ vechh[1:mg + 1]

    Msave = M.copy() if atf else None

    # [7]: H'H -> mtmp2.  First block-column (i,1) = sum_k Xi_k' Xi_{k+i-1};
    # remaining lower blocks by the recursion (i,j) = (i-1,j-1) - corr.
    rx = rxi[:, 1:m + 1, 1:m + 1]                     # (n, m, m), 0-indexed
    blk = np.zeros((g + 1, g + 1, m, m))             # blk[i][j] = HtH block (i,j)
    for i in range(1, g + 1):
        kmax = min(n - i, nlim - i + 1)
        if kmax >= 0:
            blk[i][1] = np.einsum('kab,kac->bc', rx[:kmax + 1], rx[i - 1:i + kmax])
    for i in range(2, g + 1):
        for j in range(2, i + 1):
            corr = np.zeros((m, m))
            if (n - i + 1 <= nlim) and (n - j + 1 <= nlim):
                corr = rx[n - i + 1].T @ rx[n - j + 1]
            blk[i][j] = blk[i - 1][j - 1] - corr
    HtH = np.zeros((mg, mg))
    for i in range(1, g + 1):
        for j in range(1, i + 1):
            HtH[(i - 1) * m:i * m, (j - 1) * m:j * m] = blk[i][j]
    HtH = np.tril(HtH) + np.tril(HtH, -1).T          # symmetrise from lower

    # [8]: I + M'H'HM and its Cholesky L
    ImMtHtHM = np.eye(mg) + (Mmat.T @ HtH) @ Mmat
    try:
        Lom = np.linalg.cholesky(0.5 * (ImMtHtHM + ImMtHtHM.T))
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0, a, 5
    detom = float(np.prod(np.diag(Lom)) ** 2)

    # [9]: lambda via forward substitution L lambda = vechh
    lam_vec = solve_triangular(Lom, vechh[1:mg + 1], lower=True)
    vechh[1:mg + 1] = lam_vec

    # [10]: quadratic form f1
    s1 = float(np.sum(a[1:n + 1, 1:m + 1] ** 2))
    s2 = float(np.sum(vechh[1:mg + 1] ** 2))
    f1 = s1 - s2

    # [11]: determinant factor
    f2 = detq * np.exp(np.log(detom) / n)

    # [12]: log-likelihood
    logelf = -0.5 * (n * m * (_LOG2PI + np.log(sigma2)) + n * np.log(detq)
                     + np.log(detom) + f1 / sigma2)

    # [13]: residuals
    if atf:
        # rebuild L as 1-indexed for cres
        L1 = _m1(mg, mg)
        L1[1:mg + 1, 1:mg + 1] = Lom
        cres(m, n, g, nlim, rxi, Q1, Msave, L1, vechh, a)

    return float(logelf), float(f1), float(f2), a, 0
