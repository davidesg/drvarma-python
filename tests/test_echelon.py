"""The echelon form in the Ladder: Ladder(echelon=n) — stage 3a of
docs/ECHELON_PLAN.md. Lutkepohl and Poskitt (1996, eq. 2.3): the index table
decides which cross coefficients are free, and A_0 = M_0 is lower triangular
with the AR0 terms. The pins: LP's own (1,2,1) table, the standard form
-A_0^-1 A_k against the cast, equal indices = the VARMA(n, n), and a
simulated system recovered."""
import os

import numpy as np
import pytest

from drvarma.datasets import simulate_varma
from drvarma.ladder import Ladder, LadderError, write_report


def _operator(title, order):
    if order <= 0:
        return [f"** {title}:", "0"]
    return [f"** {title}:", f"1 {order}", "**"] + ["0.000000  1"] * order


def spec(path, name, y, p, q):
    """A route-V fue .inp: annual, lambda 1, d = 0, the mean free, a free own
    AR(p)/MA(q) at zero (as sima.raw.write_inp writes them)."""
    L = ["* route V specification", "",
         "** Frequency of time series: either 1(A), 4(Q) or 12(M):", " 1",
         "** Number of observations and starting date of time series:",
         f" {len(y)}  1 1 {name}",
         "** Number of deterministic variables (including seasonal components):", "0"]
    L += _operator("Number and orders of regular AR operators", p)
    L += _operator("Number and orders of annual AR operators", 0)
    L += _operator("Number and orders of regular MA operators", q)
    L += _operator("Number and orders of anual MA operators", 0)
    L += ["** Number and frequencies of regular AR(2) operators with fixed frequency:", "0",
          "** Number and frequencies of regular MA(2) operators with fixed frequency:", "0",
          "** Mean parameter (mu):", f"{float(np.mean(y)):.6f} 1",
          "** Box-Cox lambda, regular differences and complete annual differences:",
          " 1  0  0",
          "** Individual factors of the annual difference (starting at freq 0.0):", " 0",
          "** ACF/PACF bands (0 Automatic) and reescaling factor:", " 0 1",
          "** Time series (stochastic and non-standard deterministic variables):"]
    L += [f"{v:.15g}" for v in y]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return str(path)


def specs(tmp_path, y, orders):
    return [spec(tmp_path / f"y{i + 1}.inp", f"y{i + 1}", y[:, i], p, q)
            for i, (p, q) in enumerate(orders)]


def noise(n, m, seed=0):
    return 10.0 + np.random.default_rng(seed).standard_normal((n, m))


# -- the structure ------------------------------------------------------------ #

def test_lutkepohl_poskitts_121_table(tmp_path):
    """LP §2: indices (1,2,1) give A_0 with one free entry (3<-2), the cross
    AR of row 2 at lag 2 only, row 2's MA at lags 1 and 2."""
    L = Ladder(specs(tmp_path, noise(120, 3), [(1, 1), (2, 2), (1, 1)]), echelon=(1, 2, 1))
    cross = [n for n in L.names() if "<-" in n]
    assert cross == [
        "AR0[y3<-y2]",
        "AR1[y1<-y2]", "AR1[y1<-y3]", "AR1[y3<-y1]", "AR1[y3<-y2]",
        "AR2[y2<-y1]", "AR2[y2<-y3]",
        "MA1[y1<-y2]", "MA1[y1<-y3]", "MA1[y2<-y1]", "MA1[y2<-y3]",
        "MA1[y3<-y1]", "MA1[y3<-y2]",
        "MA2[y2<-y1]", "MA2[y2<-y3]"]
    assert (L.p, L.q) == (2, 2)


def test_a_zero_index_has_only_its_contemporaneous_term(tmp_path):
    """(1,0): row 2 has no lags; n_21 = min(0 + 1, 1) = 1 = n_2 + 1 gives it
    AR0[y2<-y1]. Row 1's cross AR on y2 is gone (n_12 = 0), which is the
    collinearity BUG-0019 met in the rows= fit."""
    L = Ladder(specs(tmp_path, noise(120, 2), [(1, 1), (0, 0)]), echelon=(1, 0))
    assert [n for n in L.names() if "<-" in n] == ["AR0[y2<-y1]", "MA1[y1<-y2]"]


def test_the_cast_is_the_standard_form(tmp_path):
    """Every free value set; the cast's Phi, Theta = -A_0^-1 A_k, -A_0^-1 M_k
    of the LP-convention A(z), M(z) written from the same values."""
    L = Ladder(specs(tmp_path, noise(120, 3), [(1, 1), (2, 2), (1, 1)]), echelon=(1, 2, 1))
    rng = np.random.default_rng(1)
    v = L.pack()
    names = L.names()
    v[:] = np.where([("phi" in n or "theta" in n or "<-" in n) for n in names],
                    rng.uniform(-0.3, 0.3, v.size), v)
    mu, PHI, THETA, _qq, _w, ifault = L.cast(v)
    assert ifault == 0
    val = dict(zip(names, v))
    m, p = 3, 2
    A = np.zeros((p + 1, m, m)); M = np.zeros((p + 1, m, m))
    A[0] = np.eye(m)
    for i, s in enumerate(L.series):
        ph, th, _mu, _f = s.polynomials()
        for k, c in enumerate(ph, 1):
            A[k, i, i] = -c
        for k, c in enumerate(th, 1):
            M[k, i, i] = -c
    for n, x in val.items():
        if "<-" not in n:
            continue
        kind, k = n[:2], int(n[2:n.index("[")])
        a, b = n[n.index("[") + 1:-1].split("<-")
        i, j = int(a[1:]) - 1, int(b[1:]) - 1
        (A if kind == "AR" else M)[k, i, j] = -x
    M[0] = A[0]
    inv = np.linalg.inv(A[0])
    np.testing.assert_allclose(PHI, [-inv @ A[k] for k in range(1, p + 1)], atol=1e-12)
    np.testing.assert_allclose(THETA, [-inv @ M[k] for k in range(1, p + 1)], atol=1e-12)
    assert A[0][2, 1] != 0 and np.count_nonzero(A[0] - np.eye(m)) == 1


def test_equal_indices_are_the_varma(tmp_path):
    """Echelon (1,1) has A_0 = I and nothing restricted: the VARMA(1,1)."""
    f = specs(tmp_path, noise(150, 2, seed=3), [(1, 1), (1, 1)])
    a = Ladder(f, echelon=(1, 1)).fit()
    b = Ladder(f, 1, 1).fit()
    assert a.names == b.names
    assert a.logL == pytest.approx(b.logL, abs=1e-8)


# -- a system recovered --------------------------------------------------------- #

def test_a_simulated_10_system_is_recovered(tmp_path):
    """y1 = .6 y1(-1) + a1 + .4 a1(-1) + .3 a2(-1); y2 = .5 (y1 - a1) + a2:
    A_0 = [[1, 0], [-.5, 1]]. In the I - SUM Theta B convention the MA
    values are -.4 and -.3. 1500 observations; every estimate within 3
    standard errors, AR0 included."""
    A0 = np.array([[1.0, 0.0], [-0.5, 1.0]])
    A1 = np.array([[-0.6, 0.0], [0.0, 0.0]])
    M1 = np.array([[0.4, 0.3], [0.0, 0.0]])
    inv = np.linalg.inv(A0)
    phi, theta = [-inv @ A1], [-inv @ M1]
    sig = np.array([[1.0, 0.3], [0.3, 1.0]])
    y = simulate_varma(phi=phi, theta=theta, sigma=sig, n=1500, burnin=300, seed=11,
                       freq=1).data + 10.0
    L = Ladder(specs(tmp_path, y, [(1, 1), (0, 0)]), echelon=(1, 0))
    r = L.fit()
    est = dict(zip(r.names, zip(r.x, r.std_errors)))
    truth = {"phi_y1[B^1]": 0.6, "theta_y1[B^1]": -0.4, "MA1[y1<-y2]": -0.3,
             "AR0[y2<-y1]": 0.5}
    for n, t in truth.items():
        x, se = est[n]
        assert abs(x - t) < 3 * se, (n, x, se, t)
    lr, df, _p = L.lr_test()
    assert df == 2 + 1                         # AR0, MA1 and Q12


# -- what is refused, and what composes ---------------------------------------- #

def test_echelon_is_checked(tmp_path):
    f = specs(tmp_path, noise(120, 2), [(2, 1), (0, 0)])
    with pytest.raises(LadderError, match="degrees \\(2, 1\\) and its index is 1"):
        Ladder(f, echelon=(1, 0))
    f = specs(tmp_path, noise(120, 2), [(1, 1), (0, 0)])
    with pytest.raises(LadderError, match="2 indices"):
        Ladder(f, echelon=(1, 0, 0))
    with pytest.raises(LadderError, match="give 0, 0"):
        Ladder(f, 2, 2, echelon=(1, 0))
    with pytest.raises(LadderError, match="do not give rows"):
        Ladder(f, echelon=(1, 0), rows="y1(1,1)")
    with pytest.raises(LadderError, match="not an echelon form"):
        Ladder(f, echelon=(1, 0), cross="residual")
    with pytest.raises(LadderError, match="not a free coefficient"):
        Ladder(f, echelon=(1, 0), zeros="AR1[y1<-y2]")


def test_zeros_and_links_restrict_it_further(tmp_path):
    f = specs(tmp_path, noise(120, 2), [(1, 1), (0, 0)])
    L = Ladder(f, echelon=(1, 0), zeros="AR0[y2<-y1]")
    assert [n for n in L.names() if "<-" in n] == ["MA1[y1<-y2]"]
    L = Ladder(f, echelon=(1, 0), links="y2<-y1")
    assert [n for n in L.names() if "<-" in n] == ["AR0[y2<-y1]"]


def test_the_report_says_the_indices(tmp_path):
    f = specs(tmp_path, noise(150, 2, seed=5), [(1, 1), (0, 0)])
    L = Ladder(f, echelon=(1, 0))
    L.fit()
    out = tmp_path / "e.out"
    write_report(L, str(out))
    assert "Echelon form     : Kronecker indices (y1 1, y2 0)" in out.read_text()
