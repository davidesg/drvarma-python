"""The echelon form with a row factor -- stage 4 of docs/ECHELON_PLAN.md: a
model's seasonal and fixed-frequency factors multiply its whole row,

    F_r(B) [A(B) w]_r = G_r(B) [M(B) a]_r,

and only its regular ARMA is the row's own part, of degree <= n_r. The pins:
the split of a model into the two, the cast against the product written by
hand, the row filter, a simulated quarterly system recovered, and
own="free" (LP's row in place of a route-U model's regular part)."""
import numpy as np
import pytest

from drvarma import echelon as E
from drvarma.datasets import simulate_varma
from drvarma.ladder import Ladder, LadderError

from test_echelon import _operator


def spec(path, name, y, p, q, P=0, Q=0, freq=4, ar=None, sar=None):
    """A fue .inp: lambda 1, d = 0, the mean free, a free regular AR(p)/MA(q)
    and annual AR(P)/MA(Q) (``ar``, ``sar``: their starting values)."""
    def op(title, order, vals=None):
        if order <= 0:
            return [f"** {title}:", "0"]
        vals = vals or [0.0] * order
        return [f"** {title}:", f"1 {order}", "**"] + [f"{v:.6f}  1" for v in vals]
    L = ["* a specification with a row factor", "",
         "** Frequency of time series: either 1(A), 4(Q) or 12(M):", f" {freq}",
         "** Number of observations and starting date of time series:",
         f" {len(y)}  1 2000 {name}",
         "** Number of deterministic variables (including seasonal components):", "0"]
    L += op("Number and orders of regular AR operators", p, ar)
    L += op("Number and orders of annual AR operators", P, sar)
    L += op("Number and orders of regular MA operators", q)
    L += op("Number and orders of anual MA operators", Q)
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


def noise(n, m, seed=0):
    return 10.0 + np.random.default_rng(seed).standard_normal((n, m))


def times(c, f):
    """(1 - SUM f_j B^j) times the scalar polynomial with coefficients c
    (c[0] at lag 0), by np.convolve: the check, written apart from
    echelon.times_row_factors."""
    return np.convolve(np.r_[1.0, -np.asarray(f, float)], c)


# The quarterly system: LP's (1, 0) of test_echelon_search, with the
# factors F_1 = 1 - .5 B^4 on row 1 (an annual AR) and G_2 = 1 - .4 B^4 on
# row 2 (an annual MA). fue's values: Phi_y1[B^4] = .5, Theta_y2[B^4] = .4.
F1 = [0.0, 0.0, 0.0, 0.5]
G2 = [0.0, 0.0, 0.0, 0.4]


def system():
    A = np.zeros((2, 2, 2)); M = np.zeros((2, 2, 2))
    A[0] = [[1.0, 0.0], [-0.5, 1.0]]
    A[1] = [[-0.6, 0.0], [0.0, 0.0]]
    M[0] = A[0]
    M[1] = [[0.4, 0.3], [0.0, 0.0]]
    return (E.times_row_factors(A, [F1, []]), E.times_row_factors(M, [[], G2]))


def simulate(n, seed):
    A, M = system()
    K = max(len(A), len(M))
    A = np.concatenate([A, np.zeros((K - len(A), 2, 2))])
    M = np.concatenate([M, np.zeros((K - len(M), 2, 2))])
    phi, theta = E.standard(A, M)
    sig = np.array([[1.0, 0.3], [0.3, 1.0]])
    return simulate_varma(phi=phi, theta=theta, sigma=sig, n=n, burnin=300, seed=seed,
                          freq=4).data + 10.0


def system_specs(tmp_path, y, p1=1):
    return [spec(tmp_path / "y1.inp", "y1", y[:, 0], p1, p1, P=1),
            spec(tmp_path / "y2.inp", "y2", y[:, 1], 0, 0, Q=1)]


# -- the pieces --------------------------------------------------------------- #

def test_times_row_factors_is_the_product_by_rows():
    A, _M = system()
    assert A.shape == (6, 2, 2)                  # lags 0..1, times B^4
    base = np.array([1.0, -0.6])                 # row 1, own: A_0 + A_1 B
    np.testing.assert_allclose(A[:, 0, 0], times(base, F1))
    np.testing.assert_allclose(A[:, 1], np.r_[[[-0.5, 1.0]], np.zeros((5, 2))])
    np.testing.assert_allclose(E.times_row_factors(A, [[], []]), A)


def test_row_filter_inverts_the_ma_factor():
    """z = (F/G)(B) w satisfies G(B) z = F(B) w, after the pre-sample."""
    w = np.random.default_rng(2).standard_normal((60, 2))
    f, g = [0.2, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.4]
    z = E.row_filter(w, f, g)
    for c in range(2):
        lhs = np.convolve(np.r_[1.0, -np.array(g)], z[:, c])[:60]
        rhs = np.convolve(np.r_[1.0, -np.array(f)], w[:, c])[:60]
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_a_model_splits_into_its_regular_part_and_its_factor(tmp_path):
    """AR(1) x annual AR(1), MA(1) x annual MA(1): the two parts, whose
    products are the model's own polynomials."""
    f = spec(tmp_path / "a.inp", "a", noise(80, 1)[:, 0], 1, 1, P=1, Q=1,
             ar=[0.3], sar=[0.5])
    s = Ladder([f]).series[0]
    s.x[:] = [0.3, 0.5, -0.2, 0.4, s.x[-1]]       # ar, ar_s, ma, ma_s, mu
    phi, theta, fphi, ftheta, mu, ifault = s.split_polynomials()
    assert ifault == 0 and s.has_row_factor()
    np.testing.assert_allclose(phi, [0.3])
    np.testing.assert_allclose(fphi, [0, 0, 0, 0.5])
    np.testing.assert_allclose(theta, [-0.2])
    np.testing.assert_allclose(ftheta, [0, 0, 0, 0.4])
    ph, th, _mu, _if = s.polynomials()
    np.testing.assert_allclose(-times(np.r_[1.0, -phi], fphi)[1:], ph, atol=1e-12)
    np.testing.assert_allclose(-times(np.r_[1.0, -theta], ftheta)[1:], th, atol=1e-12)
    assert s.regular_orders() == (1, 1) and s.orders() == (5, 5)


# -- the structure ------------------------------------------------------------ #

def test_the_cast_multiplies_each_row_by_its_factor(tmp_path):
    """Every free value set: the cast's Phi, Theta are the standard form of
    the rows of [A_0 + A_1 B] times F_r(B) (and the MA's times G_r(B)),
    written by hand, AR0 included."""
    L = Ladder(system_specs(tmp_path, noise(120, 2)), echelon=(1, 0), start="zero")
    rng = np.random.default_rng(1)
    names = L.names()
    v = L.pack()
    v[:] = np.where([("phi" in n or "theta" in n or "Phi" in n or "Theta" in n
                      or "<-" in n) for n in names], rng.uniform(-0.4, 0.4, v.size), v)
    _mu, PHI, THETA, _qq, _w, ifault = L.cast(v)
    assert ifault == 0
    val = dict(zip(names, v))
    a0 = val["AR0[y2<-y1]"]
    A0 = np.array([[1.0, 0.0], [-a0, 1.0]])
    # LP's rows, lag 0 first: row 1 own AR(1) and MA(1), its MA on y2;
    # row 2 only A_0. Then the factors: F on row 1's AR, G on row 2's MA.
    rowsA = [[np.array([1.0, -val["phi_y1[B^1]"]]), np.array([0.0, 0.0])],
             [np.array([-a0]), np.array([1.0])]]
    rowsM = [[np.array([1.0, -val["theta_y1[B^1]"]]), np.array([0.0, -val["MA1[y1<-y2]"]])],
             [np.array([-a0]), np.array([1.0])]]
    fA = [[0, 0, 0, val["Phi_y1[B^4]"]], []]
    fM = [[], [0, 0, 0, val["Theta_y2[B^4]"]]]
    K = 6
    A = np.zeros((K, 2, 2)); M = np.zeros((K, 2, 2))
    for r in range(2):
        for c in range(2):
            a = times(rowsA[r][c], fA[r]); A[:a.size, r, c] = a
            b = times(rowsM[r][c], fM[r]); M[:b.size, r, c] = b
    np.testing.assert_allclose(A[0], A0); np.testing.assert_allclose(M[0], A0)
    phi, theta = E.standard(A, M)
    np.testing.assert_allclose(PHI, phi[:len(PHI)], atol=1e-12)
    np.testing.assert_allclose(THETA, theta[:len(THETA)], atol=1e-12)
    assert not np.any(np.array(phi[len(PHI):])) and not np.any(np.array(theta[len(THETA):]))


def test_without_cross_terms_it_is_the_diagonal_model(tmp_path):
    """The gate's system: the row factor times the regular part is the
    model's own, so the diagonal fit is the univariate models' sum."""
    L = Ladder(system_specs(tmp_path, noise(150, 2, seed=4)), echelon=(1, 0), start="zero")
    L._set_structure([0, 1], 0, 0, False)
    _mu, P0, T0, _q, _w, _if = L.cast(L.pack())
    L._set_structure([0, 1], L.p, L.q, False)
    _mu, P1, T1, _q, _w, _if = L.cast(L.pack())     # every cross term at zero
    np.testing.assert_allclose(P1[:len(P0)], P0, atol=1e-14)
    np.testing.assert_allclose(T1[:len(T0)], T0, atol=1e-14)


# -- a system recovered --------------------------------------------------------- #

def test_a_seasonal_system_is_recovered(tmp_path):
    """1500 quarterly observations of the system above: stage II, on the
    rows filtered by their factors, near the truth; exact ML from it, every
    estimate within 3 standard errors, the factors included."""
    y = simulate(1500, seed=11)
    L = Ladder(system_specs(tmp_path, y), echelon=(1, 0))
    r = L.fit()
    assert L.start_used == "echelon"
    st = L.echelon_stage2
    A, M = system()
    assert abs(st.A[1, 0, 0] - A[1, 0, 0]) < 0.1 and abs(st.A[0, 1, 0] - A[0, 1, 0]) < 0.1
    est = dict(zip(r.names, zip(r.x, r.std_errors)))
    truth = {"phi_y1[B^1]": 0.6, "theta_y1[B^1]": -0.4, "MA1[y1<-y2]": -0.3,
             "AR0[y2<-y1]": 0.5, "Phi_y1[B^4]": 0.5, "Theta_y2[B^4]": 0.4}
    for n, t in truth.items():
        x, se = est[n]
        assert abs(x - t) < 3 * se, (n, x, se, t)
    assert r.termcode == 1


def test_the_search_filters_each_row(tmp_path):
    """The index search on the stationary series, each row filtered by its
    factor, finds (1, 0)."""
    y = simulate(600, seed=5)
    L = Ladder(system_specs(tmp_path, y), echelon=(1, 0))
    filters = [tuple(s.split_polynomials()[2:4]) for s in L.series]
    for s, (fp, ft) in zip(L.series, filters):       # the true factors
        assert fp.size + ft.size == 4
    filters = [(np.array(F1), np.zeros(0)), (np.zeros(0), np.array(G2))]
    w = y - y.mean(0)
    assert E.search(w, n_max=2, filters=filters).indices == (1, 0)


# -- own="free", and what is refused --------------------------------------------- #

def test_own_free_gives_the_row_its_own_part(tmp_path):
    """A route-U model whose regular AR(2) does not fit in index 1: refused
    as it is; with own="free" its regular part is LP's free AR(1)/MA(1),
    the annual factor kept at the file's value."""
    y = noise(150, 2, seed=6)
    f = [spec(tmp_path / "u1.inp", "u1", y[:, 0], 2, 0, P=1, ar=[0.3, 0.1], sar=[0.4]),
         spec(tmp_path / "u2.inp", "u2", y[:, 1], 0, 0)]
    with pytest.raises(LadderError, match="regular ARMA has degrees \\(2, 0\\) and its "
                                          "index is 1.*at least 2"):
        Ladder(f, echelon=(1, 0))
    L = Ladder(f, echelon=(1, 0), own="free", start="zero")
    nm = L.names()
    assert [n for n in nm if n.endswith("u1[B^1]") or "u1[B^4]" in n] == \
        ["phi_u1[B^1]", "Phi_u1[B^4]", "theta_u1[B^1]"]
    assert dict(zip(nm, L.pack()))["Phi_u1[B^4]"] == pytest.approx(0.4)
    assert L.series[0].regular_orders() == (1, 1)
    with pytest.raises(LadderError, match="own='free' is the echelon form's"):
        Ladder(f, own="free")
    with pytest.raises(ValueError, match="own must be"):
        Ladder(f, echelon=(1, 0), own="other")
