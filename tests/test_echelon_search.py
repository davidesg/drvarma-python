"""The echelon form's index search and its least-squares start:
drvarma.echelon, Lutkepohl and Poskitt (1996) stages I-II, and
Ladder(echelon=n, start="echelon") -- stage 3b of docs/ECHELON_PLAN.md."""
import numpy as np
import pytest

from drvarma import echelon as E
from drvarma.datasets import simulate_varma
from drvarma.ladder import Ladder, LadderError

from test_echelon import specs


def system_10():
    """y1 = .6 y1(-1) + a1 + .4 a1(-1) + .3 a2(-1); y2 = .5 (y1 - a1) + a2:
    indices (1, 0), LP's convention."""
    A = np.zeros((2, 2, 2)); M = np.zeros((2, 2, 2))
    A[0] = [[1.0, 0.0], [-0.5, 1.0]]
    A[1] = [[-0.6, 0.0], [0.0, 0.0]]
    M[0] = A[0]
    M[1] = [[0.4, 0.3], [0.0, 0.0]]
    return A, M


def simulate(A, M, n, seed):
    phi, theta = E.standard(A, M)
    sig = np.array([[1.0, 0.3], [0.3, 1.0]])
    return simulate_varma(phi=phi, theta=theta, sigma=sig, n=n, burnin=300, seed=seed,
                          freq=1).data + 10.0


# -- the pieces --------------------------------------------------------------- #

def test_the_table_and_the_dimension():
    """LP's (1,2,1): 15 cross coefficients (test_echelon's names), plus the
    own 1 + 2 + 1 AR and MA."""
    assert E.index_table((1, 2, 1)) == [[1, 1, 1], [1, 2, 1], [1, 2, 1]]
    assert E.dim((1, 2, 1)) == 15 + 2 * 4
    assert E.dim((0, 0)) == 0


def test_residuals_invert_the_model():
    """The recursion's residuals of a simulated path are its innovations,
    up to the start-up and the demeaning."""
    A, M = system_10()
    phi, theta = E.standard(A, M)
    sim = simulate_varma(phi=phi, theta=theta, sigma=np.eye(2), n=300, burnin=0, seed=3,
                         freq=1)
    e = E.residuals(sim.data, A, M)
    w = sim.data - sim.data.mean(0)
    # the model applied to w gives back the residuals: A(B) w = M(B) e
    for t in range(1, 300):
        lhs = A[0] @ w[t] + A[1] @ w[t - 1]
        rhs = M[0] @ e[t] + M[1] @ e[t - 1]
        np.testing.assert_allclose(lhs, rhs, atol=1e-10)


def test_pull_inside_scales_every_root():
    A, M = system_10()
    M2 = M.copy(); M2[1] *= 4.0                    # an MA root outside
    phi, theta = E.standard(A, M2)
    r = E.max_inverse_root(theta)
    assert r > 1
    A3, M3, la, lm = E.pull_inside(A, M2)
    assert la == 1.0 and lm == pytest.approx(0.95 / r)
    _p, th3 = E.standard(A3, M3)
    assert E.max_inverse_root(th3) == pytest.approx(0.95)
    np.testing.assert_array_equal(M3[0], A3[0])
    np.testing.assert_array_equal(M3[1] == 0, M2[1] == 0)   # the zeros stay zeros


def test_stage2_recovers_a_system():
    """nobs 3000: the least-squares estimates are consistent; A_0's entry
    within 0.1 of the truth."""
    A, M = system_10()
    y = simulate(A, M, 3000, seed=4)
    h, e, _aic = E.long_var(y)
    f = E.stage2(y, e, h, (1, 0))
    assert f.A[0, 1, 0] == pytest.approx(-0.5, abs=0.1)
    assert f.A[1, 0, 0] == pytest.approx(-0.6, abs=0.1)
    assert f.M[1, 0, 1] == pytest.approx(0.3, abs=0.1)
    assert f.dim == E.dim((1, 0)) == 4


def test_excluded_positions_stay_at_zero():
    A, M = system_10()
    y = simulate(A, M, 400, seed=5)
    h, e, _aic = E.long_var(y)
    f = E.stage2(y, e, h, (1, 0), exclude={("M", 1, 0, 1)})
    assert f.M[1, 0, 1] == 0.0 and f.dim == 3


def test_the_search_finds_the_indices():
    A, M = system_10()
    y = simulate(A, M, 500, seed=6)
    s = E.search(y, n_max=2)                    # SC, full enumeration
    assert s.indices == (1, 0) and len(s.fits) == 9
    hk = E.search(y, n_max=2, full=False)
    assert hk.indices == (1, 0) and len(hk.fits) < len(s.fits)
    plain = E.search(y, n_max=2, iterate=0)
    assert plain.fits[(1, 0)].logdet != s.fits[(1, 0)].logdet
    assert "<-" in s.text(["y1", "y2"]) and "(1, 0)" in s.text()


def test_the_search_is_checked():
    y = np.random.default_rng(0).standard_normal((40, 3))
    with pytest.raises(ValueError, match="criterion"):
        E.search(y, criterion="AIC")
    with pytest.raises(ValueError, match="too few"):
        E.search(y, n_max=3)


# -- the start of exact ML ---------------------------------------------------- #

def test_the_ladder_starts_at_the_stage2_estimates(tmp_path):
    A, M = system_10()
    y = simulate(A, M, 400, seed=7)
    f = specs(tmp_path, y, [(1, 1), (0, 0)])
    L = Ladder(f, echelon=(1, 0))                # the form's default start
    assert L.start == "echelon"
    r = L.fit()
    assert L.start_used.startswith("echelon")
    x0 = dict(zip(r.names, L.x_start))
    fit = L.echelon_stage2
    assert x0["AR0[y2<-y1]"] == pytest.approx(-fit.A[0, 1, 0])
    assert x0["phi_y1[B^1]"] == pytest.approx(-fit.A[1, 0, 0])
    z = Ladder(f, echelon=(1, 0), start="zero").fit()
    assert r.logL == pytest.approx(z.logL, abs=1e-3)        # the same optimum


def test_start_echelon_needs_the_form(tmp_path):
    y = 10.0 + np.random.default_rng(1).standard_normal((120, 2))
    f = specs(tmp_path, y, [(1, 1), (1, 1)])
    with pytest.raises(LadderError, match="give echelon"):
        Ladder(f, 1, 1, start="echelon")
    assert Ladder(f, 1, 1).start == "zero"          # the VARMA keeps the C's
