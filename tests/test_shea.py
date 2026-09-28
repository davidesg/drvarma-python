"""Shea's exact likelihood (AS 242) beside elf (AS 311): `marma_c`, Ladder(lik=).

Two independent algorithms for the same exact likelihood, so each is an oracle
for the other. The C engines have it as -lik shea|both (atsw-gui lib/lik; the
csrc copy of multshea.c is synced from there), and drtran's whole battery passes
with Shea as the objective (326/326). Pins: the C oracle's values, on the same
.pre files.
"""
import glob
import os

import numpy as np
import pytest

from drvarma._engine import elf_c, marma_c
from drvarma.ladder import Ladder, _concentrated

D = os.path.join(os.path.dirname(__file__), "data", "ladder")
PAIR = [os.path.join(D, "IPC_ES_m10.pre"), os.path.join(D, "IPC_FR_msar.pre")]
M6 = sorted(glob.glob(os.path.join(D, "m6", "*.pre")))


def _point(seed=0, m=2, n=120, p=1, q=1):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, m))
    phi = np.array([[[0.5, 0.1], [0.2, 0.3]]])[:p]
    theta = np.array([[[0.4, 0.0], [0.1, 0.2]]])[:q]
    qq = np.array([[1.0, 0.3], [0.3, 2.0]])
    return m, n, p, q, np.zeros(m), phi, theta, qq, w


def test_marma_and_elf_are_the_same_likelihood():
    """At a given point, with elf untruncated: rounding, nothing else."""
    for seed in range(5):
        m, n, p, q, mu, phi, theta, qq, w = _point(seed)
        _l, f1, f2, _a, ifa = elf_c(m, n, p, q, mu, phi, theta, qq, w, 1.0, -1e-3)
        _g, g1, g2, gfa = marma_c(m, n, p, q, mu, phi, theta, qq, w)
        assert ifa == 0 and gfa == 0
        assert abs(_concentrated(m, n, f1, f2) - _concentrated(m, n, g1, g2)) < 1e-9


def test_marma_refuses_what_elf_refuses():
    """The MA frontier is elf's (chekma): a non-invertible MA is ifault 4."""
    m, n, p, q, mu, phi, _t, qq, w = _point()
    theta = np.array([[[1.2, 0.0], [0.0, 0.3]]])
    assert elf_c(m, n, p, q, mu, phi, theta, qq, w, 1.0, -1e-3)[4] == 4
    assert marma_c(m, n, p, q, mu, phi, theta, qq, w)[3] == 4


@pytest.mark.parametrize("lik", ["elf", "shea"])
def test_ladder_reaches_the_c_oracles_likelihood(lik):
    """drvarma C -lik elf|shea -m 2 on the same pair: l = 92.566119."""
    r = Ladder(PAIR, 1, 0, method=2, lik=lik).fit()
    assert r.logL == pytest.approx(92.566119, abs=2e-6)


def test_both_checks_every_point():
    L = Ladder(PAIR, 1, 0, method=2, lik="both")
    L.fit()
    c = L.lik_check
    assert c["points"] > 100 and c["one_only"] == 0
    assert c["max"] < 1e-9 and c["at_optimum"] < 1e-9


def test_m6_gate_closes_with_shea():
    """The m6 gap (-0.000428) was elf's xi truncation: with Shea, untruncated,
    the gate closes to rounding, and the likelihood is the C oracle's."""
    if len(M6) != 6:
        pytest.skip("the m6 .pre files are missing")
    L = Ladder(M6, 0, 0, diagcov=True, method=2, lik="shea")
    r = L.fit()
    assert abs(L.gate["difference"]) < 1e-9
    assert r.logL == pytest.approx(-1752.522249, abs=2e-6)


def test_lik_must_be_known():
    with pytest.raises(ValueError):
        Ladder(PAIR, 0, 0, lik="exact")


# ── the stop, as the C writes it (report_stop, atsw-gui ca398b3) ───────────

def test_ma_boundary_counts_roots_on_the_wall():
    from drvarma.ladder import _ma_boundary
    assert _ma_boundary(np.zeros((0, 2, 2))) == (0, 0)
    inside = np.array([[[0.5, 0.0], [0.0, 0.3]]])
    wall = np.array([[[1.00003, 0.0], [0.0, 0.3]]])     # chekma accepts < 1.00005
    assert _ma_boundary(inside) == (0, 2)
    assert _ma_boundary(wall) == (1, 2)


def test_the_stop_block_is_the_cs():
    """On the ES/FR pair, ARMA(1,1) cross: the C writes exactly this block."""
    from drvarma.ladder import _stop_block
    r = Ladder(PAIR, 1, 1).fit()
    assert _stop_block(r)[1:4] == [
        "  OPTIMIZER CONVERGED after 24 iterations",
        "  Objective function = 0.649411187458",
        "  Convergence criterion: norm of scaled gradient <= gradtol"]


def test_a_stop_on_the_wall_says_so():
    from drvarma.ladder import Fit, _stop_block
    r = Fit(["x"], np.zeros(1), np.zeros(1), 0.0, 1.0, 0, nit=60, termcode=3,
            fk=0.5, ma_boundary=2, ma_nroots=24)
    b = _stop_block(r)
    assert b[1] == "  OPTIMIZER STOPPED at the MA invertibility boundary after 60 iterations"
    assert "  MA boundary: 2 of 24 inverse roots at modulus >= 1" in b


def test_refit_restarts_from_a_given_point():
    """From the optimum, a refit stays there; from a moved point, it climbs back."""
    L = Ladder(PAIR, 1, 1)
    r = L.fit()
    again = L.refit(r.x)
    assert again.logL == pytest.approx(r.logL, abs=1e-6)
    moved = r.x.copy()
    moved[-1] *= 0.9
    back = L.refit(moved)
    assert back.logL >= r.logL - 1e-6
    with pytest.raises(ValueError):
        L.refit(r.x[:-1])


# ── IRF/FEVD bands for the ladder model ─────────────────────────────────────

def test_ladder_irf_bands_contain_the_point_and_restore_the_state():
    from drvarma.irf import oirf
    L = Ladder([PAIR[0], os.path.join(D, "WTI_ar1.pre")], 1, 0)
    r = L.fit()
    assert r.cov is not None and r.cov.shape == (r.npar, r.npar)
    b = L.irf_fevd_bands(12, ndraws=300)
    pt = oirf(r.phi, r.theta, r.sigma, 12)
    assert np.all((pt >= b["oirf_lo"] - 1e-12) & (pt <= b["oirf_hi"] + 1e-12))
    assert b["ndraws_used"] + b["ndraws_rejected"] == 300
    _mu, phi, _t, _q, _w, _i = L.cast(r.x)
    np.testing.assert_allclose(phi, r.phi)


def test_no_covariance_no_bands():
    L = Ladder(PAIR, 1, 0)
    r = L.fit()
    r.cov = None
    with pytest.raises(ValueError, match="no usable covariance"):
        L.irf_fevd_bands(6, ndraws=50)


# ── the pure-Python port (_as242), against the compiled one ────────────────

def test_pure_python_shea_is_the_compiled_one():
    """Random VARMA(p,q) up to m=3, p,q <= 2: the port reproduces marma_c to
    rounding (measured: 1.1e-14 relative on the logL, 2.9e-11 absolute)."""
    from drvarma._as242 import marma
    rng = np.random.default_rng(1)
    for _ in range(25):
        m = int(rng.integers(1, 4)); p = int(rng.integers(0, 3)); q = int(rng.integers(0, 3))
        q = q or (0 if p else 1)
        n = int(rng.integers(20, 120))
        phi = rng.uniform(-.4, .4, (p, m, m)) / m
        theta = rng.uniform(-.5, .5, (q, m, m)) / m
        L = rng.standard_normal((m, m)) * .5 + np.eye(m)
        args = (m, n, p, q, rng.standard_normal(m) * .1, phi, theta, L @ L.T,
                rng.standard_normal((n, m)))
        a, b = marma_c(*args), marma(*args)
        assert a[3] == b[3]
        if a[3] == 0:
            assert abs(a[0] - b[0]) <= 1e-12 * abs(a[0])
            assert abs(a[1] - b[1]) <= 1e-12 * a[1] and abs(a[2] - b[2]) <= 1e-12 * a[2]


def test_pure_python_shea_refuses_what_elf_refuses():
    from drvarma._as242 import marma
    m, n, p, q, mu, phi, _t, qq, w = _point()
    theta = np.array([[[1.2, 0.0], [0.0, 0.3]]])
    assert marma(m, n, p, q, mu, phi, theta, qq, w)[3] == 4


def test_pure_python_shea_at_the_c_oracles_optimum():
    """At the ladder's optimum on ES/FR, the port's Shea is the C's."""
    from drvarma._as242 import marma
    L = Ladder(PAIR, 1, 0, method=2, lik="shea")
    r = L.fit()
    mu, phi, theta, qq, w, _i = L.cast(r.x)
    m, n = w.shape[1], w.shape[0]
    a = marma_c(m, n, phi.shape[0], theta.shape[0], mu, phi, theta, qq, w)
    b = marma(m, n, phi.shape[0], theta.shape[0], mu, phi, theta, qq, w)
    assert b[3] == 0 and abs(a[0] - b[0]) <= 1e-12 * abs(a[0])
