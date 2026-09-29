"""Jenkins and Alavi's residual-model form (3.22): Ladder(cross='residual')."""
import glob
import os

import numpy as np
import pytest

from drvarma.ladder import Ladder

D = os.path.join(os.path.dirname(__file__), "data", "ladder")
PAIR = [os.path.join(D, "IPC_ES_m10.pre"), os.path.join(D, "IPC_FR_msar.pre")]
M6 = sorted(glob.glob(os.path.join(D, "m6", "*.pre")))


def _poly(M):
    return [np.eye(M.shape[1])] + [-M[l] for l in range(M.shape[0])]


def _mul(A, B):
    out = [np.zeros_like(A[0]) for _ in range(len(A) + len(B) - 1)]
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            out[i + j] += a @ b
    return out


def test_the_ma_is_the_univariate_ma_times_the_residual_model():
    """Row i of Theta(B) is theta_ii(B) times row i of U*(B), exactly."""
    if not M6:
        pytest.skip("m6 missing")
    L = Ladder([M6[0], M6[3]], 0, 2, cross="residual")        # EA (MA(4)), EP (MA(1))
    L.run_gate()
    L._set_structure([0, 1], 0, 2, False)
    L._cMA[:2] = np.random.default_rng(1).uniform(-.3, .3, (2, 2, 2))
    _mu, _phi, TH, _qq, _w, ifa = L.cast(L.pack())
    assert ifa == 0
    th = [L.series[i].polynomials()[1] for i in (0, 1)]
    Td = np.zeros((max(len(t) for t in th), 2, 2))
    for a, t in enumerate(th):
        Td[:len(t), a, a] = t
    U = L._cMA[:2].copy()
    for k in range(2):
        np.fill_diagonal(U[k], 0.0)
    want, got = _mul(_poly(Td), _poly(U)), _poly(TH)
    n = max(len(want), len(got))
    pad = lambda P: [P[i] if i < len(P) else np.zeros((2, 2)) for i in range(n)]  # noqa: E731
    for x, y in zip(pad(want), pad(got)):
        np.testing.assert_allclose(x, y, atol=1e-14)


def test_without_univariate_ma_the_two_forms_are_one():
    a = Ladder(PAIR, 1, 1).fit()
    r = Ladder(PAIR, 1, 1, cross="residual").fit()
    assert r.logL == pytest.approx(a.logL, abs=1e-9)


def test_on_m6_the_residual_form_stays_off_the_wall():
    """Measured 2026-09-29 on EI/EP, q = 1: additive -619.85 ON the MA wall,
    residual -624.67 interior, in half the iterations."""
    if not M6:
        pytest.skip("m6 missing")
    P2 = [os.path.join(D, "m6", "M6_EI.pre"), os.path.join(D, "m6", "M6_EP.pre")]
    r = Ladder(P2, 0, 1, cross="residual").fit()
    assert r.ma_boundary == 0 and r.termcode == 1
    assert r.logL == pytest.approx(-624.6653, abs=1e-3)


def test_cross_must_be_known():
    with pytest.raises(ValueError):
        Ladder(PAIR, 1, 1, cross="multiplicative")
