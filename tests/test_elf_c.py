"""`elf_c`: the exact likelihood scored at a GIVEN structure.

`estimate_w` fits a FREE VARMA(p, q) from the data. This is the other door:
score a Phi/Theta/Sigma the caller built — what a restricted model needs (a
transfer function, a network, anything whose structure comes from a cast).

Two things are pinned here, and both were real defects found while opening that
door:

1. **A diagonal Theta must return.** It used to hang. Not because of the
   algorithm: `macheps` is a *global* that each program's `main` sets, and
   entering through this new entry point nobody had set it, so it was 0 and the
   tolerances that depend on it stopped cutting. A diagonal Theta is not an
   exotic case — it is exactly what a per-series MA produces, which is what
   drtran's cast builds.

2. **A negative lower index must not corrupt the heap.** `elf` allocates
   `gamwa = tensor(-q+1, 0, ...)`, negative as soon as q >= 2. The NR-free
   rewrite of `nlatools.c` dropped the `t -= nrl` base offset, so `t[-1]` wrote
   before the allocation. `estimate_w` itself died on it with q >= 2.

The arbiter for the values is the pure-Python `_as311.elf`, which is the same
algorithm and was never affected.
"""

import numpy as np
import pytest

from drvarma import _as311
from drvarma._engine import elf_c, estimate_w

try:                                   # the refusal being tested is the C's
    from drvarma._drvarma_engine import lib as _lib   # noqa: F401
    _HAS_C_ENGINE = True
except ImportError:
    _HAS_C_ENGINE = False


def _elf_py(m, n, p, q, mu, phi, theta, qq, w, sigma2=1.0, xitol=-1e-3):
    """The pure-Python `elf`, called with the same structure."""
    Mu = np.zeros(m + 1); Mu[1:] = mu
    Phi = np.zeros((max(p, 1) + 1, m + 1, m + 1))
    for k in range(p):
        Phi[k + 1, 1:, 1:] = phi[k]
    Theta = np.zeros((max(q, 1) + 1, m + 1, m + 1))
    for k in range(q):
        Theta[k + 1, 1:, 1:] = theta[k]
    Qq = np.zeros((m + 1, m + 1)); Qq[1:, 1:] = qq
    W = np.zeros((n + 1, m + 1)); W[1:, 1:] = w
    return _as311.elf(m, n, p, q, Mu, Phi, Theta, Qq, W, sigma2, xitol, False)


def _caso(m=3, n=40, seed=5):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.normal(0, 1, (n, m)))


# ── 1. la Theta diagonal ─────────────────────────────────────────────────────
@pytest.mark.timeout(60)
def test_a_diagonal_theta_returns():
    """It used to hang forever: `macheps` was 0 through this entry point.

    The timeout is the assertion. Without the fix this test does not fail — it
    never ends, which is why the defect went unnoticed until a six-series system
    was estimated through here.
    """
    m, n = 3, 40
    w = _caso(m, n)
    theta = np.zeros((2, m, m))
    theta[0] = np.diag([0.64, 0.43, 0.88])
    theta[1] = np.diag([0.0, 0.0, -0.37])
    phi = np.zeros((1, m, m))

    lg, f1, f2, _a, ifault = elf_c(m, n, 1, 2, np.zeros(m), phi,
                                   np.ascontiguousarray(theta), np.eye(m), w)
    assert ifault == 0
    assert np.isfinite(lg) and f1 > 0 and f2 > 0

    esperado = _elf_py(m, n, 1, 2, np.zeros(m), phi, theta, np.eye(m), w)[0]
    assert lg == pytest.approx(esperado, abs=1e-8)


# ── 2. el indice inferior negativo ───────────────────────────────────────────
@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_estimate_w_survives_q_ge_2(q):
    """`gamwa = tensor(-q+1, 0, ...)`: el indice inferior es negativo con q >= 2.

    Sin el desplazamiento de base, esto no daba un ifault: reventaba el proceso
    con "double free or corruption".
    """
    r = estimate_w(_caso(3, 120), 1, q, include_mean=False)
    assert r["ifault"] == 0
    assert np.isfinite(r["logelf"])


# ── 3. paridad con el algoritmo de referencia ────────────────────────────────
@pytest.mark.parametrize("p,q", [(2, 1), (1, 2), (1, 4), (0, 4), (3, 0)])
def test_matches_the_pure_python_elf(p, q):
    m, n = 3, 120
    rng = np.random.default_rng(7)
    w = np.ascontiguousarray(rng.normal(0, 1, (n, m)))
    phi = rng.normal(0, 0.1, (p, m, m)) if p else np.zeros((0, m, m))
    theta = rng.normal(0, 0.1, (q, m, m)) if q else np.zeros((0, m, m))

    lg, _f1, _f2, _a, ifault = elf_c(m, n, p, q, np.zeros(m),
                                     np.ascontiguousarray(phi),
                                     np.ascontiguousarray(theta), np.eye(m), w)
    assert ifault == 0
    assert lg == pytest.approx(
        _elf_py(m, n, p, q, np.zeros(m), phi, theta, np.eye(m), w)[0], abs=1e-8)


@pytest.mark.skipif(not _HAS_C_ENGINE,
                    reason="asserts the C entry point's refusal; without the "
                           "engine `elf_c` falls back to the pure-Python "
                           "implementation, which handles p=q=0 fine")
def test_the_degenerate_p0_q0_is_refused_not_crashed():
    """A VARMA with neither AR nor MA is white noise, and the engine cannot take
    it: with g = max(p, q) = 0 it allocates a degenerate matrix and segfaults.

    The limitation is PRE-EXISTING -- `estimate_w(w, 0, 0)` dumps core too, and
    the pure-Python `_as311.elf` handles it fine -- but this entry point refuses
    it cleanly rather than pass it on. A returned ifault is an answer; a core
    dump is not.
    """
    m, n = 3, 60
    lg, _f1, _f2, _a, ifault = elf_c(m, n, 0, 0, np.zeros(m),
                                     np.zeros((0, m, m)), np.zeros((0, m, m)),
                                     np.eye(m), _caso(m, n))
    assert ifault != 0


def test_residuals_come_back_when_asked():
    m, n = 3, 60
    w = _caso(m, n, seed=11)
    theta = np.ascontiguousarray(np.zeros((1, m, m)) + np.diag([0.5, 0.3, 0.2]))
    _lg, _f1, _f2, a, ifault = elf_c(m, n, 0, 1, np.zeros(m),
                                     np.zeros((0, m, m)), theta, np.eye(m), w,
                                     atf=True)
    assert ifault == 0
    assert a.shape == (n + 1, m + 1)          # 1-indexado, como `_as311.elf`
    assert np.any(a[1:, 1:] != 0.0)
