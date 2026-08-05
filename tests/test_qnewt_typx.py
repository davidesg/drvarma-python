"""`typx` — the typical parameter size the stopping tests are relative to.

`qnewtopt.c` hardcodes it to 1 (Dennis & Schnabel A9.4.1 in its simplified form;
the full algorithm takes it as an input). That is fine while the parameters are
of order 1 and wrong when they are not, which is what breaks drtran at
`refactor=1`. These tests pin both halves of the contract:

* `typx=None` must reproduce the C **bit-for-bit** — drvarma's own default path
  must not move a single ulp;
* a float must make the two tests scale-relative.
"""
import numpy as np
import pytest

from drvarma import _qnewt


def _legacy_max1(n, x, f, g):
    return max(abs(g[i]) * (abs(x[i]) + 1.0) / (abs(f) + 1.0) for i in range(1, n + 1))


def _legacy_max2(n, xk, xkp1):
    return max(abs(xkp1[i] - xk[i]) / (abs(xkp1[i]) + 1.0) for i in range(1, n + 1))


@pytest.mark.parametrize("scale", [1.0, 1e-2, 1e-4])
def test_typx_none_is_bit_for_bit_the_c(scale):
    """The default path must be EXACTLY the old arithmetic, at any scale."""
    rng = np.random.default_rng(11)
    n = 6
    xk = np.concatenate([[0.0], rng.normal(size=n) * scale])
    xkp1 = np.concatenate([[0.0], rng.normal(size=n) * scale])
    g = np.concatenate([[0.0], rng.normal(size=n) * 1e-6])
    f = 1.0

    # umstop0's gradient test
    for gradtol in (1e-7, _legacy_max1(n, xk, f, g)):
        want = 1 if _legacy_max1(n, xk, f, g) <= gradtol else 0
        got, _ = _qnewt.umstop0(n, xk, f, g, gradtol, 100)
        assert got == want

    # umstop: force each branch through the tolerances, retcode=0, k<maxits
    m1, m2 = _legacy_max1(n, xkp1, f, g), _legacy_max2(n, xk, xkp1)
    assert _qnewt.umstop(n, xk, xkp1, f, g, 0, m1, 0.0, 1, 100, 5, 0, 0)[0] == 1
    assert _qnewt.umstop(n, xk, xkp1, f, g, 0, m1 * 0.5, m2, 1, 100, 5, 0, 0)[0] == 2
    assert _qnewt.umstop(n, xk, xkp1, f, g, 0, m1 * 0.5, m2 * 0.5,
                         1, 100, 5, 0, 0)[0] == 0


def test_typx_makes_the_gradient_test_scale_relative():
    """The defect itself: same problem, 100x smaller parameters.

    With typx=1 the gradient test statistic blows up by ~100 and stops firing,
    although the problem is identical. With typx it is invariant.
    """
    n = 3
    big = np.array([0.0, 0.5, 0.2, 0.1])
    small = big / 100.0
    g_big = np.array([0.0, 1e-8, 2e-8, 1e-8])
    g_small = g_big * 100.0            # f is scale-invariant => g scales by 100
    f = 1.0

    tol = 1e-7
    assert _qnewt.umstop0(n, big, f, g_big, tol, 100)[0] == 1
    assert _qnewt.umstop0(n, small, f, g_small, tol, 100)[0] == 0, "the defect"

    # with typx the two agree again (floor below the smallest parameter)
    assert _qnewt.umstop0(n, big, f, g_big, tol, 100, 1e-4)[0] == 1
    assert _qnewt.umstop0(n, small, f, g_small, tol, 100, 1e-4)[0] == 1


def test_typx_floor_bounds_a_parameter_at_the_origin():
    """A parameter passing through 0 must not collapse typx to 0."""
    x = np.array([0.0, 0.0, 0.5])
    assert _qnewt.typical_size(x, 1, 1e-3) == 1e-3
    assert _qnewt.typical_size(x, 2, 1e-3) == 0.5
    assert _qnewt.typical_size(x, 1, None) is None


def test_raxopt_accepts_typx():
    assert "typx" in _qnewt.raxopt.__code__.co_varnames


def test_cdgrad_step_stays_absolute():
    """`cdgrad` must NOT take typx.

    For an objective of order 1 the absolute step eta^(1/3)*max(|x|,1) ~ 6.06e-6
    is near-optimal; a step relative to a ~1e-4 parameter would be ~6e-9, whose
    cancellation error eps*|f|/h ~ 3e-7 EXCEEDS gradtol. Measured: scaling the
    step changes neither the optimum nor the termcode.
    """
    assert "typx" not in _qnewt.cdgrad.__code__.co_varnames
