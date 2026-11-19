"""start="ols": the original drvarma's start (init_varma) brought to the ladder,
the same arithmetic as the C's -start ols (escalera.c).

The ladder's start plus the parameters of the relation: least squares of the
VAR(p) on the stationary series seeds the cross AR terms (Box and Jenkins:
+Phi_k) and Q's covariances (the residuals' correlation times sqrt(Q_ii Q_jj),
the Q_ii the gate's); the own model of a .pre stays. On the US monetary VAR
(bugs/BUG-0027-repro) the zero start stops after 2 iterations below the
nested VAR(1); from here it reaches the optimum."""
import math
import os

import numpy as np
import pytest

from drvarma.ladder import Ladder

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.join(HERE, "..", "bugs", "BUG-0027-repro")
FILES = [os.path.join(REPRO, f) for f in
         ("IPI_m07.pre", "IPC_m07.pre", "WTI_m03.pre", "GS1_m01.pre")]


def _stop_at_start(L):
    """The state right after the start is set (no optimisation)."""
    class Stop(Exception):
        pass
    orig = L._take_start

    def take(x0, name):
        orig(x0, name)
        raise Stop
    L._take_start = take
    with pytest.raises(Stop):
        L.fit()
    return L


def test_the_start_adds_only_the_relation():
    """Route U: the own models and the variances stay; the cross AR and the
    covariances are the least squares'."""
    z = _stop_at_start(Ladder(FILES, 2, 0, start="preliminary"))   # to read the state
    L = _stop_at_start(Ladder(FILES, 2, 0, start="ols"))
    assert L.start_used == "ols"
    m = len(L.series)
    W = np.asarray(L.cast(L.pack())[4], float)
    own = [s.model.ar for s in L.series]
    assert own == [s.model.ar for s in z.series]               # the .pre's, untouched
    assert np.allclose(L._lvar, z._lvar)                        # the gate's Q_ii
    # the covariances: the correlation in the ladder's normalisation
    Q = np.exp(L._lvar)
    for a in range(1, m):
        for b in range(a):
            r = L._qcov[a, b] / math.sqrt(Q[a] * Q[b])
            assert -1.0 < r < 1.0


def test_the_sign_is_box_and_jenkins():
    """The log-likelihood at the start: +Phi_k (Box and Jenkins) is near the
    optimum, -Phi_k far below it."""
    L = _stop_at_start(Ladder(FILES, 2, 0, start="ols"))
    mu, phi, th, qq, w, ifa = L.cast(L.pack())
    _l, f1, f2, _a, ifa2 = L._elf(mu, phi, th, qq, w)
    assert ifa == 0 and ifa2 == 0
    n, m = w.shape
    logL = (-0.5 * m * n * (math.log(2 * math.pi) - math.log(m) - math.log(n) + 1)
            - 0.5 * n * (m * math.log(f1) + math.log(f2)))
    assert -1752 < logL < -1741.2                    # the optimum is -1741.2755


def test_the_monetary_var2_reaches_the_optimum_from_ols():
    L = Ladder(FILES, 2, 0, start="ols")
    r = L.fit()
    assert L.start_used == "ols"
    assert abs(r.logL - (-1741.2755)) < 1e-3


def test_an_inadmissible_start_falls_back_to_zero_and_says_so():
    """Jenkins and Alavi's VAR(1): the full VAR's cross terms on the own
    univariate ARs are not stationary (elf ifault 3); zero is used."""
    d = os.path.join(HERE, "data", "ladder")
    jf = [os.path.join(d, f) for f in ("muskrat.pre", "mink.pre")]
    L = Ladder(jf, 1, 0, start="ols")
    L.fit()
    assert L.start_used.startswith("zero (the ols start was not admissible)")


def test_an_unknown_start_is_refused():
    with pytest.raises(ValueError, match="'ols'"):
        Ladder(FILES, 1, 0, start="mco")


def test_route_v_keeps_the_own_models_of_the_diagonal_fit():
    """The ladder's principle on route V too: the specifications (Tiao and
    Box's, the own ARMA at zero in the file) are estimated by the diagonal fit
    first, and the ols start adds only the relation -- the own coefficients
    at the start are those of the zero start, not the least squares'."""
    d = os.path.join(HERE, "data", "ladder", "gas_v")
    files = [os.path.join(d, f) for f in ("InputGasRate_v_p1q1.inp", "CO2_v_p1q1.inp")]
    z = _stop_at_start(Ladder(files, 1, 1, start="preliminary"))   # own: the diagonal's
    L = _stop_at_start(Ladder(files, 1, 1, start="ols"))
    assert L.start_used == "ols"
    own = lambda X: {nm: v for nm, v in zip(X.names(), X.pack())   # noqa: E731
                     if nm.startswith(("phi_", "theta_", "mu_"))}
    a, b = own(z), own(L)
    assert a and all(abs(a[k] - b[k]) < 1e-12 for k in a)
    assert np.allclose(L._lvar, z._lvar)
    assert any(abs(v) > 1e-3 for v in L._cAR.ravel())                # the relation, seeded
