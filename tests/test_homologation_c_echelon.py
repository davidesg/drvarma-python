"""The re-homologation with the C of November 2026, fourth part: the echelon
form on route V (the C's -echelon, atsw-gui #213-#217). The numbers are the
C's: engines/drvarma/tests/escalera, sections 10h, 10j and 10k, and
tests/rutas (the echelon cases)."""
import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from drvarma.ladder import Ladder  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "data", "ladder")


def F(*names):
    return [os.path.join(D, n + ".pre") for n in names]


IW = F("IPC_m07", "WTI_m03")


@pytest.mark.parametrize("n, logL, npar, a0", [
    ((1, 0), -638.329701, 8, True), ((1, 1), -630.843671, 12, False),
    ((2, 1), -629.155884, 16, True), ((1, 2), -628.703020, 15, False)])
def test_the_echelon_form_of_ipc_and_wti(n, logL, npar, a0):
    L = Ladder(IW, echelon=n, route="v")
    r = L.fit()
    assert L.start == "zero" and L.start_used == "zero"       # the default, as the C
    assert r.logL == pytest.approx(logL, abs=1e-5)
    assert L.npar() == npar
    assert ("AR0[WTI<-IPC]" in r.names) is a0


def test_equal_indices_are_route_v():
    e = Ladder(IW, echelon=(1, 1), route="v").fit()
    v = Ladder(IW, 1, 1, route="v").fit()
    assert e.logL == pytest.approx(v.logL, abs=1e-6)


def test_three_and_four_series():
    r = Ladder(F("IPI_m07", "IPC_m07", "WTI_m03"), echelon=(0, 1, 1), route="v").fit()
    assert r.logL == pytest.approx(-835.732462, abs=1e-5)             # the C's
    L = Ladder(F("IPI_m07", "IPC_m07", "WTI_m03", "FF_m03"), echelon=(0, 1, 0, 1), route="v")
    r = L.fit()
    assert r.logL == pytest.approx(-1693.338788, abs=1e-5)            # tests/rutas
    assert L.npar() == 26


def test_the_stage_two_start_is_asked_for():
    """start="echelon": the same optimum and the C's iteration counts where
    the search certifies; the roots pulled inside by the C's lambdas."""
    for n, logL, nit in (((1, 0), -638.329701, 38), ((1, 1), -630.843671, 50)):
        L = Ladder(IW, echelon=n, route="v", start="echelon")
        r = L.fit()
        assert L.start_used == "echelon"
        assert r.logL == pytest.approx(logL, abs=1e-5) and r.nit == nit
    L = Ladder(IW, echelon=(2, 1), route="v", start="echelon")
    L.fit()
    assert L.start_used == "echelon (AR x 0.388^k, MA x 0.459^k: inside the unit circle)"


def test_the_row_factor_test_with_unequal_indices():
    L = Ladder(F("WTI_m03", "IPI_m07"), echelon=(1, 0), route="v")
    r = L.fit()
    assert "AR0[IPI<-WTI]" in r.names
    ft = L.factor_test()
    assert ft["df"] == 1 and ft["lr"] == pytest.approx(1.5851, abs=1e-3)   # the C's
    L = Ladder(F("IPC_m07", "WTI_m03", "IPI_m07"), echelon=(1, 1, 0), route="v")
    L.fit()
    ft = L.factor_test()
    assert ft["df"] == 2 and ft["lr"] == pytest.approx(0.9379, abs=1e-3)   # the C's


def test_own_zeros_in_the_echelon_form():
    L = Ladder(IW, echelon=(2, 1), route="v", own_zeros={"IPC": {"ar": {2}, "ma": {1}}})
    r = L.fit()
    assert r.logL == pytest.approx(-630.160565, abs=1e-5)             # the C's
    assert L.npar() == 14


def test_own_free_gives_the_same_model():
    """own="free" (the substitution before the gate, up to 0.3.x) and
    route="v" (after it, as the C) are the same model."""
    a = Ladder(IW, echelon=(2, 1), own="free", start="zero").fit()
    b = Ladder(IW, echelon=(2, 1), route="v").fit()
    assert a.logL == pytest.approx(b.logL, abs=1e-6)


def test_what_is_only_in_the_port_and_leaves():
    with pytest.warns(DeprecationWarning, match="preliminary"):
        Ladder(IW, 1, 1, start="preliminary")
    with pytest.warns(DeprecationWarning, match="rows="):
        Ladder(IW, rows={"IPC": (1, 0), "WTI": (1, 0)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        Ladder(IW, 1, 1, start="ols")
