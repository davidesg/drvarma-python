"""The re-homologation with the C of November 2026, third part: route V
through the gate (the C's -route v, atsw-gui #210), its whole-matrix seed
(#211), the row factor's test (#212) and the own zeros (#219). The numbers
are the C's: engines/drvarma/tests/escalera, sections 10f, 10g and 10l, and
tests/rutas (FF-Vg)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from drvarma.ladder import Ladder, LadderError  # noqa: E402

HERE = os.path.dirname(__file__)
D = os.path.join(HERE, "data", "ladder")


def F(*names):
    return [os.path.join(D, n + ".pre") for n in names]


GAS = [os.path.join(HERE, "..", "bugs", "BUG-0028-repro", n) for n in ("GAS_v.inp", "CO2_v.inp")]


def test_golden_var1_route_v_is_route_u():
    """Univariate models whose regular part IS an AR(p), no seasonal factors:
    route V replaces an AR(p) by a free AR(p), the same model."""
    u = Ladder(F("WTI_m03", "GS2_m05"), 1, 0, start="zero")
    ru = u.fit()
    v = Ladder(F("WTI_m03", "GS2_m05"), 1, 0, route="v", start="zero")
    rv = v.fit()
    assert rv.logL == pytest.approx(ru.logL, abs=1e-9)
    assert list(rv.names) == list(ru.names)
    assert rv.x == pytest.approx(ru.x, abs=1e-9)
    assert rv.logL == pytest.approx(-1689.702669, abs=1e-5)          # the C's
    assert v.route_notes == []


def test_golden_var3_and_the_whole_matrix_seed():
    v = Ladder(F("GS1_m01", "FF_m03"), 3, 0, route="v")
    rv = v.fit()
    assert v.start_used == "ols, the whole matrix"
    assert rv.logL == pytest.approx(-1747.072002, abs=1e-5)          # the C's, U and V
    assert rv.nit == 17                                              # the C's iterations


def test_the_row_factor_on_the_diagonal_is_the_univariate():
    """IPI is AR(3) x AR(1)_12. With p = 3 the model without cross terms is
    the univariate models (the gate's sum); with cross terms it is not route
    U's, because the AR(1)_12 multiplies the cross terms of its row too."""
    v = Ladder(F("IPI_m07", "FF_m03"), 3, 0, route="v")
    rv = v.fit()
    assert v.logL_diag == pytest.approx(v.gate["sum"], abs=1e-5)
    assert v.logL_diag == pytest.approx(-1066.090980, abs=1e-5)      # the C's
    assert rv.logL == pytest.approx(-1051.419251, abs=1e-5)          # the C's
    ru = Ladder(F("IPI_m07", "FF_m03"), 3, 0).fit()
    assert abs(ru.logL - rv.logL) > 0.5


def test_varma11_with_a_row_factor():
    """The C's -route v 1 1, and this port's echelon (1, 1): -924.846133."""
    for start, nit in (("zero", 46), ("ols", 48)):
        v = Ladder(F("IPI_m07", "WTI_m03"), 1, 1, route="v", start=start)
        rv = v.fit()
        assert rv.logL == pytest.approx(-924.846133, abs=1e-5)
        assert v.logL_diag == pytest.approx(-929.168188, abs=1e-5)
        assert v.npar() == 13 and rv.nit == nit
    assert v.route_notes == ["IPI's univariate model has a regular AR(3): this model, "
                             "with p = 1, does not contain it."]
    e = Ladder(F("IPI_m07", "WTI_m03"), echelon=(1, 1), own="free", start="zero").fit()
    assert e.logL == pytest.approx(rv.logL, abs=1e-6)


def test_the_monetary_var1():
    v = Ladder(F("IPI_m07", "IPC_m07", "WTI_m03", "FF_m03"), 1, 0, route="v")
    rv = v.fit()
    assert rv.logL == pytest.approx(-1695.296163, abs=1e-5)          # the C's (tests/rutas)
    assert v.npar() == 29 and len(v.route_notes) == 3
    ft = v.factor_test()
    assert ft["df"] == 4
    assert ft["lr"] == pytest.approx(6.1633, abs=1e-3)               # the C's


def test_the_gas_furnace_whole_matrix_seed_is_admissible():
    """VAR(6): route U's seed (cross terms only, on the diagonal fit's own
    models) is not admissible and goes back to zero; route V's is."""
    u = Ladder(GAS, 6, 0)
    ru = u.fit()
    v = Ladder(GAS, 6, 0, route="v")
    rv = v.fit()
    assert u.start_used == "zero (the ols start was not admissible)"
    assert v.start_used == "ols, the whole matrix"
    assert rv.logL == pytest.approx(-2638.762693, abs=1e-5)          # the C's
    assert ru.logL == pytest.approx(rv.logL, abs=1e-5)
    assert rv.nit == 73                                              # the C's iterations


def test_the_row_factor_test():
    """Against an independent path in the C (a hand-made file and -zero): the
    model without the factor, -925.888325 with 11 parameters."""
    v = Ladder(F("IPI_m07", "WTI_m03"), 1, 0, route="v")
    v.fit()
    ft = v.factor_test()
    assert ft["df"] == 2 and ft["free"].npar() == 11
    assert ft["logL_free"] == pytest.approx(-925.888325, abs=1e-5)
    assert ft["lr"] == pytest.approx(1.5532, abs=1e-3)
    assert ft["logL_start"] == pytest.approx(ft["logL"], abs=1e-6)
    v3 = Ladder(F("IPI_m07", "WTI_m03"), 3, 0, route="v")
    v3.fit()
    f3 = v3.factor_test()
    assert f3["df"] == 6 and f3["lr"] == pytest.approx(4.2121, abs=1e-3)


def test_own_zeros_on_route_v():
    z = Ladder(F("IPC_m07", "WTI_m03"), 2, 0, route="v", own_zeros={"WTI": {"ar": {2}}})
    rz = z.fit()
    assert z.npar() == 11
    assert rz.logL == pytest.approx(-630.694064, abs=1e-5)           # the C's
    assert "phi_WTI[B^2]" not in rz.names


def test_factor_test_keeps_an_own_zero():
    """BUG-0029: the alternative freed the own coefficient held at zero. The
    C: 2 degrees of freedom, LR 0.4849."""
    L = Ladder(F("IPI_m07", "WTI_m03"), echelon=(1, 1), own="free",
               own_zeros={"IPI": {"ma": {1}}}, start="zero")
    r = L.fit()
    assert r.logL == pytest.approx(-925.559100, abs=1e-5)
    ft = L.factor_test()
    assert ft["df"] == 2
    assert ft["lr"] == pytest.approx(0.4849, abs=1e-3)
    assert "theta_IPI[B^1]" not in ft["free"].names()


def test_route_v_refusals():
    with pytest.raises(LadderError, match="no univariate ARMA to hold"):
        Ladder(F("WTI_m03", "GS2_m05"), 1, 0, route="v", fixarma=True)
    with pytest.raises(LadderError, match="needs p or q above zero"):
        Ladder(F("WTI_m03", "GS2_m05"), 0, 0, route="v")
    with pytest.raises(ValueError):
        Ladder(F("WTI_m03", "GS2_m05"), 1, 0, route="w")
    u = Ladder(F("WTI_m03", "GS2_m05"), 1, 0)
    u.fit()
    with pytest.raises(LadderError, match="route V's"):
        u.factor_test()
