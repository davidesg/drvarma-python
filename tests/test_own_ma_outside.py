"""An own MA factor read alone outside the unit circle: the C's
own_ma_outside (escalera.c, 23-nov-2026), said in the fit and in the report.

The ladder does not take the invertible form of a coefficient (BUG-0030):
the likelihood is evaluated only where the system's MA operator is
invertible. So an own factor can end outside the circle read alone, which fue
never estimates. Muskrat and mink, VARMA(2,1), route U, from zero: the
optimiser stops (no lower point, termcode 3; not on the system's wall)
with theta_MUSKRAT = 1.5483 (the C: -551.715772); from
ols it stops on the wall at -545.562324, with no own factor outside.
"""
import os

import numpy as np
import pytest

from drvarma.ladder import Ladder, _own_ma_outside, _own_ma_text, write_report

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ladder")
MM = [os.path.join(D, "MUSKRAT_m03.pre"), os.path.join(D, "MINK_m02.pre")]   # the C's files


def _kernel():
    """The compiled likelihood, on the platform of the record: this stop is
    not an interior maximum, and the point is the path's (HOMOLOGATION.md)."""
    import platform
    import sys
    from drvarma import _engine
    return (_engine.engine_backend() == "c" and sys.platform.startswith("linux")
            and platform.machine() in ("x86_64", "AMD64"))


@pytest.mark.skipif(not _kernel(), reason="the path to this point is the compiled likelihood's, on Linux x86-64")
def test_the_muskrat_factor_beyond_the_circle_is_said(tmp_path):
    L = Ladder(MM, 2, 1, start="zero")
    r = L.fit()
    assert r.logL == pytest.approx(-551.715772, abs=1e-5)
    assert len(r.own_ma_outside) == 1
    name, kind, q, coef, md = r.own_ma_outside[0]
    assert (name, kind, q) == ("MUSKRAT", "regular MA", 1)
    assert coef.startswith("theta = 1.5482") and md == pytest.approx(1.548284, abs=1e-5)
    out = tmp_path / "r.out"
    write_report(L, str(out))
    txt = out.read_text()
    assert "OWN MA FACTORS NOT INVERTIBLE, EACH READ ALONE:" in txt
    assert "beyond the unit circle" in txt and "fue never estimates such a factor" in txt


@pytest.mark.skipif(not _kernel(), reason="the path to this point is the compiled likelihood's, on Linux x86-64")
def test_nothing_is_said_when_no_own_factor_is_outside():
    L = Ladder(MM, 2, 1, start="ols")
    r = L.fit()
    assert r.logL == pytest.approx(-545.562324, abs=1e-5)
    assert r.own_ma_outside == [] and _own_ma_text(r.own_ma_outside) == []


def test_the_check_reads_each_factor_alone():
    """MA(1) by its coefficient, a higher order by its companion's roots, a
    fixed-frequency MA by sqrt(-c2); the wall's tolerance as the C's."""
    L = Ladder(MM, 0, 0)
    s = [x for x in L.series if x.name == "MUSKRAT"][0]
    nm = s.param_names()
    k = nm.index("theta_MUSKRAT[B^1]")
    keep = s.x.copy()
    for v, n in ((0.99, 0), (0.99996, 1), (-1.2, 1)):
        s.x[k] = v
        assert len(_own_ma_outside([s])) == n
    s.x[:] = keep
    assert _own_ma_outside([s]) == []
