"""Cross orders per equation: Ladder(rows=) — the unequal row degrees of the
echelon form, without its restriction table (docs/ECHELON_PLAN.md, stage 2).
The C has no row orders, so the pins are by equivalence: a row order is the
same model as the coefficients beyond it fixed at zero, and a row of order
(0, 0) is the pair unlinked, which pins the C's -links value."""
import os

import numpy as np
import pytest

from drvarma.ladder import Ladder, LadderError, write_report

D = os.path.join(os.path.dirname(__file__), "data", "ladder")
PAIR = [os.path.join(D, "IPC_ES_m10.pre"), os.path.join(D, "IPC_FR_msar.pre")]


def test_a_null_row_is_the_cs_link():
    """IPC_FR's equation without cross terms = -links "IPC_ES<-IPC_FR"."""
    a = Ladder(PAIR, 1, 1, links="IPC_ES<-IPC_FR").fit()
    b = Ladder(PAIR, 1, 1, rows={"IPC_FR": (0, 0)}).fit()
    assert a.names == b.names and a.npar == b.npar
    assert b.logL == pytest.approx(81.114461, abs=2e-6)
    np.testing.assert_allclose(a.x, b.x, atol=1e-6)


def test_a_row_order_is_its_zeros():
    a = Ladder(PAIR, 1, 1, rows="IPC_ES(1,0)").fit()
    b = Ladder(PAIR, 1, 1, zeros="MA1[IPC_ES<-IPC_FR]").fit()
    assert a.names == b.names
    assert a.logL == pytest.approx(b.logL, abs=1e-8)


def test_p_and_q_become_the_largest_rows():
    L = Ladder(PAIR, 1, 0, rows="IPC_ES(2,1)")
    assert (L.p, L.q) == (2, 1)
    r = L.fit()
    cross = [n for n in r.names if n[:2] in ("AR", "MA") and "<-" in n]
    assert cross == ["AR1[IPC_ES<-IPC_FR]", "AR1[IPC_FR<-IPC_ES]",
                     "AR2[IPC_ES<-IPC_FR]", "MA1[IPC_ES<-IPC_FR]"]
    z = Ladder(PAIR, 2, 1, zeros="AR2[IPC_FR<-IPC_ES], MA1[IPC_FR<-IPC_ES]").fit()
    assert z.names == r.names
    assert r.logL == pytest.approx(z.logL, abs=1e-8)


def test_rows_links_and_zeros_compose():
    L = Ladder(PAIR, 2, 1, links="IPC_ES<-IPC_FR", rows="IPC_ES(2,0)",
               zeros="AR1[IPC_ES<-IPC_FR]")
    assert [n for n in L.names() if "<-" in n] == ["AR2[IPC_ES<-IPC_FR]"]


def test_the_string_and_the_dict_agree():
    a = Ladder(PAIR, 1, 1, rows="IPC_ES(2,0), IPC_FR(0,1)")
    b = Ladder(PAIR, 1, 1, rows={"IPC_ES": (2, 0), "IPC_FR": [0, 1]})
    assert a.names() == b.names()
    np.testing.assert_array_equal(a.rows, [[2, 0], [0, 1]])


def test_rows_are_checked():
    with pytest.raises(LadderError, match="not a series"):
        Ladder(PAIR, 1, 1, rows={"IPC_IT": (1, 1)})
    with pytest.raises(LadderError, match="form A\\(p,q\\)"):
        Ladder(PAIR, 1, 1, rows="IPC_ES(1)")
    with pytest.raises(LadderError, match="given twice"):
        Ladder(PAIR, 1, 1, rows="IPC_ES(1,0), IPC_ES(0,1)")
    with pytest.raises(LadderError, match=">= 0"):
        Ladder(PAIR, 1, 1, rows={"IPC_ES": (-1, 0)})
    with pytest.raises(LadderError, match="cross MA order of IPC_ES is 0"):
        Ladder(PAIR, 1, 1, rows="IPC_ES(1,0)", zeros="MA1[IPC_ES<-IPC_FR]")


def test_the_lr_counts_rows_and_zeros():
    """BUG-0017: the LR's df was (p + q) per linked pair whatever was fixed."""
    L = Ladder(PAIR, 1, 1, zeros="MA1[IPC_ES<-IPC_FR]")
    r = L.fit()
    _lr, df, _p = L.lr_test()
    assert df == 3 + 1                       # three cross coefficients + Q12
    assert df == r.npar - Ladder(PAIR, 0, 0, diagcov=True).fit().npar
    R = Ladder(PAIR, 1, 1, rows="IPC_FR(0,0)", diagcov=True)
    R.fit()
    assert R.lr_test()[1] == 2


def test_the_report_says_rows_and_zeros(tmp_path):
    L = Ladder(PAIR, 1, 1, rows="IPC_ES(1,0)", zeros="AR1[IPC_FR<-IPC_ES]")
    L.fit()
    p = tmp_path / "r.out"
    write_report(L, str(p))
    text = p.read_text()
    assert "Cross rows       : IPC_ES(1,0), IPC_FR(1,1)" in text
    assert "Fixed at zero    : AR1[IPC_FR<-IPC_ES]" in text


def test_a_log_variance_out_of_range_is_inadmissible():
    """BUG-0018: a line-search step with log(Q_ii/Q_11) > 709 raised
    OverflowError out of the fit; it is an inadmissible point, as in the C."""
    L = Ladder(PAIR, 1, 1)
    v = L.pack()
    v[L.names().index("log(Q[IPC_FR]/Q[IPC_ES])")] = 1000.0
    assert L.cast(v)[-1] == 6
    v[L.names().index("log(Q[IPC_FR]/Q[IPC_ES])")] = float("nan")
    assert L.cast(v)[-1] == 6
