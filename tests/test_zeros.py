"""Cross coefficients fixed at zero: Ladder(zeros=) — Tiao and Box's
simplification by coefficient (1981, §4). Finer than links: a pair's whole
cross dynamics zeroed by `zeros` is the same model as that pair unlinked."""
import os

import numpy as np
import pytest

from drvarma.ladder import Ladder, LadderError

D = os.path.join(os.path.dirname(__file__), "data", "ladder")
PAIR = [os.path.join(D, "IPC_ES_m10.pre"), os.path.join(D, "IPC_FR_msar.pre")]


def test_zeroing_a_pair_is_unlinking_it():
    """The C's -links "IPC_ES<-IPC_FR" (test_links pins it: logL 81.114461)
    is the full model with IPC_FR<-IPC_ES's AR1 and MA1 at zero."""
    a = Ladder(PAIR, 1, 1, links="IPC_ES<-IPC_FR").fit()
    b = Ladder(PAIR, 1, 1, zeros="AR1[IPC_FR<-IPC_ES], MA1[IPC_FR<-IPC_ES]").fit()
    assert a.names == b.names and a.npar == b.npar
    assert b.logL == pytest.approx(81.114461, abs=1e-5)
    np.testing.assert_allclose(a.x, b.x, atol=1e-6)


def test_one_coefficient_and_the_lr():
    full = Ladder(PAIR, 1, 1)
    rf = full.fit()
    L = Ladder(PAIR, 1, 1, zeros=["MA1[IPC_ES<-IPC_FR]"])
    r = L.fit()
    assert "MA1[IPC_ES<-IPC_FR]" not in r.names and r.npar == rf.npar - 1
    assert r.logL <= rf.logL + 1e-6
    assert L._cMA[0, 0, 1] == 0.0


def test_zeros_are_checked():
    with pytest.raises(LadderError, match="form AR3"):
        Ladder(PAIR, 1, 1, zeros="phi[IPC_ES]")
    with pytest.raises(LadderError, match="cross AR order is 1"):
        Ladder(PAIR, 1, 1, zeros="AR2[IPC_ES<-IPC_FR]")
    with pytest.raises(LadderError, match="two different series"):
        Ladder(PAIR, 1, 1, zeros="AR1[IPC_ES<-IPC_ES]")
    with pytest.raises(LadderError, match="not a linked pair"):
        Ladder(PAIR, 1, 1, links="IPC_ES<-IPC_FR", zeros="AR1[IPC_FR<-IPC_ES]")


def test_zeros_survive_the_preliminary_start():
    L = Ladder(PAIR, 1, 1, zeros="AR1[IPC_ES<-IPC_FR]", start="preliminary")
    L.fit()
    assert L._cAR[0, 0, 1] == 0.0
