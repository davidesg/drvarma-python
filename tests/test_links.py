"""Restricted cross terms: Ladder(links=), as the C's -links (atsw-gui).

Only the linked pairs carry cross coefficients (AR and MA, every lag up to p
and q); the others are zero. Pins: the C's values on the same .pre files.
"""
import os

import numpy as np
import pytest

from drvarma.ladder import Ladder, LadderError

D = os.path.join(os.path.dirname(__file__), "data", "ladder")
PAIR = [os.path.join(D, "IPC_ES_m10.pre"), os.path.join(D, "IPC_FR_msar.pre")]


def test_every_pair_named_is_the_unrestricted_model():
    a = Ladder(PAIR, 1, 1).fit()
    b = Ladder(PAIR, 1, 1, links="IPC_ES<-IPC_FR, IPC_FR<-IPC_ES").fit()
    assert a.names == b.names
    np.testing.assert_allclose(a.x, b.x, atol=1e-10)
    assert a.logL == pytest.approx(b.logL, abs=1e-10)


def test_one_link_is_the_cs():
    """drvarma C: IPC_ES_m10.pre IPC_FR_msar.pre 1 1 -links "IPC_ES<-IPC_FR":
    8 parameters, logL 81.114461, LR 68.9649 with df 3."""
    L = Ladder(PAIR, 1, 1, links="IPC_ES<-IPC_FR")
    r = L.fit()
    cross = [n for n in r.names if n.startswith(("AR1", "MA1"))]
    assert cross == ["AR1[IPC_ES<-IPC_FR]", "MA1[IPC_ES<-IPC_FR]"]
    assert len(r.x) == 8
    assert r.logL == pytest.approx(81.114461, abs=2e-6)
    lr, df, _p = L.lr_test()
    assert df == 3 and lr == pytest.approx(68.9649, abs=1e-3)
    assert r.phi[0, 1, 0] == 0.0 and r.theta[0, 1, 0] == 0.0     # FR <- ES absent


def test_links_as_pairs():
    L = Ladder(PAIR, 1, 0, links=[("IPC_FR", "IPC_ES")])
    assert [n for n in L.names() if n.startswith("AR1")] == ["AR1[IPC_FR<-IPC_ES]"]


@pytest.mark.parametrize("p,q,links,msg", [
    (0, 0, "IPC_ES<-IPC_FR", "needs cross dynamics"),
    (1, 0, "IPC_ES<-XX", "two different series"),
    (1, 0, "IPC_ES<-IPC_ES", "two different series"),
    (1, 0, "IPC_ES", "form A<-B")])
def test_links_refused(p, q, links, msg):
    with pytest.raises(LadderError, match=msg):
        Ladder(PAIR, p, q, links=links)
