"""LadderSeries.orders(): fue's placeholder is not an order (sima BUG-0001).

A model without ARMA carries, for fue, an AR(1) fixed at 0.0. orders() read
the length of the expanded polynomial and called a random walk «AR 1».
flour_Buffalo_rw.pre is art's (0,1,0) for Buffalo and flour_Minneapolis_ar1.pre
an AR(1), both from sima's lutkepohl_poskitt example.
"""
import os

from drvarma.ladder import load

D = os.path.join(os.path.dirname(__file__), "data", "ladder")


def test_the_placeholder_is_not_an_order():
    rw, ar1 = load([os.path.join(D, "flour_Buffalo_rw.pre"),
                    os.path.join(D, "flour_Minneapolis_ar1.pre")])
    assert rw.orders() == (0, 0)
    assert ar1.orders() == (1, 0)


def test_a_free_coefficient_seeded_at_zero_is_an_order(tmp_path):
    # art's .inp: the structure, with the parameters at zero and free
    src = open(os.path.join(D, "flour_Buffalo_rw.pre")).read()
    assert "**\n0.0000 0\n" in src
    p = tmp_path / "Buffalo_ar1.inp"
    p.write_text(src.replace("**\n0.0000 0\n", "**\n0.0000 1\n", 1))
    s, = load([str(p)])
    assert s.orders() == (1, 0)
