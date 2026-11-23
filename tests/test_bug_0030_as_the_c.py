"""BUG-0030: the ladder as the C's -- the same cast (no MA flip), the same
order of the parameters, the same arithmetic in the ols seeds."""
import os

import numpy as np
import pytest

from drvarma import _cls
from drvarma.ladder import Ladder, LadderError, load

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ladder")


def _f(*names):
    return [os.path.join(D, n + ".pre") for n in names]


def _has_flip():
    import inspect
    from fue.cast_us import cast_us_py
    return "flip" in inspect.signature(cast_us_py).parameters


needs_fue = pytest.mark.skipif(not _has_flip(), reason="needs fue >= 0.1.23 (cast_us_py(flip=False))")


@needs_fue
def test_the_order_is_the_cs():
    """esc_pack: the ARMA of every series, then the deterministic terms of
    every series, then the means; then the cross terms and Q, as before."""
    L = Ladder(_f("IPC_m07", "WTI_m03"), 1, 0, det="interv")
    names = L.names()
    kind = []
    for n in names:
        if n.startswith(("AR", "MA", "log(Q", "Q[")):
            kind.append(3)
        elif n.startswith("mu_"):
            kind.append(2)
        elif n[0] in "wd" and n[1].isdigit():
            kind.append(1)
        else:
            kind.append(0)
    assert kind == sorted(kind)                       # ARMA, det, mu, the rest
    assert {0, 1, 2, 3} <= set(kind)
    # and within a class, series by series in the files' order
    arma = [n for n, k in zip(names, kind) if k == 0]
    assert arma == sorted(arma, key=lambda n: 0 if "_IPC[" in n else 1)
    assert [n for n, k in zip(names, kind) if k == 2] == ["mu_IPC", "mu_WTI"]


@needs_fue
def test_pack_and_unpack_follow_the_names():
    L = Ladder(_f("IPC_m07", "WTI_m03"), 1, 0, det="interv")
    v = L.pack()
    names = L.names()
    assert v.size == len(names) == L.npar()
    w = np.arange(1.0, v.size + 1.0) / 100.0
    L.unpack(w)
    np.testing.assert_array_equal(L.pack(), w)
    for s in L.series:                                # each value went to its name
        own = dict(zip(s.param_names(), s.x))
        for n, x in own.items():
            if n in names:
                assert x == w[names.index(n)]
    L.unpack(v)
    np.testing.assert_array_equal(L.pack(), v)


@needs_fue
def test_a_series_layout_covers_its_x_once():
    for s in load(_f("IPI_m07", "IPC_m07", "WTI_m03", "GS1_m01", "FF_m03")):
        arma, det, mu = s.c_layout()
        assert sorted(arma + det + mu) == list(range(s.x.size))
        nm = s.param_names()
        assert all(nm[i].startswith("mu_") for i in mu)
        # the AR side whole, then the MA side (pack_ar_factors, pack_ma_factors)
        side = [0 if nm[i].lower().startswith("phi") else 1 for i in arma]
        assert side == sorted(side)


@needs_fue
def test_the_cast_does_not_take_the_reciprocal_of_an_ma1():
    """The C's shootx_esc builds the polynomial from x; an MA(1) beyond the
    unit circle stays there and the likelihood refuses it (elf, ifault 4)."""
    L = Ladder(_f("GS1_m01", "FF_m03"), 1, 1, route="v", start="zero")
    r = L.fit()
    v = L.pack()
    k = L.names().index("theta_GS1[B^1]")
    v[k] = 1.25
    mu, phi, theta, qq, w, ifa = L.cast(v)
    assert ifa == 0 and theta[0][0][0] == 1.25         # not 0.8
    assert L._elf(mu, phi, theta, qq, w)[4] == 4       # strictly non-invertible
    L.unpack(r.x)


def _kernel():
    from drvarma import _engine
    return _engine.engine_backend() == "c"


@needs_fue
@pytest.mark.skipif(not _kernel(), reason="on the MA wall the path is the compiled "
                    "likelihood's; the pure-Python fallback agrees with it to rounding only")
def test_on_the_ma_wall_the_port_ends_where_the_c_does():
    """GS1 + FF, VARMA(1,1), route V: the C stops on the invertibility wall
    at -1740.636683 after 62 iterations from zero and at -1740.622890 after
    43 from ols. With fue's flip the port gave -1742.598015 from both."""
    L = Ladder(_f("GS1_m01", "FF_m03"), 1, 1, route="v", start="zero")
    r = L.fit()
    assert r.logL == pytest.approx(-1740.636683, abs=1e-5) and r.nit == 62
    L = Ladder(_f("GS1_m01", "FF_m03"), 1, 1, route="v", start="ols")
    r = L.fit()
    assert r.logL == pytest.approx(-1740.622890, abs=1e-5) and r.nit == 43


def test_an_old_fue_is_said(monkeypatch):
    import fue.cast_us as C
    old = C.cast_us_py
    monkeypatch.setattr(C, "cast_us_py", lambda x, spec: old(x, spec))
    with pytest.raises(LadderError, match="fue >= 0.1.23"):
        Ladder(_f("GS1_m01", "FF_m03"), 1, 0).fit()


def test_the_cs_lu_solves_as_lapack_to_rounding():
    rng = np.random.default_rng(7)
    for n in (1, 2, 5, 12):
        A = rng.normal(size=(n + 3, n))
        G = (A.T @ A).tolist()
        b = rng.normal(size=n)
        want = np.linalg.solve(np.array(G), b)
        a = [row[:] for row in G]
        ip = _cls.ludcp(a, n)
        x = list(b)
        _cls.lusol(a, x, n, ip)
        np.testing.assert_allclose(x, want, rtol=1e-9, atol=1e-12)
        y = list(rng.normal(size=n))                  # the factors solve again
        want2 = np.linalg.solve(np.array(G), np.array(y))
        _cls.lusol(a, y, n, ip)
        np.testing.assert_allclose(y, want2, rtol=1e-9, atol=1e-12)


def test_the_sums_are_in_sequence():
    """What makes the seed the C's: each sum accumulated left to right. With
    these three numbers a pairwise or reordered sum gives another result."""
    col = [1e16, 1.0, -1e16, 1.0]
    assert _cls.mean(col) == ((((0.0 + 1e16) + 1.0) - 1e16) + 1.0) / 4
    Z = [[1e8, 1.0], [1.0, 1e8], [-1e8, 1.0]]
    g = _cls.gram(Z, 3, 2)
    assert g[0][1] == ((0.0 + 1e8 * 1.0) + 1.0 * 1e8) + (-1e8) * 1.0
    assert _cls.cross(Z, [1.0, 2.0, 3.0], 3, 2)[0] == ((0.0 + 1e8) + 2.0) + (-3e8)


def test_the_two_routes_write_the_covariance_seed_in_their_own_order():
    a = _cls.corr_to_cov(0.3, 1.7, 2.9, 5.0, 7.0, True)
    b = _cls.corr_to_cov(0.3, 1.7, 2.9, 5.0, 7.0, False)
    assert a == pytest.approx(b, rel=1e-15)
    assert a == 0.3 / (1.7 * 2.9) ** 0.5 * (5.0 * 7.0) ** 0.5
