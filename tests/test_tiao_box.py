"""The Wisconsin line: Tiao and Box's (1981) stepwise autoregression and Box
and Tiao's (1977) canonical analysis, against the papers' own numbers
(sima-python docs/STUDY-tiao-box.md)."""
import os

import numpy as np
import pytest

from drvarma.identification_mv import (canonical, canonical_from_moments,
                                       stepwise_ar, stepwise_order, symbols)

GAS = os.path.join(os.path.dirname(__file__), "data", "tiao_box", "gas_furnace.csv")


@pytest.fixture(scope="module")
def gas():
    return stepwise_ar(np.genfromtxt(GAS, delimiter=",", skip_header=1), 11)


# ── Tiao and Box (1981), the gas furnace ─────────────────────────────────────

def test_gas_furnace_M_reproduces_table_12b(gas):
    """Table 12(b): M(l) = 1650 665 31.7 22.5 5.6 12.9 1.8 8.0 | 3.5 0 2.0,
    on the common sample t = 12..296 with N = n - 11 - 1. The first eight to
    the printed digit (12.8 against 12.9). The last three, where
    |S(l)|/|S(l-1)| is within 1 % of one, come out 3.7, 1.0, 4.0 — a 1979
    program's precision on a log near zero; none is significant either way."""
    pub = [1650, 665, 31.7, 22.5, 5.6, 12.9, 1.8, 8.0]
    assert gas["df"] == 4 and gas["n_eff"] == 285
    for l, (ours, theirs) in enumerate(zip(gas["M"], pub), start=1):
        assert ours == pytest.approx(theirs, abs=max(0.11, 0.0005 * theirs)), l
    assert np.allclose(gas["M"][8:], [3.7, 1.0, 4.0], atol=0.06)
    assert np.all(gas["pvalue"][8:] > 0.05)


def test_gas_furnace_order_is_their_ar6(gas):
    """'the M(l) statistic suggests that an AR(6) model might be appropriate':
    the last significant M(l) is at 6, with M(5) = 5.6 not significant on the
    way."""
    assert stepwise_order(gas) == (6, [5])


def test_gas_furnace_sigma_is_table_14(gas):
    """Table 14's residual covariance matrices of the successive AR fits."""
    pub = {1: (.102, .090, .346), 2: (.037, -.004, .069), 3: (.036, -.002, .063),
           4: (.036, -.003, .059), 5: (.035, -.003, .058), 6: (.035, -.002, .057)}
    for l, (s11, s12, s22) in pub.items():
        S = gas["Sigma"][l - 1]
        assert (S[0, 0], S[0, 1], S[1, 1]) == pytest.approx((s11, s12, s22), abs=6e-4), l


def test_gas_furnace_symbols_are_table_14(gas):
    """Table 14's indicator symbols, every coefficient matrix of the AR(1)..
    AR(6) fits: at low order a spurious feedback (phi_12 significant), which
    goes from p = 3 on — the warning on reading structure from a short fit."""
    pub = {1: ["++|-+"],
           2: ["+-|++", "-+|--"],
           3: ["+.|.+", "-.|--", "+.|.+"],
           4: ["+.|.+", "-.|.-", "..|-.", "..|++"],
           5: ["+.|.+", "-.|.-", "..|-.", "..|..", "..|.."],
           6: ["+.|.+", "-.|.-", "..|-.", "..|..", "+.|..", "-.|+."]}
    for l, mats in pub.items():
        sym = symbols(gas["T_all"][l - 1], 1.0)
        got = ["".join(r[0]) + "|" + "".join(r[1]) for r in sym]
        assert got == mats, l
    # the cross effect of the output on the input (phi_12) is gone from p = 3
    for l in range(3, 7):
        assert np.all(np.abs(gas["T_all"][l - 1][:, 0, 1]) < 2.0), l


# ── Box and Tiao (1977), the hog data from the printed moments ───────────────

C0u = [[0.6831, 1.2523, 0.6535, 0.9533, 1.5224], [0, 6.1939, 3.7845, 2.0209, 5.5708],
       [0, 0, 3.6877, 0.2633, 3.4746], [0, 0, 0, 2.1407, 2.1925], [0, 0, 0, 0, 5.7206]]
C1 = np.array([[0.5864, 1.3670, 0.7513, 0.8632, 1.5151],
               [1.2038, 5.2334, 3.1639, 1.8849, 5.0392],
               [0.4616, 3.5820, 2.7173, 0.5605, 3.0633],
               [1.0108, 1.8972, 0.8338, 1.6260, 2.2508],
               [1.3993, 5.1586, 3.2153, 1.9817, 5.3246]]) * 1e4


@pytest.fixture(scope="module")
def hog():
    C0 = np.array(C0u)
    C0 = (C0 + np.triu(C0, 1).T) * 1e4
    # their C1 is E(z_{t-1} z_t'): phi = C1' C0^-1. (The eigenvalues are the
    # same either way; the eigenvectors tell the orientation.)
    phi = C1.T @ np.linalg.inv(C0)
    lam, M = canonical_from_moments(C0, phi @ C0 @ phi.T)
    return C0, phi, lam, M


def test_hog_eigenvalues_are_table_4_2(hog):
    _C0, _phi, lam, _M = hog
    assert lam == pytest.approx([0.0232, 0.1421, 0.5061, 0.6901, 0.8868], abs=3e-4)


def test_hog_eigenvectors_are_table_4_2(hog):
    """Table 4.2's eigenvectors, normalised to their largest element, and
    their unit-variance scale (the factor after the vector)."""
    _C0, _phi, _lam, M = hog
    pub = [((1.0000, 0.3876, -0.2524, -0.5896, -0.2665), 0.0284),
           ((0.2080, 1.0000, -0.8614, -0.3382, -0.3655), 0.0111),
           ((0.8925, -0.6433, -0.8277, -0.4784, 1.0000), 0.0074),
           ((-0.9358, -0.2410, -0.4391, -0.5614, 1.0000), 0.0129),
           ((0.6687, -0.1206, -0.0134, 0.0396, 1.0000), 0.0039)]
    for j, (v, scale) in enumerate(pub):
        row = M[j]                # C0 in its own units (their print is x 10^-4)
        k = int(np.argmax(np.abs(v)))
        assert row / row[k] == pytest.approx(v, abs=5e-3), j   # 4-digit moments
        assert abs(row[k]) == pytest.approx(scale, rel=0.03), j


def test_hog_variance_components_are_table_4_3(hog):
    """phi_bar = M phi M^-1: its rows square-sum to lam (3.8), and its squares
    are Table 4.3 — x4 and x5 almost on their own past, x1 and x2 nearly
    white."""
    _C0, phi, lam, M = hog
    pb = M @ phi @ np.linalg.inv(M)
    assert (pb ** 2).sum(1) == pytest.approx(lam, abs=1e-6)
    pub_diag = [0.015, 0.077, 0.401, 0.678, 0.876]
    pub_shock = [0.977, 0.858, 0.494, 0.310, 0.113]
    assert np.diag(pb ** 2) == pytest.approx(pub_diag, abs=1.5e-3)
    assert 1 - lam == pytest.approx(pub_shock, abs=1.5e-3)


# ── canonical() on data: the two ends ────────────────────────────────────────

def test_canonical_finds_the_static_relation_and_the_common_trend():
    """Their (4.9): z1 a random walk, z2 = beta z1 + a2. In levels the
    canonical analysis gives one component near 1 (the common trend) and one
    near 0 (z2 - beta z1, white)."""
    rng = np.random.default_rng(7)
    n = 400
    z1 = np.cumsum(rng.standard_normal(n))
    z2 = 0.8 * z1 + 0.5 * rng.standard_normal(n)
    out = canonical(np.column_stack([z1, z2]), 1)
    assert out["lam"][0] < 0.05 and out["lam"][1] > 0.95
    w = out["weights"][0]
    assert w[0] / w[1] == pytest.approx(-0.8, abs=0.05)
    assert out["shares"].shape == (2, 3)
    assert out["components"].shape == (n - 1, 2)
