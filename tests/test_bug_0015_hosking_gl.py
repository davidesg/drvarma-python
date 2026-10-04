"""BUG-0015 — Hosking's Q subtracts the ARMA coefficients behind the
residuals: df = m^2 s - k (Hosking 1980: m^2 (s - p - q) for a VARMA(p, q))."""
import os
import re

import numpy as np
import pytest
from scipy.stats import chi2

from drvarma.diagnostics import hosking_q, qccf

D = os.path.join(os.path.dirname(__file__), "data", "ladder")
PAIR = [os.path.join(D, "IPC_ES_m10.pre"), os.path.join(D, "IPC_FR_msar.pre")]


def test_k_comes_off_the_degrees_of_freedom():
    res = np.random.default_rng(0).standard_normal((200, 2))
    Q0, df0, p0 = hosking_q(res, 10)
    Q, df, p = hosking_q(res, 10, k=8)
    assert (df0, df) == (40, 32) and Q == Q0 and p < p0
    assert qccf(res[:, 0], res[:, 1], 10, 8)[1] == 32


def test_no_degrees_of_freedom_left_gives_no_p_value():
    res = np.random.default_rng(1).standard_normal((200, 2))
    _, df, p = hosking_q(res, 2, k=8)
    assert df == 0 and np.isnan(p)


def _sim_varma11(rng, n=300):
    phi = np.array([[.5, .2], [.1, .4]])
    th = np.array([[.4, 0.], [.2, .3]])
    a = rng.standard_normal((n + 100, 2))
    w = np.zeros_like(a)
    for t in range(1, len(a)):
        w[t] = phi @ w[t - 1] + a[t] - th @ a[t - 1]
    return w[100:]


def test_model_diagnostics_has_the_size_it_says():
    """The fitted true VARMA(1,1): Model.diagnostics uses m^2 (s - p - q), and
    over 150 replications the 5% test rejects near 5% (it was 0.3%)."""
    pytest.importorskip("drvarma._drvarma_engine")
    from drvarma import MultiSeries, Model
    rng = np.random.default_rng(7)
    rej = 0
    R = 150
    for _ in range(R):
        ms = MultiSeries(_sim_varma11(rng), freq=1, start=(1, 1), names=["A", "B"])
        # with the mean: lam=1 is (y - 1), which a zero-mean model would not fit
        mdl = Model(ms, lam=1.0, d=0, D=0, p=1, q=1, include_mean=True).fit()
        d = mdl.diagnostics(lag=12)
        assert d["hosking_df"] == 4 * (12 - 2)
        rej += d["hosking_p"] < 0.05
    assert 0.01 < rej / R < 0.10


def test_the_ladder_counts_its_arma_coefficients():
    from drvarma.ladder import Ladder
    L = Ladder(PAIR, 1, 1)
    arma = [n for n in L.names()
            if re.match(r"(phi|Phi|theta|Theta)(f|\d)*_", n) or re.match(r"(AR|MA)\d+\[", n)]
    # phi of IPC_ES, Phi of IPC_FR, and the pair's two cross AR(1) and two
    # cross MA(1): not the means, the deterministic terms nor the covariance
    assert [s.n_arma() for s in L.series] == [1, 1]
    assert L.n_arma() == len(arma) == 6
    # the series' own count even when the ladder holds them at their .pre
    assert Ladder(PAIR, 1, 1, fixarma=True).n_arma() == 6
