"""Jenkins and Alavi's (1981) identification statistics: R_k, S_k, S_k(q)."""
import numpy as np
import pytest

from drvarma.identification_mv import (corr_matrices, determinants,
                                       partial_corr_matrices,
                                       q_partial_corr_matrices, symbols)

SIG = np.array([[1.0, 0.3], [0.3, 1.0]])
P1 = np.array([[0.5, 0.3], [-0.2, 0.4]])
P2 = np.array([[-0.3, 0.1], [0.2, -0.2]])


def _sim(n, phi, theta, seed):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(SIG)
    a = (L @ rng.standard_normal((2, n + 200))).T
    w = np.zeros((n + 200, 2))
    for t in range(2, n + 200):
        w[t] = (a[t] + sum(P @ w[t - 1 - i] for i, P in enumerate(phi))
                - sum(T @ a[t - 1 - i] for i, T in enumerate(theta)))
    return w[200:]


def _std(P, w):
    d = np.sqrt(np.diag(np.cov(w.T)))
    return np.diag(1 / d) @ P @ np.diag(d)


def test_one_series_is_the_pacf():
    fue = pytest.importorskip("fue")
    x = np.convolve(np.random.default_rng(0).standard_normal(400), [1, .5, .2])[:400]
    S, se = partial_corr_matrices(x[:, None], 10)
    np.testing.assert_allclose(S[:, 0, 0], fue.pacf(x, lags=10), atol=1e-12)
    assert se == pytest.approx(1 / np.sqrt(400))


def test_var2_partial_matrices_cut_off_after_2():
    w = _sim(8000, [P1, P2], [], 1)
    S, se = partial_corr_matrices(w, 4)
    np.testing.assert_allclose(S[1], _std(P2, w), atol=0.04)
    assert np.abs(S[2:]).max() < 4 * se


def test_vma1_correlation_matrices_cut_off_after_1():
    w = _sim(8000, [], [np.array([[.6, .4], [-.3, .5]])], 2)
    R, se = corr_matrices(w, 4)
    assert np.abs(R[0]).max() > 0.3
    assert np.all(np.abs(R[1:]) < 3 * se[1:])


def test_varma11_q_conditioned_cuts_where_the_plain_one_does_not():
    """Jenkins and Alavi's reason for S_k(q): for an ARMA(1,1) neither R_k nor
    S_k cut off, S_k(1) does, and S_1(1) estimates Phi_1."""
    w = _sim(20000, [P1], [np.array([[-.4, .2], [.1, -.3]])], 0)
    Sq, se = q_partial_corr_matrices(w, 4, 1)
    S0, _ = partial_corr_matrices(w, 4)
    np.testing.assert_allclose(Sq[0], _std(P1, w), atol=0.04)
    assert np.abs(Sq[1:]).max() < 0.1 < np.abs(S0[1:]).max()
    np.testing.assert_allclose(q_partial_corr_matrices(w, 3, 0)[0], S0[:3], atol=1e-12)


def test_prewhitened_standard_error_and_symbols():
    w = np.random.default_rng(3).standard_normal((100, 2))
    R, se = corr_matrices(w, 3, prewhitened=True)
    assert np.allclose(se, 0.1)
    M = np.array([[[0.3, -0.3], [0.0, 0.1]]])
    assert symbols(M, 0.1).tolist() == [[["+", "-"], [".", "."]]]
    assert determinants(M)[0] == pytest.approx(0.03)
