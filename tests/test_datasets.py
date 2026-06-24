"""Tests for the synthetic VARMA simulator."""
import numpy as np
from drvarma.datasets import simulate_varma


def test_shapes_and_names():
    s = simulate_varma(phi=[np.array([[0.5, 0.0], [0.2, 0.4]])], n=300, seed=1)
    assert (s.nobs, s.m) == (300, 2)
    assert s.names == ["y1", "y2"]


def test_ar1_autocorrelation_recovered():
    # diagonal AR(1) with phi=0.6; lag-1 autocorr should be ~0.6
    s = simulate_varma(phi=[np.array([[0.6]])], n=20000, seed=3)
    x = s.column(0) - s.column(0).mean()
    r1 = np.dot(x[1:], x[:-1]) / np.dot(x, x)
    assert abs(r1 - 0.6) < 0.03


def test_innovation_covariance_recovered():
    sigma = np.array([[1.0, 0.5], [0.5, 2.0]])
    s = simulate_varma(phi=[np.zeros((2, 2))], sigma=sigma, n=40000, seed=5)
    # with Phi=0 the series IS the innovation sequence
    cov = np.cov(s.data.T)
    assert np.allclose(cov, sigma, atol=0.05)


def test_mean_recovered():
    s = simulate_varma(phi=[np.array([[0.3]])], mu=np.array([5.0]),
                       n=40000, seed=7)
    assert abs(s.data.mean() - 5.0) < 0.1
