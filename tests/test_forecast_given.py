"""forecast_given: forecasting a VARMA from given parameters, unit roots
allowed (the entry point for drvec's VEC written in levels; drvec-python's
tests/test_levels.py holds it to drvec's own -f on the C's goldens)."""
import math

import numpy as np
import pytest

from drvarma.forecast import (forecast_given, forecast_level_variances, forecast_w,
                              lead_sd, series_psi)
from drvarma.irf import psi_weights


def _stationary(seed=3, m=3, p=2, q=1, n=60):
    rng = np.random.default_rng(seed)
    phi = 0.15 * rng.standard_normal((p, m, m))
    theta = 0.2 * rng.standard_normal((q, m, m))
    A = rng.standard_normal((m, m))
    sigma = A @ A.T + m * np.eye(m)
    return phi, theta, sigma, rng.standard_normal((n, m)), rng.standard_normal((n, m))


def test_a_stationary_model_is_forecast_w_and_its_variance():
    phi, theta, sigma, w, a = _stationary()
    mu = np.array([0.5, -1.0, 2.0])
    out = forecast_given(phi, theta, sigma, w, a, 10, mu=mu)
    f = forecast_w(phi, theta, mu, w, a, 10)
    v, _, _ = forecast_level_variances(phi, theta, sigma, 10, 0, 0, 1)
    for c, o in enumerate(out):
        assert np.array_equal(o["w"], f[:, c])
        assert np.array_equal(o["level"], f[:, c])        # no lam: the model's units
        assert np.allclose(o["sd"] ** 2, v[1:, c, c], rtol=1e-13, atol=0)


def test_a_random_walk_with_drift():
    """Y_t = Y_{t-1} + c + a_t: Phi(1) = 0, the drift is not Phi(1) mu. The
    forecast is Y_n + h c, its s.d. sqrt(h sigma_ii)."""
    m, n, H = 2, 30, 12
    rng = np.random.default_rng(7)
    sigma = np.array([[2.0, 0.6], [0.6, 1.0]])
    c = np.array([0.25, -0.5])
    w = np.cumsum(rng.standard_normal((n, m)), axis=0)
    out = forecast_given(np.eye(m)[None], np.zeros((0, m, m)), sigma, w,
                         np.zeros((n, m)), H, const=c)
    h = np.arange(1, H + 1)
    for i, o in enumerate(out):
        assert np.allclose(o["w"], w[-1, i] + h * c[i], rtol=1e-14, atol=1e-13)
        assert np.allclose(o["sd"], np.sqrt(h * sigma[i, i]), rtol=1e-14)


def test_cointegrated_bands_from_the_psi_weights():
    """A cointegrated pair in levels, Y1 - b Y2 stationary: the s.d. is the
    psi weights of the levels system, which grow in the common trend."""
    b = 0.8
    # nabla Y2 = a2 ; W = Y1 - b Y2 = 0.5 W_{t-1} + a1, written in levels:
    # Y1_t = 0.5 Y1_{t-1} + 0.5 b Y2_{t-1} + (a1 + b a2), Y2_t = Y2_{t-1} + a2
    phi = np.array([[[0.5, 0.5 * b], [0.0, 1.0]]])
    sigma = np.array([[1.0, 0.3], [0.3, 0.5]])
    n, H = 20, 40
    w = np.zeros((n, 2))
    out = forecast_given(phi, np.zeros((0, 2, 2)), sigma, w, np.zeros((n, 2)), H)
    psi = psi_weights(phi, np.zeros((0, 2, 2)), H)
    assert abs(np.linalg.det(np.eye(2) - phi[0])) < 1e-15        # a unit root
    for c, o in enumerate(out):
        P = series_psi(psi, c, [], H)
        assert np.array_equal(o["sd"], [lead_sd(P, sigma, l) for l in range(1, H + 1)])
        assert np.all(np.diff(o["sd"]) > 0)                      # no stationary limit


def test_the_level_through_fue():
    """lam = 0: exp((w + det) / refactor); the band at +/- z s.d., transformed."""
    phi, theta, sigma, w, a = _stationary(m=2, q=0)
    det = np.array([[0.1 * h for h in range(1, 6)], [0.0] * 5])
    out = forecast_given(phi, theta, sigma, w, a, 5, det=det, lam=[0.0, 1.0],
                         refactor=[100.0, 1.0], z=1.959964)
    o = out[0]
    ys = o["w"] + det[0]
    assert np.allclose(o["level"], np.exp(ys / 100.0), rtol=1e-15)
    assert np.allclose(o["low"], np.exp((ys - 1.959964 * o["sd"]) / 100.0), rtol=1e-15)
    assert np.array_equal(out[1]["high"], out[1]["w"] + 1.959964 * out[1]["sd"])


def test_refusals():
    phi, theta, sigma, w, a = _stationary()
    with pytest.raises(ValueError, match="one residual per row"):
        forecast_given(phi, theta, sigma, w, a[1:], 3)
    with pytest.raises(ValueError, match="q = 1 needs"):
        forecast_given(phi[:0], theta, sigma, w[:0], a[:0], 3)


def test_lead_sd_is_the_ladders():
    """series_psi with an empty operator is the psi weights' row."""
    phi, theta, sigma, _w, _a = _stationary()
    psi = psi_weights(phi, theta, 6)
    P = series_psi(psi, 1, [], 6)
    assert np.array_equal(P, psi[:, 1, :].T)
    assert lead_sd(P, sigma, 1) == math.sqrt(sigma[1, 1])
