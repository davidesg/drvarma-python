"""Standard errors from Mauricio's fdhess (drvarma.stderr).

The pins come from docs/STUDY-standard-errors.md: the exact GLS of a regression
with AR(1) errors (S1), the review's IPC/WTI VAR(1) against OLS (S2) and series
of very different scale (S3). Plus the two guards: the flat direction and a
Hessian that cannot be taken (boundary / not positive definite), always said.
"""
import os

import numpy as np
import pytest

from drvarma.ladder import Ladder
from drvarma.stderr import fd_covariance, legacy_fd_std_errors
from drvarma._engine import estimate_w

HERE = os.path.dirname(__file__)
GLS_PRE = os.path.join(HERE, "data", "stderr", "ES_CPI_m10.pre")

# Exact GLS standard errors (drtran battery §1d; fue ERRORES_ESTANDAR.md).
GLS = {"mu_": 0.028502, "phi_": 0.062421, "[cos 1]": 0.068328,
       "[sin 1]": 0.068294, "[cos 2]": 0.027692, "[alter]": 0.006094}


def _pick(fit, key):
    for n, s in zip(fit.names, fit.std_errors):
        if n.startswith(key) or n.endswith(key):
            return s
    raise KeyError(key)


# --------------------------------------------------------------------------- #
def test_quadratic_objective_exact():
    """On F(x) = 1 + x'Ax/2 the covariance is 2 F(x*) A^-1 / n exactly."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    cov, std, info = fd_covariance(lambda x: 1.0 + 0.5 * x @ A @ x, np.zeros(2), n=10)
    assert info["positive_definite"]
    np.testing.assert_allclose(cov, 2.0 * np.linalg.inv(A) / 10, rtol=1e-5)


def test_fixed_index_is_nan_and_the_rest_conditional():
    A = np.diag([2.0, 5.0, 8.0])
    cov, std, _ = fd_covariance(lambda x: 1.0 + 0.5 * x @ A @ x, np.zeros(3), n=1, fixed=1)
    assert np.isnan(std[1]) and np.isnan(cov[1]).all()
    np.testing.assert_allclose(std[[0, 2]], np.sqrt(2.0 / np.array([2.0, 8.0])), rtol=1e-5)


def test_not_positive_definite_is_refused():
    cov, std, info = fd_covariance(lambda x: 1.0 + x[0] ** 2 - x[1] ** 2, np.zeros(2), n=1)
    assert cov is None and std is None and info["positive_definite"] is False


def test_boundary_is_detected():
    """A neighbour the objective rejects (sentinel 1e300) = optimum on the edge."""
    f = lambda x: 1.0e300 if x[0] > 0 else 1.0 + x[0] ** 2 + x[1] ** 2
    cov, std, info = fd_covariance(f, np.zeros(2), n=1)
    assert cov is None and info.get("boundary") is True


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("start", ["file", "perturbed"])
def test_gls_pins_fdhess(start):
    """S1: fdhess recovers the exact GLS within 0.5 %, from any start."""
    L = Ladder([GLS_PRE], 0, 0, diagcov=True, redet=True, hessian="fd")
    if start == "perturbed":
        L.series[0].x *= 1.2
    r = L.fit()
    assert r.se_method == "fdhess"
    for key, ref in GLS.items():
        assert _pick(r, key) == pytest.approx(ref, rel=5e-3), key


def test_gls_bfgs_is_path_dependent():
    """S1: the BFGS Hessian is not the curvature -- the reason for the default."""
    L = Ladder([GLS_PRE], 0, 0, diagcov=True, redet=True, hessian="bfgs")
    r = L.fit()
    assert r.se_method.startswith("bfgs")
    assert abs(_pick(r, "mu_") / GLS["mu_"] - 1) > 0.5


def test_ladder_default_is_fd():
    assert Ladder([GLS_PRE], 0, 0, diagcov=True).hessian == "fd"


# --------------------------------------------------------------------------- #
def _sim_var1(ratio, n=400, seed=3):
    rng = np.random.default_rng(seed)
    Phi = np.array([[0.3, 0.1], [0.5, 0.4]])
    C = np.linalg.cholesky(np.array([[1.0, 0.5], [0.5, 1.0]]))
    e = rng.standard_normal((n + 100, 2)) @ C.T
    w = np.zeros((n + 100, 2))
    for t in range(1, n + 100):
        w[t] = Phi @ w[t - 1] + e[t]
    return w[100:] * np.array([1.0, ratio])


def _ols_se(w):
    Y, X = w[1:], np.column_stack([np.ones(len(w) - 1), w[:-1]])
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    E = Y - X @ B
    S = E.T @ E / (len(Y) - X.shape[1])
    return np.sqrt(np.outer(np.diag(S), np.diag(np.linalg.inv(X.T @ X))))[:, 1:].ravel()


def test_scale_ratio_100_matches_ols():
    """S3: with a 1:100 scale ratio fdhess stays at OLS (BFGS collapsed to 0.01)."""
    w = _sim_var1(100.0)
    r = estimate_w(w, 1, 0, include_mean=True)
    cov, std, how, _ = legacy_fd_std_errors(r, w, 1, 0, include_mean=True)
    assert how == "fdhess"
    ratio = std[2:6] / _ols_se(w)
    assert np.all((ratio > 0.95) & (ratio < 1.05)), ratio
    assert np.isnan(std[-3])                       # qq[1,1]: the flat direction


def test_review_case_against_ols():
    """S2: the review's IPC/WTI VAR(1); phi21 SE ~ 2.40 (BFGS said 0.027)."""
    atsw = pytest.importorskip("atsw")
    pd = pytest.importorskip("pandas")
    from drvarma import MultiSeries, Model
    a = pd.read_csv(atsw.example_path("IPC_ES.csv"))
    b = pd.read_csv(atsw.example_path("WTI.csv"))
    ms = MultiSeries(np.column_stack([a.value, b.value]), freq=12, start=(2002, 1),
                     names=["IPC", "WTI"])
    mod = Model(ms, lam=0.0, d=1, p=1, include_mean=True, deseason="auto").fit()
    assert mod.hessian == "fd" and mod.se_method == "fdhess"
    se = np.asarray(mod.std_errors)[2:6]
    assert se[2] == pytest.approx(2.40, rel=0.02)
    np.testing.assert_allclose(se / _ols_se(mod._w), 0.989, atol=0.01)
