"""P4 reliability suite: parameter recovery, C-vs-Python agreement, result-dict
constraints, diagnostics formulas, determinism and edge cases.

Recovery uses the (fast) C engine at large n; the pure-Python estimator is
exercised at modest n.  Tests that need the C engine auto-skip when it is absent.
"""
import numpy as np
import pytest

from drvarma import diagnostics
from drvarma.datasets import (simulate_varma, varma_cases, is_stationary,
                              is_invertible)
from drvarma.estimate_py import estimate_w_py

has_engine = True
try:
    import drvarma._drvarma_engine  # noqa: F401
    from drvarma._engine import estimate_w
except ImportError:
    has_engine = False

needs_engine = pytest.mark.skipif(not has_engine, reason="C engine not built")

CASES = varma_cases()
WELL_ID = [c for c in CASES if c["well_identified"]]


def _arrays(c):
    phi = [np.array(P, float) for P in c["phi"]]
    theta = [np.array(T, float) for T in c["theta"]]
    return phi, theta, np.array(c["sigma"], float), np.array(c["mu"], float)


# -- parameter recovery (C engine, large n) --------------------------------- #

@needs_engine
@pytest.mark.parametrize("case", WELL_ID, ids=[c["name"] for c in WELL_ID])
def test_parameter_recovery(case):
    phi, theta, sigma, mu = _arrays(case)
    sim = simulate_varma(phi=phi, theta=theta, sigma=sigma, mu=mu,
                         n=4000, seed=123)
    r = estimate_w(sim.data, p=len(phi), q=len(theta), include_mean=True)
    assert r["ifault"] == 0
    if phi:
        assert np.max(np.abs(r["phi"] - np.array(phi))) < 0.07
    assert np.max(np.abs(r["sigma"] - sigma)) < 0.08   # mu excluded (near-unit-root)


# -- C vs pure-Python agreement --------------------------------------------- #

@needs_engine
@pytest.mark.parametrize("name,n", [("var1_m2", 800), ("varma11_m2", 600)])
def test_c_vs_python_agreement(name, n):
    case = next(c for c in CASES if c["name"] == name)
    phi, theta, sigma, mu = _arrays(case)
    sim = simulate_varma(phi=phi, theta=theta, sigma=sigma, mu=mu, n=n, seed=7)
    p, q = len(phi), len(theta)
    c = estimate_w(sim.data, p=p, q=q, include_mean=True)
    py = estimate_w_py(sim.data, p=p, q=q, include_mean=True)
    assert py["ifault"] == 0
    assert abs(py["logelf"] - c["logelf"]) < 1e-4
    assert np.max(np.abs(py["phi"] - c["phi"])) < 3e-3
    if q:
        assert np.max(np.abs(py["theta"] - c["theta"])) < 3e-3
    assert np.max(np.abs(py["sigma"] - c["sigma"])) < 3e-3


# -- result-dict constraints (pure-Python, no engine needed) ---------------- #

def test_sigma_and_cov_constraints():
    phi, theta, sigma, mu = _arrays(next(c for c in CASES if c["name"] == "var1_m2"))
    sim = simulate_varma(phi=phi, sigma=sigma, mu=mu, n=400, seed=3)
    r = estimate_w_py(sim.data, p=1, q=0, include_mean=True)
    S = r["sigma"]
    assert np.allclose(S, S.T)                       # symmetric
    assert np.all(np.linalg.eigvalsh(S) > 0)         # positive definite
    cov, std = r["cov"], r["std_errors"]
    assert np.all(np.isfinite(std)) and np.all(std >= 0)
    assert np.allclose(std, np.sqrt(np.clip(np.diag(cov), 0, None)))
    assert r["residuals"].shape == (sim.nobs, sim.m)
    assert r["params"].size == r["npar"]


@pytest.mark.parametrize("diag_ar,diag_cov,expected", [
    (False, False, 2 + 4 + 3),     # mu + full phi + full cov (m=2)
    (True, False, 2 + 2 + 3),      # diagonal AR
    (False, True, 2 + 4 + 2),      # diagonal cov
    (True, True, 2 + 2 + 2),
])
def test_npar_matches_restrictions(diag_ar, diag_cov, expected):
    sim = simulate_varma(phi=[np.array([[0.5, 0.0], [0.0, 0.4]])], n=200, seed=1)
    r = estimate_w_py(sim.data, p=1, q=0, include_mean=True,
                      diag_ar=diag_ar, diag_cov=diag_cov)
    assert r["npar"] == expected
    assert r["params"].size == expected


# -- diagnostics formulas ---------------------------------------------------- #

def test_hosking_q_matches_formula():
    rng = np.random.default_rng(0)
    res = rng.standard_normal((300, 2))
    s = 8
    Q, df, _ = diagnostics.hosking_q(res, s)
    # independent recomputation: Q = n * sum_r tr(C_r' C0^-1 C_r C0^-1)
    n = res.shape[0]
    x = res - res.mean(axis=0)
    C = [(x[:n - r].T @ x[r:]) / n for r in range(s + 1)]
    C0inv = np.linalg.inv(C[0])
    Qm = n * sum(np.trace(C[r].T @ C0inv @ C[r] @ C0inv) for r in range(1, s + 1))
    assert df == 2 * 2 * s
    assert abs(Q - Qm) < 1e-9


def test_jarque_bera_matches_formula():
    rng = np.random.default_rng(1)
    res = rng.standard_normal((400, 3))
    JB, df, _ = diagnostics.jarque_bera_mv(res)
    n, m = res.shape
    x = res - res.mean(axis=0)
    sd = np.sqrt((x ** 2).mean(axis=0))
    skew = (x ** 3).mean(axis=0) / sd ** 3
    kurt = (x ** 4).mean(axis=0) / sd ** 4 - 3.0
    JBm = float(np.sum(n * (skew ** 2 / 6.0 + kurt ** 2 / 24.0)))
    assert df == 2 * m
    assert abs(JB - JBm) < 1e-9


# -- determinism and edge cases --------------------------------------------- #

def test_simulation_determinism():
    phi = [np.array([[0.5, 0.1], [0.0, 0.4]])]
    a = simulate_varma(phi=phi, n=100, seed=42).data
    b = simulate_varma(phi=phi, n=100, seed=42).data
    c = simulate_varma(phi=phi, n=100, seed=43).data
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_estimator_is_deterministic():
    sim = simulate_varma(phi=[np.array([[0.5, 0.0], [0.0, 0.4]])], n=300, seed=9)
    r1 = estimate_w_py(sim.data, p=1, q=0, include_mean=True)
    r2 = estimate_w_py(sim.data, p=1, q=0, include_mean=True)
    assert r1["logelf"] == r2["logelf"]
    assert np.array_equal(r1["params"], r2["params"])


def test_near_unit_root_converges():
    case = next(c for c in CASES if c["name"] == "near_unit_root_m2")
    phi, _, sigma, mu = _arrays(case)
    sim = simulate_varma(phi=phi, sigma=sigma, mu=mu, n=500, seed=5)
    r = estimate_w_py(sim.data, p=1, q=0, include_mean=True)
    assert r["ifault"] == 0 and np.isfinite(r["logelf"])
    assert is_stationary(r["phi"])


def test_small_sample_converges():
    sim = simulate_varma(phi=[np.array([[0.5, 0.0], [0.0, 0.3]])], n=40, seed=2)
    r = estimate_w_py(sim.data, p=1, q=0, include_mean=True)
    assert r["ifault"] == 0 and np.isfinite(r["logelf"])
    assert r["residuals"].shape == (40, 2)


def test_registry_cases_stationary_invertible():
    for c in CASES:
        phi = [np.array(P) for P in c["phi"]]
        theta = [np.array(T) for T in c["theta"]]
        assert is_stationary(phi), c["name"]
        assert is_invertible(theta), c["name"]
