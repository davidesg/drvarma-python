"""P3 tests: pure-Python exact VAR likelihood and estimator vs the C engine."""
import os
import sys

import numpy as np
import pytest

from drvarma import load, transform, Model, report
from drvarma.datasets import simulate_varma
from drvarma.elfvarma_py import elf_var, elf_varma
from drvarma.estimate_py import estimate_w_py

C_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "drvarma_v.04.1")
IPC3 = os.path.join(C_DIR, "data", "models_group1", "IPC3.inp")

has_engine = True
try:
    import drvarma._drvarma_engine  # noqa: F401
except ImportError:
    has_engine = False

needs_engine = pytest.mark.skipif(not has_engine, reason="C engine not built")
needs_ipc3 = pytest.mark.skipif(not os.path.exists(IPC3), reason="IPC3.inp absent")


def _ipc3_w():
    ser, spec = load(IPC3)
    w, _ = transform.transform(ser.data, lam=spec.lam, d=spec.d, D=spec.D,
                               s=ser.freq, scale=100.0)
    return w


@needs_engine
@needs_ipc3
def test_elf_var_matches_c_loglik():
    from drvarma._engine import estimate_w
    w = _ipc3_w()
    r = estimate_w(w, p=3, q=0, include_mean=True)
    ll, ifault = elf_var(w, r["mu"], r["phi"], r["sigma"])
    assert ifault == 0
    assert abs(ll - r["logelf"]) < 1e-6


@needs_engine
@needs_ipc3
def test_estimate_py_matches_c():
    from drvarma._engine import estimate_w
    w = _ipc3_w()
    c = estimate_w(w, p=3, q=0, include_mean=True)
    py = estimate_w_py(w, p=3, q=0, include_mean=True)
    assert py["ifault"] == 0
    assert abs(py["logelf"] - c["logelf"]) < 1e-4
    assert np.max(np.abs(py["mu"] - c["mu"])) < 1e-3
    assert np.max(np.abs(py["phi"] - c["phi"])) < 1e-3
    assert np.max(np.abs(py["sigma"] - c["sigma"])) < 1e-3


@needs_ipc3
def test_engine_falls_back_without_extension(monkeypatch):
    # Make `import drvarma._drvarma_engine` fail so estimate_w uses pure Python.
    monkeypatch.setitem(sys.modules, "drvarma._drvarma_engine", None)
    from drvarma._engine import estimate_w
    w = _ipc3_w()
    r = estimate_w(w, p=2, q=0, include_mean=True)
    ref = estimate_w_py(w, p=2, q=0, include_mean=True)
    assert r["ifault"] == 0
    assert abs(r["logelf"] - ref["logelf"]) < 1e-8
    assert np.allclose(r["phi"], ref["phi"])


def test_synthetic_var1_recovery():
    phi_true = np.array([[0.5, 0.1], [-0.2, 0.4]])
    sigma_true = np.array([[1.0, 0.3], [0.3, 0.8]])
    sim = simulate_varma(phi=[phi_true], sigma=sigma_true, n=4000, seed=7)
    r = estimate_w_py(sim.data, p=1, q=0, include_mean=True)
    assert r["ifault"] == 0
    assert np.max(np.abs(r["phi"][0] - phi_true)) < 0.05
    assert np.max(np.abs(r["sigma"] - sigma_true)) < 0.1


def test_model_fit_via_pure_python(monkeypatch):
    monkeypatch.setitem(sys.modules, "drvarma._drvarma_engine", None)
    phi_true = np.array([[0.6, 0.0], [0.2, 0.3]])
    sim = simulate_varma(phi=[phi_true], n=1500, seed=3)
    mdl = Model(sim, lam=1.0, d=0, D=0, scale=1.0, p=1, q=0,
                include_mean=True).fit()
    assert mdl.ifault == 0
    assert np.max(np.abs(mdl.result["phi"][0] - phi_true)) < 0.1


# -- VARMA (q>0): faithful AS 311 port --------------------------------------- #

@needs_engine
@needs_ipc3
@pytest.mark.parametrize("pq", [(1, 1), (2, 1)])
def test_elf_varma_matches_c_loglik(pq):
    from drvarma._engine import estimate_w
    p, q = pq
    w = _ipc3_w()
    r = estimate_w(w, p=p, q=q, include_mean=True)
    ll, ifault, res = elf_varma(w, r["mu"], r["phi"], r["theta"], r["sigma"],
                                compute_residuals=True)
    assert ifault == 0
    assert abs(ll - r["logelf"]) < 1e-6
    assert np.max(np.abs(res - r["residuals"])) < 1e-6   # AS 311 exact residuals


def test_elf_var_matches_elf_varma_q0():
    # the fast VAR path and AS 311 (q=0) must agree
    phi = [np.array([[0.5, 0.1], [-0.2, 0.4]])]
    sigma = np.array([[1.0, 0.3], [0.3, 0.8]])
    sim = simulate_varma(phi=phi, sigma=sigma, n=300, seed=5)
    mu = sim.data.mean(axis=0)
    a, _ = elf_var(sim.data, mu, np.array(phi), sigma)
    b, ifault, _ = elf_varma(sim.data, mu, np.array(phi), np.zeros((0, 2, 2)), sigma)
    assert ifault == 0
    assert abs(a - b) < 1e-6


@needs_engine
def test_estimate_py_varma_matches_c():
    from drvarma._engine import estimate_w
    phi_t = np.array([[0.5, 0.1], [0.0, 0.4]])
    th_t = np.array([[0.3, 0.0], [0.1, 0.2]])
    sig = np.array([[1.0, 0.2], [0.2, 0.7]])
    sim = simulate_varma(phi=[phi_t], theta=[th_t], sigma=sig, n=1200, seed=11)
    c = estimate_w(sim.data, p=1, q=1, include_mean=True)
    py = estimate_w_py(sim.data, p=1, q=1, include_mean=True)
    assert py["ifault"] == 0
    assert abs(py["logelf"] - c["logelf"]) < 1e-4
    assert np.max(np.abs(py["phi"] - c["phi"])) < 1e-3
    assert np.max(np.abs(py["theta"] - c["theta"])) < 1e-3
    assert np.max(np.abs(py["sigma"] - c["sigma"])) < 1e-3


def test_estimate_py_varma_is_mle():
    # without the C engine: the fitted log-likelihood must beat the truth's
    phi_t = np.array([[0.5, 0.1], [0.0, 0.4]])
    th_t = np.array([[0.3, 0.0], [0.1, 0.2]])
    sig = np.array([[1.0, 0.2], [0.2, 0.7]])
    sim = simulate_varma(phi=[phi_t], theta=[th_t], sigma=sig, n=800, seed=2)
    py = estimate_w_py(sim.data, p=1, q=1, include_mean=True)
    ll_truth, ifault, _ = elf_varma(sim.data, sim.data.mean(axis=0),
                                    np.array([phi_t]), np.array([th_t]), sig)
    assert py["ifault"] == 0 and ifault == 0
    assert py["logelf"] >= ll_truth - 1e-6
    assert py["theta"].shape == (1, 2, 2)


def test_full_pipeline_without_engine(monkeypatch):
    # The whole Model -> forecast -> report path must work on the pure-Python
    # fallback (no compiled engine) for a VARMA model.
    monkeypatch.setitem(sys.modules, "drvarma._drvarma_engine", None)
    phi = [np.array([[0.5, 0.1], [0.0, 0.4]])]
    th = [np.array([[0.3, 0.0], [0.1, 0.2]])]
    sim = simulate_varma(phi=phi, theta=th, n=150, seed=4, names=["A", "B"])
    mdl = Model(sim, lam=1.0, d=0, D=0, scale=1.0, p=1, q=1,
                include_mean=True).fit()
    assert mdl.ifault == 0
    assert mdl.result["theta"].shape == (1, 2, 2)
    fc = mdl.forecast(6)
    assert fc.shape == (6, 2) and np.all(np.isfinite(fc))
    out = report.out_report(mdl, input_path="sim.inp", output_path="sim.out")
    assert "VARMA(1,1)" in out and "FORECAST ERROR VARIANCE" in out
    fr = report.forecast_report(mdl, 6)
    assert fr.count("Series") == 2
