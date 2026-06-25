"""P3 tests: pure-Python exact VAR likelihood and estimator vs the C engine."""
import os
import sys

import numpy as np
import pytest

from drvarma import load, transform, Model
from drvarma.datasets import simulate_varma
from drvarma.elfvarma_py import elf_var
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


def test_q_positive_not_supported():
    w = np.random.default_rng(0).standard_normal((100, 2))
    with pytest.raises(NotImplementedError):
        estimate_w_py(w, p=1, q=1, include_mean=True)
