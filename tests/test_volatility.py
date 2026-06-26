"""Tests for the conditional-volatility module vs the C binary (PP3)."""
import os
import shutil
import subprocess

import numpy as np
import pytest

from drvarma import load, Model
from drvarma import volatility as V
from drvarma.datasets import simulate_varma

C_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "drvarma_v.04.1")
IPC3 = os.path.join(C_DIR, "data", "models_group1", "IPC3.inp")
C_BIN = os.path.join(C_DIR, "bin", "drvarma")

has_engine = True
try:
    import drvarma._drvarma_engine  # noqa: F401
except ImportError:
    has_engine = False

needs_c = pytest.mark.skipif(
    not (os.path.exists(IPC3) and os.path.exists(C_BIN) and has_engine),
    reason="C binary / IPC3.inp / engine not available")


# -- maths (no C needed) ----------------------------------------------------- #

def test_estimate_phi_formula():
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    phi, thr = V.estimate_phi(d, 0.2)
    # idx = int(0.8*10)=8 -> sorted[8] (1-based) = 8.0; count(d>8)=2 -> 0.2
    assert thr == 8.0
    assert phi == 0.2


def test_moving_window_matches_sample_cov():
    rng = np.random.default_rng(0)
    res = rng.standard_normal((50, 2))
    times, H = V.moving_window_volatility(res, window=10)
    assert times[0] == 10 and times[-1] == 50
    # last window covariance equals numpy's unbiased sample covariance
    expected = np.cov(res[-10:].T, bias=False)
    assert np.allclose(H[-1], expected)


def test_exponential_weights_and_phi_zero_fallback():
    # residuals with no exceedances -> phi=0 -> equal weights, H_1 = res_1 res_1'
    res = np.ones((30, 2)) * 0.01
    phi, thr, H = V.exponential_volatility(res, np.eye(2), alpha=0.05, window=20)
    e = res[0]
    assert np.allclose(H[0], np.outer(e, e) / 20)   # equal weight 1/window at t=1


def test_volexp_engine_free_runs():
    sim = simulate_varma(phi=[np.array([[0.5, 0.1], [0.0, 0.4]])],
                         sigma=np.array([[1.0, 0.3], [0.3, 0.8]]), n=300, seed=3)
    from drvarma.estimate_py import estimate_w_py
    r = estimate_w_py(sim.data, p=1, q=0, include_mean=True)
    txt, phi, thr = V.volexp_text(r["residuals"], r["sigma"])
    assert txt.startswith("t var1 var2 cov12\n")
    assert txt.count("\n") == 1 + sim.data.shape[0]
    assert 0.0 <= phi <= 1.0


# -- byte-exact vs the C binary --------------------------------------------- #

@needs_c
def test_volexp_volmov_byte_exact(tmp_path):
    shutil.copy(IPC3, tmp_path / "v.inp")
    subprocess.run([os.path.abspath(C_BIN), str(tmp_path / "v"), "3", "0",
                    "-mean", "-volexp", "0.05", "20", "-volmov", "20"],
                   check=True, capture_output=True)
    s, spec = load(IPC3)
    mdl = Model(s, lam=spec.lam, d=spec.d, D=spec.D, p=3, q=0,
                include_mean=True).fit()
    res, sigma = mdl.result["residuals"], mdl.result["sigma"]

    py_ve, phi, thr = V.volexp_text(res, sigma, 0.05, 20)
    py_vm = V.volmov_text(res, 20)
    c_ve = open(tmp_path / "v.volexp", encoding="latin-1").read()
    c_vm = open(tmp_path / "v.volmov", encoding="latin-1").read()
    assert py_ve == c_ve
    assert py_vm == c_vm
    # the .out info line the C appends
    assert ("threshold = %s" % V._g(thr)) in open(tmp_path / "v.out",
                                                  encoding="latin-1").read()
