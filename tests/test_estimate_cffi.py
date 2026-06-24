"""P1 tests: estimation via the CFFI C engine.

- synthetic VARMA(1,0) parameter recovery (self-contained);
- agreement with the C drvarma binary on IPC3 when available.
"""
import os
import numpy as np
import pytest

pytest.importorskip("drvarma._drvarma_engine",
                    reason="C engine not built (run: python -m drvarma._build_cffi)")

from drvarma._engine import estimate_w
from drvarma.datasets import simulate_varma


def test_var1_recovery():
    Phi = np.array([[0.5, 0.0], [0.2, 0.4]])
    sim = simulate_varma(phi=[Phi], n=4000, seed=11)
    r = estimate_w(sim.data, p=1, q=0, include_mean=True)
    assert r["ifault"] == 0
    # params order: mu(2), phi[1]_11, _12, _21, _22, cov(3)
    phi_hat = r["params"][2:6].reshape(2, 2)
    assert np.allclose(phi_hat, Phi, atol=0.05), phi_hat


def test_diagonal_ar1_recovery():
    sim = simulate_varma(phi=[np.array([[0.6, 0.0], [0.0, 0.3]])], n=4000, seed=12)
    r = estimate_w(sim.data, p=1, q=0, include_mean=True, diag_ar=True, diag_cov=True)
    assert r["ifault"] == 0
    # diagonal AR: params = mu(2), phi_11, phi_22, cov_11, cov_22
    assert abs(r["params"][2] - 0.6) < 0.05
    assert abs(r["params"][3] - 0.3) < 0.05


# --- agreement with the C engine on the reference trivariate case ---
C_OUT = os.path.join(os.path.dirname(__file__), "..", "..", "drvarma_v.04.1",
                     "data", "models_group1")
IPC3 = os.path.join(C_OUT, "IPC3.inp")


C_BIN = os.path.join(os.path.dirname(__file__), "..", "..", "drvarma_v.04.1",
                     "bin", "drvarma")


@pytest.mark.skipif(not (os.path.exists(IPC3) and os.path.exists(C_BIN)),
                    reason="C drvarma binary or IPC3.inp not available")
def test_matches_c_params_ipc3(tmp_path):
    """Python-CFFI params must match the C engine on IPC3 (3 0 -mean, no deseason)."""
    import re, shutil, subprocess
    from drvarma import load, transform
    inp = tmp_path / "ref.inp"
    shutil.copy(IPC3, inp)
    subprocess.run([os.path.abspath(C_BIN), str(tmp_path / "ref"), "3", "0", "-mean"],
                   check=True, capture_output=True)
    cest = []
    for l in open(tmp_path / "ref.out", encoding="latin-1"):
        m = re.match(r"\s*(mu|phi|cov)\S*\s+([-\d.]+)\s+", l)
        if m:
            cest.append(float(m.group(2)))
    assert len(cest) == 36
    s, spec = load(IPC3)
    w, _ = transform.transform(s.data, lam=0.0, d=1, D=0, s=12, scale=100.0)
    r = estimate_w(w, p=3, q=0, include_mean=True)
    assert np.max(np.abs(r["params"] - np.array(cest))) < 1e-5
