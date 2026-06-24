"""P2 tests: multivariate forecasting + integration to levels vs the C engine."""
import os
import re
import numpy as np
import pytest

pytest.importorskip("drvarma._drvarma_engine",
                    reason="C engine not built (run: python -m drvarma._build_cffi)")

from drvarma import load, transform, MultiSeries
from drvarma.model import Model
from drvarma.datasets import simulate_varma

C_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "drvarma_v.04.1")
IPC3 = os.path.join(C_DIR, "data", "models_group1", "IPC3.inp")
C_BIN = os.path.join(C_DIR, "bin", "drvarma")


@pytest.mark.skipif(not (os.path.exists(IPC3) and os.path.exists(C_BIN)),
                    reason="C drvarma binary or IPC3.inp not available")
def test_forecast_levels_match_c(tmp_path):
    import shutil, subprocess
    shutil.copy(IPC3, tmp_path / "fc.inp")
    subprocess.run([os.path.abspath(C_BIN), str(tmp_path / "fc"), "3", "0",
                    "-mean", "-forecast", "12"], check=True, capture_output=True)
    # parse C forecast levels per series
    cl = {1: [], 2: [], 3: []}
    ser = 0
    for line in open(tmp_path / "fc.forecast", encoding="latin-1"):
        ms = re.search(r"Series\s+(\d+)", line)
        if ms:
            ser = int(ms.group(1)); continue
        m = re.match(r"\s*\d+/\d+\s+([-\d.]+)", line)
        if m and ser in cl:
            cl[ser].append(float(m.group(1)))

    s, spec = load(IPC3)
    mdl = Model(s, lam=0.0, d=1, D=0, scale=100.0, p=3, q=0, include_mean=True).fit()
    lev = mdl.forecast(12)
    for j in range(3):
        c = np.array(cl[j + 1][:12])
        assert np.max(np.abs(lev[:, j] - c)) < 1e-3


@pytest.mark.skipif(not (os.path.exists(IPC3) and os.path.exists(C_BIN)),
                    reason="C drvarma binary or IPC3.inp not available")
def test_forecast_bands_match_c(tmp_path):
    import shutil, subprocess
    shutil.copy(IPC3, tmp_path / "fb.inp")
    subprocess.run([os.path.abspath(C_BIN), str(tmp_path / "fb"), "3", "0", "-mean",
                    "-deseason", "auto", "-forecast", "12"], check=True, capture_output=True)
    cL = {1: [], 2: [], 3: []}; ser = 0
    for line in open(tmp_path / "fb.forecast", encoding="latin-1"):
        ms = re.search(r"Series\s+(\d+)", line)
        if ms:
            ser = int(ms.group(1)); continue
        mm = re.match(r"\s*\d+/\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
        if mm and ser in cL:
            cL[ser].append([float(mm.group(1)), float(mm.group(2)), float(mm.group(3))])
    s, _ = load(IPC3)
    mdl = Model(s, lam=0.0, d=1, D=0, scale=100.0, p=3, q=0,
                include_mean=True, deseason="auto").fit()
    lev, low, high = mdl.forecast(12, bands=True)
    for j in range(3):
        c = np.array(cL[j + 1][:12])
        assert np.max(np.abs(lev[:, j] - c[:, 0])) < 1e-3
        assert np.max(np.abs(low[:, j] - c[:, 1])) < 1e-3
        assert np.max(np.abs(high[:, j] - c[:, 2])) < 1e-3


def test_forecast_shapes_and_finite():
    sim = simulate_varma(phi=[np.array([[0.5, 0.0], [0.2, 0.4]])], n=300, seed=21)
    # treat as raw levels: identity transform, no differencing, scale 1
    mdl = Model(sim, lam=1.0, d=0, D=0, scale=1.0, p=1, q=0, include_mean=True).fit()
    f = mdl.forecast(6)
    assert f.shape == (6, 2)
    assert np.all(np.isfinite(f))


def test_integrate_inverts_difference():
    # a known level series: integrating its differenced "forecast" reproduces it
    rng = np.random.default_rng(0)
    level = np.cumsum(rng.normal(size=(50, 1)) + 0.1, axis=0) + 100.0
    bc = 100.0 * transform.boxcox_fwd(level, 0.0)
    # pretend the last 5 diffs are the "forecast" and integrate from origin -5
    w = transform.difference(bc, d=1)
    wf = w[-5:]
    out = transform.integrate_forecast(bc[:-5], wf, lam=0.0, scale=100.0, d=1, D=0, s=12)
    assert np.allclose(out[:, 0], level[-5:, 0], atol=1e-8)
