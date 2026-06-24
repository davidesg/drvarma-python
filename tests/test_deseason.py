"""P2 tests: harmonic seasonal adjustment vs the C engine.

Validates the delicate part: amplitudes estimated on the d=1 differenced basis,
level dummies recovered and subtracted from levels.
"""
import os
import re
import numpy as np
import pytest

from drvarma import load
from drvarma.deseason import (deseasonalize_raw, harmonics_to_dummies,
                              harmonic_regression_differenced)

C_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "drvarma_v.04.1")
IPC3 = os.path.join(C_DIR, "data", "models_group1", "IPC3.inp")
C_BIN = os.path.join(C_DIR, "bin", "drvarma")


def test_dummies_sum_to_zero():
    coeffs = np.arange(1, 12, dtype=float) * 0.1
    d = harmonics_to_dummies(coeffs, 12)
    assert len(d) == 12
    assert abs(d.sum()) < 1e-12


def test_no_seasonality_not_adjusted():
    rng = np.random.default_rng(0)
    raw = np.cumsum(rng.normal(size=(120, 1)), axis=0) + 100.0  # no seasonality
    adj, dum, info = deseasonalize_raw(raw, s=12, start_sub=1, mode="auto")
    assert not info[0]["adjusted"]
    assert np.allclose(adj, raw)            # unchanged
    assert np.allclose(dum, 0.0)


@pytest.mark.skipif(not (os.path.exists(IPC3) and os.path.exists(C_BIN)),
                    reason="C binary or IPC3.inp not available")
def test_fstats_match_c(tmp_path):
    import shutil, subprocess
    shutil.copy(IPC3, tmp_path / "d.inp")
    out = subprocess.run([os.path.abspath(C_BIN), str(tmp_path / "d"), "3", "0",
                          "-mean", "-deseason", "auto"],
                         check=True, capture_output=True, text=True).stdout
    cf = [float(m) for m in re.findall(r"Series \d+: F=([\d.]+)", out)]
    s, _ = load(IPC3)
    _, _, info = deseasonalize_raw(s.data, s=12, start_sub=1, mode="auto")
    pf = [d["f_stat"] for d in info]
    assert np.allclose(pf, cf, atol=1e-2)


@pytest.mark.skipif(not (os.path.exists(IPC3) and os.path.exists(C_BIN)),
                    reason="C binary or IPC3.inp not available")
def test_deseason_params_and_forecast_match_c(tmp_path):
    pytest.importorskip("drvarma._drvarma_engine")
    import shutil, subprocess
    from drvarma.model import Model
    shutil.copy(IPC3, tmp_path / "d.inp")
    subprocess.run([os.path.abspath(C_BIN), str(tmp_path / "d"), "3", "0",
                    "-mean", "-deseason", "auto", "-forecast", "12"],
                   check=True, capture_output=True)
    # C params (no deseason harmonics appear in the mu/phi/cov estimate list)
    cest = [float(m.group(2)) for m in
            (re.match(r"\s*(mu|phi|cov)\S*\s+([-\d.]+)\s+", l)
             for l in open(tmp_path / "d.out", encoding="latin-1")) if m]
    cl = {1: [], 2: [], 3: []}; ser = 0
    for line in open(tmp_path / "d.forecast", encoding="latin-1"):
        ms = re.search(r"Series\s+(\d+)", line)
        if ms:
            ser = int(ms.group(1)); continue
        mm = re.match(r"\s*\d+/\d+\s+([-\d.]+)", line)
        if mm and ser in cl:
            cl[ser].append(float(mm.group(1)))

    s, _ = load(IPC3)
    m = Model(s, lam=0.0, d=1, D=0, scale=100.0, p=3, q=0,
              include_mean=True, deseason="auto").fit()
    assert np.max(np.abs(m.result["params"] - np.array(cest))) < 1e-5
    lev = m.forecast(12)
    for j in range(3):
        assert np.max(np.abs(lev[:, j] - np.array(cl[j + 1][:12]))) < 1e-3
