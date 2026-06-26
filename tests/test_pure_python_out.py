"""PP5: the engine-free path is fidelity-complete.

Two locks:
  1. the pure-Python estimator reproduces the C engine across the model zoo
     (VAR & VARMA, ±deseason, diag restrictions) to ~1e-5 — convergence hardening;
  2. with the engine monkeypatched off, the *deterministic* `.out` sections
     (those driven only by the point estimates: IRF, FEVD, multivariate
     diagnostics, normalized model) are byte-identical to the C binary.

Excluded by design: the "Inverse roots" block (modulus-sorted, not chekma's QR
order) and the σ²/Q *split* under deseason (scale-ambiguous flat direction; Σ
still matches) — see docs/PURE_PYTHON_PLAN.md.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from drvarma import load, Model, report, transform
from drvarma.estimate_py import estimate_w_py

C_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "drvarma_v.04.1")
IPC3 = os.path.join(C_DIR, "data", "models_group1", "IPC3.inp")
C_BIN = os.path.join(C_DIR, "bin", "drvarma")

has_engine = True
try:
    import drvarma._drvarma_engine  # noqa: F401
except ImportError:
    has_engine = False

needs_engine = pytest.mark.skipif(not (has_engine and os.path.exists(IPC3)),
                                  reason="engine / IPC3 not available")
needs_c = pytest.mark.skipif(
    not (os.path.exists(IPC3) and os.path.exists(C_BIN)),
    reason="C binary / IPC3 not available")


def _ipc3_w(deseason=None):
    ser, spec = load(IPC3)
    levels = np.asarray(ser.data).reshape(ser.nobs, ser.m)
    if deseason:
        from drvarma.deseason import deseasonalize_raw
        levels, _, _ = deseasonalize_raw(levels, s=ser.freq,
                                         start_sub=ser.start[1], mode=deseason)
    w, _ = transform.transform(levels, lam=spec.lam, d=spec.d, D=spec.D,
                               s=ser.freq, scale=100.0)
    return w


@needs_engine
@pytest.mark.parametrize("kw,deseason", [
    (dict(p=3, q=0, include_mean=True), None),
    (dict(p=3, q=0, include_mean=True), "auto"),
    (dict(p=2, q=0, include_mean=True, diag_cov=True), None),
    (dict(p=2, q=0, include_mean=True, diag_ar=True), None),
])
def test_engine_free_estimate_matches_engine(kw, deseason):
    from drvarma._engine import estimate_w
    w = _ipc3_w(deseason)
    c = estimate_w(w, **kw)
    py = estimate_w_py(w, **kw)
    assert py["ifault"] == 0
    # point estimates / likelihood / innovation covariance match the engine.
    assert abs(py["logelf"] - c["logelf"]) < 1e-6
    assert np.max(np.abs(py["mu"] - c["mu"])) < 1e-5
    assert np.max(np.abs(py["phi"] - c["phi"])) < 1e-5
    assert np.max(np.abs(py["sigma"] - c["sigma"])) < 1e-5   # Σ = σ²·Q (split-invariant)


@needs_c
@pytest.mark.parametrize("marker_pair", [
    ("ORTHOGONALIZED IMPULSE", "ACCUMULATED IMPULSE"),
    ("ACCUMULATED IMPULSE", "LONG"),
    ("FORECAST ERROR VARIANCE", "MULTIVARIATE RESIDUAL"),
    ("MULTIVARIATE RESIDUAL", "Normalized model"),
    ("Normalized model", "Inverse roots"),
])
def test_engine_free_out_byte_exact(tmp_path, monkeypatch, marker_pair):
    # Force the pure-Python path, then diff each deterministic .out section
    # against the C binary (VAR(3): the σ²/Q split also lands byte-exact here).
    monkeypatch.setitem(sys.modules, "drvarma._drvarma_engine", None)
    shutil.copy(IPC3, tmp_path / "o.inp")
    subprocess.run([os.path.abspath(C_BIN), str(tmp_path / "o"), "3", "0", "-mean"],
                   check=True, capture_output=True)
    c_text = open(tmp_path / "o.out", encoding="latin-1").read()

    s, spec = load(IPC3)
    mdl = Model(s, lam=spec.lam, d=spec.d, D=spec.D, scale=100.0, p=3, q=0,
                include_mean=True).fit()
    py_text = report.out_report(mdl)

    def section(text, a, b):
        out, on = [], False
        for ln in text.splitlines(keepends=True):
            if a in ln:
                on = True
            elif b and b in ln and on:
                break
            if on:
                out.append(ln)
        return "".join(out)

    start, end = marker_pair
    assert section(py_text, start, end) == section(c_text, start, end)
