"""Test the residual diagnostics section of the .out (vs IPC3.out)."""
import os
import re

import numpy as np
import pytest

pytest.importorskip("pyfug", reason="pyfug not installed")
pytest.importorskip("drvarma._drvarma_engine", reason="C engine not built")

from drvarma import load, Model, report

C_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "drvarma_v.04.1")
IPC3 = os.path.join(C_DIR, "data", "models_group1", "IPC3.inp")
IPC3_OUT = os.path.join(C_DIR, "data", "models_group1", "IPC3.out")

needs_ipc3 = pytest.mark.skipif(
    not (os.path.exists(IPC3) and os.path.exists(IPC3_OUT)),
    reason="IPC3 reference files absent")

# the standardized-plot value column prints residuals to 1e-10; they differ from
# the committed .out at ~1e-9 (engine tolerance), so ignore that column.
_VAL = re.compile(r"[0-9]\.[0-9]{10}\s*$")


def _strip_val(line):
    return _VAL.sub("", line.rstrip())


@needs_ipc3
def test_residual_section_matches_ipc3_out():
    ser, spec = load(IPC3)
    mdl = Model(ser, lam=spec.lam, d=spec.d, D=spec.D, p=3, q=0,
                include_mean=True, deseason="auto").fit()
    py = report.residual_report(mdl).splitlines()

    ref_all = open(IPC3_OUT, encoding="latin-1").read().splitlines()
    # the true RESIDUAL DIAGNOSTICS section (not the MULTIVARIATE one)
    start = next(i for i, l in enumerate(ref_all)
                 if l.strip() == "RESIDUAL DIAGNOSTICS")
    ref = ref_all[start - 1:]                       # from its leading banner

    # align: py starts with blank lines then the banner
    pstart = next(i for i, l in enumerate(py) if l.strip() == "RESIDUAL DIAGNOSTICS")
    py = py[pstart - 1:]

    assert len(py) == len(ref), (len(py), len(ref))
    mism = [(i, r, p) for i, (r, p) in enumerate(zip(ref, py))
            if _strip_val(r) != _strip_val(p)]
    assert not mism, mism[:5]


@needs_ipc3
def test_out_report_auto_includes_residuals():
    ser, spec = load(IPC3)
    mdl = Model(ser, lam=spec.lam, d=spec.d, D=spec.D, p=3, q=0,
                include_mean=True, deseason="auto").fit()
    txt = report.out_report(mdl)                     # residuals="auto", pyfug present
    assert "--- Residual series a[1] (IPC_ES) ---" in txt
    assert "Cross-correlation functions between residuals" in txt
    assert "--- Residual series" not in report.out_report(mdl, residuals=False)
