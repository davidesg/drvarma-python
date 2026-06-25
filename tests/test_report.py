"""Tests for the .out/.forecast/.recursive report writers vs the C engine."""
import os
import re
import shutil
import subprocess

import pytest

pytest.importorskip("drvarma._drvarma_engine",
                    reason="C engine not built (run: python -m drvarma._build_cffi)")

from drvarma import load, Model, report

C_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "drvarma_v.04.1")
IPC3 = os.path.join(C_DIR, "data", "models_group1", "IPC3.inp")
C_BIN = os.path.join(C_DIR, "bin", "drvarma")

needs_c = pytest.mark.skipif(
    not (os.path.exists(IPC3) and os.path.exists(C_BIN)),
    reason="C drvarma binary or IPC3.inp not available")


def _section(text, start_marker, end_marker):
    """Lines from the line containing start_marker up to (excl.) end_marker."""
    out, on = [], False
    for line in text.splitlines(keepends=True):
        if start_marker in line:
            on = True
        elif end_marker and end_marker in line and on:
            break
        if on:
            out.append(line)
    return "".join(out)


def _fit_ipc3(deseason="auto"):
    s, spec = load(IPC3)
    return Model(s, lam=spec.lam, d=spec.d, D=spec.D, p=3, q=0,
                 include_mean=True, deseason=deseason).fit()


@needs_c
def test_forecast_report_byte_exact(tmp_path):
    shutil.copy(IPC3, tmp_path / "fc.inp")
    subprocess.run([os.path.abspath(C_BIN), str(tmp_path / "fc"), "3", "0",
                    "-mean", "-deseason", "auto", "-forecast", "24"],
                   check=True, capture_output=True)
    c_text = open(tmp_path / "fc.forecast", encoding="latin-1").read()
    py_text = report.forecast_report(_fit_ipc3("auto"), 24)
    assert py_text == c_text


@needs_c
@pytest.mark.parametrize("marker_pair", [
    ("ORTHOGONALIZED IMPULSE", "ACCUMULATED IMPULSE"),
    ("ACCUMULATED IMPULSE", "LONG"),
    ("FORECAST ERROR VARIANCE", "MULTIVARIATE RESIDUAL"),
    ("MULTIVARIATE RESIDUAL", "Normalized model"),
    ("Normalized model", "Inverse roots"),
])
def test_out_deterministic_sections_byte_exact(tmp_path, marker_pair):
    # These sections depend only on the (exactly-matching) point estimates.
    shutil.copy(IPC3, tmp_path / "o.inp")
    subprocess.run([os.path.abspath(C_BIN), str(tmp_path / "o"), "3", "0",
                    "-mean", "-deseason", "auto"], check=True, capture_output=True)
    c_text = open(tmp_path / "o.out", encoding="latin-1").read()
    py_text = report.out_report(_fit_ipc3("auto"))
    start, end = marker_pair
    assert _section(py_text, start, end) == _section(c_text, start, end)


@needs_c
def test_out_parameter_estimates_match(tmp_path):
    # SE/t/p carry the documented <1e-5 engine tolerance, so compare estimates.
    shutil.copy(IPC3, tmp_path / "p.inp")
    subprocess.run([os.path.abspath(C_BIN), str(tmp_path / "p"), "3", "0",
                    "-mean", "-deseason", "auto"], check=True, capture_output=True)
    c_text = open(tmp_path / "p.out", encoding="latin-1").read()
    py_text = report.out_report(_fit_ipc3("auto"))
    pat = re.compile(r"^((?:mu|phi|theta|cov)\[[^\]]*\][^ ]*)\s+([-\d.]+)\s+[-\d.]",
                     re.M)
    cpar = {m.group(1): float(m.group(2)) for m in pat.finditer(c_text)}
    ppar = {m.group(1): float(m.group(2)) for m in pat.finditer(py_text)}
    assert set(cpar) == set(ppar) and cpar
    assert max(abs(cpar[k] - ppar[k]) for k in cpar) < 1e-4


@needs_c
def test_recursive_report_matches_c(tmp_path):
    shutil.copy(IPC3, tmp_path / "r.inp")
    subprocess.run([os.path.abspath(C_BIN), str(tmp_path / "r"), "3", "0", "-mean",
                    "-forecast", "12", "-estwin", "200", "-deseason", "auto"],
                   check=True, capture_output=True)
    c_text = open(tmp_path / "r.recursive", encoding="latin-1").read()
    py_text = report.recursive_report(_fit_ipc3("auto"), estwin=200, H=12)
    # headers identical
    assert py_text.splitlines()[:3] == c_text.splitlines()[:3]
    # values match the validated engine path to <1e-4
    row = re.compile(r"(\d+/\d+)\s+(\S+)\s+(\d+)\s+([-\d.]+)")
    def parse(t):
        return {(m.group(1), m.group(2), int(m.group(3))): float(m.group(4))
                for m in (row.match(l) for l in t.splitlines()) if m}
    c, p = parse(c_text), parse(py_text)
    common = set(c) & set(p)
    assert len(common) == len(c) == len(p)
    assert max(abs(c[k] - p[k]) for k in common) < 1e-4
