"""P2 tests: fixed-parameter recursive forecasting (-estwin) vs the C engine."""
import os
import re
import numpy as np
import pytest

pytest.importorskip("drvarma._drvarma_engine", reason="C engine not built")

from drvarma import load
from drvarma.model import Model

C_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "drvarma_v.04.1")
IPC3 = os.path.join(C_DIR, "data", "models_group1", "IPC3.inp")
C_BIN = os.path.join(C_DIR, "bin", "drvarma")


def _parse_recursive(path):
    out = {}
    for line in open(path, encoding="latin-1"):
        m = re.match(r"(\d+/\d+)\s+(\S+)\s+(\d+)\s+([-\d.]+)", line)
        if m:
            out[(m.group(1), m.group(2), int(m.group(3)))] = float(m.group(4))
    return out


def _py_dict(series, rows):
    yr0, sub0 = series.start
    freq = series.freq
    names = series.names

    def date(r):
        tot = (sub0 - 1) + (r - 1)
        return f"{tot % freq + 1}/{yr0 + tot // freq}"

    return {(date(r), names[j], l): v for (r, j, l, v) in rows}


@pytest.mark.skipif(not (os.path.exists(IPC3) and os.path.exists(C_BIN)),
                    reason="C binary or IPC3.inp not available")
@pytest.mark.parametrize("deseason_flag", [[], ["-deseason", "auto"]])
def test_recursive_matches_c(tmp_path, deseason_flag):
    import shutil, subprocess
    shutil.copy(IPC3, tmp_path / "r.inp")
    subprocess.run([os.path.abspath(C_BIN), str(tmp_path / "r"), "3", "0", "-mean",
                    "-forecast", "12", "-estwin", "200"] + deseason_flag,
                   check=True, capture_output=True)
    c = _parse_recursive(tmp_path / "r.recursive")

    s, _ = load(IPC3)
    des = "auto" if deseason_flag else None
    rows = Model(s, lam=0.0, d=1, D=0, scale=100.0, p=3, q=0,
                 include_mean=True, deseason=des).recursive_forecast(estwin=200, H=12)
    py = _py_dict(s, rows)
    common = set(py) & set(c)
    assert len(common) == len(c) == len(py)
    assert max(abs(py[k] - c[k]) for k in common) < 1e-4


@pytest.mark.skipif(not os.path.exists(C_BIN),
                    reason="C binary not available")
def test_recursive_varma_q_positive_matches_c(tmp_path):
    # PP4: q>0 recursive forecasts vs the C binary (MA term uses the
    # estimation-window residuals, zero beyond, exactly as forecast_mean).
    import subprocess
    from drvarma.datasets import simulate_varma
    from drvarma.inp import save, InpSpec
    phi = [np.array([[0.6, 0.1], [0.0, 0.5]])]
    th = [np.array([[-0.3, 0.0], [0.1, -0.2]])]
    sig = np.array([[1.0, 0.3], [0.3, 0.8]])
    sim = simulate_varma(phi=phi, theta=th, sigma=sig, n=260,
                         mu=np.array([50.0, 30.0]), seed=42, names=["Y1", "Y2"])
    save(tmp_path / "sq.inp", sim, InpSpec(lam=1.0, d=0, D=0))
    subprocess.run([os.path.abspath(C_BIN), str(tmp_path / "sq"), "1", "1", "-mean",
                    "-forecast", "6", "-estwin", "240"],
                   check=True, capture_output=True)
    c = _parse_recursive(tmp_path / "sq.recursive")

    s, spec = load(tmp_path / "sq.inp")
    rows = Model(s, lam=spec.lam, d=spec.d, D=spec.D, scale=1.0, p=1, q=1,
                 include_mean=True).recursive_forecast(estwin=240, H=6)
    py = _py_dict(s, rows)
    common = set(py) & set(c)
    assert len(common) == len(c) == len(py)
    assert max(abs(py[k] - c[k]) for k in common) < 1e-4
