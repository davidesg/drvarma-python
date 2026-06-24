"""P2 tests: multivariate residual diagnostics vs the C engine."""
import os
import re
import numpy as np
import pytest

pytest.importorskip("drvarma._drvarma_engine",
                    reason="C engine not built")

from drvarma import load, transform
from drvarma.model import Model
from drvarma.diagnostics import hosking_q, jarque_bera_mv
from drvarma.datasets import simulate_varma

C_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "drvarma_v.04.1")
IPC3 = os.path.join(C_DIR, "data", "models_group1", "IPC3.inp")
C_BIN = os.path.join(C_DIR, "bin", "drvarma")


@pytest.mark.skipif(not (os.path.exists(IPC3) and os.path.exists(C_BIN)),
                    reason="C binary or IPC3.inp not available")
def test_hosking_jb_match_c(tmp_path):
    import shutil, subprocess
    shutil.copy(IPC3, tmp_path / "d.inp")
    subprocess.run([os.path.abspath(C_BIN), str(tmp_path / "d"), "3", "0", "-mean"],
                   check=True, capture_output=True)
    out = open(tmp_path / "d.out", encoding="latin-1").read()
    cQ = float(re.search(r"Q\(\d+\)\s*=\s*([\d.]+),\s*p-value", out).group(1))
    cJB = float(re.search(r"JB\(\d+\)\s*=\s*([\d.]+),\s*p-value", out).group(1))
    lag = int(re.search(r"Portmanteau Test \(lag (\d+)\)", out).group(1))

    s, _ = load(IPC3)
    mdl = Model(s, lam=0.0, d=1, D=0, scale=100.0, p=3, q=0, include_mean=True).fit()
    d = mdl.diagnostics(lag=lag)
    assert abs(d["hosking_Q"] - cQ) < 1e-3
    assert abs(d["JB"] - cJB) < 1e-3


def test_jb_normal_residuals():
    # white standard-normal residuals -> small JB, large p-value
    rng = np.random.default_rng(0)
    res = rng.standard_normal((5000, 2))
    JB, df, p = jarque_bera_mv(res)
    assert df == 4
    assert p > 0.01


def test_hosking_white_noise():
    rng = np.random.default_rng(1)
    res = rng.standard_normal((4000, 2))
    Q, df, p = hosking_q(res, 10)
    assert df == 40
    assert p > 0.01  # no autocorrelation -> not rejected
