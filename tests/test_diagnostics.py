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
from _cdiag import C_DIAG_BIN as C_BIN


@pytest.mark.skipif(not (os.path.exists(IPC3) and os.path.exists(C_BIN)),
                    reason="C binary or IPC3.inp not available")
def test_hosking_jb_match_c(tmp_path):
    import shutil, subprocess
    shutil.copy(IPC3, tmp_path / "d.inp")
    subprocess.run([os.path.abspath(C_BIN), str(tmp_path / "d"), "3", "0", "-mean"],
                   check=True, capture_output=True)
    out = open(tmp_path / "d.out", encoding="latin-1").read()
    cQ, cdf = re.search(r"Q\((\d+)\)\s*=\s*([\d.]+),\s*p-value", out).group(2, 1)
    cJB = float(re.search(r"JB\(\d+\)\s*=\s*([\d.]+),\s*p-value", out).group(1))
    lag = int(re.search(r"Portmanteau Test \(lag (\d+)\)", out).group(1))
    assert lag == 14                       # freq + 2, not floor(sqrt(n))

    s, _ = load(IPC3)
    mdl = Model(s, lam=0.0, d=1, D=0, scale=100.0, p=3, q=0, include_mean=True).fit()
    d = mdl.diagnostics()
    assert d["hosking_lag"] == lag         # the default is the C's rule
    assert abs(d["hosking_Q"] - float(cQ)) < 1e-3
    assert d["hosking_df"] == int(cdf)     # m^2 s - k, as lib/diagmv
    assert abs(d["JB"] - cJB) < 1e-3       # Lutkepohl, as lib/diagmv


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
