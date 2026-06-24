"""P2 tests: impulse responses (OIRF) and FEVD vs the C engine."""
import os
import re
import numpy as np
import pytest

pytest.importorskip("drvarma._drvarma_engine", reason="C engine not built")

from drvarma import load, transform
from drvarma.model import Model
from drvarma import irf as irfmod

C_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "drvarma_v.04.1")
IPC3 = os.path.join(C_DIR, "data", "models_group1", "IPC3.inp")
C_BIN = os.path.join(C_DIR, "bin", "drvarma")


def _rows(lines, header_re):
    """Parse blocks: header line matching header_re (captures key) then numeric rows."""
    blocks = {}
    key = None
    for ln in lines:
        mh = re.match(header_re, ln)
        if mh:
            key = mh.group(1); blocks[key] = []; continue
        if key is not None:
            m = re.match(r"\s*\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", ln)
            if m:
                blocks[key].append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
            elif "Horizon" in ln or ln.strip() == "":
                continue
            else:
                key = None
    return blocks


@pytest.mark.skipif(not (os.path.exists(IPC3) and os.path.exists(C_BIN)),
                    reason="C binary or IPC3.inp not available")
def test_oirf_and_fevd_match_c(tmp_path):
    import shutil, subprocess
    shutil.copy(IPC3, tmp_path / "i.inp")
    subprocess.run([os.path.abspath(C_BIN), str(tmp_path / "i"), "3", "0", "-mean"],
                   check=True, capture_output=True)
    txt = open(tmp_path / "i.out", encoding="latin-1").read().splitlines()
    names = ["IPC_ES", "IPC_FR", "IPC_DE"]

    # non-cumulative IRF section (before ACCUMULATED)
    cut = next((i for i, l in enumerate(txt) if "ACCUMULATED" in l), len(txt))
    irf_c = _rows(txt[:cut], r"Shock to (\S+):")
    # FEVD section (after FORECAST ERROR VARIANCE DECOMPOSITION)
    fstart = next(i for i, l in enumerate(txt) if "FORECAST ERROR VARIANCE" in l)
    fevd_c = _rows(txt[fstart:], r"^(IPC_\S+):")

    s, _ = load(IPC3)
    mdl = Model(s, lam=0.0, d=1, D=0, scale=100.0, p=3, q=0, include_mean=True).fit()
    H = len(irf_c[names[0]])
    O = mdl.irf(H - 1)                      # (H, m, m): O[h, i, j]
    for jname, j in zip(names, range(3)):
        arr = np.array(irf_c[jname])        # rows h, cols i (response var)
        assert np.max(np.abs(arr - O[:, :, j])) < 1e-3

    Hf = len(fevd_c[names[0]])
    F = mdl.fevd(Hf)                        # (Hf, m, m): F[H-1, i, j]
    for iname, i in zip(names, range(3)):
        arr = np.array(fevd_c[iname])       # rows H, cols j (shock)
        assert np.max(np.abs(arr - F[:, i, :])) < 0.05


def test_oirf0_is_cholesky():
    sigma = np.array([[1.0, 0.5], [0.5, 2.0]])
    O = irfmod.oirf(phi=np.zeros((1, 2, 2)), theta=np.zeros((0, 2, 2)),
                    sigma=sigma, horizon=3)
    assert np.allclose(O[0] @ O[0].T, sigma)        # OIRF[0] = chol(sigma)


def test_fevd_rows_sum_100():
    sigma = np.array([[1.0, 0.3], [0.3, 1.5]])
    F = irfmod.fevd(phi=np.array([[[0.5, 0.1], [0.0, 0.4]]]),
                    theta=np.zeros((0, 2, 2)), sigma=sigma, horizon=8)
    assert np.allclose(F.sum(axis=2), 100.0)
