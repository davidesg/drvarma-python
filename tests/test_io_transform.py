"""P0 tests: multivariate .inp round-trip and the Box-Cox/differencing transform.

Validates against the real reference file from the C repo (IPC3.inp).
"""
import os
import numpy as np
import pytest

from drvarma import load, save, MultiSeries, InpSpec, transform

# reference .inp from the C engine repo (sibling directory)
C_REPO = os.path.join(os.path.dirname(__file__), "..", "..",
                      "drvarma_v.04.1", "data", "models_group1")
IPC3 = os.path.join(C_REPO, "IPC3.inp")


@pytest.mark.skipif(not os.path.exists(IPC3), reason="reference IPC3.inp not found")
def test_load_ipc3():
    s, spec = load(IPC3)
    assert (s.m, s.nobs, s.freq) == (3, 216, 12)
    assert s.start == (2002, 1)
    assert s.names == ["IPC_ES", "IPC_FR", "IPC_DE"]
    assert (spec.lam, spec.d, spec.D) == (0.0, 1, 0)
    # first/last data rows
    assert np.allclose(s.data[0], [69.53, 81.94, 77.70])
    assert np.allclose(s.data[-1], [98.096, 104.98, 100.0])


@pytest.mark.skipif(not os.path.exists(IPC3), reason="reference IPC3.inp not found")
def test_roundtrip(tmp_path):
    s, spec = load(IPC3)
    out = tmp_path / "rt.inp"
    save(str(out), s, spec)
    s2, spec2 = load(str(out))
    assert (s2.m, s2.nobs) == (s.m, s.nobs)
    assert s2.names == s.names
    assert (spec2.lam, spec2.d, spec2.D) == (spec.lam, spec.d, spec.D)
    assert np.allclose(s2.data, s.data, atol=1e-4)


@pytest.mark.skipif(not os.path.exists(IPC3), reason="reference IPC3.inp not found")
def test_transform_dlog_scale100():
    s, spec = load(IPC3)
    w, bc = transform.transform(s.data, lam=0.0, d=1, D=0, s=12, scale=100.0)
    assert w.shape == (s.nobs - 1, s.m)
    # first differenced value of series 1: 100*(log(69.59)-log(69.53))
    expected = 100.0 * (np.log(69.59) - np.log(69.53))
    assert abs(w[0, 0] - expected) < 1e-9


def test_difference_orders():
    x = np.arange(1, 11, dtype=float).reshape(-1, 1)  # 1..10
    assert np.allclose(transform.difference(x, d=1), 1.0)          # first diff = 1
    assert transform.difference(x, d=1).shape == (9, 1)
    assert transform.difference(x, d=1, D=1, s=3).shape == (9 - 3, 1)


def test_boxcox_roundtrip():
    x = np.array([1.0, 5.0, 20.0, 100.0])
    for lam in (0.0, 0.5, 1.0):
        assert np.allclose(transform.boxcox_inv(transform.boxcox_fwd(x, lam), lam), x)
