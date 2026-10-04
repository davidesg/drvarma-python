"""BUG-0004 — the pure-Python fallback estimates the full VARMA(p,q), not only
q=0: on a VARMA(1,1) it gives the C engine's logelf and its iterations."""
import numpy as np
import pytest

from drvarma import _engine
from drvarma.estimate_py import estimate_w_py

pytest.importorskip("drvarma._drvarma_engine")


def _varma11(n=300, seed=3):
    rng = np.random.default_rng(seed)
    phi = np.array([[0.5, 0.1], [0.0, 0.3]])
    theta = np.array([[0.4, 0.0], [0.2, 0.3]])
    a = rng.standard_normal((n + 50, 2))
    w = np.zeros_like(a)
    for t in range(1, len(a)):
        w[t] = phi @ w[t - 1] + a[t] - theta @ a[t - 1]
    return w[50:]


def test_el_respaldo_estima_el_varma_completo(monkeypatch):
    monkeypatch.delenv("DRVARMA_NO_ENGINE", raising=False)
    w = _varma11()
    c = _engine.estimate_w(w, 1, 1)
    py = estimate_w_py(w, 1, 1)
    assert c["ifault"] == 0 and py["ifault"] == 0
    assert py["logelf"] == pytest.approx(c["logelf"], abs=1e-8)
    assert py["nit"] == c["nit"]
    assert np.abs(py["theta"]).max() > 0.05     # the MA part is estimated
