"""BUG-0032: DRVARMA_NO_ENGINE asks for the port at every entry point, not only
in estimate_w: elf_c and marma_c (the ladder's, and sima's) computed in C while
engine_backend() answered "python"."""
import numpy as np
import pytest

import drvarma
from drvarma import _engine, _as311


def test_la_variable_llega_a_elf_c_y_marma_c(monkeypatch):
    m, n, p = 2, 60, 1
    w = np.random.default_rng(32).normal(size=(n, m))
    phi = np.full((p, m, m), 0.05)
    called = []
    real = _as311.elf

    def spy(*a, **k):
        called.append("port")
        return real(*a, **k)

    monkeypatch.setattr(_as311, "elf", spy)
    monkeypatch.setenv("DRVARMA_NO_ENGINE", "1")
    assert drvarma.engine_backend() == "python"
    assert "DRVARMA_NO_ENGINE" in drvarma.engine_load_error()
    _engine.elf_c(m, n, p, 0, np.zeros(m), phi, np.zeros((0, m, m)), np.eye(m), w)
    assert called == ["port"]
    lg, *_ = _engine.marma_c(m, n, p, 0, np.zeros(m), phi, np.zeros((0, m, m)),
                             np.eye(m), w)
    assert np.isfinite(lg)


def test_sin_la_variable_elf_c_usa_el_motor(monkeypatch):
    pytest.importorskip("drvarma._drvarma_engine")
    monkeypatch.delenv("DRVARMA_NO_ENGINE", raising=False)
    assert drvarma.engine_backend() == "c"
    assert drvarma.engine_load_error() is None
